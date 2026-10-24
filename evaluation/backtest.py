"""
evaluation/backtest.py -- Walk-forward backtest of the base xP model.

Why this exists: the only accuracy number this project has ever reported
(metrics.json: RMSE 2.16) is one global figure over a small, early-season
slice. A single global RMSE is dominated by the ~60% of player-gameweeks
that are zeros and blanks -- easy to predict, worth nothing to a manager
deciding a starting XI -- and says nothing about whether the model can
find the players who actually haul. This script measures that properly,
*before* touching the model itself, so any future change has a real
baseline to beat.

Scope, stated plainly: this evaluates `raw_xp` -- the bare trained
LinearRegression in Models/linear_reg_v1.pkl -- not `final_xp` (the full
rule-adjusted number the live app shows). Reconstructing the expert rule
layer's historical context (fixture difficulty, rotation risk, DEFCON rate
as they stood *for that specific past gameweek*) is a separate, larger
undertaking than this. The same scoping call was already made for the
in-app Player History tab and the recap feature it replaced -- see
src/predictor.py::make_predictions and app.py's Season Points Explorer.

Data and leakage: all of it comes from src/features.py::fetch_full_history(),
unmodified. That function already (a) sums both legs of a double gameweek
instead of dropping one (fixed once, in the CSV pipeline itself -- reusing
it here rather than re-parsing CSVs is what keeps that fix in force), and
(b) computes every rolling_3_* feature via `.shift(1).rolling(3)` grouped by
player+season, so a gameweek's features only ever depend on strictly
earlier gameweeks. This script adds no feature engineering of its own and
therefore cannot reintroduce the leakage bug the old Backtest tab had --
it just slices that already-correct walk-forward table by gameweek and
predicts.

Usage:
    python evaluation/backtest.py --season 2025-26 --model-name linear_reg_v1
"""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features import fetch_full_history, fetch_full_history_with_understat, ROLLING_HORIZONS, ROLLED_PLAYER_STATS, ROLLED_TEAM_STATS
from src.predictor import load_model, load_position_models
from src.understat import PLAYER_STATS as US_PLAYER_STATS, TEAM_STATS as US_TEAM_STATS

FEATURES = [
    'rolling_3_minutes', 'rolling_3_ict_index', 'rolling_3_creativity',
    'rolling_3_influence', 'rolling_3_threat', 'rolling_3_total_points', 'value'
]

# Named, cumulative feature sets for the (a)/(b)/(c) expansion in
# src/features.py, so each step can be retrained and backtested the same
# way and compared like-for-like. Built from the same ROLLING_HORIZONS /
# ROLLED_*_STATS constants src/features.py uses to generate the columns,
# so this list can't silently drift out of sync with what actually exists.
MULTI_HORIZON_FEATURES = [
    f'rolling_{w}_{s}' for s in ROLLED_PLAYER_STATS for w in ROLLING_HORIZONS
] + ['value']

TEAM_OPP_FEATURE_COLS = (
    [f'team_rolling_{w}_{s}' for s in ROLLED_TEAM_STATS for w in ROLLING_HORIZONS] + ['league_rank']
)
TEAM_OPP_FEATURES = (
    MULTI_HORIZON_FEATURES
    + TEAM_OPP_FEATURE_COLS
    + [f'opp_{c}' for c in TEAM_OPP_FEATURE_COLS]
)

# Step (c): Understat features, on top of everything in step_b. Column
# names must match src/understat.py::build_understat_rolling_features
# exactly ('us_rolling_*' for player stats, 'us_team_rolling_*'/
# 'opp_us_team_rolling_*' for team/opponent) -- built from US_PLAYER_STATS/
# US_TEAM_STATS imported directly from that module for the same reason
# MULTI_HORIZON_FEATURES imports from src.features: so this can't silently
# drift out of sync with what the fetch function actually produces.
UNDERSTAT_PLAYER_FEATURES = [f'us_rolling_{w}_{s}' for s in US_PLAYER_STATS for w in ROLLING_HORIZONS]
UNDERSTAT_TEAM_FEATURE_COLS = [f'us_team_rolling_{w}_{s}' for s in US_TEAM_STATS for w in ROLLING_HORIZONS]
UNDERSTAT_FEATURES = (
    TEAM_OPP_FEATURES
    + UNDERSTAT_PLAYER_FEATURES
    + UNDERSTAT_TEAM_FEATURE_COLS
    + [f'opp_{c}' for c in UNDERSTAT_TEAM_FEATURE_COLS]
)

FEATURE_SETS = {
    'baseline7': FEATURES,                    # original 7, retrained on fetch_full_history's own train split
    'step_a_multihorizon': MULTI_HORIZON_FEATURES,
    'step_b_team_opp': TEAM_OPP_FEATURES,
    'step_c_understat': UNDERSTAT_FEATURES,
}

BUCKET_ORDER = ['Zeros', 'Blanks', 'Tickers', 'Haulers']
POSITION_ORDER = ['GKP', 'DEF', 'MID', 'FWD']


def bucket_outcome(minutes, points):
    """
    Zeros:   did not play (minutes == 0) -- always 0 points in practice.
    Blanks:  played, <= 2 points.
    Tickers: played, 3-4 points.
    Haulers: played, >= 5 points.
    """
    if minutes <= 0:
        return 'Zeros'
    if points <= 2:
        return 'Blanks'
    if points <= 4:
        return 'Tickers'
    return 'Haulers'


def _rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def _mae(y_true, y_pred):
    return float(mean_absolute_error(y_true, y_pred))


def _metrics_by(df, group_col, order=None):
    rows = []
    for key, g in df.groupby(group_col):
        rows.append({
            group_col: key,
            'n': int(len(g)),
            'rmse': _rmse(g['actual'], g['predicted']),
            'mae': _mae(g['actual'], g['predicted']),
        })
    if order:
        rank = {k: i for i, k in enumerate(order)}
        rows.sort(key=lambda r: rank.get(r[group_col], len(order)))
    return rows


def run_backtest(season, model_name="linear_reg_v1", fetch_fn=fetch_full_history):
    """
    Backtests the shipped Models/linear_reg_v1.pkl as-is -- the original
    "how good is what's actually deployed" measurement. See
    train_and_backtest() for evaluating a candidate feature set that isn't
    (yet) the shipped model.
    """
    model = load_model()
    if model is None:
        raise RuntimeError("Could not load Models/linear_reg_v1.pkl")

    df_hist = fetch_fn()
    if df_hist.empty:
        raise RuntimeError("fetch_full_history() returned no data -- check network access")

    season_df = df_hist[df_hist['season'] == season].copy()
    if season_df.empty:
        available = sorted(df_hist['season'].unique())
        raise RuntimeError(f"No historical data for season {season!r}. Available: {available}")

    season_df['position'] = season_df['position'].replace({'GK': 'GKP'})

    # Walk-forward is already guaranteed by fetch_full_history's rolling_3_*
    # columns (see module docstring) -- this is just a predict() over rows
    # that already only encode prior-gameweek information.
    season_df = season_df.dropna(subset=FEATURES).reset_index(drop=True)
    X = season_df[FEATURES].fillna(0)
    season_df['predicted'] = model.predict(X)
    season_df['actual'] = season_df['total_points']

    scope_note = (
        "Evaluates raw_xp (the bare LinearRegression model in "
        "Models/linear_reg_v1.pkl), not final_xp (the full rule-adjusted "
        "prediction shown in the live app). Reconstructing the expert "
        "rule layer's historical context is out of scope -- see the "
        "module docstring in evaluation/backtest.py."
    )
    return _evaluate_predictions(season_df, season, model_name, len(FEATURES), scope_note)


def run_position_ensemble_backtest(season, model_name="position_models_v1", fetch_fn=fetch_full_history):
    """
    Backtests the current production model -- the per-position
    XGBoost + RandomForest median ensemble in Models/position_models_v1.pkl
    (see training/train_position_models.py) -- the same way run_backtest()
    scores the legacy single LinearRegression, so the two are directly
    comparable via print_side_by_side() below. Routes each row to its
    position's own ensemble rather than a single model.predict() call.
    """
    bundle = load_position_models()
    if bundle is None:
        raise RuntimeError(
            "Could not load Models/position_models_v1.pkl -- run "
            "training/train_position_models.py first"
        )

    df_hist = fetch_fn()
    if df_hist.empty:
        raise RuntimeError("fetch_full_history() returned no data -- check network access")

    season_df = df_hist[df_hist['season'] == season].copy()
    if season_df.empty:
        available = sorted(df_hist['season'].unique())
        raise RuntimeError(f"No historical data for season {season!r}. Available: {available}")

    season_df['position'] = season_df['position'].replace({'GK': 'GKP'})
    feature_cols = bundle['features']
    season_df = season_df.dropna(subset=feature_cols).reset_index(drop=True)

    predicted = pd.Series(np.nan, index=season_df.index)
    for position, ensemble in bundle['positions'].items():
        mask = season_df['position'] == position
        if not mask.any():
            continue
        X = season_df.loc[mask, feature_cols].fillna(0)
        predicted.loc[mask] = ensemble.predict(X)
    unmatched_positions = int(predicted.isna().sum())
    if unmatched_positions:
        # A row whose position has no trained ensemble (shouldn't happen --
        # GKP/DEF/MID/FWD are exhaustive -- but 0.0 is the same safe
        # fallback make_predictions() uses elsewhere for the unexpected
        # case, not a silent crash).
        predicted = predicted.fillna(0.0)
    season_df['predicted'] = predicted
    season_df['actual'] = season_df['total_points']

    scope_note = (
        "Evaluates the production per-position ensemble (XGBoost + "
        "RandomForest, median-blended, one model per GKP/DEF/MID/FWD, "
        "trained with OpenFPL-style balanced sample weighting) against raw "
        "total_points -- same scope as run_backtest() (not final_xp; see "
        "that function's docstring)."
    )
    return _evaluate_predictions(season_df, season, model_name, len(feature_cols), scope_note)


def run_component_backtest(season, model_name="component_models_v1", fetch_fn=fetch_full_history):
    """
    Backtests the component prediction pipeline (src/components.py,
    Models/component_models_v1.pkl -- see
    training/train_component_models.py) the same way run_position_ensemble_
    backtest() scores the per-position ensemble, so the two are directly
    comparable via print_side_by_side(). Calls
    src/components.py::predict_all_components() directly -- the exact same
    function the live app calls (src/predictor.py::_make_predictions_
    component) -- so "what gets backtested" and "what the app predicts"
    can never drift apart.
    """
    from src.components import predict_all_components
    from src.predictor import load_component_models

    bundle = load_component_models()
    if bundle is None:
        raise RuntimeError(
            "Could not load Models/component_models_v1.pkl -- run "
            "training/train_component_models.py first"
        )

    df_hist = fetch_fn()
    if df_hist.empty:
        raise RuntimeError("fetch_full_history() returned no data -- check network access")

    season_df = df_hist[df_hist['season'] == season].copy()
    if season_df.empty:
        available = sorted(df_hist['season'].unique())
        raise RuntimeError(f"No historical data for season {season!r}. Available: {available}")

    season_df['position'] = season_df['position'].replace({'GK': 'GKP'})
    season_df = season_df.reset_index(drop=True)

    components = predict_all_components(season_df, bundle)
    season_df['predicted'] = components['raw_xp'].values
    season_df['actual'] = season_df['total_points']

    scope_note = (
        "Evaluates the component prediction pipeline's raw_xp (predict_"
        "all_components()'s summed output -- minutes, attacking, clean "
        "sheet, goals conceded, saves, DEFCON, bonus, cards) against raw "
        "total_points, same scope as run_position_ensemble_backtest() -- "
        "not final_xp (which additionally applies the injured/suspended/"
        "manual-override wipe; see src/predictor.py's docstring). CAVEAT: "
        "the bonus component (training/train_component_models.py::"
        "fit_bonus_model) is trained on this exact season, not held out --"
        " its contribution to the numbers below is in-sample, unlike every "
        "other component's. See that function's docstring for why."
    )
    return _evaluate_predictions(season_df, season, model_name, len(components.columns), scope_note)


_LAST_UNDERSTAT_MATCH_REPORT = {}


def _fetch_with_understat():
    """
    Adapts fetch_full_history_with_understat()'s (df, match_report) return
    into the plain-df signature train_and_backtest()'s fetch_fn expects,
    stashing the match report where main()/print_report() can pick it up
    and print/verify it -- the 90% match-rate gate has to be checked
    against the exact same fetch this backtest run used, not a separate
    one-off script.
    """
    df, report = fetch_full_history_with_understat()
    _LAST_UNDERSTAT_MATCH_REPORT.clear()
    _LAST_UNDERSTAT_MATCH_REPORT.update(report)
    return df


def train_and_backtest(feature_set_name, test_season="2025-26", train_seasons=None,
                        model_name=None, fetch_fn=None):
    """
    Trains a fresh LinearRegression on `feature_set_name` (see FEATURE_SETS)
    over `train_seasons` and backtests it on `test_season`, via the exact
    same bucket/position/rank-correlation scoring as run_backtest().

    This is deliberately NOT the shipped Models/linear_reg_v1.pkl and does
    not touch it -- it exists to answer "if we retrained on this expanded
    feature set, would it help", per step, before any decision to actually
    change the shipped model. Algorithm is held fixed at LinearRegression
    across every feature-set step specifically so a delta between steps
    reflects the features, not a confounded algorithm change.

    Note: fetch_full_history() only carries 2022-23 onward, so this train
    split (default 2022-23 to 2024-25) is one season shorter than the
    shipped model's own training data (which also had 2021-22). Results
    from this function are therefore comparable *to each other* across
    feature-set steps, but not a perfect apples-to-apples reproduction of
    the shipped pickle's own training run -- use the 'baseline7' feature
    set as the fair reference point for the other steps, not the shipped
    model's previously-recorded numbers.
    """
    if train_seasons is None:
        train_seasons = ['2022-23', '2023-24', '2024-25']
    if model_name is None:
        model_name = feature_set_name
    if fetch_fn is None:
        fetch_fn = fetch_full_history

    feature_cols = FEATURE_SETS[feature_set_name]

    df_hist = fetch_fn()
    if df_hist.empty:
        raise RuntimeError("fetch_full_history() returned no data -- check network access")

    missing = [c for c in feature_cols if c not in df_hist.columns]
    if missing:
        raise RuntimeError(f"Feature set {feature_set_name!r} references column(s) not produced by fetch_full_history(): {missing}")

    train_df = df_hist[df_hist['season'].isin(train_seasons)].dropna(subset=feature_cols).copy()
    test_df = df_hist[df_hist['season'] == test_season].dropna(subset=feature_cols).reset_index(drop=True).copy()
    if train_df.empty or test_df.empty:
        raise RuntimeError(f"Empty train or test split for train_seasons={train_seasons}, test_season={test_season!r}")

    test_df['position'] = test_df['position'].replace({'GK': 'GKP'})

    model = LinearRegression()
    model.fit(train_df[feature_cols].fillna(0), train_df['total_points'])
    test_df['predicted'] = model.predict(test_df[feature_cols].fillna(0))
    test_df['actual'] = test_df['total_points']

    scope_note = (
        f"Retrained LinearRegression on feature set '{feature_set_name}' "
        f"({len(feature_cols)} features), train seasons {train_seasons}, "
        f"NOT the shipped Models/linear_reg_v1.pkl. Evaluates the model's "
        f"own raw prediction, not a rule-adjusted final_xp -- see "
        f"train_and_backtest() docstring in evaluation/backtest.py."
    )
    return _evaluate_predictions(test_df, test_season, model_name, len(feature_cols), scope_note)


def _evaluate_predictions(season_df, season, model_name, n_features, scope_note):
    """
    Shared scoring path for both run_backtest (shipped pickle) and
    train_and_backtest (retrained on a candidate feature set) -- takes a
    season_df that already has 'predicted' and 'actual' columns and
    produces the exact same results dict shape either way, so different
    steps/models are always comparable apples-to-apples.
    """
    season_df = season_df.copy()
    season_df['bucket'] = [
        bucket_outcome(m, p) for m, p in zip(season_df['minutes'], season_df['actual'])
    ]

    # --- Ranking quality, computed per gameweek then averaged ---
    # Absolute error is dominated by scale (a hauler's error dwarfs a
    # zero's); rank correlation and top-10 overlap measure the thing this
    # app is actually used for -- "who should I pick this week" -- which
    # cares about relative ordering, not point-for-point accuracy.
    spearman_per_gw = []
    top10_overlap_per_gw = []
    for gw, g in season_df.groupby('GW'):
        if len(g) < 10 or g['actual'].nunique() < 2 or g['predicted'].nunique() < 2:
            continue
        rho, _ = spearmanr(g['predicted'], g['actual'])
        if not np.isnan(rho):
            spearman_per_gw.append(float(rho))
        top_pred = set(g.nlargest(10, 'predicted').index)
        top_actual = set(g.nlargest(10, 'actual').index)
        top10_overlap_per_gw.append(len(top_pred & top_actual))

    results = {
        'season': season,
        'model_name': model_name,
        'generated_at_utc': datetime.now(timezone.utc).isoformat(),
        'n_gameweeks': int(season_df['GW'].nunique()),
        'n_player_gameweeks': int(len(season_df)),
        'n_features': n_features,
        'overall': {
            'n': int(len(season_df)),
            'rmse': _rmse(season_df['actual'], season_df['predicted']),
            'mae': _mae(season_df['actual'], season_df['predicted']),
        },
        'by_bucket': _metrics_by(season_df, 'bucket', BUCKET_ORDER),
        'by_position': _metrics_by(season_df, 'position', POSITION_ORDER),
        'spearman_rank_correlation_mean_per_gw': (
            float(np.mean(spearman_per_gw)) if spearman_per_gw else None
        ),
        'spearman_n_gameweeks': len(spearman_per_gw),
        'top10_overlap_mean_per_gw': (
            float(np.mean(top10_overlap_per_gw)) if top10_overlap_per_gw else None
        ),
        'top10_overlap_n_gameweeks': len(top10_overlap_per_gw),
        'bucket_definitions': {
            'Zeros': 'minutes == 0 (did not play)',
            'Blanks': 'played, actual points <= 2',
            'Tickers': 'played, actual points 3-4',
            'Haulers': 'actual points >= 5',
        },
        'scope_note': scope_note,
        'reference_published_one_gw_ahead_rmse': {
            'note': (
                "For context only -- NOT a direct comparison until this "
                "harness itself is verified against these. Different "
                "leagues/seasons/eligible-player pools are not guaranteed "
                "to line up with the buckets here."
            ),
            'openfpl_arxiv_2508.09992': {'Zeros': 0.818, 'Blanks': 1.291, 'Tickers': 1.517, 'Haulers': 5.142},
            'fpl_review_paid_model': {'Zeros': 0.689, 'Blanks': 1.189, 'Tickers': 1.594, 'Haulers': 5.172},
        },
    }
    return results, season_df


def print_report(results):
    print("=" * 78)
    print(f" BACKTEST: {results['model_name']}  |  season {results['season']}")
    print("=" * 78)
    print(f"Gameweeks: {results['n_gameweeks']}   Player-gameweeks: {results['n_player_gameweeks']}")
    print()
    print(f"{'':10s} {'n':>8s} {'RMSE':>8s} {'MAE':>8s}")
    o = results['overall']
    print(f"{'Overall':10s} {o['n']:>8d} {o['rmse']:>8.3f} {o['mae']:>8.3f}")
    print()
    print("By outcome bucket:")
    for r in results['by_bucket']:
        print(f"  {r['bucket']:10s} {r['n']:>8d} {r['rmse']:>8.3f} {r['mae']:>8.3f}")
    print()
    print("By position:")
    for r in results['by_position']:
        print(f"  {r['position']:10s} {r['n']:>8d} {r['rmse']:>8.3f} {r['mae']:>8.3f}")
    print()
    sp = results['spearman_rank_correlation_mean_per_gw']
    if sp is not None:
        print(f"Spearman rank correlation (mean per GW, n={results['spearman_n_gameweeks']}): {sp:.3f}")
    else:
        print("Spearman rank correlation: n/a (not enough qualifying gameweeks)")
    ov = results['top10_overlap_mean_per_gw']
    if ov is not None:
        print(f"Top-10 overlap, predicted vs actual (mean per GW, n={results['top10_overlap_n_gameweeks']}): {ov:.1f} / 10")
    else:
        print("Top-10 overlap: n/a")
    print("=" * 78)
    print("Reference only, not yet compared (see scope_note in the JSON):")
    ref = results['reference_published_one_gw_ahead_rmse']
    for label in ('openfpl_arxiv_2508.09992', 'fpl_review_paid_model'):
        vals = ref[label]
        print(f"  {label:28s} " + " / ".join(f"{k} {v}" for k, v in vals.items()))


# One-line description per model_name, printed alongside it in
# print_side_by_side() below -- looked up rather than hardcoded per call
# site, since this function is now used for two different comparisons
# (legacy-vs-position-ensemble, position-ensemble-vs-component).
_MODEL_DESCRIPTIONS = {
    'linear_reg_v1': 'single all-positions LinearRegression',
    'position_models_v1': 'per-position XGBoost+RandomForest median ensemble',
    'component_models_v1': 'component pipeline (minutes/attack/CS/DEFCON/bonus/cards summed)',
}


def print_side_by_side(old_results, new_results):
    """
    Side-by-side bucket/position table for two backtest results from
    _evaluate_predictions() -- used both for the legacy-LinearRegression-
    vs-position-ensemble comparison and the position-ensemble-vs-
    component-pipeline comparison (see run_component_backtest()).

    The bar for a model-architecture comparison here is Haulers and
    Tickers RMSE, not the overall number: a design built to stop
    predicting everyone as roughly average (balanced sample weighting,
    component-level prediction) is expected to trade some Zeros/overall
    accuracy for it. If Zeros RMSE ticks up while Haulers/Tickers improve,
    that is the expected, correct trade -- this function reports it
    plainly rather than only surfacing the metric that happens to look
    better.
    """
    old_desc = _MODEL_DESCRIPTIONS.get(old_results['model_name'], '')
    new_desc = _MODEL_DESCRIPTIONS.get(new_results['model_name'], '')
    print("=" * 100)
    print(f" MODEL COMPARISON  |  season {old_results['season']}")
    print(f" OLD: {old_results['model_name']}" + (f"  ({old_desc})" if old_desc else ""))
    print(f" NEW: {new_results['model_name']}" + (f"  ({new_desc})" if new_desc else ""))
    print("=" * 100)
    header = f"{'':10s} {'n':>7s} | {'OLD RMSE':>9s} {'NEW RMSE':>9s} {'d RMSE':>8s} | {'OLD MAE':>9s} {'NEW MAE':>9s} {'d MAE':>8s}"
    print(header)

    def _row(label, o, n):
        d_rmse = n['rmse'] - o['rmse']
        d_mae = n['mae'] - o['mae']
        print(
            f"{label:10s} {o['n']:>7d} | {o['rmse']:>9.3f} {n['rmse']:>9.3f} {d_rmse:>+8.3f} | "
            f"{o['mae']:>9.3f} {n['mae']:>9.3f} {d_mae:>+8.3f}"
        )

    _row('Overall', old_results['overall'], new_results['overall'])
    print()
    print("By outcome bucket (bar: Haulers, Tickers):")
    old_by_bucket = {r['bucket']: r for r in old_results['by_bucket']}
    new_by_bucket = {r['bucket']: r for r in new_results['by_bucket']}
    for b in BUCKET_ORDER:
        if b in old_by_bucket and b in new_by_bucket:
            _row(b, old_by_bucket[b], new_by_bucket[b])
    print()
    print("By position:")
    old_by_pos = {r['position']: r for r in old_results['by_position']}
    new_by_pos = {r['position']: r for r in new_results['by_position']}
    for p in POSITION_ORDER:
        if p in old_by_pos and p in new_by_pos:
            _row(p, old_by_pos[p], new_by_pos[p])
    print()
    sp_old, sp_new = old_results['spearman_rank_correlation_mean_per_gw'], new_results['spearman_rank_correlation_mean_per_gw']
    ov_old, ov_new = old_results['top10_overlap_mean_per_gw'], new_results['top10_overlap_mean_per_gw']
    print(f"Spearman rank corr (mean/GW): OLD {sp_old:.3f}  NEW {sp_new:.3f}" if sp_old is not None and sp_new is not None else "Spearman rank corr: n/a")
    print(f"Top-10 overlap (mean/GW):     OLD {ov_old:.1f}/10  NEW {ov_new:.1f}/10" if ov_old is not None and ov_new is not None else "Top-10 overlap: n/a")
    print("=" * 100)
    print(
        "Judge this by Haulers and Tickers RMSE. A Zeros/overall RMSE\n"
        "regression alongside Haulers/Tickers improvement is the expected,\n"
        "correct trade-off for a model built to stop predicting everyone as\n"
        "average -- not a bug to chase back down."
    )


def main():
    parser = argparse.ArgumentParser(description="Walk-forward backtest of the base xP model, bucketed by outcome.")
    parser.add_argument('--season', default='2025-26', help="Test season label as used in src/features.py (e.g. 2025-26)")
    parser.add_argument('--model-name', default=None, help="Label used in the output filename and JSON")
    parser.add_argument(
        '--compare', action='store_true',
        help="Backtest both Models/linear_reg_v1.pkl and Models/position_models_v1.pkl on --season and print them side by side."
    )
    parser.add_argument(
        '--compare-components', action='store_true',
        help="Backtest Models/position_models_v1.pkl (before) against the component pipeline, Models/component_models_v1.pkl (after), on --season and print them side by side."
    )
    parser.add_argument(
        '--feature-set', default=None, choices=sorted(FEATURE_SETS.keys()),
        help="If set, retrains a fresh LinearRegression on this feature set instead of backtesting the shipped pickle."
    )
    parser.add_argument(
        '--train-seasons', default='2022-23,2023-24,2024-25',
        help="Comma-separated seasons to train on when --feature-set is used."
    )
    args = parser.parse_args()

    if args.compare:
        results_old, _ = run_backtest(args.season, 'linear_reg_v1')
        results_new, _ = run_position_ensemble_backtest(args.season, 'position_models_v1')
        out_dir = PROJECT_ROOT / "evaluation" / "results"
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / f"{args.season}_linear_reg_v1.json", 'w') as f:
            json.dump(results_old, f, indent=2)
        with open(out_dir / f"{args.season}_position_models_v1.json", 'w') as f:
            json.dump(results_new, f, indent=2)
        print_side_by_side(results_old, results_new)
        print(f"\nResults written to {out_dir}/{args.season}_linear_reg_v1.json and {out_dir}/{args.season}_position_models_v1.json")
        return

    if args.compare_components:
        results_before, _ = run_position_ensemble_backtest(args.season, 'position_models_v1')
        results_after, _ = run_component_backtest(args.season, 'component_models_v1')
        out_dir = PROJECT_ROOT / "evaluation" / "results"
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / f"{args.season}_position_models_v1.json", 'w') as f:
            json.dump(results_before, f, indent=2)
        with open(out_dir / f"{args.season}_component_models_v1.json", 'w') as f:
            json.dump(results_after, f, indent=2)
        print_side_by_side(results_before, results_after)
        print(f"\nResults written to {out_dir}/{args.season}_position_models_v1.json and {out_dir}/{args.season}_component_models_v1.json")
        return

    if args.feature_set:
        model_name = args.model_name or args.feature_set
        train_seasons = [s.strip() for s in args.train_seasons.split(',')]
        # step_c_understat needs the Understat-enriched fetch; every other
        # feature set uses fetch_full_history via train_and_backtest's own
        # default so this stays a no-op for steps (a)/(b).
        fetch_fn = _fetch_with_understat if args.feature_set == 'step_c_understat' else None
        results, _ = train_and_backtest(args.feature_set, args.season, train_seasons, model_name, fetch_fn=fetch_fn)
        if args.feature_set == 'step_c_understat' and _LAST_UNDERSTAT_MATCH_REPORT:
            print("Understat match rates (must clear 90% gate):")
            worst = 1.0
            for season, r in _LAST_UNDERSTAT_MATCH_REPORT.items():
                if 'error' in r:
                    print(f"  {season}: ERROR -- {r['error']}")
                    worst = 0.0
                    continue
                pr, tr = r.get('player_match_rate', 0.0), r.get('team_match_rate', 0.0)
                print(f"  {season}: player={pr:.1%}  team={tr:.1%}")
                worst = min(worst, pr, tr)
            print(f"  -> {'PASS' if worst >= 0.90 else 'FAIL'} (worst match rate {worst:.1%} {'>=' if worst >= 0.90 else '<'} 90%)")
            print()
    else:
        model_name = args.model_name or 'linear_reg_v1'
        results, _ = run_backtest(args.season, model_name)

    out_dir = PROJECT_ROOT / "evaluation" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.season}_{model_name}.json"
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)

    print_report(results)
    print(f"\nResults written to {out_path}")


if __name__ == "__main__":
    main()
