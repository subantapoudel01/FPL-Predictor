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
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features import fetch_full_history
from src.predictor import load_model

FEATURES = [
    'rolling_3_minutes', 'rolling_3_ict_index', 'rolling_3_creativity',
    'rolling_3_influence', 'rolling_3_threat', 'rolling_3_total_points', 'value'
]

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
    Returns (results_dict, season_df). Exposed as a function (not just a
    __main__ script) so a future model version can be evaluated the same
    way by passing a different model_name / model object without editing
    this file.
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
        'scope_note': (
            "Evaluates raw_xp (the bare LinearRegression model in "
            "Models/linear_reg_v1.pkl), not final_xp (the full rule-adjusted "
            "prediction shown in the live app). Reconstructing the expert "
            "rule layer's historical context is out of scope -- see the "
            "module docstring in evaluation/backtest.py."
        ),
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


def main():
    parser = argparse.ArgumentParser(description="Walk-forward backtest of the base xP model, bucketed by outcome.")
    parser.add_argument('--season', default='2025-26', help="Season label as used in src/features.py (e.g. 2025-26)")
    parser.add_argument('--model-name', default='linear_reg_v1', help="Label used in the output filename and JSON")
    args = parser.parse_args()

    results, _ = run_backtest(args.season, args.model_name)

    out_dir = PROJECT_ROOT / "evaluation" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.season}_{args.model_name}.json"
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)

    print_report(results)
    print(f"\nResults written to {out_path}")


if __name__ == "__main__":
    main()
