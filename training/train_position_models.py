"""
training/train_position_models.py -- trains the four per-position ensembles
(GKP/DEF/MID/FWD, XGBoost + RandomForest, median-blended) that replace the
single all-positions LinearRegression previously shipped as
Models/linear_reg_v1.pkl.

Why per-position: a goalkeeper's points come overwhelmingly from saves and
clean sheets, a forward's from xG and shot volume -- one linear model
averaged across all four positions can't represent both relationships, and
in practice minutes-played alone recovers ~91% of that single model's R^2
(it's mostly learning "did this player start", not "how well did they
play" once they did). Splitting by position lets each model learn the
scoring pattern that actually applies to it.

Why XGBoost + RandomForest, not one model: two tree-based learners with
different bias/variance profiles disagree in different ways on the same
data; taking the median of their predictions is a cheap, robust way to
damp down whichever one is overfitting a quirk in the training data,
without the complexity of a full stacking model.

Why sample weights (see compute_position_sample_weights): without them, a
regressor trained on plain MSE has every incentive to predict close to the
mean for every player, because hauls (big point totals) are rare and each
one contributes little to the average loss even when badly missed -- that
collapse-to-the-mean failure mode is exactly what this replaces. Following
OpenFPL's approach: bin the target into position-specific quantile bins
(a class boundary, not a real class), balance sample weight against those
bins, clip at the 95th percentile, rescale to unit mean so the *average*
weight stays 1 and this only reshapes the loss surface rather than
inflating it.

Why team-grouped (not random) CV folds: a random split can put some of a
team's players in the training fold and others in the validation fold for
the same run of matches, so the validation score partly measures "did the
model memorize this team's current run of form" rather than "does it
generalize" -- GroupKFold on team keeps every row for a given team-season
entirely on one side of each split, so hyperparameters are chosen for
generalization, not memorization.

Why 2021-22..2024-25 to train, 2025-26 held out entirely: the shipped
linear_reg_v1.pkl was trained on the same four seasons (see
evaluation/backtest.py's train_and_backtest docstring for how it documents
its own, shorter split). 2025-26 is reserved for
evaluation/backtest.py's prospective evaluation and is never touched here
-- no feature computed from it, no fold drawn from it, no hyperparameter
chosen by looking at it.

Feature choice: trained on the same 7 features process_data() already
computes live for the deployed app (rolling_3_minutes/ict_index/
creativity/influence/threat/total_points, value) -- the same 7
linear_reg_v1.pkl used. The richer multi-horizon / team-opponent feature
set validated in evaluation/backtest.py's step_b_team_opp is NOT used here
on purpose: those columns only exist in fetch_full_history()'s historical
pipeline, not in the live per-gameweek process_data() pipeline (see
CLAUDE.md's field-whitelist warning -- shipping a model that expects a
feature the live app doesn't compute is exactly that bug class, just one
level up). Extending process_data() to compute those features live is a
separate, larger piece of work; this keeps today's ensemble a true,
safe drop-in replacement for the live app.

Usage:
    python training/train_position_models.py
    python training/train_position_models.py --n-iter 20 --cv-folds 5
"""
import argparse
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold, RandomizedSearchCV
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBRegressor

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ensemble import MedianEnsembleRegressor
from src.features import fetch_full_history

FEATURES = [
    'rolling_3_minutes', 'rolling_3_ict_index', 'rolling_3_creativity',
    'rolling_3_influence', 'rolling_3_threat', 'rolling_3_total_points', 'value'
]

TRAIN_SEASONS = ['2021-22', '2022-23', '2023-24', '2024-25']
EVAL_SEASON = '2025-26'  # never touched here -- see evaluation/backtest.py

POSITIONS = ['GKP', 'DEF', 'MID', 'FWD']

# OpenFPL-style: more bins for positions whose scoring distribution has
# more distinct "shapes" worth separating (MID covers everything from a
# defensive midfielder's clean-sheet points to a set-piece taker's hauls);
# GKP's scoring is close to bimodal (played a full match and kept a clean
# sheet, or didn't) so 2 bins is enough resolution to be worth balancing.
POSITION_BIN_COUNTS = {'GKP': 2, 'DEF': 3, 'MID': 4, 'FWD': 3}

XGB_PARAM_DIST = {
    'n_estimators': [100, 200, 300, 400],
    'max_depth': [3, 4, 5, 6],
    'learning_rate': [0.01, 0.03, 0.05, 0.1],
    'subsample': [0.7, 0.85, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'min_child_weight': [1, 3, 5, 10],
}

RF_PARAM_DIST = {
    'n_estimators': [150, 300, 450],
    # max_depth deliberately excludes None (unbounded): on DEF/MID's
    # 35k-45k rows an unbounded tree with min_samples_leaf=1 grows to
    # roughly one leaf per training row, which is both a textbook
    # overfit and, at n_estimators up to 450, the single biggest
    # contributor to a slow search (observed directly -- with None in
    # this list, one position's search didn't finish in 8+ minutes; capped
    # at 16, the same search completes in well under a minute per
    # position). It rarely wins on CV RMSE anyway, so the cap costs little.
    'max_depth': [4, 6, 8, 12, 16],
    'min_samples_leaf': [2, 5, 10, 20],
    'max_features': ['sqrt', 0.5, 0.8, 1.0],
}


def compute_position_sample_weights(y, n_bins):
    """
    OpenFPL-style sample weighting -- see module docstring. Bins the
    continuous point total into `n_bins` position-specific quantile bins,
    treats bin membership as a pseudo-class for
    compute_sample_weight("balanced"), clips at the 95th percentile, and
    rescales to unit mean.
    """
    y_arr = np.asarray(y, dtype=float).reshape(-1, 1)
    kbd = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='quantile')
    bins = kbd.fit_transform(y_arr).ravel().astype(int)
    weights = compute_sample_weight(class_weight='balanced', y=bins)
    cap = np.percentile(weights, 95)
    weights = np.minimum(weights, cap)
    weights = weights / weights.mean()
    return weights


def team_group_cv(groups, n_splits):
    """
    GroupKFold keyed on team -- see module docstring for why this has to
    be grouped, not random. Caps n_splits at the number of distinct teams
    actually present so a thin position (GKP has far fewer rows than MID,
    though usually still ~20+ distinct teams across 4 seasons) never asks
    for more folds than groups.
    """
    n_splits = max(2, min(n_splits, groups.nunique()))
    return n_splits, GroupKFold(n_splits=n_splits)


def tune_and_fit(estimator_cls, param_dist, X, y, weights, cv, n_iter, random_state, base_kwargs):
    """
    RandomizedSearchCV over the team-grouped folds (scored by plain,
    unweighted RMSE -- the sample weights bias what the model *fits* to,
    not how a candidate hyperparameter set is *judged*, so the search
    still selects for genuine generalization). Refits the winning
    hyperparameters once on the full training set, with weights, since
    RandomizedSearchCV's own refit path doesn't thread sample_weight
    through consistently across estimator types.
    """
    base = estimator_cls(**base_kwargs)
    search = RandomizedSearchCV(
        base, param_dist, n_iter=n_iter, cv=cv,
        scoring='neg_root_mean_squared_error', random_state=random_state,
        n_jobs=-1, refit=False,
    )
    search.fit(X, y, sample_weight=weights)
    best = estimator_cls(**{**base_kwargs, **search.best_params_})
    best.fit(X, y, sample_weight=weights)
    return best, search.best_params_, -search.best_score_


def train_position(pos_df, position, n_iter, cv_folds, random_state=42):
    X = pos_df[FEATURES].fillna(0)
    y = pos_df['total_points'].astype(float)
    groups = pos_df['team']

    weights = compute_position_sample_weights(y, POSITION_BIN_COUNTS[position])
    n_splits, gkf = team_group_cv(groups, cv_folds)
    cv = list(gkf.split(X, y, groups=groups))

    # n_jobs=1 on the base estimator, -1 on the outer RandomizedSearchCV:
    # parallelizing at both levels means every one of the search's own
    # worker processes ALSO spawns its own full set of threads for that
    # single fit, oversubscribing the machine's cores many times over and
    # making the whole search far *slower* than single-level parallelism,
    # not faster (observed directly -- a run with n_jobs=-1 on both levels
    # was still short of finishing GKP's search after 30+ minutes on a
    # 32-core machine; fixing this is what actually made runs complete).
    xgb_model, xgb_params, xgb_cv_rmse = tune_and_fit(
        XGBRegressor, XGB_PARAM_DIST, X, y, weights, cv, n_iter, random_state,
        base_kwargs=dict(random_state=random_state, n_jobs=1, tree_method='hist'),
    )
    rf_model, rf_params, rf_cv_rmse = tune_and_fit(
        RandomForestRegressor, RF_PARAM_DIST, X, y, weights, cv, n_iter, random_state,
        base_kwargs=dict(random_state=random_state, n_jobs=1),
    )

    ensemble = MedianEnsembleRegressor([xgb_model, rf_model], FEATURES)
    meta = {
        'n_rows': int(len(pos_df)), 'n_teams': int(groups.nunique()), 'cv_folds': n_splits,
        'xgb_best_params': xgb_params, 'xgb_cv_rmse': float(xgb_cv_rmse),
        'rf_best_params': rf_params, 'rf_cv_rmse': float(rf_cv_rmse),
    }
    return ensemble, meta


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--n-iter', type=int, default=12, help="RandomizedSearchCV iterations per model per position (default 12)")
    parser.add_argument('--cv-folds', type=int, default=5, help="Team-grouped CV folds, capped per-position at distinct teams present (default 5)")
    parser.add_argument('--out', default=None, help="Output path (default Models/position_models_v1.pkl)")
    parser.add_argument(
        '--cache-file', default=None,
        help=(
            "Optional local parquet path to cache fetch_full_history()'s "
            "result to (and reuse from, if it already exists) -- purely a "
            "convenience for re-running training against a flaky/slow "
            "upstream (raw.githubusercontent.com has no SLA) without "
            "re-fetching every time. Not used unless passed."
        ),
    )
    args = parser.parse_args()

    cache_path = Path(args.cache_file) if args.cache_file else None
    if cache_path and cache_path.exists():
        print(f"Loading cached historical data from {cache_path}...")
        df = pd.read_parquet(cache_path)
    else:
        print(f"Fetching historical data ({', '.join(TRAIN_SEASONS)} for training; {EVAL_SEASON} held out)...")
        t0 = time.time()
        df = fetch_full_history()
        if df.empty:
            raise RuntimeError("fetch_full_history() returned no data -- check network access")
        print(f"  fetched {len(df)} player-gameweeks across {sorted(df['season'].unique())} in {time.time() - t0:.1f}s")
        if cache_path:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(cache_path)
            print(f"  cached to {cache_path}")

    df = df[df['season'].isin(TRAIN_SEASONS)].copy()
    df['position'] = df['position'].replace({'GK': 'GKP'})
    if EVAL_SEASON in df['season'].unique():
        raise RuntimeError(f"{EVAL_SEASON} leaked into the training frame -- refusing to train")

    bundle = {
        'positions': {},
        'features': FEATURES,
        'meta': {'train_seasons': TRAIN_SEASONS, 'eval_season': EVAL_SEASON},
    }
    for position in POSITIONS:
        pos_df = df[df['position'] == position].dropna(subset=FEATURES + ['total_points']).reset_index(drop=True)
        if pos_df.empty:
            print(f"{position}: no rows, skipping")
            continue
        print(f"Training {position} ({len(pos_df)} rows, {pos_df['team'].nunique()} distinct teams)...")
        t0 = time.time()
        ensemble, meta = train_position(pos_df, position, args.n_iter, args.cv_folds)
        print(f"  done in {time.time() - t0:.1f}s -- XGB CV RMSE {meta['xgb_cv_rmse']:.3f}, RF CV RMSE {meta['rf_cv_rmse']:.3f}")
        bundle['positions'][position] = ensemble
        bundle['meta'][position] = meta

    out_path = Path(args.out) if args.out else PROJECT_ROOT / "Models" / "position_models_v1.pkl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, out_path)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
