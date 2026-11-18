"""
training/train_component_models.py -- fits every component the prediction
pipeline sums through the real 2026/27 scoring matrix (see
src/components.py for the inference side, src/predictor.py for how it
replaces the old rule layer).

Why component models instead of one points regression + hand-tuned rules:
a goalkeeper's points come from saves and clean sheets, a forward's from
xG -- summing genuinely separate, correctly-scaled predictions per scoring
event is both more accurate and, critically, auditable: every number in
the reasoning string this produces is something a model actually computed,
not a fixture-bonus or premium-multiplier constant chosen by feel.

Deliberately simple models throughout (LogisticRegression / linear
calibration), not XGBoost + RandomizedSearchCV: these targets are
individually low-dimensional and well-behaved (a handful of features each,
binary or near-linear relationships), and the previous per-position-model
training run (training/train_position_models.py) spent real wall-clock
time on hyperparameter search for benefits this pipeline doesn't need --
fast, robust, and easy to sanity-check beats marginally-better-tuned here.

Train/serve feature parity: every input column referenced below is either
identically named on both fetch_full_history() (historical, walk-forward)
and process_data() (live) output, or goes through
src/components.py::derive_component_inputs() first, which is the one place
the historical-vs-live translation happens. See that function's docstring
for exactly which raw columns each "_input" name maps to.

Train seasons / eval season: 2021-22..2024-25 held-out-clean for every
component EXCEPT bonus -- see fit_bonus_model() below, which is the one
component genuinely fit on EVAL_SEASON itself, for a real data reason
(2026/27 recalibrated BPS, and DEFCON has no history before 2025-26 at
all), not an oversight. main()'s leak guard only checks the seasons the
*other* components trained on (_train_rows(df), which excludes
EVAL_SEASON) precisely because it CANNOT also cover fit_bonus_model's
deliberate exception without raising on every run -- it prints that
exception loudly instead (see main()) rather than silently passing it.
Concretely: evaluation/backtest.py's reported bonus-component accuracy
for 2025-26 is in-sample, not out-of-sample, unlike every other
component's. This is a known, stated limitation of comparing the
component pipeline against the position ensemble on that season, not
fixed here -- a genuinely clean fix needs a within-season walk-forward
refit of the bonus model (train on gameweeks strictly before the one
being scored), which is a larger change than this file currently does
for any other component either.

DEFCON has no history before 2025-26 (the rule didn't exist), so its
walk-forward hit-rate estimator's *position-level prior* is fit from
2025-26 itself, inside fetch_full_history() -- see
src/features.py::_defcon_walk_forward_hit_rate's docstring for why that
specific, narrow exception is unavoidable rather than a leakage bug.

Usage:
    python training/train_component_models.py
"""
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import log_loss

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.components import derive_component_inputs
from src.features import fetch_full_history

TRAIN_SEASONS = ['2021-22', '2022-23', '2023-24', '2024-25']
EVAL_SEASON = '2025-26'  # never touched here -- see evaluation/backtest.py
DEFCON_ONLY_SEASON = '2025-26'  # bonus model's DEFCON-interaction term -- see module docstring

MINUTES_FEATURES = [
    'rolling_3_minutes', 'value', 'squad_depth_count', 'starts_rate_input',
    'is_gkp', 'is_def', 'is_mid', 'is_fwd',
]
CS_FEATURES = [
    'team_rolling_5_team_goals_against', 'opp_team_rolling_5_team_goals_for',
    'team_rolling_10_team_goals_against', 'opp_team_rolling_10_team_goals_for',
]
BONUS_FEATURES = ['bps_rate_input', 'is_gkp', 'is_def', 'is_mid', 'is_fwd']  # + p_defcon, added at inference (see predict_bonus)


def _train_rows(df):
    return df[df['season'].isin(TRAIN_SEASONS)].copy()


def fit_minutes_models(df):
    """
    Two LogisticRegressions: P(plays), P(60+ minutes). Target rows are
    every player-gameweek in the training seasons (including genuine
    zeros -- a player who didn't play IS the P(plays)=0 signal, dropping
    those rows would just teach the model everyone always plays).
    """
    rows = derive_component_inputs(_train_rows(df))
    X = rows[MINUTES_FEATURES].fillna(0)
    y_plays = (rows['minutes'] > 0).astype(int)
    y_60 = (rows['minutes'] >= 60).astype(int)

    plays_model = LogisticRegression(max_iter=1000).fit(X, y_plays)
    plays60_model = LogisticRegression(max_iter=1000).fit(X, y_60)

    # Log loss on P(60+), the metric this component was asked to be
    # evaluated on separately -- in-sample here (this function only fits;
    # evaluation/backtest.py's component backtest scores it out-of-sample
    # on the held-out season).
    p60_pred = plays60_model.predict_proba(X)[:, 1]
    ll = log_loss(y_60, p60_pred)
    print(f"  minutes: P(plays) base rate {y_plays.mean():.3f}, P(60+) base rate {y_60.mean():.3f}, in-sample log loss {ll:.4f}")

    return {'plays': plays_model, 'plays60': plays60_model}


def fit_attack_calibration(df):
    """
    Linear calibration: actual goals(assists) this match ~ slope * xG90(xA90)
    input + intercept. A deliberately simple 1-feature regression per
    target -- the walk-forward rolling xG90/xA90 rate is already a strong
    signal (it's FPL's own underlying shot-based model); this just corrects
    any systematic scale bias against realized goals/assists rather than
    re-deriving the relationship from scratch.

    Filtered to played rows: a benched player pairs a 0 input (no minutes
    to derive a rate from) with a genuine 0 actual goal/assist -- not
    wrong, but an uninformative (0, 0) pair that only dilutes the fit
    rather than corrupting it the way fit_gc_calibration's team-level
    target did; filtering it out here too for a cleaner, faster fit.
    """
    rows = _train_rows(df)
    rows = rows[rows['minutes'] > 0]
    inputs = derive_component_inputs(rows)

    def _fit_one(x_col, y_col):
        X = inputs[[x_col]].fillna(0)
        y = rows[y_col].fillna(0)
        reg = LinearRegression().fit(X, y)
        return float(reg.coef_[0]), float(reg.intercept_)

    # Calibrated against what actually happened (goals_scored/assists),
    # not against expected_goals/expected_assists again -- the input rate
    # already IS the expected_goals-derived signal; regressing it against
    # itself would be a near-identity fit that calibrates nothing.
    g_slope, g_intercept = _fit_one('attack_xg90_input', 'goals_scored')
    a_slope, a_intercept = _fit_one('attack_xa90_input', 'assists')
    print(f"  attack: goals calib slope={g_slope:.3f} intercept={g_intercept:.3f}; assists calib slope={a_slope:.3f} intercept={a_intercept:.3f}")
    return {'goals_slope': g_slope, 'goals_intercept': g_intercept, 'assists_slope': a_slope, 'assists_intercept': a_intercept}


def fit_clean_sheet_model(df):
    """
    LogisticRegression: P(team keeps a clean sheet this match) from own/
    opponent rolling goals-for/against -- replaces the old hand-tuned
    "0.30 + strength_delta * 0.08" formula with coefficients actually
    fit against real outcomes.
    """
    rows = _train_rows(df).dropna(subset=CS_FEATURES)
    X = rows[CS_FEATURES].fillna(0)
    y = (rows['team_goals_against'] == 0).astype(int)
    model = LogisticRegression(max_iter=1000).fit(X, y)
    p_pred = model.predict_proba(X)[:, 1]
    print(f"  clean sheet: base rate {y.mean():.3f}, in-sample log loss {log_loss(y, p_pred):.4f}")
    return model


def fit_gc_calibration(df):
    """
    Filtered to rows where the player actually played: `gc_xgc90_input` is
    0 for a benched player (no minutes to derive a rate from -- see
    derive_component_inputs' min_minutes guard), but the TEAM's actual
    goals conceded that match is a real number regardless of whether this
    particular player played. Fitting on every row pairs a real number of
    "0 input" benched-player rows against nonzero team outcomes that have
    nothing to do with them, which drags the fit toward "just predict the
    league-average concession rate" (observed directly: correlation of
    0.03 and a near-flat fit before this filter). Downstream, a benched
    player's GC prediction is scaled to ~0 anyway via minutes_fraction
    (see predict_clean_sheet_and_gc) -- this filter just stops those rows
    from corrupting the fit that number is calibrated from.
    """
    rows = _train_rows(df)
    rows = rows[rows['minutes'] > 0]
    inputs = derive_component_inputs(rows)
    X = inputs[['gc_xgc90_input']].fillna(0)
    y = rows['team_goals_against'].fillna(0)
    reg = LinearRegression().fit(X, y)
    print(f"  goals conceded: calib slope={reg.coef_[0]:.3f} intercept={reg.intercept_:.3f} (n={len(rows)}, played rows only)")
    return {'slope': float(reg.coef_[0]), 'intercept': float(reg.intercept_)}


def fit_saves_calibration(df):
    rows = _train_rows(df)
    rows = rows[rows['position'].isin(['GKP', 'GK']) & (rows['minutes'] > 0)]
    inputs = derive_component_inputs(rows)
    X = inputs[['saves90_input']].fillna(0)
    y = rows['saves'].fillna(0)
    reg = LinearRegression().fit(X, y)
    print(f"  saves: calib slope={reg.coef_[0]:.3f} intercept={reg.intercept_:.3f} (n={len(rows)} GKP rows)")
    return {'slope': float(reg.coef_[0]), 'intercept': float(reg.intercept_)}


def fit_bonus_model(df):
    """
    LinearRegression: actual bonus points ~ rolling BPS rate + p_defcon
    (the DEFCON hit-probability itself, not an independent draw) --
    letting the fitted coefficient on p_defcon absorb whatever real
    negative interaction the 2026/27 BPS recalibration created between
    heavy defensive-action matches and maximum bonus, rather than assuming
    independence. Fit on 2025-26 only (see module docstring: that's the
    only season whose BPS reflects the current recalibration, and the only
    season with a real, non-prior-only p_defcon signal to learn against).
    """
    rows = df[df['season'] == DEFCON_ONLY_SEASON].copy()
    inputs = derive_component_inputs(rows)
    X = inputs[BONUS_FEATURES].fillna(0).copy()
    X['p_defcon'] = rows['defcon_hit_prob'].fillna(0).values
    y = rows['bonus'].fillna(0)
    reg = LinearRegression().fit(X, y)
    coef_str = ", ".join(f"{c}={v:.3f}" for c, v in zip(list(X.columns), reg.coef_))
    print(f"  bonus: {coef_str}, intercept={reg.intercept_:.3f} (n={len(rows)}, {DEFCON_ONLY_SEASON} only)")
    return reg


def main():
    print(f"Fetching historical data ({', '.join(TRAIN_SEASONS)} for training; {EVAL_SEASON} held out)...")
    t0 = time.time()
    df = fetch_full_history()
    if df.empty:
        raise RuntimeError("fetch_full_history() returned no data -- check network access")
    df['position'] = df['position'].replace({'GK': 'GKP'})
    print(f"  fetched {len(df)} player-gameweeks in {time.time() - t0:.1f}s")
    if EVAL_SEASON in _train_rows(df)['season'].unique():
        raise RuntimeError(f"{EVAL_SEASON} leaked into the training frame -- refusing to train")
    print(
        f"  NOTE: fit_bonus_model() is the one exception to the above -- it "
        f"trains on {EVAL_SEASON} itself (see this file's module docstring "
        f"for why), so the bonus component's reported {EVAL_SEASON} accuracy "
        f"is in-sample, not out-of-sample like every other component's."
    )

    bundle = {}
    print("Fitting minutes model...")
    bundle['minutes'] = fit_minutes_models(df)
    print("Fitting attack calibration...")
    bundle['attack_calibration'] = fit_attack_calibration(df)
    print("Fitting clean sheet model...")
    bundle['clean_sheet'] = fit_clean_sheet_model(df)
    print("Fitting goals-conceded calibration...")
    bundle['gc_calibration'] = fit_gc_calibration(df)
    print("Fitting saves calibration...")
    bundle['saves_calibration'] = fit_saves_calibration(df)
    print("Fitting bonus model...")
    bundle['bonus'] = fit_bonus_model(df)

    out_path = PROJECT_ROOT / "Models" / "component_models_v1.pkl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, out_path)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
