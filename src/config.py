"""
src/config.py -- small, explicit feature flags. Not user-facing settings;
this file exists so a model-architecture change can be rolled back by
flipping one constant here, without touching app.py or predictor.py.

USE_COMPONENT_MODEL: True routes predictions through the component
pipeline (src/components.py, one model per scoring event -- minutes,
attacking, clean sheet, DEFCON, bonus, cards -- summed through the real
2026/27 scoring matrix). False falls back to the previous per-position
XGBoost+RandomForest ensemble (Models/position_models_v1.pkl) and its old
rule layer (fixture-difficulty bonus, premium multiplier, star-captain
bonus, rotation tax). See evaluation/backtest.py's component-vs-
position-ensemble comparison for why this defaults to whichever value it
does -- the comparison result is what should decide this flag, not a
guess made before running it.
"""
USE_COMPONENT_MODEL = False
# Set to False as of the first component-vs-position-ensemble backtest
# (evaluation/backtest.py --compare-components, season 2025-26): the
# component pipeline improved Zeros/Blanks/overall RMSE but made the two
# metrics explicitly used as the bar WORSE -- Tickers RMSE +0.248 (+19%),
# Haulers RMSE +0.413 (+7%) -- plus a real drop in rank correlation
# (0.693 -> 0.672) and top-10 overlap (1.2 -> 1.0/GW). See
# evaluation/results/2025-26_component_models_v1.json and the sibling
# position_models_v1.json for the exact numbers, and the conversation this
# was built in for the likely cause: the component calibrations are plain
# linear regressions, which regress toward the mean on a heavy-tailed
# target -- the opposite of what the position-ensemble's balanced sample
# weighting was specifically built to do for exactly the Haulers/Tickers
# buckets. Flip this back to True only after that's addressed and
# re-verified, not on the strength of the Zeros/Blanks/overall improvement
# alone.
#
# A second, independent issue a later code review found and left
# unresolved (also to be fixed before flipping this): src/components.py::
# predict_bonus's p_defcon feature means two different things depending on
# whether the row is historical or live -- fit_bonus_model trains against
# the historical walk-forward Beta-Binomial estimate (defcon_hit_prob),
# but predict_defcon's live branch feeds the same feature name a Poisson
# threshold-crossing estimate from a season-aggregate rate instead. The
# backtest never exercises the live branch, so this distribution mismatch
# wouldn't show up there even after the Haulers/Tickers regression above
# is fixed -- it needs its own live-vs-historical calibration (matching
# the pattern already used for the attack/goals-conceded/saves
# components) before this flag is trustworthy in production.
