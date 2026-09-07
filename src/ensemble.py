"""
src/ensemble.py -- MedianEnsembleRegressor, the wrapper around the
XGBoost + RandomForest pair trained per-position by
training/train_position_models.py.

Lives in src/, not training/, specifically so the production app (which
only ever unpickles Models/position_models_v1.pkl, never imports the
training module) can still resolve this class -- joblib/pickle need the
exact same importable module path at load time that existed at dump time.
"""
import numpy as np


class MedianEnsembleRegressor:
    """
    Predicts the median across N fitted regressors (here: one XGBoost, one
    RandomForest, per position). With exactly two models the median is
    arithmetically the same as the mean -- kept as an explicit median
    rather than a mean because "median across the ensemble" is the
    intended semantics if a third model is ever added to a position, and
    median is the more outlier-resistant combining rule for that case.
    """

    def __init__(self, models, feature_cols):
        self.models = models
        self.feature_cols = feature_cols

    def predict(self, X):
        if hasattr(X, "columns"):
            X = X[self.feature_cols].fillna(0)
        preds = np.column_stack([m.predict(X) for m in self.models])
        return np.median(preds, axis=1)
