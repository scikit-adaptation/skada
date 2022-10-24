import numpy as np
from sklearn.neighbors import KernelDensity

from .base import BaseDAEstimator, clone


class ReweightDensity(BaseDAEstimator):
    def __init__(
        self,
        base_estimator,
        weight_estimator=None,
    ):
        super().__init__(base_estimator)

        if weight_estimator is None:
            weight_estimator = KernelDensity()

        self.weight_estimator = weight_estimator

    def predict_adapt(self, X, y, X_target, y_target=None):
        """Predict adaptation (weights, sample or labels)"""
        ws = self.weight_estimator_source_.score_samples(X)
        wt = self.weight_estimator_target_.score_samples(X)
        weights = np.exp(wt - ws)
        weights /= weights.sum()

        return X, y, weights

    def fit_adapt(self, X, y, X_target, y_target=None):
        """Fit adaptation parameters"""
        self.weight_estimator_source_ = clone(self.weight_estimator)
        self.weight_estimator_target_ = clone(self.weight_estimator)
        self.weight_estimator_source_.fit(X)
        self.weight_estimator_target_.fit(X_target)
