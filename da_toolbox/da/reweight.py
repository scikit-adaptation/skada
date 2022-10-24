from abc import ABC, abstractmethod
from .base import BaseDAEstimator, clone
import numpy as np

from sklearn.neighbors import KernelDensity

class ReweightDensity(BaseDAEstimator):
    def __init__(
        self,
        base_estimator,
        weight_estimator=None,
    ):
        self.base_estimator = base_estimator

        super().__init__(base_estimator)

        if weight_estimator is None:
            weight_estimator = KernelDensity()

        self.weight_estimator_source = clone(weight_estimator)
        self.weight_estimator_target = clone(weight_estimator)

    def predict_adapt(self, X, y, X_target, y_target=None):
        """Predict adaptation (weights, sample or labels)"""
        ws = self.weight_estimator_source.score_sample(X)
        wt = self.weight_estimator_target.score_sample(X)
        weights = np.exp(wt - ws)
        weights = weights/weights.sum()

        return X, y, weights

    @abstractmethod
    def fit_adapt(self, X, y, X_target, y_target=None):
        """Fit adaptation parameters"""
        self.weight_estimator_source.fit(X)
        self.weight_estimator_target.fit(X_target)
