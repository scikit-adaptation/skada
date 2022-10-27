
# Author: Remi Flamary <remi.flamary@polytechnique.edu>
#         Alexandre Gramfort <firstname.lastname@inria.fr>
#
# License: MIT License

from abc import ABC, abstractmethod

from sklearn.base import clone


class BaseDAEstimator(ABC):
    def __init__(
        self,
        base_estimator,
    ):
        self.base_estimator = base_estimator

    def fit(self, X, y, X_target, y_target=None):
        """Fit the DA model on data"""
        base_estimator = clone(self.base_estimator)
        # fit adaptation parameters
        self.fit_adapt(X, y, X_target, y_target)
        # Adapt sample, labels or weights
        X_adapt, y_adapt, weights_adapt = self.predict_adapt(X, y, X_target)
        # fit estimator on adapted data
        if weights_adapt is None:
            base_estimator.fit(X_adapt, y_adapt)
        else:
            # XXX should check if the estimator has a sample_weight parameter
            base_estimator.fit(X_adapt, y_adapt, sample_weight=weights_adapt)
        self.base_estimator_ = base_estimator

    def predict_adapt(self, X, y, X_target, y_target=None):
        """Predict adaptation (weights, sample or labels)"""
        return X, y, None

    @abstractmethod
    def fit_adapt(self, X, y, X_target, y_target=None):
        """Fit adaptation parameters"""
        pass

    def predict(self, X):
        base_estimator = self.base_estimator_
        return base_estimator.predict(X)

    def score(self, X, y):
        base_estimator = self.base_estimator_
        return base_estimator.score(X, y)
