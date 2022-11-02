
# Author: Remi Flamary <remi.flamary@polytechnique.edu>
#         Alexandre Gramfort <firstname.lastname@inria.fr>
#
# License: MIT License

from abc import ABC, abstractmethod

from sklearn.base import BaseEstimator, clone
from sklearn.utils.metaestimators import available_if
from sklearn.utils.validation import check_is_fitted

from sklearn.base import clone


def _estimator_has(attr):
    """Check if we can delegate a method to the underlying estimator.

    First, we check the first fitted classifier if available, otherwise we
    check the unfitted classifier.
    """
    return lambda estimator: (
        hasattr(estimator.base_estimator_, attr)
        if hasattr(estimator, "base_estimator_")
        else hasattr(estimator.base_estimator, attr)
    )


class BaseDAEstimator(BaseEstimator):
    """Base class for al DA estimators.

    Similar API than sklearn except that X_target is given during the fit.
    """

    def __init__(self):
        pass

    @abstractmethod
    def fit(self, X, y, X_target, y_target=None):
        """Fit the DA model on data"""

    @abstractmethod
    def predict(self, X):
        """Predict on target data"""
        pass

    def fit_predict(self, X, y, X_target, y_target=None):
        """Fit the DA model on data and predict on X_target"""
        self.fit(X, y, X_target, y_target)
        return self.predict(X_target)

    @abstractmethod
    def score(self, X, y):
        """Score te performance of the estimator"""


class BaseDataAdaptEstimator(BaseDAEstimator):
    """Base class for Data Adapation DA estimators.

    Those estimators  work in two steps:

    1. Estimate a transformation of the source data so that it becomes for
       similar to the target data.
    2. Fit a base estimator on the adapted data; this estimator can then be used
       on new target data.

    This class is very general wand can be used for reweighting, mapping of the
    source data but also with label propagation strategies on target data.

    """

    def __init__(
        self,
        base_estimator,
    ):
        super().__init__()
        self.base_estimator = base_estimator

    def fit(self, X, y, X_target, y_target=None):
        """Fit the DA model on data"""
        base_estimator = clone(self.base_estimator)
        # fit adaptation parameters
        self.fit_adapt(X, y, X_target, y_target)
        # Adapt sample, labels or weights
        X_adapt, y_adapt, weights_adapt = self.predict_adapt(X, y, X_target, y_target)

        # fit estimator on adapted data
        if weights_adapt is None:
            base_estimator.fit(X_adapt, y_adapt)
        else:
            # XXX should check if the estimator has a sample_weight parameter
            base_estimator.fit(X_adapt, y_adapt, sample_weight=weights_adapt)
        self.base_estimator_ = base_estimator

    @abstractmethod
    def predict_adapt(self, X, y, X_target, y_target=None):
        """Predict adaptation (weights, sample or labels)"""
        return X, y, None

    @abstractmethod
    def fit_adapt(self, X, y, X_target, y_target=None):
        """Fit adaptation parameters"""
        pass

    def predict(self, X):
        check_is_fitted(self)
        base_estimator = self.base_estimator_
        return base_estimator.predict(X)

    @available_if(_estimator_has("predict_proba"))
    def predict_proba(self, X):
        check_is_fitted(self)
        base_estimator = self.base_estimator_
        return base_estimator.predict_proba(X)

    def score(self, X, y):
        base_estimator = self.base_estimator_
        return base_estimator.score(X, y)
