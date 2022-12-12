
# Author: Theo Gnassounou <theo.gnassounou@inria.fr>
#         Remi Flamary <remi.flamary@polytechnique.edu>
#         Alexandre Gramfort <alexandre.gramfort@inria.fr>
#
# License: BSD 3-Clause

from abc import abstractmethod

from sklearn.base import BaseEstimator, clone
from sklearn.utils.metaestimators import available_if
from sklearn.utils.validation import check_is_fitted


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
    """Base class for all DA estimators.

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
        return self

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


class BaseSubspaceEstimator(BaseDataAdaptEstimator):
    """Base class for Subspace Data Adaptation DA estimators.

    This class is a more specific base of BaseDataAdaptEstimator
    for subspace problems which ask a function transform for
    source and target domains.
    """

    def __init__(
        self,
        base_estimator,
    ):
        super().__init__(base_estimator)

    def predict(self, X, domain='target'):
        check_is_fitted(self)
        base_estimator = self.base_estimator_
        X_transform = self.transform(X, domain)
        return base_estimator.predict(X_transform)

    @available_if(_estimator_has("predict_proba"))
    def predict_proba(self, X, domain='target'):
        check_is_fitted(self)
        base_estimator = self.base_estimator_
        X_transform = self.transform(X, domain)
        return base_estimator.predict_proba(X_transform)

    def score(self, X, y, domain='target'):
        base_estimator = self.base_estimator_
        X_transform = self.transform(X, domain)
        return base_estimator.score(X_transform, y)

    @abstractmethod
    def transform(self, X, domain='target'):
        return X
