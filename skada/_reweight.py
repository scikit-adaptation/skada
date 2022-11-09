
# Author: Remi Flamary <remi.flamary@polytechnique.edu>
#         Alexandre Gramfort <firstname.lastname@inria.fr>
#
# License: BSD 3-Clause

import numpy as np
from sklearn.neighbors import KernelDensity
from sklearn.linear_model import LogisticRegression
from scipy.stats import multivariate_normal

from .base import BaseDataAdaptEstimator, clone


class ReweightDensity(BaseDataAdaptEstimator):
    """Estimator based on reweighting samples using density estimation.

    Parameters
    ----------
    base_estimator : estimator object
        The base estimator to fit on reweighted data.
    weight_estimator : estimator object, optional
        The estimator to use to estimate the densities of source and target
        observations. If None, a KernelDensity estimator is used.

    Attributes
    ----------
    weight_estimator_source_ : object
        The estimator object fitted on the source data.
    weight_estimator_target_ : object
        The estimator object fitted on the target data.
    """

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
        """Predict adaptation (weights, sample or labels).

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The source data.
        y : array-like, shape (n_samples,)
            The source labels.
        X_target : array-like, shape (n_samples, n_features)
            The target data.
        y_target : array-like, shape (n_samples,), optional
            The target labels.

        Returns
        -------
        X_t : array-like, shape (n_samples, n_components)
            The data (same as X).
        y_t : array-like, shape (n_samples,)
            The labels (same as y).
        weights : array-like, shape (n_samples,)
            The weights of the samples.
        """
        ws = self.weight_estimator_source_.score_samples(X)
        wt = self.weight_estimator_target_.score_samples(X)
        weights = np.exp(wt - ws)
        weights /= weights.sum()

        return X, y, weights

    def fit_adapt(self, X, y, X_target, y_target=None):
        """Fit adaptation parameters.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The source data.
        y : array-like, shape (n_samples,)
            The source labels.
        X_target : array-like, shape (n_samples, n_features)
            The target data.
        y_target : array-like, shape (n_samples,), optional
            The target labels.

        Returns
        -------
        self : object
            Returns self.
        """
        self.weight_estimator_source_ = clone(self.weight_estimator)
        self.weight_estimator_target_ = clone(self.weight_estimator)
        self.weight_estimator_source_.fit(X)
        self.weight_estimator_target_.fit(X_target)


class GaussianReweightDensity(BaseDataAdaptEstimator):
    """Gaussian approximation reweighting method.

    See [1]_ for details.

    Parameters
    ----------
    base_estimator: sklearn estimator
        estimator used for fitting and prediction

    Attributes
    ----------
    `mean_source_` : array-like, shape (n_features,)
        Mean of the source data.
    `cov_source_` : array-like, shape (n_features, n_features)
        Mean of the source data.
    `mean_target_` : array-like, shape (n_features,)
        Mean of the target data.
    `cov_target_` : array-like, shape (n_features, n_features)
        Covariance of the target data.

    References
    ----------
    .. [1]  Hidetoshi Shimodaira. Improving predictive inference under
            covariate shift by weighting the log-likelihood function.
            In Journal of Statistical Planning and Inference, 2000.
    """

    def __init__(
        self,
        base_estimator,
    ):
        super().__init__(base_estimator)

    def predict_adapt(self, X, y, X_target, y_target=None):
        """Predict adaptation (weights, sample or labels).

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The source data.
        y : array-like, shape (n_samples,)
            The source labels.
        X_target : array-like, shape (n_samples, n_features)
            The target data.
        y_target : array-like, shape (n_samples,), optional
            The target labels.

        Returns
        -------
        X_t : array-like, shape (n_samples, n_components)
            The data (same as X).
        y_t : array-like, shape (n_samples,)
            The labels (same as y).
        weights : array-like, shape (n_samples,)
            The weights of the samples.
        """

        gaussian_target = multivariate_normal.pdf(
            X, self.mean_target_, self.cov_target_
        )
        gaussian_source = multivariate_normal.pdf(
            X, self.mean_source_, self.cov_source_
        )

        weights = gaussian_target / gaussian_source

        return X, y, weights

    def fit_adapt(self, X, y, X_target, y_target=None):
        """Fit adaptation parameters.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The source data.
        y : array-like, shape (n_samples,)
            The source labels.
        X_target : array-like, shape (n_samples, n_features)
            The target data.
        y_target : array-like, shape (n_samples,), optional
            The target labels.

        Returns
        -------
        self : object
            Returns self.
        """
        # XXX : at some point we should support more than the empirical cov
        self.mean_source_ = X.mean(axis=0)
        self.cov_source_ = np.cov(X.T)
        self.mean_target_ = X_target.mean(axis=0)
        self.cov_target_ = np.cov(X_target.T)


class ClassifierReweightDensity(BaseDataAdaptEstimator):
    """Gaussian approximation reweighting method.

    See [1]_ for details.

    Parameters
    ----------
    base_estimator : sklearn estimator
        Estimator used for fitting and prediction.
    domain_classifier : sklearn classifier, optional
        Classifier used to predict the domains. If None, a
        LogisticRegression is used.

    Attributes
    ----------
    `domain_classifier_` : object
        The classifier object fitted on the source and target data.

    References
    ----------
    .. [1] Hidetoshi Shimodaira. Improving predictive inference under
           covariate shift by weighting the log-likelihood function.
           In Journal of Statistical Planning and Inference, 2000.
    """

    def __init__(
        self,
        base_estimator,
        domain_classifier=None,
    ):
        super().__init__(base_estimator)

        if domain_classifier is None:
            domain_classifier = LogisticRegression()

        self.domain_classifier = domain_classifier

    def predict_adapt(self, X, y, X_target, y_target=None):
        """Predict adaptation (weights, sample or labels).

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The source data.
        y : array-like, shape (n_samples,)
            The source labels.
        X_target : array-like, shape (n_samples, n_features)
            The target data.
        y_target : array-like, shape (n_samples,), optional
            The target labels.

        Returns
        -------
        X_t : array-like, shape (n_samples, n_components)
            The data (same as X).
        y_t : array-like, shape (n_samples,)
            The labels (same as y).
        weights : array-like, shape (n_samples,)
            The weights of the samples.
        """
        weights = self.domain_classifier_.predict_proba(X)[:, 1]
        return X, y, weights

    def fit_adapt(self, X, y, X_target, y_target=None):
        """Fit adaptation parameters.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The source data.
        y : array-like, shape (n_samples,)
            The source labels.
        X_target : array-like, shape (n_samples, n_features)
            The target data.
        y_target : array-like, shape (n_samples,), optional
            The target labels.

        Returns
        -------
        self : object
            Returns self.
        """
        self.domain_classifier_ = clone(self.domain_classifier)
        y_domain = np.concatenate((len(X) * [0], len(X_target) * [1]))
        self.domain_classifier_.fit(np.concatenate((X, X_target)), y_domain)
