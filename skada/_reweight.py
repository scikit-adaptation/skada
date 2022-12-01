
# Author: Remi Flamary <remi.flamary@polytechnique.edu>
#         Alexandre Gramfort <firstname.lastname@inria.fr>
#
# License: BSD 3-Clause
import warnings

import numpy as np
from scipy.stats import multivariate_normal

from sklearn.neighbors import KernelDensity
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import pairwise_kernels

from .base import BaseDataAdaptEstimator, clone

EPS = np.finfo(float).eps


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


class DiscriminatorReweightDensity(BaseDataAdaptEstimator):
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


class KLIEP(BaseDataAdaptEstimator):
    """Kullback-Leibler Importance Estimation Procedure (KLIEP).

    See [3]_ for details.

    Parameters
    ----------
    base_estimator : sklearn estimator
        Estimator used for fitting and prediction.
    kparam : float or array like
        Parameters for the kernels.
        If array like, compute the likelihood cross validation to choose
        the best parameters for the kernels.
        If float, solve the optimisation for the given kernels' parameters.
    n_subsets : int, default=5
        Number of subsets of target data used for the likelihood cross validation.
    n_centers : int, default=100
        Number of kernel centers defining their number.
    tol : float, default=1e-6
        Tolerance for the stopping criterion in the optimization.
    max_iter : int, default=1000
        Number of maximum iteration before stopping the optimization.
    random_state : int, RandomState instance or None, default=None
        Determines random number generation for dataset creation. Pass an int
        for reproducible output across multiple function calls.

    Attributes
    ----------
    `best_kparam_` : float
        The best parameters for the kernel chosen with the likelihood
        cross validation if several parameters are given as input.
    `alpha_` : float
        Solution of the optimisation problem.
    `centers_` : list
        List of the target data taken as centers for the kernels.

    References
    ----------
    .. [3] Masashi Sugiyama et. al. Direct Importance Estimation with Model Selection
           and Its Application to Covariate Shift Adaptation.
           In NeurIPS, 2007.
    """

    def __init__(
        self,
        base_estimator,
        kparam,
        n_subsets=5,
        n_centers=100,
        tol=1e-6,
        max_iter=1000,
        random_state=42,
    ):
        super().__init__(base_estimator)

        self.kparam = kparam
        self.n_subsets = n_subsets
        self.n_centers = n_centers
        self.tol = tol
        self.max_iter = max_iter
        self.rng = np.random.RandomState(random_state)

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
        A = pairwise_kernels(X, self.centers_, metric="rbf", gamma=self.best_kparam_)
        weights = A @ self.alpha_
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
        if isinstance(self.kparam, list):
            self.best_kparam_ = self._likelihood_cross_validation(
                self.kparam, X, X_target
            )
        else:
            self.best_kparam_ = self.kparam
        self.alpha_, self.centers_ = self._weights_optimisation(
            self.best_kparam_, X, X_target
        )

    def _weights_optimisation(self, kparam, X, X_target):
        n_targets = len(X_target)
        n_centers = np.min((n_targets, self.n_centers))
        centers = X_target[self.rng.choice(np.arange(n_targets), n_centers)]
        A = pairwise_kernels(X_target, centers, metric="rbf", gamma=kparam)
        b = pairwise_kernels(X, centers, metric="rbf", gamma=kparam)
        b = np.mean(b, axis=0)
        alpha = np.ones(n_centers)
        obj = np.sum(np.log(A @ alpha))
        for it in range(self.max_iter):
            old_obj = obj
            alpha += EPS * A.T @ (1 / (A @ alpha))
            alpha += (1 - b @ alpha) * b / (b @ b)
            alpha = (alpha > 0) * alpha
            alpha = alpha / (b @ alpha)
            obj = np.sum(np.log(A @ alpha + EPS))
            if np.abs(obj - old_obj) < self.tol:
                break
        if it+1 == self.max_iter:
            warnings.warn("Maximum iteration reached before convergence.")
        return alpha, centers

    def _likelihood_cross_validation(self, kparams, X, X_target):
        """Compute the likelihood cross validation to choose the
           best parameter for the kernel
        """
        J = []
        index = np.arange(len(X_target))
        self.rng.shuffle(index)
        index_subsets = np.array_split(index, self.n_subsets)
        for kparam in kparams:
            Jr = []
            for s, index_subset in enumerate(index_subsets):
                alpha, centers = self._weights_optimisation(kparam, X, X_target[
                    np.concatenate(index_subsets[:s] + index_subsets[s:])
                ])
                A = pairwise_kernels(
                    X_target[index_subset], centers, metric="rbf", gamma=kparam
                )
                weights = A @ alpha
                Jr.append(np.mean(np.log(weights + EPS)))
            J.append(np.mean(Jr))
        best_kparam_ = kparams[np.argmax(J)]

        return best_kparam_
