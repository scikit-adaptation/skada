# Author: Theo Gnassounou <theo.gnassounou@inria.fr>
#         Remi Flamary <remi.flamary@polytechnique.edu>
#         Alexandre Gramfort <alexandre.gramfort@inria.fr>
#
# License: BSD 3-Clause
import warnings

import numpy as np
from scipy.stats import multivariate_normal

from sklearn.neighbors import KernelDensity
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.model_selection import check_cv
from sklearn.utils import check_random_state

from .base import BaseDataAdaptEstimator, clone
from ._utils import _estimate_covariance

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
    reg : 'auto' or float, default="auto"
        The regularization parameter of the covariance estimator.
        Possible values:

          - None: no shrinkage.
          - 'auto': automatic shrinkage using the Ledoit-Wolf lemma.
          - float between 0 and 1: fixed shrinkage parameter.

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
        reg='auto'
    ):
        super().__init__(base_estimator)
        self.reg = reg

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
        self.mean_source_ = X.mean(axis=0)
        self.cov_source_ = _estimate_covariance(X, shrinkage=self.reg)
        self.mean_target_ = X_target.mean(axis=0)
        self.cov_target_ = _estimate_covariance(X_target, shrinkage=self.reg)


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
        return self


class KLIEP(BaseDataAdaptEstimator):
    """Kullback-Leibler Importance Estimation Procedure (KLIEP).

    The idea of KLIEP is to find an importance estimate w(x) such that
    the Kullback-Leibler (KL) divergence between the source input density
    p_source(x) to its estimate p_target(x) = w(x)p_source(x) is minimized.

    See [3]_ for details.

    Parameters
    ----------
    base_estimator : sklearn estimator
        Estimator used for fitting and prediction.
    gamma : float or array like
        Parameters for the kernels.
        If array like, compute the likelihood cross validation to choose
        the best parameters for the RBF kernel.
        If float, solve the optimisation for the given kernel parameter.
    cv : int, cross-validation generator or an iterable, default=5
        Determines the cross-validation splitting strategy.
        If it is an int it is the number of folds for the cross validation.
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
    `best_gamma_` : float
        The best gamma parameter for the RBF kernel chosen with the likelihood
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
        gamma,  # XXX use the auto/scale mode as done with sklearn SVC
        cv=5,
        n_centers=100,
        tol=1e-6,
        max_iter=1000,
        random_state=None,
    ):
        super().__init__(base_estimator)
        self.gamma = gamma
        self.cv = cv
        self.n_centers = n_centers
        self.tol = tol
        self.max_iter = max_iter
        self.random_state = random_state

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
        A = pairwise_kernels(X, self.centers_, metric="rbf", gamma=self.best_gamma_)
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
        if isinstance(self.gamma, list):
            self.best_gamma_ = self._likelihood_cross_validation(
                self.gamma, X, X_target
            )
        else:
            self.best_gamma_ = self.gamma
        self.alpha_, self.centers_ = self._weights_optimisation(
            self.best_gamma_, X, X_target
        )

    def _weights_optimisation(self, gamma, X, X_target):
        """Optimisation loop."""
        rng = check_random_state(self.random_state)
        n_targets = len(X_target)
        n_centers = np.min((n_targets, self.n_centers))

        centers = X_target[rng.choice(np.arange(n_targets), n_centers)]
        A = pairwise_kernels(X_target, centers, metric="rbf", gamma=gamma)
        b = pairwise_kernels(X, centers, metric="rbf", gamma=gamma)
        b = np.mean(b, axis=0)

        alpha = np.ones(n_centers)
        obj = np.sum(np.log(A @ alpha))
        for _ in range(self.max_iter):
            old_obj = obj
            alpha += EPS * A.T @ (1 / (A @ alpha))
            alpha += (1 - b @ alpha) * b / (b @ b)
            alpha = (alpha > 0) * alpha
            alpha /= b @ alpha
            obj = np.sum(np.log(A @ alpha + EPS))
            if np.abs(obj - old_obj) < self.tol:
                break
        else:
            warnings.warn("Maximum iteration reached before convergence.")

        return alpha, centers

    def _likelihood_cross_validation(self, gammas, X, X_target):
        """Compute the likelihood cross validation to choose the
        best parameter for the kernel.
        """
        log_liks = []
        rng = check_random_state(self.random_state)

        index = np.arange(len(X_target))
        rng.shuffle(index)
        cv = check_cv(self.cv)
        for this_gamma in gammas:
            this_log_lik = []
            for train, test in cv.split(X_target):
                alpha, centers = self._weights_optimisation(
                    this_gamma,
                    X,
                    X_target[train],
                )
                A = pairwise_kernels(
                    X_target[test], centers, metric="rbf", gamma=this_gamma
                )
                weights = A @ alpha
                this_log_lik.append(np.mean(np.log(weights + EPS)))
            log_liks.append(np.mean(this_log_lik))
        best_gamma_ = gammas[np.argmax(log_liks)]

        return best_gamma_
