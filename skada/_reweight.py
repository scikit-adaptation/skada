# Author: Theo Gnassounou <theo.gnassounou@inria.fr>
#         Remi Flamary <remi.flamary@polytechnique.edu>
#         Oleksii Kachaiev <kachayev@gmail.com>
#
# License: BSD 3-Clause

import warnings

import numpy as np
from scipy.stats import multivariate_normal
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.model_selection import check_cv
from sklearn.neighbors import KernelDensity
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_is_fitted
from sklearn.neighbors import KNeighborsClassifier

from .base import AdaptationOutput, BaseAdapter, clone
from .utils import check_X_domain, source_target_split, extract_source_indices
from ._utils import _estimate_covariance
from ._pipeline import make_da_pipeline


EPS = np.finfo(float).eps


class ReweightDensityAdapter(BaseAdapter):
    """Adapter based on re-weighting samples using density estimation.

    Parameters
    ----------
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

    def __init__(self, weight_estimator=None):
        super().__init__()
        self.weight_estimator = weight_estimator or KernelDensity()

    def fit(self, X, y=None, sample_domain=None):
        """Fit adaptation parameters.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The source data.
        y : array-like, shape (n_samples,)
            The source labels.
        sample_domain : array-like, shape (n_samples,)
            The domain labels (same as sample_domain).

        Returns
        -------
        self : object
            Returns self.
        """
        X, sample_domain = check_X_domain(X, sample_domain)
        X_source, X_target = source_target_split(X, sample_domain=sample_domain)

        source_estimator = clone(self.weight_estimator)
        source_estimator.fit(X_source)
        target_estimator = clone(self.weight_estimator)
        target_estimator.fit(X_target)
        self.weight_estimator_source_ = source_estimator
        self.weight_estimator_target_ = target_estimator
        return self

    def adapt(self, X, y=None, sample_domain=None):
        """Predict adaptation (weights, sample or labels).

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The source data.
        y : array-like, shape (n_samples,)
            The source labels.
        sample_domain : array-like, shape (n_samples,)
            The domain labels (same as sample_domain).

        Returns
        -------
        output : :class:`skada.base.AdaptationOutput`
            Dictionary-like object, with the following attributes.

            X_t : array-like, shape (n_samples, n_components)
                The data (same as X).
            weights : array-like, shape (n_samples,)
                The weights of the samples.
        """
        check_is_fitted(self)
        X, sample_domain = check_X_domain(X, sample_domain)
        source_idx = extract_source_indices(sample_domain)

        # xxx(okachaiev): move this to API
        if source_idx.sum() > 0:
            source_idx, = np.where(source_idx)
            ws = self.weight_estimator_source_.score_samples(X[source_idx])
            wt = self.weight_estimator_target_.score_samples(X[source_idx])
            source_weights = np.exp(wt - ws)
            source_weights /= source_weights.sum()
            weights = np.zeros(X.shape[0], dtype=source_weights.dtype)
            weights[source_idx] = source_weights
        else:
            weights = None
        return AdaptationOutput(X=X, sample_weight=weights)


def ReweightDensity(
    base_estimator=None,
    weight_estimator=None,
):
    """Density re-weighting pipeline adapter and estimator.

    Parameters
    ----------
    base_estimator : sklearn estimator, default=None
        estimator used for fitting and prediction
    weight_estimator : estimator object, optional
        The estimator to use to estimate the densities of source and target
        observations. If None, a KernelDensity estimator is used.

    Returns
    -------
    pipeline : sklearn pipeline
        Pipeline containing the ReweightDensity adapter and the base estimator.
    """
    if base_estimator is None:
        base_estimator = LogisticRegression().set_fit_request(sample_weight=True)

    return make_da_pipeline(
        ReweightDensityAdapter(weight_estimator=weight_estimator),
        base_estimator,
    )


class GaussianReweightDensityAdapter(BaseAdapter):
    """Gaussian approximation re-weighting method.

    See [1]_ for details.

    Parameters
    ----------
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

    def __init__(self, reg='auto'):
        super().__init__()
        self.reg = reg

    def fit(self, X, y=None, sample_domain=None):
        """Fit adaptation parameters.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The source data.
        y : array-like, shape (n_samples,)
            The source labels.
        sample_domain : array-like, shape (n_samples,)
            The domain labels (same as sample_domain).

        Returns
        -------
        self : object
            Returns self.
        """
        X, sample_domain = check_X_domain(X, sample_domain)
        X_source, X_target = source_target_split(X, sample_domain=sample_domain)

        self.mean_source_ = X_source.mean(axis=0)
        self.cov_source_ = _estimate_covariance(X_source, shrinkage=self.reg)
        self.mean_target_ = X_target.mean(axis=0)
        self.cov_target_ = _estimate_covariance(X_target, shrinkage=self.reg)
        return self

    def adapt(self, X, y=None, sample_domain=None):
        """Predict adaptation (weights, sample or labels).

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The source data.
        y : array-like, shape (n_samples,)
            The source labels.
        sample_domain : array-like, shape (n_samples,)
            The domain labels (same as sample_domain).

        Returns
        -------
        output : :class:`skada.base.AdaptationOutput`
            Dictionary-like object, with the following attributes.

            X_t : array-like, shape (n_samples, n_components)
                The data (same as X).
            weights : array-like, shape (n_samples,)
                The weights of the samples.
        """
        check_is_fitted(self)
        X, sample_domain = check_X_domain(X, sample_domain)
        source_idx = extract_source_indices(sample_domain)

        # xxx(okachaiev): move this to API
        if source_idx.sum() > 0:
            source_idx, = np.where(source_idx)
            gaussian_target = multivariate_normal.pdf(
                X[source_idx], self.mean_target_, self.cov_target_
            )
            gaussian_source = multivariate_normal.pdf(
                X[source_idx], self.mean_source_, self.cov_source_
            )
            source_weights = gaussian_target / gaussian_source
            weights = np.zeros(X.shape[0], dtype=source_weights.dtype)
            weights[source_idx] = source_weights
        else:
            weights = None
        return AdaptationOutput(X=X, sample_weight=weights)


def GaussianReweightDensity(
    base_estimator=None,
    reg='auto',
):
    """Gaussian approximation re-weighting pipeline adapter and estimator.

    see [1]_ for details.

    Parameters
    ----------
    base_estimator : sklearn estimator, default=None
        estimator used for fitting and prediction
    reg : 'auto' or float, default="auto"
        The regularization parameter of the covariance estimator.
        Possible values:

          - None: no shrinkage.
          - 'auto': automatic shrinkage using the Ledoit-Wolf lemma.
          - float between 0 and 1: fixed shrinkage parameter.

    Returns
    -------
    pipeline : sklearn pipeline
        Pipeline containing the GaussianReweightDensity adapter and the
        base estimator.

    References
    ----------
    .. [1] Hidetoshi Shimodaira. Improving predictive inference under
           covariate shift by weighting the log-likelihood function.
           In Journal of Statistical Planning and Inference, 2000.
    """
    if base_estimator is None:
        base_estimator = LogisticRegression().set_fit_request(sample_weight=True)

    return make_da_pipeline(
        GaussianReweightDensityAdapter(reg=reg),
        base_estimator,
    )


class DiscriminatorReweightDensityAdapter(BaseAdapter):
    """Gaussian approximation re-weighting method.

    See [1]_ for details.

    Parameters
    ----------
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

    def __init__(self, domain_classifier=None):
        super().__init__()
        self.domain_classifier = domain_classifier or LogisticRegression()

    def fit(self, X, y=None, sample_domain=None):
        """Fit adaptation parameters.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The source data.
        y : array-like, shape (n_samples,)
            The source labels.
        sample_domain : array-like, shape (n_samples,)
            The domain labels (same as sample_domain).

        Returns
        -------
        self : object
            Returns self.
        """
        X, sample_domain = check_X_domain(X, sample_domain)
        source_idx = extract_source_indices(sample_domain)
        source_idx, = np.where(source_idx)
        y_domain = np.ones(X.shape[0], dtype=np.int32)
        y_domain[source_idx] = 0
        domain_classifier = clone(self.domain_classifier)
        domain_classifier.fit(X, y_domain)
        self.domain_classifier_ = domain_classifier
        return self

    def adapt(self, X, y=None, sample_domain=None, **kwargs):
        """Predict adaptation (weights, sample or labels).

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The source data.
        y : array-like, shape (n_samples,)
            The source labels.
        sample_domain : array-like, shape (n_samples,)
            The domain labels (same as sample_domain).

        Returns
        -------
        output : :class:`skada.base.AdaptationOutput`
            Dictionary-like object, with the following attributes.

            X_t : array-like, shape (n_samples, n_components)
                The data (same as X).
            weights : array-like, shape (n_samples,)
                The weights of the samples.
        """
        check_is_fitted(self)
        X, sample_domain = check_X_domain(X, sample_domain)
        source_idx = extract_source_indices(sample_domain)

        # xxx(okachaiev): move this to API
        if source_idx.sum() > 0:
            source_idx, = np.where(source_idx)
            source_weights = self.domain_classifier_.predict_proba(X[source_idx])[:, 1]
            weights = np.zeros(X.shape[0], dtype=source_weights.dtype)
            weights[source_idx] = source_weights
        else:
            weights = None
        return AdaptationOutput(X=X, sample_weight=weights)


def DiscriminatorReweightDensity(
    base_estimator=None,
    domain_classifier=None
):
    """Discriminator re-weighting pipeline adapter and estimator.

    see [1]_ for details.

    Parameters
    ----------
    base_estimator : sklearn estimator, default=None
        estimator used for fitting and prediction
    domain_classifier : sklearn classifier, optional
        Classifier used to predict the domains. If None, a
        LogisticRegression is used.

    Returns
    -------
    pipeline : sklearn pipeline
        Pipeline containing the DiscriminatorReweightDensity adapter and the
        base estimator.

    References
    ----------
    .. [1] Hidetoshi Shimodaira. Improving predictive inference under
           covariate shift by weighting the log-likelihood function.
           In Journal of Statistical Planning and Inference, 2000.
    """
    if base_estimator is None:
        base_estimator = LogisticRegression().set_fit_request(sample_weight=True)

    return make_da_pipeline(
        DiscriminatorReweightDensityAdapter(
            domain_classifier=domain_classifier
        ),
        base_estimator,
    )


class KLIEPAdapter(BaseAdapter):
    """Kullback-Leibler Importance Estimation Procedure (KLIEP).

    The idea of KLIEP is to find an importance estimate w(x) such that
    the Kullback-Leibler (KL) divergence between the source input density
    p_source(x) to its estimate p_target(x) = w(x)p_source(x) is minimized.

    See [3]_ for details.

    Parameters
    ----------
    gamma : float or array like
        Parameters for the kernels.
        If array like, compute the likelihood cross validation to choose
        the best parameters for the RBF kernel.
        If float, solve the optimization for the given kernel parameter.
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
        Solution of the optimization problem.
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
        gamma,  # XXX use the auto/scale mode as done with sklearn SVC
        cv=5,
        n_centers=100,
        tol=1e-6,
        max_iter=1000,
        random_state=None,
    ):
        super().__init__()
        self.gamma = gamma
        self.cv = cv
        self.n_centers = n_centers
        self.tol = tol
        self.max_iter = max_iter
        self.random_state = random_state

    def fit(self, X, y=None, sample_domain=None, **kwargs):
        """Fit adaptation parameters.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The source data.
        y : array-like, shape (n_samples,)
            The source labels.
        sample_domain : array-like, shape (n_samples,)
            The domain labels (same as sample_domain).

        Returns
        -------
        self : object
            Returns self.
        """
        X, sample_domain = check_X_domain(
            X,
            sample_domain,
            allow_multi_source=True,
            allow_multi_target=True
        )
        X_source, X_target = source_target_split(X, sample_domain=sample_domain)

        if isinstance(self.gamma, list):
            self.best_gamma_ = self._likelihood_cross_validation(
                self.gamma, X_source, X_target
            )
        else:
            self.best_gamma_ = self.gamma
        self.alpha_, self.centers_ = self._weights_optimization(
            self.best_gamma_, X_source, X_target
        )
        return self

    def _weights_optimization(self, gamma, X_source, X_target):
        """Optimization loop."""
        rng = check_random_state(self.random_state)
        n_targets = len(X_target)
        n_centers = np.min((n_targets, self.n_centers))

        centers = X_target[rng.choice(np.arange(n_targets), n_centers)]
        A = pairwise_kernels(X_target, centers, metric="rbf", gamma=gamma)
        b = pairwise_kernels(X_source, centers, metric="rbf", gamma=gamma)
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

    def _likelihood_cross_validation(self, gammas, X_source, X_target):
        """Compute the likelihood cross validation to choose the
        best parameter for the kernel.
        """
        log_liks = []
        # xxx(okachaiev): should this be done when fitting?
        rng = check_random_state(self.random_state)
        index = np.arange(len(X_target))
        rng.shuffle(index)
        cv = check_cv(self.cv)
        for this_gamma in gammas:
            this_log_lik = []
            for train, test in cv.split(X_target):
                alpha, centers = self._weights_optimization(
                    this_gamma,
                    X_source,
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

    def adapt(self, X, y=None, sample_domain=None, **kwargs):
        """Predict adaptation (weights, sample or labels).

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The source data.
        y : array-like, shape (n_samples,)
            The source labels.
        sample_domain : array-like, shape (n_samples,)
            The domain labels (same as sample_domain).

        Returns
        -------
        output : :class:`skada.base.AdaptationOutput`
            Dictionary-like object, with the following attributes.

            X_t : array-like, shape (n_samples, n_components)
                The data (same as X).
            weights : array-like, shape (n_samples,)
                The weights of the samples.
        """
        check_is_fitted(self)
        X, sample_domain = check_X_domain(X, sample_domain)
        source_idx = extract_source_indices(sample_domain)

        if source_idx.sum() > 0:
            source_idx, = np.where(source_idx)
            A = pairwise_kernels(
                X[source_idx],
                self.centers_,
                metric="rbf",
                gamma=self.best_gamma_
            )
            source_weights = A @ self.alpha_
            weights = np.zeros(X.shape[0], dtype=source_weights.dtype)
            weights[source_idx] = source_weights
        else:
            weights = None
        return AdaptationOutput(X=X, sample_weight=weights)


def KLIEP(
    base_estimator=None,
    gamma=1.0,
    cv=5,
    n_centers=100,
    tol=1e-6,
    max_iter=1000,
    random_state=None,
):
    """KLIEP pipeline adapter and estimator.

    see [1]_ for details.

    Parameters
    ----------
    base_estimator : sklearn estimator, default=LogisticRegression()
        estimator used for fitting and prediction
    gamma : float or array like
        Parameters for the kernels.
        If array like, compute the likelihood cross validation to choose
        the best parameters for the RBF kernel.
        If float, solve the optimization for the given kernel parameter.
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

    Returns
    -------
    pipeline : sklearn pipeline
        Pipeline containing the KLIEP adapter and the base estimator.

    References
    ----------
    .. [1] Masashi Sugiyama et. al. Direct Importance Estimation with Model Selection
           and Its Application to Covariate Shift Adaptation.
           In NeurIPS, 2007.
    """
    if base_estimator is None:
        base_estimator = LogisticRegression().set_fit_request(sample_weight=True)
    return make_da_pipeline(
        KLIEPAdapter(
            gamma=gamma, cv=cv, n_centers=n_centers, tol=tol,
            max_iter=max_iter, random_state=random_state
        ),
        base_estimator,
    )

class NearestNeighborDensityAdapter(BaseAdapter):
    """Adapter based on re-weighting samples using a 1NN,

    See: [Loog, 2012] Loog, M. (2012). 
    Nearest neighbor-based importance weighting. 
    In 2012 IEEE International Workshop on Machine 
    Learning for Signal Processing, pages 1–6. IEEE.

    Parameters
    ----------
    base_estimator : estimator object, optional
        The estimator to use to estimate the densities of source and target
        observations. If None, a KNeighborsClassifier(n_neighbors=1) estimator
        is used.

    Attributes
    ----------
    weight_estimator_source_ : object
        The estimator object fitted on the source data.
    weight_estimator_target_ : object
        The estimator object fitted on the target data.
    """

    def __init__(self, base_estimator=None):
        super().__init__()
        self.base_estimator = base_estimator or KNeighborsClassifier(
            n_neighbors=1)

    def fit(self, X, y=None, sample_domain=None):
        """Fit adaptation parameters.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The source data.
        y : array-like, shape (n_samples,)
            The source labels.
        sample_domain : array-like, shape (n_samples,)
            The domain labels (same as sample_domain).

        Returns
        -------
        self : object
            Returns self.
        """
        X, sample_domain = check_X_domain(X, sample_domain)
        X_source, X_target = source_target_split(X, sample_domain=sample_domain)
        self.X_source_fit = X_source
        indices_source = np.arange(X_source.shape[0])

        self.estimator = clone(self.base_estimator)
        self.estimator.fit(X_source, indices_source)

        self.weight_estimator_source_ = self.estimator
        self.weight_estimator_target_ = self.estimator

        return self

    def adapt(self, X, y=None, sample_domain=None):
        """Predict adaptation (weights, sample or labels).

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The source data.
        y : array-like, shape (n_samples,)
            The source labels.
        sample_domain : array-like, shape (n_samples,)
            The domain labels (same as sample_domain).

        Returns
        -------
        output : :class:`skada.base.AdaptationOutput`
            Dictionary-like object, with the following attributes.

            X_t : array-like, shape (n_samples, n_components)
                The data (same as X).
            weights : array-like, shape (n_samples,)
                The weights of the samples.
        """
        check_is_fitted(self)
        X, sample_domain = check_X_domain(X, sample_domain)
        source_idx = extract_source_indices(sample_domain)

        # xxx(okachaiev): move this to API
        if source_idx.sum() > 0:
            source_idx, = np.where(source_idx)
            indices_source = np.arange(X[source_idx].shape[0])
            if np.array_equal(self.X_source_fit, X[source_idx]):
                estimator = self.estimator
            else:
                estimator = clone(self.base_estimator)
                estimator.fit(X[source_idx], indices_source)
            predictions = estimator.predict(X[~source_idx])
            weights = np.array(
                [np.count_nonzero(predictions == i) for i in indices_source])
        else:
            weights = None
        return AdaptationOutput(X=X, sample_weight=weights)


def NearestNeighborReweightDensity(
    base_estimator=None,
    weight_estimator=None,
):
    """Density re-weighting pipeline adapter and estimator.

    Parameters
    ----------
    base_estimator : sklearn estimator, default=None
        estimator used for fitting and prediction
    weight_estimator : estimator object, optional
        The estimator to use to estimate the densities of source and target
        observations. If None, a KernelDensity estimator is used.

    Returns
    -------
    pipeline : sklearn pipeline
        Pipeline containing the ReweightDensity adapter and the base estimator.
    """
    if base_estimator is None:
        base_estimator = LogisticRegression().set_fit_request(sample_weight=True)

    return make_da_pipeline(
        NearestNeighborDensityAdapter(base_estimator=weight_estimator),
        base_estimator,
    )
