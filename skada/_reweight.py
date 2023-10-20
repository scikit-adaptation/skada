import warnings

import numpy as np
from scipy.stats import multivariate_normal
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.model_selection import check_cv
from sklearn.neighbors import KernelDensity
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_is_fitted

from .base import BaseAdapter, DomainAwareEstimator, SingleAdapterMixin, SingleEstimatorMixin, clone
from ._utils import _estimate_covariance, check_X_y_domain, check_X_domain


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
        X_t : array-like, shape (n_samples, n_components)
            The data (same as X).
        y_t : array-like, shape (n_samples,)
            The labels (same as y).
        weights : array-like, shape (n_samples,)
            The weights of the samples.
        """
        X_source, y_source, X_target, y_target = check_X_y_domain(
            X,
            y,
            sample_domain,
            allow_multi_source=False,
            allow_multi_target=False
        )
        ws = self.weight_estimator_source_.score_samples(X_source)
        wt = self.weight_estimator_target_.score_samples(X_source)
        weights = np.exp(wt - ws)
        weights /= weights.sum()
        return X, y, sample_domain, weights

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
        # xxx(okachaiev): that's the reason we need a way to cache this call
        X, X_target = check_X_domain(
            X,
            sample_domain,
            allow_multi_source=False,
            allow_multi_target=False
        )
        self.weight_estimator_source_ = clone(self.weight_estimator)
        self.weight_estimator_target_ = clone(self.weight_estimator)
        self.weight_estimator_source_.fit(X)
        self.weight_estimator_target_.fit(X_target)


class ReweightDensity(SingleEstimatorMixin, SingleAdapterMixin, DomainAwareEstimator):

    def __init__(self, weight_estimator=None, **kwargs):
        self.weight_estimator = weight_estimator
        super().__init__(base_adapter=ReweightDensityAdapter(weight_estimator), **kwargs)


class GaussianReweightDensityAdapter(BaseAdapter):
    """Gaussian approximation re-weighting method.

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

    def __init__(self, reg='auto'):
        super().__init__()
        self.reg = reg

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
        X_t : array-like, shape (n_samples, n_components)
            The data (same as X).
        y_t : array-like, shape (n_samples,)
            The labels (same as y).
        weights : array-like, shape (n_samples,)
            The weights of the samples.
        """
        check_is_fitted(self)
        # xxx(okachaiev): what should happen to target here?
        # i guess we just return it "as it". so it might be
        # good to have helper like 'AdapterOnSource' + 'TargetAsIs'
        # keeping this in mind, we have to consider option for
        # the adapter to setup where and how it should be used
        X_source, y_source, X_target, y_target = check_X_y_domain(
            X,
            y,
            sample_domain,
            allow_multi_source=False,
            allow_multi_target=False
        )
        gaussian_target = multivariate_normal.pdf(
            X_source, self.mean_target_, self.cov_target_
        )
        gaussian_source = multivariate_normal.pdf(
            X_source, self.mean_source_, self.cov_source_
        )
        weights = gaussian_target / gaussian_source
        return X, y, sample_domain, weights

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
        X, X_target = check_X_domain(
            X,
            sample_domain,
            allow_multi_source=False,
            allow_multi_target=False
        )
        self.mean_source_ = X.mean(axis=0)
        self.cov_source_ = _estimate_covariance(X, shrinkage=self.reg)
        self.mean_target_ = X_target.mean(axis=0)
        self.cov_target_ = _estimate_covariance(X_target, shrinkage=self.reg)


# xxx(okachaiev): chain of subclasses is an incredibly fragile design decision
# try to use meta classes instead (same how it's done for routing)
class GaussianReweightDensity(SingleEstimatorMixin, SingleAdapterMixin, DomainAwareEstimator):

    def __init__(self, reg='auto', **kwargs):
        self.reg = reg
        super().__init__(base_adapter=GaussianReweightDensityAdapter(reg), **kwargs)


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
        X_t : array-like, shape (n_samples, n_components)
            The data (same as X).
        y_t : array-like, shape (n_samples,)
            The labels (same as y).
        weights : array-like, shape (n_samples,)
            The weights of the samples.
        """
        X_source, y_source, X_target, y_target = check_X_y_domain(
            X,
            y,
            sample_domain,
            allow_multi_source=False,
            allow_multi_target=False
        )
        weights = self.domain_classifier_.predict_proba(X_source)[:, 1]
        return X, y, sample_domain, weights

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
        X, y, X_target, y_target = check_X_y_domain(
            X,
            y,
            sample_domain,
            allow_multi_source=False,
            allow_multi_target=False
        )
        self.domain_classifier_ = clone(self.domain_classifier)
        # xxx(okachaiev): is this actually correct???
        y_domain = np.concatenate((len(X) * [0], len(X_target) * [1]))
        self.domain_classifier_.fit(np.concatenate((X, X_target)), y_domain)
        return self


class DiscriminatorReweightDensity(SingleEstimatorMixin, SingleAdapterMixin, DomainAwareEstimator):

    def __init__(self, domain_classifier=None, **kwargs):
        self.domain_classifier = domain_classifier
        base_adapter = DiscriminatorReweightDensityAdapter(domain_classifier)
        super().__init__(base_adapter=base_adapter, **kwargs)


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
        X_t : array-like, shape (n_samples, n_components)
            The data (same as X).
        y_t : array-like, shape (n_samples,)
            The labels (same as y).
        weights : array-like, shape (n_samples,)
            The weights of the samples.
        """
        X_source, y_source, X_target, y_target = check_X_y_domain(
            X,
            y,
            sample_domain,
            allow_multi_source=False,
            allow_multi_target=False
        )
        A = pairwise_kernels(X, self.centers_, metric="rbf", gamma=self.best_gamma_)
        weights = A @ self.alpha_
        return X, y, sample_domain, weights

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
        X, X_target = check_X_domain(
            X,
            sample_domain,
            allow_multi_source=False,
            allow_multi_target=False
        )
        if isinstance(self.gamma, list):
            self.best_gamma_ = self._likelihood_cross_validation(
                self.gamma, X, X_target
            )
        else:
            self.best_gamma_ = self.gamma
        self.alpha_, self.centers_ = self._weights_optimization(
            self.best_gamma_, X, X_target
        )

    def _weights_optimization(self, gamma, X, X_target):
        """Optimization loop."""
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
                alpha, centers = self._weights_optimization(
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


class KLIEP(SingleEstimatorMixin, SingleAdapterMixin, DomainAwareEstimator):

    def __init__(
        self,
        gamma,  # XXX use the auto/scale mode as done with sklearn SVC
        cv=5,
        n_centers=100,
        tol=1e-6,
        max_iter=1000,
        random_state=None,
        **kwargs
    ):
        self.gamma = gamma
        self.cv = cv
        self.n_centers = n_centers
        self.tol = tol
        self.max_iter = max_iter
        self.random_state = random_state
        base_adapter = KLIEPAdapter(
            gamma,
            cv=cv,
            n_centers=n_centers,
            tol=tol,
            max_iter=max_iter,
            random_state=random_state
        )
        super().__init__(base_adapter=base_adapter, **kwargs)
