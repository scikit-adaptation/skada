# Author: Theo Gnassounou <theo.gnassounou@inria.fr>
#         Remi Flamary <remi.flamary@polytechnique.edu>
#         Oleksii Kachaiev <kachayev@gmail.com>
#         Bueno Ruben <ruben.bueno@polytechnique.edu>
#
# License: BSD 3-Clause

import warnings

import numpy as np
from scipy.stats import multivariate_normal
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import pairwise_kernels, KERNEL_PARAMS
from sklearn.model_selection import check_cv
from sklearn.neighbors import KernelDensity
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_is_fitted
from sklearn.neighbors import KNeighborsClassifier

from .base import AdaptationOutput, BaseAdapter, clone
from .utils import check_X_domain, source_target_split, extract_source_indices, qp_solve
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

            X : array-like, shape (n_samples, n_components)
                The data (same as X).
            sample_weight : array-like, shape (n_samples,)
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
        return AdaptationOutput(X, sample_weight=weights)


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

            X : array-like, shape (n_samples, n_components)
                The data (same as X).
            sample_weight : array-like, shape (n_samples,)
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
        return AdaptationOutput(X, sample_weight=weights)


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

            X : array-like, shape (n_samples, n_components)
                The data (same as X).
            sample_weight : array-like, shape (n_samples,)
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
        return AdaptationOutput(X, sample_weight=weights)


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

            X : array-like, shape (n_samples, n_components)
                The data (same as X).
            sample_weight : array-like, shape (n_samples,)
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
        return AdaptationOutput(X, sample_weight=weights)


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

    def __init__(
            self,
            weights='uniform',
            algorithm='auto',
            leaf_size=30,
            p=2,
            metric='minkowski',
            metric_params=None,
            n_jobs=None,
            laplace_smoothing=False):
        super().__init__()
        self.laplace_smoothing = laplace_smoothing
        self.base_estimator = KNeighborsClassifier(
            n_neighbors=1,
            weights=weights,
            algorithm=algorithm,
            leaf_size=leaf_size,
            p=p,
            metric=metric,
            metric_params=metric_params,
            n_jobs=n_jobs)

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

        self.estimator_ = clone(self.base_estimator)
        self.estimator_.fit(X_source, indices_source)

        return self

    def get_weights(self, Xs, Xt):
        indices_source = np.arange(Xs.shape[0])
        estimator = clone(self.base_estimator)
        estimator.fit(Xs, indices_source)
        predictions = estimator.predict(Xt)

        unique, counts = np.unique(predictions, return_counts=True)
        weights = np.ones(Xs.shape[0]) * float(self.laplace_smoothing)
        weights[unique] += counts
        return weights

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
                estimator = self.estimator_
            else:
                estimator = clone(self.base_estimator)
                estimator.fit(X[source_idx], indices_source)
            weights = np.ones(X.shape[0])
            weights[source_idx] = self.get_weights(X[source_idx], X[~source_idx])
        else:
            weights = None
        return AdaptationOutput(X=X, sample_weight=weights)


def NearestNeighborReweightDensity(
    base_estimator=None,
    laplace_smoothing=False,
    weights='uniform',
    algorithm='auto',
    leaf_size=30,
    p=2,
    metric='minkowski',
    metric_params=None,
    n_jobs=None,
):
    """Density re-weighting pipeline adapter and estimator.

    The last 7 parameters are the parametters from the 1NN estimator that
    will be used to estimate the weights in the `adapt` method

    Parameters
    ----------
    base_estimator : sklearn estimator, default=None
        estimator used for fitting and prediction

    laplace_smoothing : bool, default=False, optional
        True if we want to use laplace smoothing, and
        thus adding 1 to all our weights (to prevent some
        of them to be 0)

    weights : {'uniform', 'distance'}, callable or None, default='uniform'
        Weight function used in prediction.  Possible values:

        - 'uniform' : uniform weights.  All points in each neighborhood
          are weighted equally.
        - 'distance' : weight points by the inverse of their distance.
          in this case, closer neighbors of a query point will have a
          greater influence than neighbors which are further away.
        - [callable] : a user-defined function which accepts an
          array of distances, and returns an array of the same shape
          containing the weights.

        Refer to the example entitled
        :ref:`sphx_glr_auto_examples_neighbors_plot_classification.py`
        showing the impact of the `weights` parameter on the decision
        boundary.

    algorithm : {'auto', 'ball_tree', 'kd_tree', 'brute'}, default='auto'
        Algorithm used to compute the nearest neighbors:

        - 'ball_tree' will use :class:`BallTree`
        - 'kd_tree' will use :class:`KDTree`
        - 'brute' will use a brute-force search.
        - 'auto' will attempt to decide the most appropriate algorithm
          based on the values passed to :meth:`fit` method.

        Note: fitting on sparse input will override the setting of
        this parameter, using brute force.

    leaf_size : int, default=30
        Leaf size passed to BallTree or KDTree.  This can affect the
        speed of the construction and query, as well as the memory
        required to store the tree.  The optimal value depends on the
        nature of the problem.

    p : float, default=2
        Power parameter for the Minkowski metric. When p = 1, this is equivalent
        to using manhattan_distance (l1), and euclidean_distance (l2) for p = 2.
        For arbitrary p, minkowski_distance (l_p) is used. This parameter is expected
        to be positive.

    metric : str or callable, default='minkowski'
        Metric to use for distance computation. Default is "minkowski", which
        results in the standard Euclidean distance when p = 2. See the
        documentation of `scipy.spatial.distance
        <https://docs.scipy.org/doc/scipy/reference/spatial.distance.html>`_ and
        the metrics listed in
        :class:`~sklearn.metrics.pairwise.distance_metrics` for valid metric
        values.

        If metric is "precomputed", X is assumed to be a distance matrix and
        must be square during fit. X may be a :term:`sparse graph`, in which
        case only "nonzero" elements may be considered neighbors.

        If metric is a callable function, it takes two arrays representing 1D
        vectors as inputs and must return one value indicating the distance
        between those vectors. This works for Scipy's metrics, but is less
        efficient than passing the metric name as a string.

    metric_params : dict, default=None
        Additional keyword arguments for the metric function.

    n_jobs : int, default=None
        The number of parallel jobs to run for neighbors search.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.
        Doesn't affect :meth:`fit` method.

    Returns
    -------
    pipeline : sklearn pipeline
        Pipeline containing the ReweightDensity adapter and the base estimator.
    """
    if base_estimator is None:
        base_estimator = LogisticRegression().set_fit_request(sample_weight=True)

    return make_da_pipeline(
        NearestNeighborDensityAdapter(
            base_estimator=weight_estimator,
            laplace_smoothing=laplace_smoothing),
        base_estimator,
    )


class KMMAdapter(BaseAdapter):
    """Kernel Mean Matching (KMM).

    The idea of KMM is to find an importance estimate w(x) such that
    the Maximum Mean Discrepancy (MMD) divergence between the target
    input density p_target(x) and the reweighted source input density
    w(x)p_source(x) is minimized.

    See [1]_ for details.

    Parameters
    ----------
    kernel : str, default="rbf"
        Kernel
    gamma : float, None
        Parameters for the kernels.
    degree : int, 3
        Parameters for the kernels.
    coef0 : float, default
        Parameters for the kernels.
    B : float, default=1000.
        Weight upper bound.
    eps : float, default=None
        KMM tolerance parameter. If `None`, eps is set to
        (sqrt(n_samples_source) - 1) / sqrt(n_samples_source).
    tol : float, default=1e-6
        Tolerance for the stopping criterion in the optimization.
    max_iter : int, default=100
        Number of maximum iteration before stopping the optimization.
    smooth_weights : bool, default=False
        If True, the weights are "smoothed" using the kernel function.

    Attributes
    ----------
    `source_weights_` : array-like, shape (n_samples,)
        The learned source weights.
    `X_source_` : array-like, shape (n_samples, n_features)
        The source data.

    References
    ----------
    .. [1] J. Huang, A. Gretton, K. Borgwardt, B. Schölkopf and A. J. Smola.
           'Correcting sample selection bias by unlabeled data.'
           In NIPS, 2007.
    """

    def __init__(
        self,
        kernel="rbf",
        gamma=None,
        degree=3,
        coef0=1,
        B=1000.,
        eps=None,
        tol=1e-6,
        max_iter=1000,
        smooth_weights=False,
    ):
        super().__init__()
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.B = B
        self.eps = eps
        self.tol = tol
        self.max_iter = max_iter
        self.smooth_weights = smooth_weights

        if kernel not in KERNEL_PARAMS:
            kernel_list = str(list(KERNEL_PARAMS.keys()))
            raise ValueError("`kernel` argument should be included in %s,"
                             " got '%s'" % (kernel_list, str(kernel)))

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

        self.source_weights_ = self._weights_optimization(X_source, X_target)
        self.X_source_ = X_source

        return self

    def _weights_optimization(self, X_source, X_target):
        """Weight optimization"""
        Kss = pairwise_kernels(X_source,
                               metric=self.kernel,
                               filter_params=True,
                               gamma=self.gamma,
                               degree=self.degree,
                               coef0=self.coef0)
        Kst = pairwise_kernels(X_source,
                               X_target,
                               metric=self.kernel,
                               filter_params=True,
                               gamma=self.gamma,
                               degree=self.degree,
                               coef0=self.coef0)
        Ns = Kss.shape[0]
        kappa = Ns * Kst.mean(axis=1)

        if self.eps is None:
            eps = (np.sqrt(Ns) - 1) / np.sqrt(Ns)
        else:
            eps = self.eps

        A = np.stack([np.ones(Ns), -np.ones(Ns)], axis=0)
        b = np.array([Ns*(1+eps), -Ns*(1-eps)])

        weights, _ = qp_solve(Kss, -kappa, A, b,
                              lb=np.zeros(Ns),
                              ub=np.ones(Ns)*self.B,
                              tol=self.tol,
                              max_iter=self.max_iter)

        weights = np.array(weights).ravel()
        return weights

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
        X, sample_domain = check_X_domain(
            X,
            sample_domain
        )

        source_idx = extract_source_indices(sample_domain)

        if source_idx.sum() > 0:
            if (np.array_equal(self.X_source_, X[source_idx])
               and not self.smooth_weights):
                source_weights = self.source_weights_
            else:
                K = pairwise_kernels(X[source_idx], self.X_source_,
                                     metric=self.kernel, filter_params=True,
                                     gamma=self.gamma, degree=self.degree,
                                     coef0=self.coef0)
                source_weights = K.dot(self.source_weights_)
            source_idx = np.where(source_idx)
            weights = np.zeros(X.shape[0], dtype=source_weights.dtype)
            weights[source_idx] = source_weights
        else:
            weights = None
        return AdaptationOutput(X=X, sample_weight=weights)


def KMM(
    base_estimator=None,
    kernel="rbf",
    gamma=None,
    degree=3,
    coef0=1,
    B=1000.,
    eps=None,
    tol=1e-6,
    max_iter=1000,
    smooth_weights=False,
):
    """KMM pipeline adapter and estimator.

    see [1]_ for details.

    Parameters
    ----------
    base_estimator : sklearn estimator, default=LogisticRegression()
        estimator used for fitting and prediction
    kernel : str, default="rbf"
        Kernel
    gamma : float, None
        Parameters for the kernels.
    degree : int, 3
        Parameters for the kernels.
    coef0 : float, default
        Parameters for the kernels.
    B : float, default=1000.
        Weight upper bound.
    eps : float, default=None
        KMM tolerance parameter. If `None`, eps is set to
        (sqrt(n_samples_source) - 1) / sqrt(n_samples_source).
    tol : float, default=1e-6
        Tolerance for the stopping criterion in the optimization.
    max_iter : int, default=100
        Number of maximum iteration before stopping the optimization.
    smooth_weights : bool, default=False
        If True, the weights are "smoothed" using the kernel function.
        Pipeline containing the KMM adapter and the base estimator.

    References
    ----------
    .. [1] J. Huang, A. Gretton, K. Borgwardt, B. Schölkopf and A. J. Smola.
           'Correcting sample selection bias by unlabeled data.'
           In NIPS, 2007.
    """
    if base_estimator is None:
        base_estimator = LogisticRegression().set_fit_request(sample_weight=True)
    return make_da_pipeline(
        KMMAdapter(
            kernel=kernel,
            gamma=gamma,
            degree=degree,
            coef0=coef0,
            B=B,
            eps=eps,
            tol=tol,
            max_iter=max_iter,
            smooth_weights=smooth_weights
        ),
        base_estimator,
    )
