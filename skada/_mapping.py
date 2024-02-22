# Author: Theo Gnassounou <theo.gnassounou@inria.fr>
#         Remi Flamary <remi.flamary@polytechnique.edu>
#         Oleksii Kachaiev <kachayev@gmail.com>
#
# License: BSD 3-Clause

from abc import abstractmethod

import numpy as np
from ot import da
from scipy.optimize import minimize
from sklearn.metrics.pairwise import pairwise_kernels

from .base import BaseAdapter, clone
from .utils import (
    check_X_domain,
    check_X_y_domain,
    source_target_split,
    source_target_merge
)
from ._utils import (
    _estimate_covariance,
    _find_y_type
)

from ._pipeline import make_da_pipeline
from sklearn.svm import SVC


class BaseOTMappingAdapter(BaseAdapter):
    """Base class for all DA estimators implemented using OT mapping.

    Each implementation has to provide `_create_transport_estimator` callback
    to create OT object using parameters saved in the constructor.
    """

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
        X, y, sample_domain = check_X_y_domain(X, y, sample_domain)
        X, X_target, y, y_target = source_target_split(
            X, y, sample_domain=sample_domain
        )
        transport = self._create_transport_estimator()
        self.ot_transport_ = clone(transport)
        self.ot_transport_.fit(Xs=X, ys=y, Xt=X_target, yt=y_target)
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
            The domain labels.

        Returns
        -------
        X_t : array-like, shape (n_samples, n_components)
            The data transformed to the target subspace.
        y_t : array-like, shape (n_samples,)
            The labels (same as y).
        weights : array-like, shape (n_samples,)
            The weights of the samples.
        """
        # xxx(okachaiev): implement auto-infer for sample_domain
        X, sample_domain = check_X_domain(
            X,
            sample_domain,
            allow_multi_source=True,
            allow_multi_target=True
        )
        X_source, X_target = source_target_split(X, sample_domain=sample_domain)
        # in case of prediction we would get only target samples here,
        # thus there's no need to perform any transformations
        if X_source.shape[0] > 0:
            X_source = self.ot_transport_.transform(Xs=X_source)
        X_adapt, _ = source_target_merge(
            X_source, X_target, sample_domain=sample_domain
        )
        return X_adapt

    @abstractmethod
    def _create_transport_estimator(self):
        pass


class OTMappingAdapter(BaseOTMappingAdapter):
    """Domain Adaptation Using Optimal Transport.

    Parameters
    ----------
    metric : str, optional (default="sqeuclidean")
        The ground metric for the Wasserstein problem
    norm : {'median', 'max', 'log', 'loglog'} (default=None)
        If given, normalize the ground metric to avoid numerical errors that
        can occur with large metric values.
    max_iter : int, optional (default=100_000)
        The maximum number of iterations before stopping OT algorithm if it
        has not converged.

    Attributes
    ----------
    ot_transport_ : object
        The OT object based on Earth Mover's distance
        fitted on the source and target data.

    References
    ----------
    .. [1] N. Courty, R. Flamary, D. Tuia and A. Rakotomamonjy,
           Optimal Transport for Domain Adaptation, in IEEE
           Transactions on Pattern Analysis and Machine Intelligence
    """

    def __init__(
        self,
        metric="sqeuclidean",
        norm=None,
        max_iter=100_000,
    ):
        super().__init__()
        self.metric = metric
        self.norm = norm
        self.max_iter = max_iter

    def _create_transport_estimator(self):
        return da.EMDTransport(
            metric=self.metric,
            norm=self.norm,
            max_iter=self.max_iter,
        )


def OTMapping(
    base_estimator=None,
    metric="sqeuclidean",
    norm=None,
    max_iter=100000
):
    """OTmapping pipeline with adapter and estimator.

    see [1]_ for details.

    Parameters
    ----------
    base_estimator : object, optional (default=None)
        The base estimator to fit on the target dataset.
    metric : str, optional (default="sqeuclidean")
        The ground metric for the Wasserstein problem
    norm : {'median', 'max', 'log', 'loglog'} (default=None)
        If given, normalize the ground metric to avoid numerical errors that
        can occur with large metric values.
    max_iter : int, optional (default=100_000)
        The maximum number of iterations before stopping OT algorithm if it
        has not converged.

    Returns
    -------
    pipeline : Pipeline
        Pipeline containing OTMapping adapter and base estimator.

    References
    ----------
    .. [1] N. Courty, R. Flamary, D. Tuia and A. Rakotomamonjy,
           Optimal Transport for Domain Adaptation, in IEEE
           Transactions on Pattern Analysis and Machine Intelligence
    """
    if base_estimator is None:
        base_estimator = SVC(kernel="rbf")

    return make_da_pipeline(
        OTMappingAdapter(metric=metric, norm=norm, max_iter=max_iter),
        base_estimator,
    )


class EntropicOTMappingAdapter(BaseOTMappingAdapter):
    """Domain Adaptation Using Optimal Transport.

    Parameters
    ----------
    reg_e : float, default=1
        Entropic regularization parameter.
    metric : str, optional (default="sqeuclidean")
        The ground metric for the Wasserstein problem.
    norm : {'median', 'max', 'log', 'loglog'} (default=None)
        If given, normalize the ground metric to avoid numerical errors that
        can occur with large metric values.
    max_iter : int, float, optional (default=1000)
        The minimum number of iteration before stopping the optimization
        of the Sinkhorn algorithm if it has not converged
    tol : float, optional (default=10e-9)
        The precision required to stop the optimization of the Sinkhorn
        algorithm.

    Attributes
    ----------
    ot_transport_ : object
        The OT object based on Sinkhorn Algorithm
        fitted on the source and target data.

    References
    ----------
    .. [1] N. Courty, R. Flamary, D. Tuia and A. Rakotomamonjy,
           Optimal Transport for Domain Adaptation, in IEEE
           Transactions on Pattern Analysis and Machine Intelligence
    """

    def __init__(
        self,
        reg_e=1.,
        metric="sqeuclidean",
        norm=None,
        max_iter=1000,
        tol=10e-9,
    ):
        super().__init__()
        self.reg_e = reg_e
        self.metric = metric
        self.norm = norm
        self.max_iter = max_iter
        self.tol = tol

    def _create_transport_estimator(self):
        return da.SinkhornTransport(
            reg_e=self.reg_e,
            metric=self.metric,
            norm=self.norm,
            max_iter=self.max_iter,
            tol=self.tol,
        )


def EntropicOTMapping(
    base_estimator=None,
    metric="sqeuclidean",
    norm=None,
    max_iter=1000,
    reg_e=1.,
    tol=1e-8,
):
    """EntropicOTMapping pipeline with adapter and estimator.

    see [1]_ for details.

    Parameters
    ----------
    base_estimator : object, optional (default=None)
        The base estimator to fit on the target dataset.
    reg_e : float, default=1
        Entropic regularization parameter.
    metric : str, optional (default="sqeuclidean")
        The ground metric for the Wasserstein problem.
    norm : {'median', 'max', 'log', 'loglog'} (default=None)
        If given, normalize the ground metric to avoid numerical errors that
        can occur with large metric values.
    max_iter : int, float, optional (default=1000)
        The minimum number of iteration before stopping the optimization
        of the Sinkhorn algorithm if it has not converged
    tol : float, optional (default=10e-9)
        The precision required to stop the optimization of the Sinkhorn
        algorithm.

    Returns
    -------
    pipeline : Pipeline
        Pipeline containing EntropicOTMapping adapter and base estimator.

    References
    ----------
    .. [1] N. Courty, R. Flamary, D. Tuia and A. Rakotomamonjy,
           Optimal Transport for Domain Adaptation, in IEEE
           Transactions on Pattern Analysis and Machine Intelligence
    """
    if base_estimator is None:
        base_estimator = SVC(kernel="rbf")

    return make_da_pipeline(
        EntropicOTMappingAdapter(
            metric=metric,
            norm=norm,
            max_iter=max_iter,
            reg_e=reg_e,
            tol=tol
        ),
        base_estimator,
    )


class ClassRegularizerOTMappingAdapter(BaseOTMappingAdapter):
    """Domain Adaptation Using Optimal Transport.

    Parameters
    ----------
    reg_e : float, default=1
        Entropic regularization parameter.
    reg_cl : float, default=0.1
        Class regularization parameter.
    norm : str, default="lpl1"
        Norm use for the regularizer of the class labels.
        If "lpl1", use the lp l1 norm.
        If "l1l2", use the l1 l2 norm.
    metric : str, optional (default="sqeuclidean")
        The ground metric for the Wasserstein problem
    max_iter : int, float, optional (default=10)
        The minimum number of iteration before stopping the optimization
        algorithm if it has not converged
    max_inner_iter : int, float, optional (default=200)
        The number of iteration in the inner loop
    tol : float, optional (default=10e-9)
        Stop threshold on error (inner sinkhorn solver) (>0)

    Attributes
    ----------
    ot_transport_ : object
        The OT object based on Sinkhorn Algorithm
        + class regularization fitted on the source
        and target data.

    References
    ----------
    .. [1] N. Courty, R. Flamary, D. Tuia and A. Rakotomamonjy,
           Optimal Transport for Domain Adaptation, in IEEE
           Transactions on Pattern Analysis and Machine Intelligence
    """

    def __init__(
        self,
        reg_e=1.,
        reg_cl=0.1,
        norm="lpl1",
        metric="sqeuclidean",
        max_iter=10,
        max_inner_iter=200,
        tol=10e-9,
    ):
        super().__init__()
        self.reg_e = reg_e
        self.reg_cl = reg_cl
        self.norm = norm
        self.metric = metric
        self.max_iter = max_iter
        self.max_inner_iter = max_inner_iter
        self.tol = tol

    def _create_transport_estimator(self):
        assert self.norm in ["lpl1", "l1l2"], "Unknown norm"

        if self.norm == "lpl1":
            transport_cls = da.SinkhornLpl1Transport
        elif self.norm == "l1l2":
            transport_cls = da.SinkhornL1l2Transport
        return transport_cls(
            reg_e=self.reg_e,
            reg_cl=self.reg_cl,
            metric=self.metric,
            max_iter=self.max_iter,
            max_inner_iter=self.max_inner_iter,
            tol=self.tol,
        )


def ClassRegularizerOTMapping(
    base_estimator=SVC(kernel="rbf"),
    metric="sqeuclidean",
    norm="lpl1",
    max_iter=10,
    max_inner_iter=200,
    reg_e=1.,
    reg_cl=0.1,
    tol=1e-8,
):
    """ClassRegularizedOTMapping pipeline with adapter and estimator.

    see [1]_ for details.

    Parameters
    ----------
    base_estimator : object, optional (default=SVC(kernel="rbf"))
        The base estimator to fit on the target dataset.
    reg_e : float, default=1
        Entropic regularization parameter.
    reg_cl : float, default=0.1
        Class regularization parameter.
    norm : str, default="lpl1"
        Norm use for the regularizer of the class labels.
        If "lpl1", use the lp l1 norm.
        If "l1l2", use the l1 l2 norm.
    metric : str, optional (default="sqeuclidean")
        The ground metric for the Wasserstein problem
    max_iter : int, float, optional (default=10)
        The minimum number of iteration before stopping the optimization
        algorithm if it has not converged
    max_inner_iter : int, float, optional (default=200)
        The number of iteration in the inner loop
    tol : float, optional (default=10e-9)
        Stop threshold on error (inner sinkhorn solver) (>0)

    Returns
    -------
    pipeline : Pipeline
        Pipeline containing ClassRegularizerOTMapping adapter and base estimator.

    References
    ----------
    .. [1] N. Courty, R. Flamary, D. Tuia and A. Rakotomamonjy,
           Optimal Transport for Domain Adaptation, in IEEE
           Transactions on Pattern Analysis and Machine Intelligence
    """
    ot_mapping = make_da_pipeline(
        ClassRegularizerOTMappingAdapter(
            metric=metric,
            norm=norm,
            max_iter=max_iter,
            max_inner_iter=max_inner_iter,
            reg_e=reg_e,
            reg_cl=reg_cl,
            tol=tol
        ),
        base_estimator,
    )
    return ot_mapping


class LinearOTMappingAdapter(BaseOTMappingAdapter):
    """Domain Adaptation Using Optimal Transport.

    Parameters
    ----------
    reg : float, (default=1e-08)
        regularization added to the diagonals of covariances.
    bias: bool, optional (default=True)
        estimate bias.

    Attributes
    ----------
    ot_transport_ : object
        The OT object based on linear operator between empirical
        distributions fitted on the source
        and target data.
    """

    def __init__(self, reg=1e-08, bias=True):
        super().__init__()
        self.reg = reg
        self.bias = bias

    def _create_transport_estimator(self):
        return da.LinearTransport(reg=self.reg, bias=self.bias)


def LinearOTMapping(
    base_estimator=None,
    reg=1.,
    bias=True,
):
    """Returns a the linear OT mapping method with adapter and estimator.

    see [1]_ for details.

    Parameters
    ----------
    base_estimator : object, optional (default=None)
        The base estimator to fit on the target dataset.
    reg : float, (default=1e-08)
        regularization added to the diagonals of covariances.
    bias: bool, optional (default=True)
        estimate bias.

    Returns
    -------
    pipeline : Pipeline
        Pipeline containing linear OT mapping adapter and base estimator.

    References
    ----------
    .. [1] N. Courty, R. Flamary, D. Tuia and A. Rakotomamonjy,
           Optimal Transport for Domain Adaptation, in IEEE
           Transactions on Pattern Analysis and Machine Intelligence
    """
    if base_estimator is None:
        base_estimator = SVC(kernel="rbf")

    return make_da_pipeline(
        LinearOTMappingAdapter(
            reg=reg,
            bias=bias,
        ),
        base_estimator,
    )


def _sqrtm(C):
    r"""Square root of SPD matrices.

    The matrix square root of a SPD matrix C is defined by:

    .. math::
        \mathbf{D} =
        \mathbf{V} \left( \mathbf{\Lambda} \right)^{1/2} \mathbf{V}^\top

    where :math:`\mathbf{\Lambda}` is the diagonal matrix of eigenvalues
    and :math:`\mathbf{V}` the eigenvectors of :math:`\mathbf{C}`.

    Parameters
    ----------
    C : ndarray, shape (n, n)
        SPD matrix.

    Returns
    -------
    D : ndarray, shape (n, n)
        Matrix inverse square root of C.
    """
    eigvals, eigvecs = np.linalg.eigh(C)
    return (eigvecs * np.sqrt(eigvals)) @ eigvecs.T


def _invsqrtm(C):
    r"""Inverse square root of SPD matrices.

    The matrix inverse square root of a SPD matrix C is defined by:

    .. math::
        \mathbf{D} =
        \mathbf{V} \left( \mathbf{\Lambda} \right)^{-1/2} \mathbf{V}^\top

    where :math:`\mathbf{\Lambda}` is the diagonal matrix of eigenvalues
    and :math:`\mathbf{V}` the eigenvectors of :math:`\mathbf{C}`.

    Parameters
    ----------
    C : ndarray, shape (n, n)
        SPD matrix.

    Returns
    -------
    D : ndarray, shape (n, n)
        Matrix inverse square root of C.
    """
    eigvals, eigvecs = np.linalg.eigh(C)
    return (eigvecs * 1. / np.sqrt(eigvals)) @ eigvecs.T


class CORALAdapter(BaseAdapter):
    """Estimator based on Correlation Alignment [1]_.

    Parameters
    ----------
    reg : 'auto' or float, default="auto"
        The regularization parameter of the covariance estimator.
        Possible values:

          - None: no shrinkage).
          - 'auto': automatic shrinkage using the Ledoit-Wolf lemma.
          - float between 0 and 1: fixed shrinkage parameter.

    Attributes
    ----------
    cov_source_inv_sqrt_: array, shape (n_features, n_features)
        Inverse of the square root of covariance of the source data with regularization.
    cov_target_sqrt_: array, shape (n_features, n_features)
        Square root of covariance of the target data with regularization.

    References
    ----------
    .. [1] Baochen Sun, Jiashi Feng, and Kate Saenko.
           Correlation Alignment for Unsupervised Domain Adaptation.
           In Advances in Computer Vision and Pattern Recognition, 2017.
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
        X, sample_domain = check_X_domain(
            X,
            sample_domain,
            allow_multi_source=True,
            allow_multi_target=True
        )
        X_source, X_target = source_target_split(X, sample_domain=sample_domain)

        cov_source_ = _estimate_covariance(X_source, shrinkage=self.reg)
        cov_target_ = _estimate_covariance(X_target, shrinkage=self.reg)
        self.cov_source_inv_sqrt_ = _invsqrtm(cov_source_)
        self.cov_target_sqrt_ = _sqrtm(cov_target_)
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
        X_t : array-like, shape (n_samples, n_features)
            The data transformed to the target space.
        y_t : array-like, shape (n_samples,)
            The labels (same as y).
        weights : None
            No weights are returned here.
        """
        X, sample_domain = check_X_domain(
            X,
            sample_domain,
            allow_multi_source=True,
            allow_multi_target=True
        )
        X_source, X_target = source_target_split(X, sample_domain=sample_domain)

        X_source_adapt = np.dot(X_source, self.cov_source_inv_sqrt_)
        X_source_adapt = np.dot(X_source_adapt, self.cov_target_sqrt_)
        X_adapt, _ = source_target_merge(
            X_source_adapt, X_target, sample_domain=sample_domain
        )
        return X_adapt


def CORAL(
    base_estimator=None,
    reg="auto",
):
    """CORAL pipeline with adapter and estimator.

    see [1]_ for details.

    Parameters
    ----------
    base_estimator : object, optional (default=None)
        The base estimator to fit on the target dataset.
    reg : 'auto' or float, default="auto"
        The regularization parameter of the covariance estimator.
        Possible values:

          - None: no shrinkage).
          - 'auto': automatic shrinkage using the Ledoit-Wolf lemma.
          - float between 0 and 1: fixed shrinkage parameter.

    Returns
    -------
    pipeline : Pipeline
        Pipeline containing CORAL adapter and base estimator.

    References
    ----------
    .. [1] Baochen Sun, Jiashi Feng, and Kate Saenko.
           Correlation Alignment for Unsupervised Domain Adaptation.
           In Advances in Computer Vision and Pattern Recognition, 2017.
    """
    if base_estimator is None:
        base_estimator = SVC(kernel="rbf")

    return make_da_pipeline(
        CORALAdapter(reg=reg),
        base_estimator,
    )


class MMDConSMappingAdapter(BaseAdapter):
    """MMDConSMapping adapter.

    MMDConSMapping finds a linear transformation that minimizes the Maximum Mean
    Discrepancy (MMD) between the source and target domains, such that
    $X^t = W(y^s) \\odot X^s + B(y^s)$, where $W(y^s)$ and $B(y^s)$ are the scaling
    and bias of the linear transformation, respectively.

    See Section 4 of [4]_ for details.

    Parameters
    ----------
    gamma : float or array like
        Parameters for the kernels.
    reg_k : float, default=1e-10
        Regularization parameter for the labels kernel matrix.
    reg_m : float, default=1e-10
        Regularization parameter for the mapping parameters.
    tol : float, default=1e-6
        Tolerance for the stopping criterion in the optimization.
    max_iter : int, default=1000
        Number of maximum iteration before stopping the optimization.

    Attributes
    ----------
    `X_source_` : array-like, shape (n_samples, n_features)
        The source data used for fitting.
    `y_source_` : array-like, shape (n_samples,)
        The source labels used for fitting.

    References
    ----------
    .. [4] Kun Zhang et. al. Domain Adaptation under Target and Conditional Shift
           In ICML, 2013.
    """

    def __init__(self, gamma, reg_k=1e-10, reg_m=1e-10, tol=1e-6, max_iter=1000):
        super().__init__()
        self.gamma = gamma
        self.reg_k = reg_k
        self.reg_m = reg_m
        self.tol = tol
        self.max_iter = max_iter
        self.W_ = None
        self.B_ = None

    def _mapping_optimization(self, X_source, X_target, y_source):
        """Mapping optimization"""
        m, n = X_source.shape[0], X_target.shape[0]
        d = X_source.shape[1]

        # check y is discrete or continuous
        discrete = _find_y_type(y_source) == "classification"

        # compute omega
        L = pairwise_kernels(y_source.reshape(-1, 1), metric="rbf", gamma=self.gamma)
        omega = L @ np.linalg.inv(L + self.reg_k * np.eye(m))

        # compute R
        if discrete:
            classes = np.unique(y_source)
            R = np.zeros((X_target.shape[0], len(classes)))
            for i, c in enumerate(classes):
                R[:, i] = (y_source == c).astype(int)
        else:
            R = L @ np.linalg.inv(L + self.reg_k * np.eye(m))

        # solve the optimization problem
        # min_{G, H} MMD(W \odot X^s + B, X^t)
        # s.t. W = RG, B = RH

        def flatten(G, H):
            return np.concatenate((G.flatten(), H.flatten()))

        def unflatten(x):
            k = R.shape[1]
            return x[:k * d].reshape(k, d), x[k * d:].reshape(k, d)

        def func(x):
            G, H = unflatten(x)
            W = R @ G
            B = R @ H

            X_new = W * X_source + B

            K = pairwise_kernels(X_new, metric="rbf", gamma=self.gamma)
            K_cross = pairwise_kernels(X_target, X_new, metric="rbf", gamma=self.gamma)
            J_cons = (1 / (m ** 2)) * np.sum(omega @ K @ omega.T)
            J_cons -= (2 / (m * n)) * np.sum(K_cross @ omega.T)

            J_reg = (1 / m) * (np.linalg.norm(W - 1, ord="fro") ** 2)
            J_reg += (1 / n) * (np.linalg.norm(B, ord="fro") ** 2)

            return J_cons + self.reg_m * J_reg

        def jac(x):
            G, H = unflatten(x)
            return flatten(np.zeros(G.shape), np.zeros(H.shape))

        results = minimize(
            func,
            x0=flatten(np.ones((R.shape[1], d)), np.zeros((R.shape[1], d))),
            method="SLSQP",
            jac=jac,
            tol=self.tol,
            options={"maxiter": self.max_iter}
        )

        G, H = unflatten(results.x)
        W = R @ G
        B = R @ H

        return W, B

    def fit(self, X, y, sample_domain=None, **kwargs):
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
        X, y, sample_domain = check_X_y_domain(X, y, sample_domain)
        X_source, X_target, y_source, _ = source_target_split(
            X,
            y,
            sample_domain=sample_domain
        )
        self.W_, self.B_ = self._mapping_optimization(X_source, X_target, y_source)

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
        X_source, X_target, _, _ = source_target_split(
            X, sample_domain=sample_domain
        )

        # source_idx = extract_source_indices(sample_domain)

        # if source_idx.sum() > 0:
        #     if np.array_equal(self.X_source_, X[source_idx]):
        #         source_weights = self.weights_
        #     else:
        #         # get the nearest neighbor in the source domain
        #         K = pairwise_kernels(
        #             X[source_idx],
        #             self.X_source_,
        #             metric="rbf",
        #             gamma=self.gamma
        #         )
        #         idx = np.argmax(K, axis=1)
        #         source_weights = self.weights_[idx]
        #     source_idx = np.where(source_idx)
        #     weights = np.zeros(X.shape[0], dtype=source_weights.dtype)
        #     weights[source_idx] = source_weights
        # else:
        #     weights = None
        # return AdaptationOutput(X=X, sample_weight=weights)

        X_source_adapt = self.W_ * X + self.B_
        X_adapt, _ = source_target_merge(
            X_source_adapt, X_target, sample_domain=sample_domain
        )

        return X_adapt
