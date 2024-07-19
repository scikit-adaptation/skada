# Author: Theo Gnassounou <theo.gnassounou@inria.fr>
#         Remi Flamary <remi.flamary@polytechnique.edu>
#         Oleksii Kachaiev <kachayev@gmail.com>
#         Antoine Collas <contact@antoinecollas.fr>
#
# License: BSD 3-Clause

from abc import abstractmethod

import numpy as np
from ot import da
from ot.gaussian import bures_wasserstein_barycenter, bures_wasserstein_mapping
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.svm import SVC

from ._pipeline import make_da_pipeline
from ._utils import Y_Type, _estimate_covariance, _find_y_type
from .base import BaseAdapter, clone
from .utils import (
    check_X_domain,
    extract_domains_indices,
    extract_source_indices,
    per_domain_split,
    source_target_merge,
    source_target_split,
    torch_minimize,
)


class BaseOTMappingAdapter(BaseAdapter):
    """Base class for all DA estimators implemented using OT mapping.

    Each implementation has to provide `_create_transport_estimator` callback
    to create OT object using parameters saved in the constructor.
    """

    def fit(self, X, y=None, *, sample_domain=None):
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
        X, X_target, y, y_target = source_target_split(
            X, y, sample_domain=sample_domain
        )
        transport = self._create_transport_estimator()
        self.ot_transport_ = clone(transport)
        self.ot_transport_.fit(Xs=X, ys=y, Xt=X_target, yt=y_target)
        return self

    def fit_transform(self, X, y=None, *, sample_domain=None, **params):
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
        """
        self.fit(X, y, sample_domain=sample_domain)
        return self.transform(X, sample_domain=sample_domain, allow_source=True)

    def transform(
        self, X, y=None, *, sample_domain=None, allow_source=False, **params
    ) -> np.ndarray:
        # xxx(okachaiev): implement auto-infer for sample_domain
        X, sample_domain = check_X_domain(
            X,
            sample_domain,
            allow_source=allow_source,
            allow_multi_source=True,
            allow_multi_target=True,
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

    See [6]_ for details.

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
    .. [6] N. Courty, R. Flamary, D. Tuia and A. Rakotomamonjy,
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


def OTMapping(base_estimator=None, metric="sqeuclidean", norm=None, max_iter=100000):
    """OTmapping pipeline with adapter and estimator.

    See [6]_ for details.

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
    .. [6] N. Courty, R. Flamary, D. Tuia and A. Rakotomamonjy,
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

    See [6]_ for details.

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
    .. [6] N. Courty, R. Flamary, D. Tuia and A. Rakotomamonjy,
           Optimal Transport for Domain Adaptation, in IEEE
           Transactions on Pattern Analysis and Machine Intelligence
    """

    def __init__(
        self,
        reg_e=1.0,
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
    reg_e=1.0,
    tol=1e-8,
):
    """EntropicOTMapping pipeline with adapter and estimator.

    see [6]_ for details.

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
    .. [6] N. Courty, R. Flamary, D. Tuia and A. Rakotomamonjy,
           Optimal Transport for Domain Adaptation, in IEEE
           Transactions on Pattern Analysis and Machine Intelligence
    """
    if base_estimator is None:
        base_estimator = SVC(kernel="rbf")

    return make_da_pipeline(
        EntropicOTMappingAdapter(
            metric=metric, norm=norm, max_iter=max_iter, reg_e=reg_e, tol=tol
        ),
        base_estimator,
    )


class ClassRegularizerOTMappingAdapter(BaseOTMappingAdapter):
    """Domain Adaptation Using Optimal Transport.

    See [6]_ for details.

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
    .. [6] N. Courty, R. Flamary, D. Tuia and A. Rakotomamonjy,
           Optimal Transport for Domain Adaptation, in IEEE
           Transactions on Pattern Analysis and Machine Intelligence
    """

    def __init__(
        self,
        reg_e=1.0,
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
    reg_e=1.0,
    reg_cl=0.1,
    tol=1e-8,
):
    """ClassRegularizedOTMapping pipeline with adapter and estimator.

    see [6]_ for details.

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
    .. [6] N. Courty, R. Flamary, D. Tuia and A. Rakotomamonjy,
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
            tol=tol,
        ),
        base_estimator,
    )
    return ot_mapping


class LinearOTMappingAdapter(BaseOTMappingAdapter):
    """Domain Adaptation Using Optimal Transport.

    Uses Gaussian Monge mapping to align source and target domains as proposed
    in [7].

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

    References
    ----------
    .. [7] Flamary, R., Lounici, K., & Ferrari, A. (2019). Concentration bounds
        for linear monge mapping estimation and optimal transport domain
        adaptation. arXiv preprint arXiv:1905.10155.
    """

    def __init__(self, reg=1e-08, bias=True):
        super().__init__()
        self.reg = reg
        self.bias = bias

    def _create_transport_estimator(self):
        return da.LinearTransport(reg=self.reg, bias=self.bias)


def LinearOTMapping(
    base_estimator=None,
    reg=1.0,
    bias=True,
):
    """Returns a the linear OT mapping method with adapter and estimator.

    Uses Gaussian Monge mapping to align source and target domains as proposed
    in [7].

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
    .. [6] N. Courty, R. Flamary, D. Tuia and A. Rakotomamonjy,
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


def _get_cov_mean(X, w=None, bias=True):
    """Returns covariance and mean

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        The source data.
    w : array-like, shape (n_samples,)
        The weights of the samples.
    bias: bool, optional (default=True)
        estimate bias (mean).

    Returns
    -------
    cov : array-like, shape (n_features, n_features)
        The covariance matrix.
    mean : array-like, shape (n_features,)
        The mean vector.
    """
    if w is None:
        w = np.ones(X.shape[0])
    if bias:
        mean = np.average(X, axis=0, weights=w)
    else:
        mean = np.zeros(X.shape[1])
    X = X - mean
    cov = np.dot(w * X.T, X) / np.sum(w)
    return cov, mean


class MultiLinearMongeAlignmentAdapter(BaseAdapter):
    """Aligns multiple domains using Gaussian Monge mapping to a barycenter.

    The method is a simplified extension of [29] using the Bures-Wasserstein
    distance and mapping of [7] to align multiple source domains to a
    barycenter. The sued of barycenter alignment with gaussien assumption was
    proposed in [30].

    Parameters
    ----------
    reg : float, optional (default=1e-08)
        Regularization parameter added to the diagonal of the covariance.
    bias : bool, optional (default=True)
        Estimate bias.
    test_time : bool, optional (default=False)
        If True, the estimator can be updated at test time to map new
        target domains unseen during training

    Attributes
    ----------
    cov_means_sources_ : dict
        Dictionary of covariance and mean for each source domain.
    cov_means_targets_ : dict
        Dictionary of covariance and mean for each target domain.
    barycenter_ : tuple
        Barycenter of the source domains (mean, cov).
    _mappings_ : dict
        Dictionary of mappings for each domain.

    References
    ----------
    .. [29] Montesuma, Eduardo Fernandes, and Fred Maurice Ngole Mboula.
        "Wasserstein barycenter for multi-source domain adaptation." In Proceedings
        of the IEEE/CVF conference on computer vision and pattern recognition, pp.
        16785-16793. 2021.

    .. [7] Flamary, R., Lounici, K., & Ferrari, A. (2019). Concentration bounds
        for linear monge mapping estimation and optimal transport domain
        adaptation. arXiv preprint arXiv:1905.10155.

    .. [30] Gnassounou, Theo, Rémi Flamary, and Alexandre Gramfort. "Convolution
        Monge Mapping Normalization for learning on sleep data." Advances in
        Neural Information Processing Systems 36 (2024).

    """

    def __init__(self, reg=1e-08, bias=True, test_time=False):
        super().__init__()
        self.reg = reg
        self.bias = bias
        self.test_time = test_time

    def fit(self, X, y=None, *, sample_domain=None):
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
        sources, targets = per_domain_split(X, y, None, sample_domain=sample_domain)

        self.cov_means_sources_ = {
            domain: _get_cov_mean(X, w, bias=self.bias)
            for domain, (X, y, w) in sources.items()
        }

        self.cov_means_targets_ = {
            domain: _get_cov_mean(X, w, bias=self.bias)
            for domain, (X, y, w) in targets.items()
        }

        C = np.stack([cov for cov, mean in self.cov_means_sources_.values()])
        m = np.stack([mean for cov, mean in self.cov_means_sources_.values()])

        self.barycenter_ = bures_wasserstein_barycenter(
            m,
            C,
            eps=self.reg,
        )

        self.mappings_ = {
            domain: bures_wasserstein_mapping(
                mean,
                self.barycenter_[0],
                cov,
                self.barycenter_[1],
            )
            for domain, (cov, mean) in self.cov_means_sources_.items()
        }

        mapping_target = {
            domain: bures_wasserstein_mapping(
                mean,
                self.barycenter_[0],
                cov,
                self.barycenter_[1],
            )
            for domain, (cov, mean) in self.cov_means_targets_.items()
        }

        self.mappings_.update(mapping_target)

        return self

    def fit_transform(self, X, y=None, sample_domain=None, **params):
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
        """
        self.fit(X, y, sample_domain=sample_domain)
        return self.transform(X, sample_domain=sample_domain, allow_source=True)

    def transform(
        self, X, y=None, *, sample_domain=None, allow_source=False, **params
    ) -> np.ndarray:
        X, sample_domain = check_X_domain(
            X, sample_domain, allow_multi_source=True, allow_multi_target=True
        )
        idx = extract_domains_indices(sample_domain)
        X_adapt = X.copy()

        for domain, sel in idx.items():
            A, b = self.mappings_[domain]
            X_adapt[sel] = X[sel].dot(A) + b

        return X_adapt


def MultiLinearMongeAlignment(
    base_estimator=None, reg=1e-08, bias=True, test_time=False
):
    """MultiLinearMongeAlignment pipeline with adapter and estimator.

    The method is a simplified extension of [29] using the Bures-Wasserstein
    distance and mapping of [7] to align multiple source domains to a
    barycenter. The sued of barycenter alignment with gaussien assumption was
    proposed in [30].

    Parameters
    ----------
    base_estimator : object, optional (default=None)
        The base estimator to fit on the target dataset.
    reg : float, optional (default=1e-08)
        Regularization parameter added to the diagonal of the covariance.
    bias : bool, optional (default=True)
        Estimate bias.
    test_time : bool, optional (default=False)
        If True, the estimator can be updated at test time to map new
        target domains unseen during training

    Returns
    -------
    pipeline : Pipeline
        Pipeline containing MultiLinearMongeAlignment adapter and base
        estimator.

    References
    ----------
    .. [29] Montesuma, Eduardo Fernandes, and Fred Maurice Ngole Mboula.
        "Wasserstein barycenter for multi-source domain adaptation." In Proceedings
        of the IEEE/CVF conference on computer vision and pattern recognition, pp.
        16785-16793. 2021.

    .. [7] Flamary, R., Lounici, K., & Ferrari, A. (2019). Concentration bounds
        for linear monge mapping estimation and optimal transport domain
        adaptation. arXiv preprint arXiv:1905.10155.

    .. [30] Gnassounou, Theo, Rémi Flamary, and Alexandre Gramfort. "Convolution
        Monge Mapping Normalization for learning on sleep data." Advances in
        Neural Information Processing Systems 36 (2024).
    """
    if base_estimator is None:
        base_estimator = LogisticRegression()

    return make_da_pipeline(
        MultiLinearMongeAlignmentAdapter(reg=reg, bias=bias, test_time=test_time),
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
    return (eigvecs * 1.0 / np.sqrt(eigvals)) @ eigvecs.T


class CORALAdapter(BaseAdapter):
    """Estimator based on Correlation Alignment [1]_.

    See [5]_ for details.

    Parameters
    ----------
    reg : 'auto' or float, default="auto"
        The regularization parameter of the covariance estimator.
        Possible values:

          - None: no shrinkage).
          - 'auto': automatic shrinkage using the Ledoit-Wolf lemma.
          - float between 0 and 1: fixed shrinkage parameter.
    assume_centered: bool, default=False
        If True, data are not centered before computation.

    Attributes
    ----------
    mean_source_: array, shape (n_features,)
    mean_target_: array, shape (n_features,)
    cov_source_inv_sqrt_: array, shape (n_features, n_features)
        Inverse of the square root of covariance of the source data with regularization.
    cov_target_sqrt_: array, shape (n_features, n_features)
        Square root of covariance of the target data with regularization.

    References
    ----------
    .. [5] Baochen Sun, Jiashi Feng, and Kate Saenko.
           Correlation Alignment for Unsupervised Domain Adaptation.
           In Advances in Computer Vision and Pattern Recognition, 2017.
    """

    def __init__(self, reg="auto", assume_centered=False):
        super().__init__()
        self.reg = reg
        self.assume_centered = assume_centered

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
            X, sample_domain, allow_multi_source=True, allow_multi_target=True
        )
        X_source, X_target = source_target_split(X, sample_domain=sample_domain)

        self.mean_source_ = np.mean(X_source, axis=0)
        self.mean_target_ = np.mean(X_target, axis=0)
        cov_source_ = _estimate_covariance(
            X_source, shrinkage=self.reg, assume_centered=self.assume_centered
        )
        cov_target_ = _estimate_covariance(
            X_target, shrinkage=self.reg, assume_centered=self.assume_centered
        )
        self.cov_source_inv_sqrt_ = _invsqrtm(cov_source_)
        self.cov_target_sqrt_ = _sqrtm(cov_target_)
        return self

    def fit_transform(self, X, y=None, *, sample_domain=None, **params):
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
        """
        self.fit(X, y, sample_domain=sample_domain)
        return self.transform(X, sample_domain=sample_domain, allow_source=True)

    def transform(
        self, X, y=None, *, sample_domain=None, allow_source=False, **params
    ) -> np.ndarray:
        X, sample_domain = check_X_domain(
            X,
            sample_domain,
            allow_source=allow_source,
            allow_multi_source=True,
            allow_multi_target=True,
        )
        X_source_adapt, X_target_adapt = source_target_split(
            X, sample_domain=sample_domain
        )

        # Adapt the source data
        if X_source_adapt.shape[0] > 0:
            # Center data
            if not self.assume_centered:
                X_source_adapt = X_source_adapt - self.mean_source_

            # Whitening and coloring source data
            X_source_adapt = np.dot(X_source_adapt, self.cov_source_inv_sqrt_)
            X_source_adapt = np.dot(X_source_adapt, self.cov_target_sqrt_)

        # Adapt the target data
        if X_target_adapt.shape[0] > 0 and not self.assume_centered:
            X_target_adapt = X_target_adapt - self.mean_target_

        X_adapt, _ = source_target_merge(
            X_source_adapt, X_target_adapt, sample_domain=sample_domain
        )
        return X_adapt


def CORAL(
    base_estimator=None,
    reg="auto",
    assume_centered=False,
):
    """CORAL pipeline with adapter and estimator.

    See [5]_ for details.

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
    assume_centered: bool, default=False
        If True, data are not centered before computation.

    Returns
    -------
    pipeline : Pipeline
        Pipeline containing CORAL adapter and base estimator.

    References
    ----------
    .. [5] Baochen Sun, Jiashi Feng, and Kate Saenko.
           Correlation Alignment for Unsupervised Domain Adaptation.
           In Advances in Computer Vision and Pattern Recognition, 2017.
    """
    if base_estimator is None:
        base_estimator = SVC(kernel="rbf")

    return make_da_pipeline(
        CORALAdapter(reg=reg, assume_centered=assume_centered),
        base_estimator,
    )


# xxx(okachaiev): we should move this to 'skada.deep.*' I guess
# to avoid defining things that won't work anyways
class MMDLSConSMappingAdapter(BaseAdapter):
    r"""Location-Scale mapping minimizing the MMD with a Gaussian kernel.

    MMDLSConSMapping finds a linear transformation that minimizes the Maximum Mean
    Discrepancy (MMD) between the source and target domains, such that
    $X^t = W(y^s) \\odot X^s + B(y^s)$, where $W(y^s)$ and $B(y^s)$ are the scaling
    and bias of the linear transformation, respectively.

    See Section 4 of [21]_ for details.

    Parameters
    ----------
    gamma : float
        Parameter for the Gaussian kernel.
    reg_k : float, default=1e-10
        Regularization parameter for the labels kernel matrix.
    reg_m : float, default=1e-10
        Regularization parameter for the mapping parameters.
    tol : float, default=1e-5
        Tolerance for the stopping criterion in the optimization.
    max_iter : int, default=100
        Number of maximum iteration before stopping the optimization.

    Attributes
    ----------
    `W_` : array-like, shape (n_samples, n_features)
        The scaling matrix.
    `B_` : array-like, shape (n_samples, n_features)
        The bias matrix.
    `G_` : array-like, shape (n_classes, n_features) or (n_samples, n_features)
        The learned kernel scaling matrix.
    `H_` : array-like, shape (n_classes, n_features) or (n_samples, n_features)
        The learned kernel bias matrix.
    `X_source_` : array-like, shape (n_samples, n_features)
        The source data.

    References
    ----------
    .. [21] Kun Zhang et. al. Domain Adaptation under Target and Conditional Shift
           In ICML, 2013.
    """

    def __init__(self, gamma, reg_k=1e-10, reg_m=1e-10, tol=1e-5, max_iter=100):
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
        try:
            import torch
        except ImportError:
            raise ImportError(
                "MMDLSConSMappingAdapter requires pytorch to be installed."
            )

        # check y is discrete or continuous
        self.discrete_ = discrete = _find_y_type(y_source) == Y_Type.DISCRETE

        # convert to pytorch tensors
        X_source = torch.tensor(X_source, dtype=torch.float64)
        X_target = torch.tensor(X_target, dtype=torch.float64)
        y_source = torch.tensor(
            y_source, dtype=torch.int64 if discrete else torch.float64
        )

        # get shapes
        m, n = X_source.shape[0], X_target.shape[0]
        d = X_source.shape[1]

        # compute omega
        L = torch.exp(-self.gamma * torch.cdist(X_source, X_source, p=2))
        omega = L @ torch.linalg.inv(L + self.reg_k * torch.eye(m))

        # compute R
        if discrete:
            self.classes_ = classes = torch.unique(y_source).numpy()
            R = torch.zeros((m, len(classes)), dtype=torch.float64)
            for i, c in enumerate(classes):
                R[:, i] = (y_source == c).int()
        else:
            self.classes_ = None
            R = L @ torch.linalg.inv(L + self.reg_k * torch.eye(m))

        # solve the optimization problem
        # min_{G, H} MMD(W \odot X^s + B, X^t)
        # s.t. W = RG, B = RH
        k = R.shape[1]

        def func(G, H):
            W = R @ G
            B = R @ H

            X_new = W * X_source + B

            K = torch.exp(-self.gamma * torch.cdist(X_new, X_new, p=2))
            K_cross = torch.exp(-self.gamma * torch.cdist(X_target, X_new, p=2))
            J_cons = (1 / (m**2)) * torch.sum(omega @ K @ omega.T)
            J_cons -= (2 / (m * n)) * torch.sum(K_cross @ omega.T)

            J_reg = (1 / m) * (torch.sum((W - 1) ** 2) + torch.sum(B**2))

            return J_cons + self.reg_m * J_reg

        # optimize using torch solver
        G = torch.ones((k, d), dtype=torch.float64, requires_grad=True)
        H = torch.zeros((k, d), dtype=torch.float64, requires_grad=True)

        (G, H), _ = torch_minimize(func, (G, H), tol=self.tol, max_iter=self.max_iter)

        R = R.detach().numpy()
        W = R @ G
        B = R @ H

        return W, B, G, H

    def fit(self, X, y, sample_domain=None):
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
        # xxx(okachaiev): we can't test X_y here because y might
        # have NaNs, thought it might be better to keep this as an
        # argument of a checker
        X, sample_domain = check_X_domain(X, sample_domain)
        X_source, X_target, y_source, _ = source_target_split(
            X, y, sample_domain=sample_domain
        )
        self.X_source_ = X_source

        self.W_, self.B_, self.G_, self.H_ = self._mapping_optimization(
            X_source, X_target, y_source
        )

        return self

    def fit_transform(self, X, y=None, sample_domain=None, **params):
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
        """
        self.fit(X, y, sample_domain=sample_domain)
        return self.transform(X, sample_domain=sample_domain, allow_source=True)

    def transform(
        self, X, y=None, *, sample_domain=None, allow_source=False, **params
    ) -> np.ndarray:
        X, sample_domain = check_X_domain(X, sample_domain, allow_source=allow_source)

        source_idx = extract_source_indices(sample_domain)
        X_source, X_target = X[source_idx], X[~source_idx]
        if X_source.shape[0] == 0:
            X_source_adapt = X_source
        else:
            if np.array_equal(self.X_source_, X[source_idx]):
                W, B = self.W_, self.B_
            else:
                if self.discrete_ and y is not None:
                    # recompute the mapping
                    X, sample_domain = check_X_domain(X, sample_domain)
                    source_idx = extract_source_indices(sample_domain)
                    y_source = y[source_idx]
                    classes = self.classes_
                    R = np.zeros((source_idx.sum(), len(classes)))
                    for i, c in enumerate(classes):
                        R[:, i] = (y_source == c).astype(int)
                    W, B = R @ self.G_, R @ self.H_
                else:
                    # assign the nearest neighbor's mapping to the source samples
                    C = pairwise_distances(X[source_idx], self.X_source_)
                    idx = np.argmin(C, axis=1)
                    W, B = self.W_[idx], self.B_[idx]
            X_source_adapt = W * X_source + B
        X_adapt, _ = source_target_merge(
            X_source_adapt, X_target, sample_domain=sample_domain
        )
        return X_adapt


def MMDLSConSMapping(
    base_estimator=None, gamma=1.0, reg_k=1e-10, reg_m=1e-10, tol=1e-5, max_iter=100
):
    """MMDLSConSMapping pipeline with adapter and estimator.

    See [21]_ for details.

    Parameters
    ----------
    base_estimator : object, optional (default=None)
        The base estimator to fit on the target dataset.
    gamma : float
        Parameter for the Gaussian kernel.
    reg_k : float, default=1e-10
        Regularization parameter for the labels kernel matrix.
    reg_m : float, default=1e-10
        Regularization parameter for the mapping parameters.
    tol : float, default=1e-5
        Tolerance for the stopping criterion in the optimization.
    max_iter : int, default=100
        Number of maximum iteration before stopping the optimization.

    Returns
    -------
    pipeline : Pipeline
        Pipeline containing CORAL adapter and base estimator.

    References
    ----------
    .. [21] Kun Zhang et. al. Domain Adaptation under Target and Conditional Shift
            In ICML, 2013.
    """
    if base_estimator is None:
        base_estimator = SVC(kernel="rbf")

    return make_da_pipeline(
        MMDLSConSMappingAdapter(
            gamma=gamma, reg_k=reg_k, reg_m=reg_m, tol=tol, max_iter=max_iter
        ),
        base_estimator,
    )
