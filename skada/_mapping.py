# Author: Theo Gnassounou <theo.gnassounou@inria.fr>
#         Remi Flamary <remi.flamary@polytechnique.edu>
#         Alexandre Gramfort <alexandre.gramfort@inria.fr>
#
# License: BSD 3-Clause

import numpy as np

from ot import da

from .base import BaseDataAdaptEstimator, clone
from ._utils import _estimate_covariance


class OTmapping(BaseDataAdaptEstimator):
    """Domain Adaptation Using Optimal Transport.

    Parameters
    ----------
    base_estimator : estimator object
        The base estimator to fit on reweighted data.

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
            The data transformed to the target subspace.
        y_t : array-like, shape (n_samples,)
            The labels (same as y).
        weights : array-like, shape (n_samples,)
            The weights of the samples.
        """
        X_ = self.ot_transport_.transform(Xs=X)
        weights = None
        return X_, y, weights

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
        self.ot_transport_ = clone(da.EMDTransport())
        self.ot_transport_.fit(Xs=X, Xt=X_target)
        return self


class EntropicOTmapping(OTmapping):
    """Domain Adaptation Using Optimal Transport.

    Parameters
    ----------
    base_estimator : estimator object
        The base estimator to fit on reweighted data.
    reg_e : float, default=1
        Entropic regularization parameter.
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
        base_estimator,
        reg_e=1,
        max_iter=1000,
        tol=10e-9
    ):
        super().__init__(base_estimator)
        self.reg_e = reg_e
        self.max_iter = max_iter
        self.tol = tol

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

        self.ot_transport_ = clone(
            da.SinkhornTransport(
                reg_e=self.reg_e, max_iter=self.max_iter, tol=self.tol
            )
        )
        self.ot_transport_.fit(Xs=X, Xt=X_target)
        return self


class ClassRegularizerOTmapping(OTmapping):
    """Domain Adaptation Using Optimal Transport.

    Parameters
    ----------
    base_estimator : estimator object
        The base estimator to fit on reweighted data.
    reg_e : float, default=1
        Entropic regularization parameter.
    reg_cl : float, default=0.1
        Class regularization parameter.
    norm : tuple, default="lpl1"
        Norm use for the regularizer of the class labels.
        If "lpl1", use the lp l1 norm.
        If "l1l2", use the l1 l2 norm.

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
        base_estimator,
        reg_e=1,
        reg_cl=0.1,
        norm="lpl1"
    ):
        super().__init__(base_estimator)
        self.reg_e = reg_e
        self.reg_cl = reg_cl
        self.norm = norm

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

        assert self.norm in ["lpl1", "l1l2"], "Unknown norm"

        if self.norm == "lpl1":
            self.ot_transport_ = clone(da.SinkhornLpl1Transport(
                reg_e=self.reg_e, reg_cl=self.reg_cl
            ))
        elif self.norm == "l1l2":
            self.ot_transport_ = clone(da.SinkhornL1l2Transport(
                reg_e=self.reg_e, reg_cl=self.reg_cl
            ))

        self.ot_transport_.fit(Xs=X, ys=y, Xt=X_target)
        return self


class LinearOTmapping(OTmapping):
    """Domain Adaptation Using Optimal Transport.

    Parameters
    ----------
    base_estimator : estimator object
        The base estimator to fit on reweighted data.
    reg : float, default=1e-08
        regularization added to the diagonals of covariances.

    Attributes
    ----------
    ot_transport_ : object
        The OT object based on linear operator between empirical
        distributions fitted on the source
        and target data.
    """

    def __init__(
        self,
        base_estimator,
        reg=1e-08,
    ):
        super().__init__(base_estimator)
        self.reg = reg

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

        self.ot_transport_ = clone(da.LinearTransport(reg=self.reg))

        self.ot_transport_.fit(Xs=X, ys=y, Xt=X_target)
        return self


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


class CORAL(BaseDataAdaptEstimator):
    """Estimator based on Correlation Alignment [1]_.

    Parameters
    ----------
    base_estimator : estimator object
        The base estimator to fit on reweighted data.
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
        X_t : array-like, shape (n_samples, n_features)
            The data transformed to the target space.
        y_t : array-like, shape (n_samples,)
            The labels (same as y).
        weights : None
            No weights are returned here.
        """
        X_ = np.dot(X, self.cov_source_inv_sqrt_)
        X_ = np.dot(X_, self.cov_target_sqrt_)
        weights = None
        return X_, y, weights

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
        cov_source_ = _estimate_covariance(X, shrinkage=self.reg)
        cov_target_ = _estimate_covariance(X_target, shrinkage=self.reg)
        self.cov_source_inv_sqrt_ = _invsqrtm(cov_source_)
        self.cov_target_sqrt_ = _sqrtm(cov_target_)
        return self
