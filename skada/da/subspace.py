# Author: Theo Gnassounou <theo.gnassounou@inria.fr>
#
# License: MIT License

import numpy as np

from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import pairwise_kernels

from .base import BaseSubspaceEstimator


class SubspaceAlignment(BaseSubspaceEstimator):
    """Estimator based on reweighting samples using density estimation.

    Parameters
    ----------
    base_estimator : estimator object
        The base estimator to fit on reweighted data.
    n_components : int, default=None
        The numbers of components to learn with PCA.
        Should be less or equal to the number of features
        of the data.

    References
    ----------
    .. [1]  Basura Fernando et. al. Unsupervised Visual
            Domain Adaptation Using Subspace Alignment.
            In IEEE International Conference on Computer Vision, 2013.
    """
    def __init__(
        self,
        base_estimator,
        n_components=None,
    ):
        super().__init__(base_estimator)

        self.n_components = n_components

    def predict_adapt(self, X, y, X_target, y_target=None):
        """Predict adaptation (weights, sample or labels)"""
        weights = None
        self.M_ = np.dot(self.pca_source_.components_, self.pca_target_.components_.T)
        X_ = np.dot(self.pca_source_.transform(X), self.M_)
        return X_, y, weights

    def fit_adapt(self, X, y, X_target, y_target=None):
        """Fit adaptation parameters"""
        assert self.n_components <= X.shape[1], "n_components higher than n_features"

        self.pca_source_ = PCA(self.n_components).fit(X)
        self.pca_target_ = PCA(self.n_components).fit(X_target)

    def transform(self, X, domain='target'):
        """Transform the data in the new subspace
        Parameters
        ----------
        domain : string, default='target
            The domain from where come the data.
        """
        assert domain in ['target', 'source']

        if domain == 'source':
            X_transform = np.dot(self.pca_source_.transform(X), self.M_)
        else:
            X_transform = self.pca_target_.transform(X)
        return X_transform


class TCA(BaseSubspaceEstimator):
    """Estimator based on reweighting samples using density estimation.

    Parameters
    ----------
    base_estimator : estimator object
        The base estimator to fit on reweighted data.
    kernel : kernel object, default='rbf'
        The kernel computed between data.
    n_components : int, default=None
        The numbers of components to learn with PCA.
        Should be less or equal to the number of samples
        of the source and target data.
    mu : float, default=0.1
        The parameter of the regularization in
        the optimization problem.

    References
    ----------
    .. [1]  Sinno Jialin Pan et. al. Domain Adaptation via
            Transfer Component Analysi. In IEEE Transactions
            on Neural Networks, 2011.
    """

    def __init__(
        self,
        base_estimator,
        kernel='rbf',
        n_components=None,
        mu=0.1
    ):
        super().__init__(base_estimator)

        self.kernel = kernel
        self.n_components = n_components
        self.mu = mu

    def predict_adapt(self, X, y, X_target, y_target=None):
        """Predict adaptation (weights, sample or labels)"""
        X_ = (self.K_ @ self.eigvects_)[:len(X)]
        weights = None

        return X_, y, weights

    def fit_adapt(self, X, y, X_target, y_target=None):
        """Fit adaptation parameters"""
        self.X_source_ = X
        self.X_target_ = X_target

        Kss = pairwise_kernels(X, metric=self.kernel)
        Ktt = pairwise_kernels(X_target, metric=self.kernel)
        Kst = pairwise_kernels(X, X_target, metric=self.kernel)
        K = np.block([[Kss, Kst], [Kst.T, Ktt]])
        self.K_ = K

        ns = len(X)
        nt = len(X_target)
        Lss = 1/ns**2 * np.ones((ns, ns))
        Ltt = 1/nt**2 * np.ones((nt, nt))
        Lst = -1/(ns*nt) * np.ones((ns, nt))
        L = np.block([[Lss, Lst], [Lst.T, Ltt]])

        H = np.eye(ns+nt) - 1/(ns + nt) * np.ones((ns + nt, ns + nt))

        A = np.eye(ns + nt) + self.mu * K @ L @ K
        B = K @ H @ K
        solution = np.linalg.solve(A, B)

        eigvals, eigvects = np.linalg.eigh(solution)

        selected_components = np.argsort(np.abs(eigvals))[::-1][:self.n_components]
        self.eigvects_ = np.real(eigvects[:, selected_components])

    def transform(self, X, domain='target'):
        """Transform the data in the new subspace
        Parameters
        ----------
        domain : string, default='target
            The domain from where come the data.
        """
        Ks = pairwise_kernels(X, self.X_source_, metric=self.kernel)
        Kt = pairwise_kernels(X, self.X_target_, metric=self.kernel)
        K = np.concatenate((Ks, Kt), axis=1)

        return (K @ self.eigvects_)[:len(X)]
