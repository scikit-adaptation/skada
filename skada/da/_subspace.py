# Author: Theo Gnassounou <theo.gnassounou@inria.fr>
#
# License: MIT License

import numpy as np

from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import pairwise_kernels

from .base import BaseSubspaceEstimator


class SubspaceAlignment(BaseSubspaceEstimator):
    """Domain Adaptation Using Subspace Alignment.

    See [1]_ for details.

    Parameters
    ----------
    base_estimator : estimator object
        The base estimator to fit on reweighted data.
    n_components : int, default=None
        The numbers of components to learn with PCA.
        Should be less or equal to the number of features
        of the data.

    Attributes
    ----------
    `pca_source_` : object
        The PCA object fitted on the source data.
    `pca_target_` : object
        The PCA object fitted on the target data.

    References
    ----------
    .. [1] Basura Fernando et. al. Unsupervised Visual
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
        weights = None
        self.M_ = np.dot(self.pca_source_.components_, self.pca_target_.components_.T)
        X_ = np.dot(self.pca_source_.transform(X), self.M_)
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
        assert self.n_components <= X.shape[1], "n_components higher than n_features"

        self.pca_source_ = PCA(self.n_components).fit(X)
        self.pca_target_ = PCA(self.n_components).fit(X_target)
        return self

    def transform(self, X, domain='target'):
        """Transform the data in the new subspace

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The data to transform.
        domain : str, default='target'
            The domain to transform the data to.

        Returns
        -------
        X_t : array-like, shape (n_samples, n_components)
            The transformed data.
        """
        assert domain in ['target', 'source']

        if domain == 'source':
            X_transform = np.dot(self.pca_source_.transform(X), self.M_)
        else:
            X_transform = self.pca_target_.transform(X)
        return X_transform


class TransferComponentAnalysis(BaseSubspaceEstimator):
    """Transfer Component Analysis.

    See [1]_ for details.

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
        The parameter of the regularization in the optimization
        problem.

    Attributes
    ----------
    `X_source_` : array
        Source data used for the optimisation problem.
    `X_target_` : array
        Target data used for the optimisation problem.
    `K_` : array
        Kernel distance between the data (source and target).
    `eigvects_` : array
        Highest n_components eigen vectors of the solution
        of the optimisation problem used to project
        in the new subspace.

    References
    ----------
    .. [1] Sinno Jialin Pan et. al. Domain Adaptation via
           Transfer Component Analysis. In IEEE Transactions
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
        weights : None
            No weights are returned here.
        """
        X_ = (self.K_ @ self.eigvects_)[:len(X)]
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
        return self

    def transform(self, X, domain='target'):
        """Transform the data in the new subspace.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The data to transform.
        domain : string, default='target
            The domain from where come the data.

        Returns
        -------
        X_t : array-like, shape (n_samples, n_components)
            The transformed data.
        """
        Ks = pairwise_kernels(X, self.X_source_, metric=self.kernel)
        Kt = pairwise_kernels(X, self.X_target_, metric=self.kernel)
        K = np.concatenate((Ks, Kt), axis=1)

        return (K @ self.eigvects_)[:len(X)]
