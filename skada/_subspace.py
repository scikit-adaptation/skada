# Author: Theo Gnassounou <theo.gnassounou@inria.fr>
#         Remi Flamary <remi.flamary@polytechnique.edu>
#         Oleksii Kachaiev <kachayev@gmail.com>
#         Ruben Bueno <ruben.bueno@polytechnique.edu>
#         Antoine Collas <contact@antoinecollas.fr>
#
# License: BSD 3-Clause


import warnings

import numpy as np
import scipy.linalg
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.utils import check_random_state

from ._pipeline import make_da_pipeline
from .base import BaseAdapter
from .utils import (
    check_X_domain,
    extract_source_indices,
    source_target_merge,
    source_target_split,
    torch_minimize,
)


class SubspaceAlignmentAdapter(BaseAdapter):
    """Domain Adaptation Using Subspace Alignment.

    See [8]_ for details.

    Parameters
    ----------
    n_components : int, default=None
        The numbers of components to learn with PCA.
        If n_components is not set all components are kept::

            n_components == min(n_samples, n_features)
    random_state : int, RandomState instance or None, default=None
        Determines random number generation for dataset creation. Pass an int
        for reproducible output across multiple function calls.

    Attributes
    ----------
    `pca_source_` : object
        The PCA object fitted on the source data.
    `pca_target_` : object
        The PCA object fitted on the target data.

    References
    ----------
    .. [8] Basura Fernando et. al. Unsupervised Visual
           Domain Adaptation Using Subspace Alignment.
           In IEEE International Conference on Computer Vision, 2013.
    """

    def __init__(
        self,
        n_components=None,
        random_state=None,
    ):
        super().__init__()
        self.n_components = n_components
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
            allow_multi_target=True,
        )
        X_source, X_target = source_target_split(X, sample_domain=sample_domain)

        if self.n_components is None:
            n_components = min(X.shape[0], X.shape[1])
        else:
            n_components = self.n_components
        self.random_state_ = check_random_state(self.random_state)
        self.pca_source_ = PCA(n_components, random_state=self.random_state_).fit(
            X_source
        )
        self.pca_target_ = PCA(n_components, random_state=self.random_state_).fit(
            X_target
        )
        self.n_components_ = n_components
        self.M_ = np.dot(self.pca_source_.components_, self.pca_target_.components_.T)
        return self

    def transform(
        self, X, y=None, *, sample_domain=None, allow_source=False, **params
    ) -> np.ndarray:
        """Perform adaptation on given samples (weights, sample or labels).

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
            The data transformed to the target subspace.
        """
        X, sample_domain = check_X_domain(
            X,
            sample_domain,
            allow_source=allow_source,
            allow_multi_source=True,
            allow_multi_target=True,
        )
        X_source, X_target = source_target_split(X, sample_domain=sample_domain)

        if X_source.shape[0]:
            X_source = np.dot(self.pca_source_.transform(X_source), self.M_)
        if X_target.shape[0]:
            X_target = self.pca_target_.transform(X_target)
        X_adapt, _ = source_target_merge(
            X_source, X_target, sample_domain=sample_domain
        )
        return X_adapt

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
        X_t : array-like, shape (n_samples, n_components)
            The data transformed to the target subspace.
        """
        self.fit(X, y, sample_domain=sample_domain)
        params["allow_source"] = True
        return self.transform(X, y, sample_domain=sample_domain, **params)


def SubspaceAlignment(
    base_estimator=None,
    n_components=None,
    random_state=None,
):
    """Domain Adaptation Using Subspace Alignment.

    See [8]_ for details.

    Parameters
    ----------
    base_estimator : object, default=None
        estimator used for fitting and prediction
    n_components : int, default=None
        The numbers of components to learn with PCA.
        If n_components is not set all components are kept::

            n_components == min(n_samples, n_features)
    random_state : int, RandomState instance or None, default=None
        Determines random number generation for dataset creation. Pass an int
        for reproducible output across multiple function calls.

    Returns
    -------
    pipeline : Pipeline
        A pipeline containing a SubspaceAlignmentAdapter.

    References
    ----------
    .. [8] Basura Fernando et. al. Unsupervised Visual
           Domain Adaptation Using Subspace Alignment.
           In IEEE International Conference on Computer Vision, 2013.
    """
    if base_estimator is None:
        base_estimator = SVC()

    return make_da_pipeline(
        SubspaceAlignmentAdapter(
            n_components=n_components,
            random_state=random_state,
        ),
        base_estimator,
    )


class TransferComponentAnalysisAdapter(BaseAdapter):
    """Transfer Component Analysis.

    See [9]_ for details.

    Parameters
    ----------
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
        Source data used for the optimization problem.
    `X_target_` : array
        Target data used for the optimization problem.
    `K_` : array
        Kernel distance between the data (source and target).
    `eigvects_` : array
        Highest n_components eigenvectors of the solution
        of the optimization problem used to project
        in the new subspace.

    References
    ----------
    .. [9] Sinno Jialin Pan et. al. Domain Adaptation via
           Transfer Component Analysis. In IEEE Transactions
           on Neural Networks, 2011.
    """

    def __init__(self, kernel="rbf", n_components=None, mu=0.1):
        super().__init__()
        self.kernel = kernel
        self.n_components = n_components
        self.mu = mu

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
        X, sample_domain = check_X_domain(
            X,
            sample_domain,
            allow_multi_source=True,
            allow_multi_target=True,
        )
        self.X_source_, self.X_target_ = source_target_split(
            X, sample_domain=sample_domain
        )

        Kss = pairwise_kernels(self.X_source_, metric=self.kernel)
        Ktt = pairwise_kernels(self.X_target_, metric=self.kernel)
        Kst = pairwise_kernels(self.X_source_, self.X_target_, metric=self.kernel)
        K = np.block([[Kss, Kst], [Kst.T, Ktt]])
        self.K_ = K

        ns = self.X_source_.shape[0]
        nt = self.X_target_.shape[0]
        Lss = 1 / ns**2 * np.ones((ns, ns))
        Ltt = 1 / nt**2 * np.ones((nt, nt))
        Lst = -1 / (ns * nt) * np.ones((ns, nt))
        L = np.block([[Lss, Lst], [Lst.T, Ltt]])

        H = np.eye(ns + nt) - 1 / (ns + nt) * np.ones((ns + nt, ns + nt))

        A = np.eye(ns + nt) + self.mu * K @ L @ K
        B = K @ H @ K
        solution = np.linalg.solve(A, B)

        eigvals, eigvects = np.linalg.eigh(solution)

        if self.n_components is None:
            n_components = min(X.shape[0], X.shape[1])
        else:
            n_components = self.n_components
        selected_components = np.argsort(np.abs(eigvals))[::-1][:n_components]
        self.eigvects_ = np.real(eigvects[:, selected_components])
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
        X_t : array-like, shape (n_samples, n_components)
            The data transformed to the target subspace.
        """
        self.fit(X, y, sample_domain=sample_domain)
        params["allow_source"] = True
        return self.transform(X, y, sample_domain=sample_domain, **params)

    def transform(
        self, X, y=None, *, sample_domain=None, allow_source=False, **params
    ) -> np.ndarray:
        """Perform adaptation on given samples (weights, sample or labels).

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
            The data transformed to the target subspace.
        """
        X, sample_domain = check_X_domain(
            X,
            sample_domain,
            allow_source=allow_source,
            allow_multi_source=True,
            allow_multi_target=True,
        )
        X_source, X_target = source_target_split(X, sample_domain=sample_domain)

        if np.array_equal(X_source, self.X_source_) and np.array_equal(
            X_target, self.X_target_
        ):
            X_ = (self.K_ @ self.eigvects_)[: X.shape[0]]
        else:
            Ks = pairwise_kernels(X, self.X_source_, metric=self.kernel)
            Kt = pairwise_kernels(X, self.X_target_, metric=self.kernel)
            K = np.concatenate((Ks, Kt), axis=1)
            X_ = (K @ self.eigvects_)[: X.shape[0]]
        return X_


def TransferComponentAnalysis(
    base_estimator=None, kernel="rbf", n_components=None, mu=0.1
):
    """Domain Adaptation Using Transfer Component Analysis.

    See [9]_ for details.

    Parameters
    ----------
    base_estimator : object, default=None
        estimator used for fitting and prediction
    kernel : kernel object, default='rbf'
        The kernel computed between data.
    n_components : int, default=None
        The numbers of components to learn with PCA.
        Should be less or equal to the number of samples
        of the source and target data.
    mu : float, default=0.1
        The parameter of the regularization in the optimization
        problem.

    Returns
    -------
    pipeline : Pipeline
        A pipeline containing a TransferComponentAnalysisAdapter.

    References
    ----------
    .. [9] Sinno Jialin Pan et. al. Domain Adaptation via
           Transfer Component Analysis. In IEEE Transactions
           on Neural Networks, 2011.
    """
    if base_estimator is None:
        base_estimator = SVC()

    return make_da_pipeline(
        TransferComponentAnalysisAdapter(
            kernel=kernel, n_components=n_components, mu=mu
        ),
        base_estimator,
    )


class TransferJointMatchingAdapter(BaseAdapter):
    """Domain Adaptation Using TJM: Transfer Joint Matching.

    See [26]_ for details.

    Parameters
    ----------
    n_components : int, default=None
        The numbers of components to learn with PCA.
        Should be less or equal to the number of samples
        of the source and target data.
    tradeoff : float, default=1e-2
        The tradeoff constant for the TJM algorithm.
        It serves to trade off feature matching and instance
        reweighting.
    max_iter : int>0, default=100
        The maximal number of iteration before stopping when
        fitting.
    kernel : kernel object, default='rbf'
        The kernel computed between data.
    tol : float, default=0.01
        The threshold for the differences between losses on two iteration
        before the algorithm stops
    verbose : bool, default=False
        If True, print the loss value at each iteration.

    Attributes
    ----------
    None

    References
    ----------
    .. [26]  [Long et al., 2014] Long, M., Wang, J., Ding, G., Sun, J., and Yu, P.
             (2014). Transfer joint matching for unsupervised domain adaptation.
             In IEEE Conference on Computer Vision and Pattern Recognition (CVPR),
             pages 1410–1417

    """

    def __init__(
        self,
        n_components=None,
        tradeoff=1e-2,
        max_iter=100,
        kernel="rbf",
        tol=0.01,
        verbose=False,
    ):
        super().__init__()
        self.n_components = n_components
        self.tradeoff = tradeoff
        self.kernel = kernel
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose

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
        X_t : array-like, shape (n_samples, n_components)
            The data transformed to the target subspace.
        """
        self.fit(X, y, sample_domain=sample_domain)
        params["allow_source"] = True
        return self.transform(X, y, sample_domain=sample_domain, **params)

    def transform(
        self, X, y=None, *, sample_domain=None, allow_source=False, **params
    ) -> np.ndarray:
        """Perform adaptation on given samples (weights, sample or labels).

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
            The data transformed to the target subspace.
        """
        X, sample_domain = check_X_domain(
            X,
            sample_domain,
            allow_source=allow_source,
            allow_multi_source=True,
            allow_multi_target=True,
        )
        X_source, X_target = source_target_split(X, sample_domain=sample_domain)

        if np.array_equal(X_source, self.X_source_) and np.array_equal(
            X_target, self.X_target_
        ):
            K = self._get_kernel_matrix(X_source, X_target)
            X_ = K @ self.A_
        else:
            Ks = pairwise_kernels(X, self.X_source_, metric=self.kernel)
            Kt = pairwise_kernels(X, self.X_target_, metric=self.kernel)
            K = np.concatenate((Ks, Kt), axis=1)
            X_ = K @ self.A_
        return X_

    def _get_mmd_matrix(self, ns, nt, sample_domain):
        Mss = (1 / (ns**2)) * np.ones((ns, ns))
        Mtt = (1 / (nt**2)) * np.ones((nt, nt))
        Mst = -(1 / (ns * nt)) * np.ones((ns, nt))
        M = np.block([[Mss, Mst], [Mst.T, Mtt]])
        return M

    def _get_kernel_matrix(self, X_source, X_target):
        Kss = pairwise_kernels(X_source, metric=self.kernel)
        Ktt = pairwise_kernels(X_target, metric=self.kernel)
        Kst = pairwise_kernels(X_source, X_target, metric=self.kernel)
        K = np.block([[Kss, Kst], [Kst.T, Ktt]])
        return K

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
        X, sample_domain = check_X_domain(
            X,
            sample_domain,
            allow_multi_source=True,
            allow_multi_target=True,
        )
        X_source, X_target = source_target_split(X, sample_domain=sample_domain)

        if self.n_components is None:
            n_components = min(X.shape[0], X.shape[1])
        else:
            n_components = self.n_components
        self.X_source_ = X_source
        self.X_target_ = X_target

        n = X.shape[0]
        source_mask = extract_source_indices(sample_domain)

        H = np.identity(n) - 1 / n * np.ones((n, n))
        K = self._get_kernel_matrix(X_source, X_target)
        M = self._get_mmd_matrix(X_source.shape[0], X_target.shape[0], sample_domain)
        M /= np.linalg.norm(M, ord="fro")
        G = np.identity(n)

        EPS_eigval = 1e-10
        last_loss = -2 * self.tol
        for i in range(self.max_iter):
            # update A
            B = K @ M @ K.T + self.tradeoff * G
            C = K @ H @ K.T
            B = B + EPS_eigval * np.identity(n)
            C = C + EPS_eigval * np.identity(n)
            phi, A = scipy.linalg.eigh(B, C)
            phi = phi + EPS_eigval
            indices = np.argsort(phi)[:n_components]
            phi, A = phi[indices], A[:, indices]
            error_eigv = np.linalg.norm(B @ A - C @ A @ np.diag(phi))
            if error_eigv > 1e-5:
                warnings.warn(
                    "The solution of the generalized eigenvalue problem "
                    "is not accurate."
                )

            # update G
            A_norms = np.linalg.norm(A, axis=1)
            G = np.zeros(n, dtype=np.float64)
            G[A_norms != 0] = 1 / (2 * A_norms[A_norms != 0] + EPS_eigval)
            G[~source_mask] = 1
            G = np.diag(G)

            loss = np.trace(A.T @ K @ M @ K @ A)
            reg = (
                np.sum(np.linalg.norm(A[source_mask], axis=1))
                + np.linalg.norm(A[~source_mask]) ** 2
            )
            loss_total = loss + self.tradeoff * reg
            # print objective function and constraint satisfaction
            if self.verbose:
                print(
                    f"iter {i}: loss={loss_total:.3e}, loss_mmd={loss:.3e}, "
                    f"reg={reg:.3e}"
                )
                mat = A.T @ K @ H @ K.T @ A
                cond = np.allclose(mat, np.identity(n_components))
                dist = np.linalg.norm(mat - np.identity(n_components))
                print(f"Constraint satisfaction: {cond}, dist={dist:.3e}")
                print(f"Error of generalized eigendecomposition: {error_eigv:.3e}")

            if last_loss == 0 or np.abs(last_loss - loss_total) / last_loss < self.tol:
                break
            else:
                last_loss = loss_total

        self.A_ = A

        return self


def TransferJointMatching(
    base_estimator=None,
    n_components=None,
    tradeoff=1e-2,
    kernel="rbf",
    max_iter=100,
    tol=0.01,
):
    """

    Parameters
    ----------
    base_estimator : object, default=None
        estimator used for fitting and prediction
    n_components : int, default=None
        The numbers of components to learn with PCA.
        Should be less or equal to the number of samples
        of the source and target data.
    tradeoff : float, default=1e-2
        The tradeoff constant for the TJM algorithm.
        It serves to trade off feature matching and instance
        reweighting.
    max_iter : int>0, default=100
        The maximal number of iteration before stopping when
        fitting.
    kernel : kernel object, default='rbf'
        The kernel computed between data.
    tol :

    Returns
    -------
    pipeline : Pipeline
        A pipeline containing a TransferJointMatchingAdapter.

    References
    ----------
    .. [26]  [Long et al., 2014] Long, M., Wang, J., Ding, G., Sun, J., and Yu, P.
             (2014). Transfer joint matching for unsupervised domain adaptation.
             In IEEE Conference on Computer Vision and Pattern Recognition (CVPR),
             pages 1410–1417
    """
    if base_estimator is None:
        base_estimator = SVC()

    return make_da_pipeline(
        TransferJointMatchingAdapter(
            tradeoff=tradeoff,
            n_components=n_components,
            kernel=kernel,
            max_iter=max_iter,
            tol=tol,
        ),
        base_estimator,
    )


class TransferSubspaceLearningAdapter(BaseAdapter):
    """Domain Adaptation Using TSL: Transfer Subspace Learning.

    See [27]_ for details.

    Parameters
    ----------
    n_components : int, default=None
        The numbers of components to learn.
        Should be less or equal to the number of samples
        of the source and target data.
    base_method : str, default='flda'
        The method used to learn the subspace.
        Possible values are 'pca', 'flda', and 'lpp'.
    length_scale : float, default=2
        The length scale of the rbf kernel used in
        'lpp' method.
    mu : float, default=0.1
        The parameter of the regularization in the optimization
        problem.
    reg : float, default=0.01
        The regularization parameter of the covariance estimator.
        Possible values:
          - None: no shrinkage.
          - float between 0 and 1: fixed shrinkage parameter.
    max_iter : int>0, default=100
        The maximal number of iteration before stopping when
        fitting.
    tol : float, default=0.01
        The threshold for the differences between losses on two iteration
        before the algorithm stops
    verbose : bool, default=False
        If True, print the final gradient norm.

    Attributes
    ----------
    W_ : array of shape (n_features, n_components)
        The learned projection matrix.

    References
    ----------
    .. [27]  [Si et al., 2010] Si, S., Tao, D. and Geng, B.
             Bregman Divergence-Based Regularization
             for Transfer Subspace Learning.
             In IEEE Transactions on Knowledge and Data Engineering.
             pages 929-942
    """

    def __init__(
        self,
        n_components=None,
        base_method="flda",
        length_scale=2,
        mu=0.1,
        reg=0.01,
        max_iter=100,
        tol=0.01,
        verbose=False,
    ):
        super().__init__()
        self.n_components = n_components
        _accepted_base_methods = ["pca", "flda", "lpp"]
        if base_method not in _accepted_base_methods:
            raise ValueError(f"base_method should be in {_accepted_base_methods}")
        self.base_method = base_method
        self.length_scale = length_scale
        self.mu = mu
        if reg is not None and (reg < 0 or reg > 1):
            raise ValueError("reg should be None or between 0 and 1.")
        self.reg = 0 if reg is None else reg
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose

    def _torch_cov(self, X):
        """Compute the covariance matrix of X using torch."""
        torch = self.torch
        reg = self.reg

        n_samples, d = X.shape
        X = X - torch.mean(X, dim=0)
        cov = X.T @ X / n_samples
        cov = (1 - reg) * cov + reg * torch.trace(cov) * torch.eye(d)

        return cov

    def _D(self, W, X_source, X_target):
        """Divergence objective function"""
        torch = self.torch

        Z_source = X_source @ W
        Z_target = X_target @ W

        sigma_1 = self._torch_cov(Z_source)
        sigma_2 = self._torch_cov(Z_target)
        sigma_11 = 2 * sigma_1
        sigma_12 = sigma_1 + sigma_2
        sigma_22 = 2 * sigma_2

        L_11 = torch.linalg.cholesky(torch.linalg.inv(sigma_11))
        L_12 = torch.linalg.cholesky(torch.linalg.inv(sigma_12))
        L_22 = torch.linalg.cholesky(torch.linalg.inv(sigma_22))
        Kss = torch.exp(-0.5 * torch.cdist(Z_source @ L_11, Z_source @ L_11))
        Kst = torch.exp(-0.5 * torch.cdist(Z_source @ L_12, Z_target @ L_12))
        Ktt = torch.exp(-0.5 * torch.cdist(Z_target @ L_22, Z_target @ L_22))

        return torch.mean(Kss) + torch.mean(Ktt) - 2 * torch.mean(Kst)

    def _F(self, W, X_source, y_source):
        """Subspace learning objective function"""
        torch = self.torch
        base_method = self.base_method

        if base_method == "pca":
            cov = self._torch_cov(X_source)
            loss = -torch.trace(W.T @ cov @ W)
        elif base_method == "flda":
            classes = torch.unique(y_source)
            classes_means = torch.stack(
                [torch.mean(X_source[y_source == c], dim=0) for c in classes]
            )
            classes_n_samples = torch.stack([torch.sum(y_source == c) for c in classes])
            classes_means = classes_means * torch.sqrt(classes_n_samples).reshape(-1, 1)
            S_W = self._torch_cov(classes_means)
            S_B = self._torch_cov(X_source)
            loss = torch.trace(W.T @ S_W @ W) / torch.trace(W.T @ S_B @ W)
        elif base_method == "lpp":
            # E is the Gaussian kernel if (y_source)_i == (y_source)_j and 0 otherwise
            E = torch.exp(-torch.cdist(X_source, X_source) / self.length_scale)
            E = E * (y_source[:, None] == y_source[None, :])
            D = torch.diag(torch.sum(E, dim=1))
            loss = -2 * torch.trace(W.T @ X_source.T @ (D - E) @ X_source @ W)

        return loss

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
        try:
            import torch

            self.torch = torch
        except ImportError:
            raise ImportError(
                "TransferSubspaceLearningAdapter requires pytorch to be installed."
            )

        X, sample_domain = check_X_domain(
            X,
            sample_domain,
            allow_multi_source=True,
            allow_multi_target=True,
        )
        X_source, X_target, y_source, _ = source_target_split(
            X, y, sample_domain=sample_domain
        )

        if self.n_components is None:
            n_components = min(X.shape[0], X.shape[1])
        else:
            n_components = self.n_components

        # Convert data to torch tensors
        X_source = torch.tensor(X_source, dtype=torch.float64)
        y_source = torch.tensor(y_source)
        X_target = torch.tensor(X_target, dtype=torch.float64)

        # Solve the optimization problem
        # min_W F(W) + mu * D(W)
        # s.t. W^T W = I

        def _orth(W):
            if type(W) is np.ndarray:
                W = np.linalg.qr(W)[0]
            else:
                W = torch.linalg.qr(W)[0]
            return W

        def func(W):
            W = _orth(W)
            loss = self._F(W, X_source, y_source)
            loss = loss + self.mu * self._D(W, X_source, X_target)
            return loss

        # Optimize using torch solver
        W = torch.eye(X.shape[1], dtype=torch.float64, requires_grad=True)
        W = W[:, :n_components]
        W, _ = torch_minimize(
            func, W, tol=self.tol, max_iter=self.max_iter, verbose=self.verbose
        )
        W = _orth(W)

        # store W
        self.W_ = W

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
        X_t : array-like, shape (n_samples, n_components)
            The data transformed to the target subspace.
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
        X_source, X_target = source_target_split(X, sample_domain=sample_domain)

        if X_source.shape[0]:
            X_source = np.dot(X_source, self.W_)
        if X_target.shape[0]:
            X_target = np.dot(X_target, self.W_)
        # xxx(okachaiev): this could be done through a more high-level API
        X_adapt, _ = source_target_merge(
            X_source, X_target, sample_domain=sample_domain
        )
        return X_adapt


def TransferSubspaceLearning(
    base_estimator=None,
    n_components=None,
    base_method="flda",
    length_scale=2,
    mu=0.1,
    reg=0.01,
    max_iter=100,
    tol=0.01,
    verbose=False,
):
    """Domain Adaptation Using Transfer Subspace Learning.

    Parameters
    ----------
    n_components : int, default=None
        The numbers of components to learn.
        Should be less or equal to the number of samples
        of the source and target data.
    base_method : str, default='flda'
        The method used to learn the subspace.
        Possible values are 'pca', 'flda', and 'lpp'.
    length_scale : float, default=2
        The length scale of the rbf kernel used in
        'lpp' method.
    mu : float, default=0.1
        The parameter of the regularization in the optimization
        problem.
    reg : float, default=0.01
        The regularization parameter of the covariance estimator.
        Possible values:
          - None: no shrinkage.
          - float between 0 and 1: fixed shrinkage parameter.
    max_iter : int>0, default=100
        The maximal number of iteration before stopping when
        fitting.
    tol : float, default=0.01
        The threshold for the differences between losses on two iteration
        before the algorithm stops
    verbose : bool, default=False
        If True, print the final gradient norm.

    Returns
    -------
    pipeline : Pipeline
        A pipeline containing a TransferSubspaceLearning estimator.

    References
    ----------
    .. [27]  [Si et al., 2010] Si, S., Tao, D. and Geng, B.
             Bregman Divergence-Based Regularization
             for Transfer Subspace Learning.
             In IEEE Transactions on Knowledge and Data Engineering.
             pages 929-942
    """
    if base_estimator is None:
        base_estimator = KNeighborsClassifier(n_neighbors=1)

    return make_da_pipeline(
        TransferSubspaceLearningAdapter(
            n_components=n_components,
            base_method=base_method,
            length_scale=length_scale,
            mu=mu,
            reg=reg,
            max_iter=max_iter,
            tol=tol,
            verbose=verbose,
        ),
        base_estimator,
    )
