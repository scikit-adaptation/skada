# Author: Theo Gnassounou <theo.gnassounou@inria.fr>
#         Remi Flamary <remi.flamary@polytechnique.edu>
#         Oleksii Kachaiev <kachayev@gmail.com>
#         Ruben Bueno <ruben.bueno@polytechnique.edu>
#
# License: BSD 3-Clause

import warnings

import numpy as np
import scipy.linalg
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.svm import SVC
from sklearn.utils import check_random_state

from ._pipeline import make_da_pipeline
from .base import BaseAdapter
from .utils import check_X_domain, source_target_merge, source_target_split


class SubspaceAlignmentAdapter(BaseAdapter):
    """Domain Adaptation Using Subspace Alignment.

    See [1]_ for details.

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
    .. [1] Basura Fernando et. al. Unsupervised Visual
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
            The data transformed to the target subspace.
        y_t : array-like, shape (n_samples,)
            The labels (same as y).
        sample_domain : array-like, shape (n_samples,)
            The domain labels transformed to the target subspace
            (same as sample_domain).
        weights : None
            No weights are returned here.
        """
        X, sample_domain = check_X_domain(
            X,
            sample_domain,
            allow_multi_source=True,
            allow_multi_target=True,
        )
        X_source, X_target = source_target_split(X, sample_domain=sample_domain)

        if X_source.shape[0]:
            X_source = np.dot(self.pca_source_.transform(X_source), self.M_)
        if X_target.shape[0]:
            X_target = np.dot(self.pca_target_.transform(X_target), self.M_)
        # xxx(okachaiev): this could be done through a more high-level API
        X_adapt, _ = source_target_merge(
            X_source, X_target, sample_domain=sample_domain
        )
        return X_adapt

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
            n_components = min(min(X_source.shape), min(X_target.shape))
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


def SubspaceAlignment(
    base_estimator=None,
    n_components=None,
    random_state=None,
):
    """Domain Adaptation Using Subspace Alignment.

    See [1]_ for details.

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
    .. [1] Basura Fernando et. al. Unsupervised Visual
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

    See [1]_ for details.

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
    .. [1] Sinno Jialin Pan et. al. Domain Adaptation via
           Transfer Component Analysis. In IEEE Transactions
           on Neural Networks, 2011.
    """

    def __init__(self, kernel="rbf", n_components=None, mu=0.1):
        super().__init__()
        self.kernel = kernel
        self.n_components = n_components
        self.mu = mu

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
            The data transformed to the target subspace.
        sample_domain : array-like, shape (n_samples,)
            The domain labels transformed to the target subspace
            (same as sample_domain).
        weights : None
            No weights are returned here.
        """
        X, sample_domain = check_X_domain(
            X,
            sample_domain,
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

    See [1]_ for details.

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
    .. [1] Sinno Jialin Pan et. al. Domain Adaptation via
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

    See [1]_ for details.

    Parameters
    ----------
    n_components : int, default=None
        The numbers of components to learn with PCA.
        Should be less or equal to the number of samples
        of the source and target data.
    random_state : int, default=None
        The seed for random number generation.
    tradeoff : float, default=0
        The tradeoff constant for the TJM algorithm.
        It serves to trade off feature matching and instance
        reweighting.
    max_iter : int>0, default=100
        The maximal number of iteration before stopping when
        fitting.
    kernel : kernel object, default='rbf'
        The kernel computed between data.
    verbose : bool, default=False
        If True, print the loss value at each iteration.

    Attributes
    ----------
    None

    References
    ----------
    .. [1]
            [Long et al., 2014] Long, M., Wang, J., Ding, G., Sun, J., and Yu, P.
            (2014). Transfer joint matching for unsupervised domain adaptation.
            In IEEE Conference on Computer Vision and Pattern Recognition (CVPR),
            pages 1410–1417

    """

    def __init__(
        self,
        n_components=None,
        random_state=None,
        tradeoff=0,
        max_iter=100,
        kernel="rbf",
        verbose=False,
    ):
        super().__init__()
        self.n_components = n_components
        self.tradeoff = tradeoff
        self.random_state = random_state
        self.kernel = kernel
        self.max_iter = max_iter
        self.verbose = verbose

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
            The data transformed to the target subspace.
        y_t : array-like, shape (n_samples,)
            The labels (same as y).
        sample_domain : array-like, shape (n_samples,)
            The domain labels transformed to the target subspace
            (same as sample_domain).
        weights : None
            No weights are returned here.
        """
        X, sample_domain = check_X_domain(
            X,
            sample_domain,
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
        Mss = 1 / ns**2 * np.ones((ns, ns))
        Mtt = 1 / nt**2 * np.ones((nt, nt))
        Mst = -1 / (ns * nt) * np.ones((ns, nt))
        M = np.block([[Mss, Mst], [Mst.T, Mtt]])
        return M

    def _get_kernel_matrix(self, X_source, X_target):
        Kss = pairwise_kernels(X_source, metric=self.kernel)
        Ktt = pairwise_kernels(X_target, metric=self.kernel)
        Kst = pairwise_kernels(X_source, X_target, metric=self.kernel)
        K = np.block([[Kss, Kst], [Kst.T, Ktt]])
        return K

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
            n_components = min(min(X_source.shape), min(X_target.shape))
        else:
            n_components = self.n_components
        self.random_state_ = check_random_state(self.random_state)

        n = X.shape[0]
        H = np.identity(n) - 1 / n * np.ones((n, n))
        M = self._get_mmd_matrix(X_source.shape[0], X_target.shape[0], sample_domain)
        self.X_source_ = X_source
        self.X_target_ = X_target
        K = self._get_kernel_matrix(X_source, X_target)
        M /= np.linalg.norm(M, ord="fr" "o")
        G = np.identity(n)

        # minimization of the objective function
        # \min_{A} tr(A^T K M K A) + tradeoff * (||A_s||_{2, 1} + ||A_t||_F^2)
        # s.t. A^T K H K^T A = I
        for i in range(self.max_iter):
            B = self.tradeoff * G + K @ M @ K + 1e-14 * np.identity(n)
            C = K @ H @ K + 1e-14 * np.identity(n)
            phi, A = scipy.linalg.eigh(B, C)
            phi = phi + 1e-14
            indices = np.argsort(phi)[:n_components]
            A = A[:, indices]
            for j in range(n):
                if sample_domain[j] < 0:
                    G[j, j] = 1
                else:
                    a = A[j]
                    if np.array_equal(a, np.zeros(a.shape)):
                        G[j, j] = 0
                    else:
                        G[j, j] = 1 / (2 * np.linalg.norm(a))
            if self.verbose:
                loss = np.trace(A.T @ K @ M @ K @ A)
                reg = (
                    np.linalg.norm(A[: X_source.shape[0]], ord=2, axis=1).sum()
                    + np.linalg.norm(A[X_source.shape[0] :], ord="fr" "o") ** 2
                )
                loss_total = loss + self.tradeoff * reg
                print(
                    f"iter {i}: loss={loss_total:.4f}, loss_mmd={loss:.4f}, "
                    f"reg={reg:.4f}"
                )
        self.A_ = A

        if not np.allclose(A.T @ K @ H @ K.T @ A, np.identity(n_components)):
            warnings.warn(
                "The solution does not satisfy the constraint " "A^T K H K^T A = I."
            )

        return self


def TransferJointMatching(
    base_estimator=None,
    random_state=None,
    n_components=1,
    tradeoff=0,
    kernel="rbf",
    max_iter=10,
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
    random_state : int, default=None
        The seed for random number generation.
    tradeoff : float, default=0
        The tradeoff constant for the TJM algorithm.
        It serves to trade off feature matching and instance
        reweighting.
    max_iter : int>0, default=100
        The maximal number of iteration before stopping when
        fitting.
    kernel : kernel object, default='rbf'
        The kernel computed between data.

    Returns
    -------
    pipeline : Pipeline
        A pipeline containing a TransferJointMatchingAdapter.

    References
    ----------
    .. [1]
        [Long et al., 2014] Long, M., Wang, J., Ding, G., Sun, J., and Yu, P.
        (2014). Transfer joint matching for unsupervised domain adaptation.
        In IEEE Conference on Computer Vision and Pattern Recognition (CVPR),
        pages 1410–1417
    """
    if base_estimator is None:
        base_estimator = SVC()

    return make_da_pipeline(
        TransferJointMatchingAdapter(
            random_state=random_state,
            tradeoff=tradeoff,
            n_components=n_components,
            kernel=kernel,
            max_iter=max_iter,
        ),
        base_estimator,
    )
