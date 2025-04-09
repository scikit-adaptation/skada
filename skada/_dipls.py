import numpy as np
import scipy
from scipy.linalg import eigh
from scipy.sparse import issparse
from scipy.stats import norm
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

from .base import DAEstimator
from .utils import source_target_split


def convex_relaxation(X_s, X_t):
    """
    Perform convex relaxation of the covariance difference matrix.

    This relaxation involves computing the eigenvalue decomposition
    of the symmetric covariance difference matrix, inverting the signs
    of negative eigenvalues, and reconstructing the matrix. This corresponds
    to an upper bound on the covariance difference between source and target domains.

    Parameters
    ----------
    X_s : ndarray of shape (n_source_samples, n_features)
        Feature data from the source domain.

    X_t : ndarray of shape (n_target_samples, n_features)
        Feature data from the target domain.

    Returns
    -------
    D : ndarray of shape (n_features, n_features)
        Relaxed covariance difference matrix.

    References
    ----------
    Ramin Nikzad-Langerodi et al., "Domain-Invarian_t Regression
    under Beer-Lambert's Law", Proc. ICMLA, 2019.

    Examples
    --------
    >>> import numpy as np
    >>> from skada._dipls import convex_relaxation
    >>> X_s = np.random.random((100, 10))
    >>> X_t = np.random.random((100, 10))
    >>> D = convex_relaxation(X_s, X_t)
    """
    # Ensure input array_s are numerical
    X_s = np.asarray(X_s, dtype=np.float64)
    X_t = np.asarray(X_t, dtype=np.float64)

    # Check for NaN or infinite values
    if not np.all(np.isfinite(X_s)) or not np.all(np.isfinite(X_t)):
        raise ValueError(
            "Input array_s must not con_tain NaN or infinite values. one sample."
        )

    # Check for complex data
    if np.iscomplexobj(X_s) or np.iscomplexobj(X_t):
        raise ValueError("Complex data not supported.")

    # Preliminaries
    n_s = np.shape(X_s)[0]
    n_t = np.shape(X_t)[0]
    x = np.vstack([X_s, X_t])
    x = x[..., :] - np.mean(x, 0)

    # Compute difference between source and target covariance matrices
    rot = 1 / n_s * X_s.T @ X_s - 1 / n_t * X_t.T @ X_t

    # Convex Relaxation
    w, v = eigh(rot)
    eigs = np.abs(w)
    eigs = np.diag(eigs)
    D = v @ eigs @ v.T

    return D


def dipals(X_s, y_s, X_t, A, reg_param, heuristic: bool = False, target_domain=0):
    """
    Perform (Multiple) Domain-Invarian_t Partial Least Squares (di-PLS) regression.

    This method fits a PLS regression model using labeled source domain and
    unlabeled target domain data such that domain-specific feature distributions
    are aligned in terms of second moment differences.

    Parameters
    ----------
    X_s : ndarray of shape (n_source_samples, n_features)
        Source domain feature data.

    y_s : ndarray of shape (n_samples,)
        Response variable associated with the source domain.

    X_t : ndarray of shape (n_target_samples, n_features) or list of ndarray
        Target domain feature data. Multiple domains can be provided as a list.

    A : int
        Number of latent variables to use in the model.

    reg_param : float or tuple of len(reg_param)=A
        Regularization parameter. If a single value is provided, the same
        regularization is applied to all laten_t variables.

    heuristic : bool, default=False
        If True, automatically determine the regularization parameter to equally
        balance fitting to Y and minimizing domain discrepancy.

    target_domain : int, default=0
        Specifies which target domain the model should apply to, where 0
        indicates the source domain.

    Returns
    -------
    b : ndarray of shape (n_features, 1)
        Regression coefficient vector.

    b0 : float
        intercept of the regression model.

    References
    ----------
    [37] Nikzad-Langerodi, R., Zellinger, W., Saminger-Platz, S., & Moser, B. A. (2020).
         Domain adaptation for regression under Beer–Lambert’s law.
         Knowledge-Based Systems, 210, 106447.

    Examples
    --------
    >>> import numpy as np
    >>> from skada._dipls import dipals
    >>> X_s = np.random.random((50, 10))
    >>> y_s = np.random.random((50, 1))
    >>> X_t = np.random.random((50, 10))
    >>> results = dipals(X_s, y_s, X_t, 2, 0.1)
    """
    (n_s, k) = np.shape(X_s)

    # If multiple target domains are passed
    if isinstance(X_t, list):
        P_t = []
        T_t = []

        for z in range(len(X_t)):
            T_ti = np.zeros([np.shape(X_t[z])[0], A])
            P_ti = np.zeros([k, A])

            P_t.append(P_ti)
            T_t.append(T_ti)

    else:
        (n_t, k) = np.shape(X_t)
        T_t = np.zeros([n_t, A])
        P_t = np.zeros([k, A])

    T_s = np.zeros([n_s, A])
    P_s = np.zeros([k, A])
    W = np.zeros([k, A])
    C = np.zeros([A, 1])
    opt_l = np.zeros(A)
    discrepancy = np.zeros(A)
    Iden_t = np.eye(k)

    # Compute LVs
    for i in range(A):
        if (
            isinstance(reg_param, tuple) and len(reg_param) == A
        ):  # Separate regularization params for each LV
            lA = reg_param[i]

        elif isinstance(
            reg_param, (float, int, np.int64)
        ):  # The same regularization param for each LV
            lA = reg_param

        else:
            raise ValueError(
                "The regularization parameter must be either a single value "
                "or an A-tuple."
            )

        # Compute Domain-Invariant Weight Vector
        w_pls = (y_s.T @ X_s) / (y_s.T @ y_s)  # Ordinary PLS solution

        if lA != 0 or heuristic is True:  # In case of regularization
            if isinstance(X_t, np.ndarray):
                # Convex relaxation of covariance difference matrix
                D = convex_relaxation(X_s, X_t)

            # Multiple target domains
            elif isinstance(X_t, list):
                ndoms = len(X_t)
                D = np.zeros([k, k])

                for z in range(ndoms):
                    d = convex_relaxation(X_s, X_t[z])
                    D = D + d

            else:
                print(
                    "X_t must either be a matrix or list of "
                    "(appropriately dimensioned) matrices"
                )

            if heuristic is True:  # Regularization parameter heuristic
                w_pls = w_pls / np.linalg.norm(w_pls)
                gamma = (np.linalg.norm(X_s - np.outer(y_s, w_pls)) ** 2) / (
                    w_pls @ D @ w_pls.T
                )
                opt_l[i] = gamma
                lA = gamma

            reg = Iden_t + lA / (y_s.T @ y_s) * D
            w = scipy.linalg.solve(reg.T, w_pls.T, assume_a="sym").T

            # Normalize w
            w = w / np.linalg.norm(w)

            # Absolute difference between variance of source and
            # target domain projections
            discrepancy[i] = (w @ D @ w.T).item()

        else:
            if isinstance(X_t, list):
                D = convex_relaxation(X_s, X_t[0])

            else:
                D = convex_relaxation(X_s, X_t)

            w = w_pls / np.linalg.norm(w_pls)
            discrepancy[i] = (w @ D @ w.T).item()

        # Compute scores
        t_s = X_s @ w.T

        if isinstance(X_t, list):
            t_t = []

            for z in range(len(X_t)):
                t_ti = X_t[z] @ w.T
                t_t.append(t_ti)

        else:
            t_t = X_t @ w.T

        # Regress y on t
        c = (y_s.reshape(-1, 1).T @ t_s) / (t_s.T @ t_s)

        # Compute loadings
        p_s = (t_s.T @ X_s) / (t_s.T @ t_s)
        if isinstance(X_t, list):
            p_t = []

            for z in range(len(X_t)):
                p_ti = (t_t[z].T @ X_t[z]) / (t_t[z].T @ t_t[z])
                p_t.append(p_ti)

        else:
            p_t = (t_t.T @ X_t) / (t_t.T @ t_t)

        # Deflate X and y (Gram-Schmidt orthogonalization)
        X_s = X_s - np.outer(t_s, p_s)

        if isinstance(X_t, list):
            for z in range(len(X_t)):
                X_t[z] = X_t[z] - np.outer(t_t[z], p_t[z])

        else:
            X_t = X_t - np.outer(t_t, p_t)

        y_s = y_s - c * t_s

        # Store w,t,p,c
        W[:, i] = w
        T_s[:, i] = t_s.reshape(n_s)
        P_s[:, i] = p_s.reshape(k)
        C[i] = c

        if isinstance(X_t, list):
            for z in range(len(X_t)):
                P_t[z][:, i] = p_t[z].reshape(k)
                T_t[z][:, i] = t_t[z].reshape(np.shape(X_t[z])[0])

        else:
            P_t[:, i] = p_t.reshape(k)
            T_t[:, i] = t_t.reshape(n_t)

    if isinstance(reg_param, tuple):  # Check if multiple regularization
        # parameters are passed (one for each LV)
        if target_domain == 0:  # Multiple target domains (Domain unknown)
            b = W @ (np.linalg.inv(P_s.T @ W)) @ C

        elif isinstance(X_t, np.ndarray):  # Single target domain
            b = W @ (np.linalg.inv(P_t.T @ W)) @ C

        elif isinstance(X_t, list):  # Multiple target domains (Domain known)
            b = W @ (np.linalg.inv(P_t[target_domain - 1].T @ W)) @ C

    else:
        b = W @ (np.linalg.inv(P_t.T @ W)) @ C

    return b


class DIPLS(DAEstimator):
    """
    Domain-Invariant Partial Least Squares (DIPLS) algorithm for domain adaptation.

    This class implements the DIPLS algorithm, which aligns feature
    distributions of source and target domain data in terms of second order
    moment differences while identifying latent variables with high covariance
    between input data and the response variable.

    Parameters
    ----------
    A : int
        Number of laten_t variables to be used in the model.

    reg_param : float or tuple with len(reg_param)=A, default=0
        Regularization parameter. If a single value is provided, the same
        regularization is applied to all latent variables.

    cen_tering : bool, default=True
            If True, source and target domain data are mean-cen_tered.

    heuristic : bool, default=False
        If True, the regularization parameter is set to a heuristic value that
        balances fitting the output variable y and minimizing domain discrepancy.

    target_domain : int, default=0
        If multiple target domains are passed, target_domain specifies
        for which of the target domains the model should apply.
        If target_domain=0, the model applies to the source domain,
        if target_domain=1, it applies to the first target domain, and so on.

    rescale : Union[str, ndarray], default='Target'
            Determines rescaling of the test data. If 'Target' or 'Source',
            the test data will be rescaled to the mean of X_t or X_s, respectively.
            If an ndarray is provided, the test data will be rescaled to the mean
            of the provided array.

    Attributes
    ----------
    mu_s_ : ndarray of shape (n_features,)
        Mean of columns in `X_s`.

    mu_t_ : ndarray of shape (n_features,) or ndarray of shape (n_domains, n_features)
        Mean of columns in `X_t`, averaged per target domain if multiple domains exist.

    b_ : ndarray of shape (n_features, 1)
        Regression coefficient vector.

    b0_ : float
        intercept of the regression model.

    is_fitted_ : bool, default=False
        Whether the model has been fitted to data.


    References
    ----------
    [37] Nikzad-Langerodi, R., Zellinger, W., Saminger-Platz, S., & Moser, B. A. (2020).
         Domain adaptation for regression under Beer–Lambert’s law.
         Knowledge-Based Systems, 210, 106447.

    Examples
    --------
    >>> import numpy as np
    >>> from skada import DIPLS
    >>> X_s = np.random.rand(100, 10)
    >>> y_s = np.random.rand(100)
    >>> X_t = np.random.rand(50, 10)
    >>> y_t = -np.random.rand(50)
    >>> X = np.vstack([X_s, X_t])
    >>> y = np.hstack([y_s, y_t])
    >>> sample_domain = np.concatenate((np.full(100, 1), np.full(50, -1)))
    >>> model = DIPLS(A=5, reg_param=10)
    >>> model.fit(X, y, sample_domain=sample_domain)
    DIPLS(A=5, reg_param=10)
    >>> X_test = np.array([5, 7, 4, 3, 2, 1, 6, 8, 9, 10])
    >>> yhat = model.predict(X_test)
    """

    def __init__(
        self,
        A=2,
        reg_param=0,
        centering=True,
        heuristic=False,
        target_domain=0,
        rescale="Target",
    ):
        super().__init__()
        self.A = A
        self.reg_param = reg_param
        self.centering = centering
        self.heuristic = heuristic
        self.target_domain = target_domain
        self.rescale = rescale

    def fit(self, X, y=None, sample_domain=None, *, sample_weight=None):
        """
        Fit the DIPLS model.

        This method fits the domain-invariant partial least squares (di-PLS) model
        using the provided source and target domain data. It can handle both single
        and multiple target domains.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Labeled input data from the source domain.

        y : ndarray of shape (n_samples, 1)
            Response variable corresponding to the input data `x`.

        sample_domain : array-like of shape (n_samples,), default=None
            Domain labels for the samples. If None, all samples are assumed to be
            from the same domain.

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights. If None, all samples are assumed to have equal weight.

        Returns
        -------
        self : object
            FiT_ted model instance.
        """
        # Check for sparse input
        if issparse(X):
            raise ValueError(
                "Sparse input is not supported. Please convert your data to "
                "dense format."
            )

        # Validate input array_s
        X, y = check_X_y(
            X,
            y,
            ensure_2d=True,
            allow_nd=False,
            accept_large_sparse=False,
            accept_sparse=False,
            force_all_finite=True,
        )

        # Split source and target data
        X_s, X_t, y_s, _, _, _ = source_target_split(
            X, y, sample_weight, sample_domain=sample_domain
        )

        # Preliminaries
        n_s_, n_features_in_ = X_s.shape
        n_s_, _ = X_s.shape

        self.b0_ = np.mean(y_s)

        # Mean centering
        if self.centering:
            self.mu_s_ = np.mean(X_s, axis=0)
            X_s = X_s - self.mu_s_
            y_s = y_s - self.b0_

            # Multiple target domains
            if isinstance(X_t, list):
                n_t_, _ = X_t[0].shape
                self.mu_t_ = [np.mean(x, axis=0) for x in X_t]
                X_t = [x - mu for x, mu in zip(X_t, self.mu_t_)]
            else:
                n_t_, _ = X_t.shape
                self.mu_t_ = np.mean(X_t, axis=0)
                X_t = X_t - self.mu_t_

        # Fit model
        self.b = dipals(
            X_s,
            y_s,
            X_t,
            self.A,
            self.reg_param,
            heuristic=self.heuristic,
            target_domain=self.target_domain,
        )

        self.is_fitted_ = True
        return self

    def predict(self, X, sample_domain=None, *, sample_weight=None):
        """
        Predict y using the fiT_ted DIPLS model.

        This method predicts the response variable for the provided test data using
        the fitted domain-invariant partial least squares (di-PLS) model.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test data matrix to perform the prediction on.

        sample_domain : array-like of shape (n_samples,), default=None
            Domain labels for the samples. If None, all samples are assumed to be
            from the same domain.

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights. If None, all samples are assumed to have equal weight.

        Returns
        -------
        yhat : ndarray of shape (n_samples_test,)
            Predicted response values for the test data.

        """
        check_is_fitted(self)

        # Check for sparse input
        if issparse(X):
            raise ValueError(
                "Sparse input is not supported. Please convert your data to "
                "dense format."
            )

        # Validate input array
        X = check_array(X, ensure_2d=False, allow_nd=False, force_all_finite=True)

        # Rescale Test data
        if isinstance(self.rescale, str):
            if self.rescale == "Target":
                if isinstance(X, list):
                    if self.target_domain == 0:
                        X_test = X[..., :] - self.mu_s_
                    else:
                        X_test = X[..., :] - self.mu_t_[self.target_domain - 1]
                else:
                    X_test = X[..., :] - self.mu_t_
            elif self.rescale == "Source":
                X_test = X[..., :] - self.mu_s_
            elif self.rescale == "none":
                X_test = X
        elif isinstance(self.rescale, np.ndarray):
            X_test = X[..., :] - np.mean(self.rescale, 0)
        else:
            raise Exception("rescale must either be Source, Target or a Dataset")

        yhat = X_test @ self.b_ + self.b0_

        # Ensure the shape of yhat matches the shape of y
        yhat = np.ravel(yhat)

        return yhat


def genspec(length, mu, sigma, mag, noise=0):
    """
    Generate a spectrum-like signal with optional random noise.

    Parameters
    ----------
    length : int
        Length of the generated signal.

    mu : float
        Mean of the Gaussian.

    sigma : float
        Standard deviation of the Gaussian.

    mag : float
        Magnitude of the Gaussian.


    Returns
    -------
    signal : ndarray of shape (length,)
        The generated Gaussian signal with noise.

    Examples
    --------
    >>> from skada._dipls import genspec
    >>> import numpy as np
    >>> import scipy.stats
    >>> signal = genspec(100, 50, 10, 5, noise=0.1)
    """
    s = mag * norm.pdf(np.arange(length), mu, sigma)
    n = noise * np.random.rand(length)
    signal = s + n

    return signal
