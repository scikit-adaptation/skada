import numpy as np
import scipy
from scipy.linalg import eigh
from scipy.sparse import issparse
from scipy.stats import norm
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

from .base import DAEstimator
from .utils import source_target_split


def convex_relaxation(xs, xt):
    """
    Perform convex relaxation of the covariance difference matrix.

    This relaxation involves computing the eigenvalue decomposition
    of the symmetric covariance difference matrix, inverting the signs
    of negative eigenvalues, and reconstructing the matrix. This corresponds
    to an upper bound on the covariance difference between source and target domains.

    Parameters
    ----------
    xs : ndarray of shape (n_source_samples, n_features)
        Feature data from the source domain.

    xt : ndarray of shape (n_target_samples, n_features)
        Feature data from the target domain.

    Returns
    -------
    D : ndarray of shape (n_features, n_features)
        Relaxed covariance difference matrix.

    References
    ----------
    Ramin Nikzad-Langerodi et al., "Domain-Invariant Regression
    under Beer-Lambert's Law", Proc. ICMLA, 2019.

    Examples
    --------
    >>> import numpy as np
    >>> from skada._dipls import convex_relaxation
    >>> xs = np.random.random((100, 10))
    >>> xt = np.random.random((100, 10))
    >>> D = convex_relaxation(xs, xt)
    """
    # Ensure input arrays are numerical
    xs = np.asarray(xs, dtype=np.float64)
    xt = np.asarray(xt, dtype=np.float64)

    # Check for NaN or infinite values
    if not np.all(np.isfinite(xs)) or not np.all(np.isfinite(xt)):
        raise ValueError(
            "Input arrays must not contain NaN or infinite values. one sample."
        )

    # Check for complex data
    if np.iscomplexobj(xs) or np.iscomplexobj(xt):
        raise ValueError("Complex data not supported.")

    # Preliminaries
    ns = np.shape(xs)[0]
    nt = np.shape(xt)[0]
    x = np.vstack([xs, xt])
    x = x[..., :] - np.mean(x, 0)

    # Compute difference between source and target covariance matrices
    rot = 1 / ns * xs.T @ xs - 1 / nt * xt.T @ xt

    # Convex Relaxation
    w, v = eigh(rot)
    eigs = np.abs(w)
    eigs = np.diag(eigs)
    D = v @ eigs @ v.T

    return D


def dipals(x, y, xs, xt, A, reg_param, heuristic: bool = False, target_domain=0):
    """
    Perform (Multiple) Domain-Invariant Partial Least Squares (di-PLS) regression.

    This method fits a PLS regression model using labeled source domain and
    unlabeled target domain data such that domain-specific feature distributions
    are aligned in terms of second moment differences.

    Parameters
    ----------
    x : ndarray of shape (n_samples, n_features)
        Labeled source domain data.

    y : ndarray of shape (n_samples,)
        Response variable associated with the source domain.

    xs : ndarray of shape (n_source_samples, n_features)
        Source domain feature data.

    xt : ndarray of shape (n_target_samples, n_features) or list of ndarray
        Target domain feature data. Multiple domains can be provided as a list.

    A : int
        Number of latent variables to use in the model.

    reg_param : float or tuple of len(reg_param)=A
        Regularization parameter. If a single value is provided, the same
        regularization is applied to all latent variables.

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
        Intercept of the regression model.

    T : ndarray of shape (n_samples, A)
        Training data projections (scores).

    Ts : ndarray of shape (n_source_samples, A)
        Source domain projections (scores).

    Tt : ndarray of shape (n_target_samples, A)
        Target domain projections (scores).

    W : ndarray of shape (n_features, A)
        Weight matrix.

    P : ndarray of shape (n_features, A)
        Loadings matrix corresponding to x.

    Ps : ndarray of shape (n_features, A)
        Loadings matrix corresponding to xs.

    Pt : ndarray of shape (n_features, A)
        Loadings matrix corresponding to xt.

    E : ndarray of shape (n_source_samples, n_features)
        Residuals of source domain data.

    Es : ndarray of shape (n_source_samples, n_features)
        Source domain residual matrix.

    Et : ndarray of shape (n_target_samples, n_features)
        Target domain residual matrix.

    Ey : ndarray of shape (n_source_samples, 1)
        Residuals of response variable in the source domain.

    C : ndarray of shape (A, 1)
        Regression vector relating source projections to the response variable.

    opt_l : ndarray of shape (A, 1)
        Heuristically determined regularization parameter for each latent variable.

    discrepancy : ndarray
        The variance discrepancy between source and target domain projections.

    References
    ----------
    [37] Nikzad-Langerodi, R., Zellinger, W., Saminger-Platz, S., & Moser, B. A. (2020).
         Domain adaptation for regression under Beer–Lambert’s law.
         Knowledge-Based Systems, 210, 106447.

    Examples
    --------
    >>> import numpy as np
    >>> from skada._dipls import dipals
    >>> x = np.random.random((100, 10))
    >>> y = np.random.random((100, 1))
    >>> xs = np.random.random((50, 10))
    >>> xt = np.random.random((50, 10))
    >>> results = dipals(x, y, xs, xt, 2, 0.1)
    """
    # Get array dimensions
    (n, k) = np.shape(x)
    (ns, k) = np.shape(xs)

    # If multiple target domains are passed
    if isinstance(xt, list):
        Pt = []
        Tt = []

        for z in range(len(xt)):
            Tti = np.zeros([np.shape(xt[z])[0], A])
            Pti = np.zeros([k, A])

            Pt.append(Pti)
            Tt.append(Tti)

    else:
        (nt, k) = np.shape(xt)
        Tt = np.zeros([nt, A])
        Pt = np.zeros([k, A])

    T = np.zeros([n, A])
    P = np.zeros([k, A])
    Ts = np.zeros([ns, A])
    Ps = np.zeros([k, A])
    W = np.zeros([k, A])
    C = np.zeros([A, 1])
    opt_l = np.zeros(A)
    discrepancy = np.zeros(A)
    Ident = np.eye(k)

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
        w_pls = (y.T @ x) / (y.T @ y)  # Ordinary PLS solution
        # w_pls = w_pls.reshape(1, -1)

        if lA != 0 or heuristic is True:  # In case of regularization
            if isinstance(xt, np.ndarray):
                # Convex relaxation of covariance difference matrix
                D = convex_relaxation(xs, xt)

            # Multiple target domains
            elif isinstance(xt, list):
                # print('Relaxing domains ... ')
                ndoms = len(xt)
                D = np.zeros([k, k])

                for z in range(ndoms):
                    d = convex_relaxation(xs, xt[z])
                    D = D + d

            else:
                print(
                    "xt must either be a matrix or list of "
                    "(appropriately dimensioned) matrices"
                )

            if heuristic is True:  # Regularization parameter heuristic
                w_pls = w_pls / np.linalg.norm(w_pls)
                gamma = (np.linalg.norm(x - np.outer(y, w_pls)) ** 2) / (
                    w_pls @ D @ w_pls.T
                )
                opt_l[i] = gamma
                lA = gamma

            reg = Ident + lA / (y.T @ y) * D
            w = scipy.linalg.solve(reg.T, w_pls.T, assume_a="sym").T

            # Normalize w
            w = w / np.linalg.norm(w)

            # Absolute difference between variance of source and
            # target domain projections
            discrepancy[i] = (w @ D @ w.T).item()

        else:
            if isinstance(xt, list):
                D = convex_relaxation(xs, xt[0])

            else:
                D = convex_relaxation(xs, xt)

            w = w_pls / np.linalg.norm(w_pls)
            discrepancy[i] = (w @ D @ w.T).item()

        # Compute scores
        t = x @ w.T
        ts = xs @ w.T

        if isinstance(xt, list):
            tt = []

            for z in range(len(xt)):
                tti = xt[z] @ w.T
                tt.append(tti)

        else:
            tt = xt @ w.T

        # Regress y on t
        c = (y.reshape(-1, 1).T @ t) / (t.T @ t)

        # Compute loadings
        p = (t.T @ x) / (t.T @ t)
        ps = (ts.T @ xs) / (ts.T @ ts)
        if isinstance(xt, list):
            pt = []

            for z in range(len(xt)):
                pti = (tt[z].T @ xt[z]) / (tt[z].T @ tt[z])
                pt.append(pti)

        else:
            pt = (tt.T @ xt) / (tt.T @ tt)

        # Deflate X and y (Gram-Schmidt orthogonalization)
        x = x - np.outer(t, p)
        xs = xs - np.outer(ts, ps)

        if isinstance(xt, list):
            for z in range(len(xt)):
                xt[z] = xt[z] - np.outer(tt[z], pt[z])

        else:
            xt = xt - np.outer(tt, pt)

        y = y - c * t

        # Store w,t,p,c
        W[:, i] = w
        T[:, i] = t.reshape(n)
        Ts[:, i] = ts.reshape(ns)
        P[:, i] = p.reshape(k)
        Ps[:, i] = ps.reshape(k)
        C[i] = c

        if isinstance(xt, list):
            for z in range(len(xt)):
                Pt[z][:, i] = pt[z].reshape(k)
                Tt[z][:, i] = tt[z].reshape(np.shape(xt[z])[0])

        else:
            Pt[:, i] = pt.reshape(k)
            Tt[:, i] = tt.reshape(nt)

    if isinstance(reg_param, tuple):  # Check if multiple regularization
        # parameters are passed (one for each LV)
        if target_domain == 0:  # Multiple target domains (Domain unknown)
            b = W @ (np.linalg.inv(P.T @ W)) @ C

        elif isinstance(xt, np.ndarray):  # Single target domain
            b = W @ (np.linalg.inv(Pt.T @ W)) @ C

        elif isinstance(xt, list):  # Multiple target domains (Domain known)
            b = W @ (np.linalg.inv(Pt[target_domain - 1].T @ W)) @ C

    else:
        b = W @ (np.linalg.inv(Pt.T @ W)) @ C

    # Store residuals
    E = x
    Es = xs
    Et = xt
    Ey = y

    return b, T, Ts, Tt, W, P, Ps, Pt, E, Es, Et, Ey, C, opt_l, discrepancy


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
        Number of latent variables to be used in the model.

    reg_param : float or tuple with len(reg_param)=A, default=0
        Regularization parameter. If a single value is provided, the same
        regularization is applied to all latent variables.

    centering : bool, default=True
            If True, source and target domain data are mean-centered.

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
            the test data will be rescaled to the mean of xt or xs, respectively.
            If an ndarray is provided, the test data will be rescaled to the mean
            of the provided array.

    Attributes
    ----------
    n_ : int
        Number of samples in `x`.

    ns_ : int
        Number of samples in `xs`.

    nt_ : int
        Number of samples in `xt`.

    n_features_in_ : int
        Number of features (variables) in `x`.

    mu_ : ndarray of shape (n_features,)
        Mean of columns in `x`.

    mu_s_ : ndarray of shape (n_features,)
        Mean of columns in `xs`.

    mu_t_ : ndarray of shape (n_features,) or ndarray of shape (n_domains, n_features)
        Mean of columns in `xt`, averaged per target domain if multiple domains exist.

    b_ : ndarray of shape (n_features, 1)
        Regression coefficient vector.

    b0_ : float
        Intercept of the regression model.

    T_ : ndarray of shape (n_samples, A)
        Training data projections (scores).

    Ts_ : ndarray of shape (n_source_samples, A)
        Source domain projections (scores).

    Tt_ : ndarray of shape (n_target_samples, A)
        Target domain projections (scores).

    W_ : ndarray of shape (n_features, A)
        Weight matrix.

    P_ : ndarray of shape (n_features, A)
        Loadings matrix corresponding to x.

    Ps_ : ndarray of shape (n_features, A)
        Loadings matrix corresponding to xs.

    Pt_ : ndarray of shape (n_features, A)
        Loadings matrix corresponding to xt.

    E_ : ndarray of shape (n_source_samples, n_features)
        Residuals of source domain data.

    Es_ : ndarray of shape (n_source_samples, n_features)
        Source domain residual matrix.

    Et_ : ndarray of shape (n_target_samples, n_features)
        Target domain residual matrix.

    Ey_ : ndarray of shape (n_source_samples, 1)
        Residuals of response variable in the source domain.

    C_ : ndarray of shape (A, 1)
        Regression vector relating source projections to the response variable.

    opt_l_ : ndarray of shape (A, 1)
        Heuristically determined regularization parameter for each latent variable.

    discrepancy_ : ndarray
        The variance discrepancy between source and target domain projections.

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
    >>> x = np.random.rand(100, 10)
    >>> y = np.random.rand(100)
    >>> xs = x
    >>> xt = np.random.rand(50, 10)
    >>> yt = np.random.rand(50)
    >>> X = np.vstack([xs, xt])
    >>> Y = np.hstack([y, yt])
    >>> class_variable = np.concatenate((np.full(100, 1), np.full(50, -1)))
    >>> model = DIPLS(A=5, reg_param=10)
    >>> model.fit(X, Y, sample_domain=class_variable)
    DIPLS(A=5, reg_param=10)
    >>> xtest = np.array([5, 7, 4, 3, 2, 1, 6, 8, 9, 10])
    >>> yhat = model.predict(xtest)
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
            Fitted model instance.
        """
        # Check for sparse input
        if issparse(X):
            raise ValueError(
                "Sparse input is not supported. Please convert your data to "
                "dense format."
            )

        # Validate input arrays
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
        Xs, Xt, ys, yt, ws, wt = source_target_split(
            X, y, sample_weight, sample_domain=sample_domain
        )

        # Preliminaries
        self.n_, self.n_features_in_ = Xs.shape
        self.ns_, _ = Xs.shape
        self.x_ = Xs
        self.y_ = ys
        self.xs_ = Xs
        self.xt_ = Xt
        self.b0_ = np.mean(self.y_)

        # Mean centering
        if self.centering:
            self.mu_ = np.mean(self.x_, axis=0)
            self.mu_s_ = np.mean(self.xs_, axis=0)
            self.x_ = self.x_ - self.mu_
            self.xs_ = self.xs_ - self.mu_s_
            y = self.y_ - self.b0_

            # Multiple target domains
            if isinstance(self.xt_, list):
                self.nt_, _ = Xt[0].shape
                self.mu_t_ = [np.mean(x, axis=0) for x in self.xt_]
                self.xt_ = [x - mu for x, mu in zip(self.xt_, self.mu_t_)]
            else:
                self.nt_, _ = Xt.shape
                self.mu_t_ = np.mean(self.xt_, axis=0)
                self.xt_ = self.xt_ - self.mu_t_
        else:
            y = self.y_

        x = self.x_
        xs = self.xs_
        xt = self.xt_

        # Fit model
        results = dipals(
            x,
            y,
            xs,
            xt,
            self.A,
            self.reg_param,
            heuristic=self.heuristic,
            target_domain=self.target_domain,
        )
        (
            self.b_,
            self.T_,
            self.Ts_,
            self.Tt_,
            self.W_,
            self.P_,
            self.Ps_,
            self.Pt_,
            self.E_,
            self.Es_,
            self.Et_,
            self.Ey_,
            self.C_,
            self.opt_l_,
            self.discrepancy_,
        ) = results

        self.is_fitted_ = True
        return self

    def predict(self, X, sample_domain=None, *, sample_weight=None):
        """
        Predict y using the fitted DIPLS model.

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
                if isinstance(self.xt_, list):
                    if self.target_domain == 0:
                        Xtest = X[..., :] - self.mu_s_
                    else:
                        Xtest = X[..., :] - self.mu_t_[self.target_domain - 1]
                else:
                    Xtest = X[..., :] - self.mu_t_
            elif self.rescale == "Source":
                Xtest = X[..., :] - self.mu_
            elif self.rescale == "none":
                Xtest = X
        elif isinstance(self.rescale, np.ndarray):
            Xtest = X[..., :] - np.mean(self.rescale, 0)
        else:
            raise Exception("rescale must either be Source, Target or a Dataset")

        yhat = Xtest @ self.b_ + self.b0_

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
