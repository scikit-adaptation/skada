# Author: Theo Gnassounou <theo.gnassounou@inria.fr>
#
# License: BSD 3-Clause

from ot import da

from .base import BaseDataAdaptEstimator, clone


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
        reg_e=1
    ):
        super().__init__(base_estimator)
        self.reg_e = reg_e

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

        self.ot_transport_ = clone(da.SinkhornTransport(reg_e=self.reg_e))
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


class CORAL(BaseDataAdaptEstimator):
    """Estimator based on reweighting samples using density estimation.
    Parameters
    ----------
    base_estimator : estimator object
        The base estimator to fit on reweighted data.
    weight_estimator : estimator object, optional
        The estimator to use to estimate the densities of source and target
        observations. If None, a KernelDensity estimator is used.

    Attributes
    ----------
    cov_source_inv_sqrt_: array
        Inverse of the square root of covariance of the source data with regularization.
    cov_target_sqrt_: array
        Square root of covariance of the target data with regularization.

    References
    ----------
    .. [1] Baochen Sun, Jiashi Feng, and Kate Saenko.
           Correlation Alignment for Unsupervised
           Domain Adaptation. In dvances in Computer
           Vision and Pattern Recognition, 2017.
    """

    def __init__(
        self,
        base_estimator,
        reg=0.1
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
        X_t : array-like, shape (n_samples, n_components)
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
        cov_source_ = np.cov(X.T) + self.reg * np.eye(X.shape[1])
        cov_target_ = np.cov(X_target.T) + self.reg * np.eye(X_target.shape[1])
        self.cov_source_inv_sqrt_ = scipy.linalg.inv(scipy.linalg.sqrtm(cov_source_))
        self.cov_target_sqrt_ = scipy.linalg.sqrtm(cov_target_)
