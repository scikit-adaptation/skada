# Author: Theo Gnassounou <theo.gnassounou@inria.fr>
#
# License: MIT License

from .base import BaseDataAdaptEstimator

from ot import da


class EMDTransport(BaseDataAdaptEstimator):
    """Domain Adaptation Using Optimal Transport.

    Parameters
    ----------
    base_estimator : estimator object
        The base estimator to fit on reweighted data.

    Attributes
    ----------
    ot_emd_ : object
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
        X_ = self.ot_emd_.transform(Xs=X)
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

        self.ot_emd_ = da.EMDTransport().fit(Xs=X, Xt=X_target)
        return self


class SinkhornTransport(BaseDataAdaptEstimator):
    """Domain Adaptation Using Optimal Transport.

    Parameters
    ----------
    base_estimator : estimator object
        The base estimator to fit on reweighted data.
    reg_e : float, default=1
        Entropic regularization parameter.

    Attributes
    ----------
    ot_sinkhorn_ : object
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
        X_ = self.ot_sinkhorn_.transform(Xs=X)
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

        self.ot_sinkhorn_ = da.SinkhornTransport(
            reg_e=self.reg_e
        ).fit(Xs=X, Xt=X_target)
        return self


class SinkhornLpl1Transport(BaseDataAdaptEstimator):
    """Domain Adaptation Using Optimal Transport.

    Parameters
    ----------
    base_estimator : estimator object
        The base estimator to fit on reweighted data.
    reg_e : float, default=1
        Entropic regularization parameter.
    reg_cl : float, default=0.1
        Class regularization parameter.

    Attributes
    ----------
    ot_lpl1_ : object
        The OT object based on Sinkhorn Algorithm
        + LpL1 class regularization fitted on the source
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
        reg_cl=0.1
    ):
        super().__init__(base_estimator)
        self.reg_e = reg_e
        self.reg_cl = reg_cl

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
        X_ = self.ot_lpl1_.transform(Xs=X)
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

        self.ot_lpl1_ = da.SinkhornLpl1Transport(
            reg_e=self.reg_e, reg_cl=self.reg_cl
        ).fit(Xs=X, ys=y, Xt=X_target)
        return self


class SinkhornL1l2Transport(BaseDataAdaptEstimator):
    """Domain Adaptation Using Optimal Transport.

    Parameters
    ----------
    base_estimator : estimator object
        The base estimator to fit on reweighted data.
    reg_e : float, default=1
        Entropic regularization parameter.
    reg_cl : float, default=0.1
        Class regularization parameter.

    Attributes
    ----------
    ot_l1l2_ : object
        The OT object based on Sinkhorn Algorithm
        + L1L2 class regularization fitted on the source
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
        reg_cl=0.1
    ):
        super().__init__(base_estimator)
        self.reg_e = reg_e
        self.reg_cl = reg_cl

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
        X_ = self.ot_l1l2_.transform(Xs=X)
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

        self.ot_l1l2_ = da.SinkhornL1l2Transport(
            reg_e=self.reg_e, reg_cl=self.reg_cl
        ).fit(Xs=X, ys=y, Xt=X_target)
        return self
