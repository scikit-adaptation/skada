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


class EntropicOTmapping(BaseDataAdaptEstimator):
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

        self.ot_transport_ = clone(da.SinkhornTransport(reg_e=self.reg_e))
        self.ot_transport_.fit(Xs=X, Xt=X_target)
        return self


class ClassRegularizerOTmapping(BaseDataAdaptEstimator):
    """Domain Adaptation Using Optimal Transport.

    Parameters
    ----------
    base_estimator : estimator object
        The base estimator to fit on reweighted data.
    reg_e : float, default=1
        Entropic regularization parameter.
    reg_cl : float, default=0.1
        Class regularization parameter.
    norm : tuple, default="Lpl1"
        Norm use for the regularizer of the class labels.

    Attributes
    ----------
    ot_transport_ : object
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
        reg_cl=0.1,
        norm="Lpl1"
    ):
        super().__init__(base_estimator)
        self.reg_e = reg_e
        self.reg_cl = reg_cl
        self.norm = norm

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
            No weights is this case.
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
        assert self.norm in ["Lpl1", "L1l2"], "Unknown norm"

        if self.norm == "Lpl1":
            self.ot_transport_ = clone(da.SinkhornLpl1Transport(
                reg_e=self.reg_e, reg_cl=self.reg_cl
            ))
        elif self.norm == "L1l2":
            self.ot_transport_ = clone(da.SinkhornL1l2Transport(
                reg_e=self.reg_e, reg_cl=self.reg_cl
            ))

        self.ot_transport_.fit(Xs=X, ys=y, Xt=X_target)
        return self
