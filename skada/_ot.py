# Author: Remi Flamary <remi.flamary@polytechnique.edu>
#
# License: BSD 3-Clause


import warnings

import numpy as np
import ot
from sklearn.base import clone
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import SVC
from sklearn.utils.metaestimators import available_if
from sklearn.utils.validation import check_is_fitted

from ._pipeline import make_da_pipeline
from ._utils import Y_Type, _find_y_type
from .base import BaseAdapter, DAEstimator
from .utils import check_X_y_domain, per_domain_split, source_target_split


def get_jdot_class_cost_matrix(Ys, Xt, estimator=None, metric="multinomial"):
    """Cost matrix for joint distribution optimal transport classification problem.

    Parameters
    ----------
    Ys : array-like of shape (n_samples,n_classes)
        Source domain labels one hot encoded.
    Xt : array-like of shape (m_samples, n_features)
        Target domain samples.
    estimator : object
        The already fitted estimator to be used for the classification task. This
        estimator should optimize a classification loss corresponding to the
        given metric and provide compatible predict method (decision_function of
        predict_proba). If None, a constant prediction is used.
    metric : str, default='multinomial'
        The metric to use for the cost matrix. Can be 'multinomial' for cross-entropy
        cost/ multinomial logistic regression or 'hinge' for hinge cost (SVM/SVC).

    Returns
    -------
    M : array-like of shape (n_samples, m_samples)
        The cost matrix.

    References
    ----------
    [10] N. Courty, R. Flamary, A. Habrard, A. Rakotomamonjy, Joint Distribution
         Optimal Transportation for Domain Adaptation, Neural Information Processing
         Systems (NIPS), 2017.

    """
    if estimator is None:
        M = np.ones((Ys.shape[0], Xt.shape[0])) * 10
        return M

    if metric == "multinomial":
        if hasattr(estimator, "predict_log_proba"):
            Yt_pred = estimator.predict_log_proba(Xt)
            M = -np.sum(Ys[:, None, :] * Yt_pred[None, :, :], 2)
        elif hasattr(estimator, "predict_proba"):
            Yt_pred = estimator.predict_proba(Xt)
            M = -np.sum(Ys[:, None, :] * np.log(Yt_pred[None, :, :] + 1e-16), 2)
        else:
            raise ValueError(
                "Estimator must have predict_proba or predict_log_proba"
                " method for cce loss"
            )

    elif metric == "hinge":
        Ys = 2 * Ys - 1  # make Y -1/1 for hinge loss

        if hasattr(estimator, "decision_function"):
            Yt_pred = estimator.decision_function(Xt)
            if len(Yt_pred.shape) == 1:
                Yt_pred = np.repeat(Yt_pred.reshape(-1, 1), 2, axis=1)
            M = np.maximum(0, 1 - Ys[:, None, :] * Yt_pred[None, :, :]).sum(2)
        else:
            raise ValueError(
                "Estimator must have decision_function method for hinge loss"
            )
    else:
        raise ValueError("Unknown metric")

    return M


def get_data_jdot_class(Xt, Yth, labels, thr_weights=1e-6):
    """Get data for the joint distribution optimal transport classification problem.

    This function will repeat sample to allow for training on uncertain labels.

    Parameters
    ----------
    Xt : array-like of shape (m_samples, n_features)
        Target domain samples.
    Yth : array-like of shape (n_samples,n_classes)
        Transported source domain labels one hot encoded.
    labels : array-like of shape (n_classes,)
        The labels of the classification problem.
    thr_weights : float, default=1e-6
        The relative threshold for the weights

    Returns
    -------
    Xh : array-like of shape (n_samples, n_features)
        The transported source domain samples.
    yh : array-like of shape (n_samples,)
        The transported source domain labels.
    wh : array-like of shape (n_samples,)
        The transported source domain weights.

    References
    ----------
    [10] N. Courty, R. Flamary, A. Habrard, A. Rakotomamonjy, Joint Distribution
         Optimal Transportation for Domain Adaptation, Neural Information Processing
         Systems (NIPS), 2017.

    """
    thr = thr_weights * np.max(Yth)

    Xh = np.repeat(Xt, Yth.shape[1], axis=0)
    yh = np.tile(labels, Yth.shape[0])
    wh = Yth.flatten()

    # remove samples with low weights
    ind = wh > thr
    Xh = Xh[ind]
    yh = yh[ind]
    wh = wh[ind]

    return Xh, yh, wh


def get_tgt_loss_jdot_class(Xh, yh, wh, estimator, metric="multinomial"):
    """Get target loss for joint distribution optimal transport classification problem.

    Parameters
    ----------
    Xh : array-like of shape (n_samples, n_features)
        The transported source domain samples.
    yh : array-like of shape (n_samples,)
        The transported source domain labels.
    wh : array-like of shape (n_samples,)
        The transported source domain weights.
    estimator : object
        The already fitted estimator to be used for the classification task. This
        estimator should optimize a classification loss corresponding to the
        given metric and provide compatible predict method (decision_function of
        predict_proba).
    metric : str, default='multinomial'
        The metric to use for the cost matrix. Can be 'multinomial' for cross-entropy
        cost/ multinomial logistic regression or 'hinge' for hinge cost
        (SVM/SVC).

    Returns
    -------
    loss : float
        The target labels losses.

    References
    ----------
    [10] N. Courty, R. Flamary, A. Habrard, A. Rakotomamonjy, Joint Distribution
         Optimal Transportation for Domain Adaptation, Neural Information Processing
         Systems (NIPS), 2017.

    """
    if metric == "multinomial":
        if hasattr(estimator, "predict_log_proba"):
            Yh_pred = estimator.predict_log_proba(Xh)
            loss = -np.sum(yh * Yh_pred, 1).dot(wh)
        elif hasattr(estimator, "predict_proba"):
            Yh_pred = estimator.predict_proba(Xh)
            loss = -np.sum(yh * np.log(Yh_pred + 1e-16), 1).dot(wh)
        else:
            raise ValueError(
                "Estimator must have predict_proba or predict_log_proba method"
                " for multinomial loss"
            )

    elif metric == "hinge":
        yh = 2 * yh - 1  # make Y -1/1 for hinge loss

        if hasattr(estimator, "decision_function"):
            Yh_pred = estimator.decision_function(Xh)
            if len(Yh_pred.shape) == 1:  # handle binary classification
                Yh_pred = np.repeat(Yh_pred.reshape(-1, 1), 2, axis=1)
            loss = np.sum(np.maximum(0, 1 - yh * Yh_pred), 1).dot(wh)
        else:
            raise ValueError(
                "Estimator must have decision_function method for hinge loss"
            )
    else:
        raise ValueError("Unknown metric")

    return loss


def solve_jdot_regression(
    base_estimator,
    Xs,
    ys,
    Xt,
    alpha=0.5,
    ws=None,
    wt=None,
    n_iter_max=100,
    tol=1e-5,
    verbose=False,
    **kwargs,
):
    """Solve the joint distribution optimal transport regression problem [10]

    .. warning::
        This estimator assumes that the loss function optimized by the base
        estimator is the quadratic loss. For instance, the base estimator should
        optimize and L2 loss (e.g. LinearRegression() or Ridge() or even
        MLPRegressor ()). While any estimator providing the necessary prediction
        functions can be used, the convergence of the fixed point is not guaranteed
        and behavior can be unpredictable.

    Parameters
    ----------
    base_estimator : object
        The base estimator to be used for the regression task. This estimator
        should solve a least squares regression problem (regularized or not)
        to correspond to JDOT theoretical regression problem but other
        approaches can be used with the risk that the fixed point might not converge.
    Xs : array-like of shape (n_samples, n_features)
        Source domain samples.
    ys : array-like of shape (n_samples,)
        Source domain labels.
    Xt : array-like of shape (m_samples, n_features)
        Target domain samples.
    alpha : float, default=0.5
        The trade-off parameter between the feature and label loss in OT metric
    ws : array-like of shape (n_samples,)
        Source domain weights (will ne normalized to sum to 1).
    wt : array-like of shape (m_samples,)
        Target domain weights (will ne normalized to sum to 1).
    n_iter_max: int
        Max number of JDOT alternat optimization iterations.
    tol: float>0
        Tolerance for loss variations (OT and mse) stopping iterations.
    verbose: bool
        Print loss along iterations if True.as_integer_ratio
    kwargs : dict
        Additional parameters to be passed to the base estimator.


    Returns
    -------
    estimator : object
        The fitted estimator.
    lst_loss_ot : list
        The list of OT losses at each iteration.
    lst_loss_tgt_labels : list
        The list of target labels losses at each iteration.
    sol : object
        The solution of the OT problem.

    References
    ----------
    [10] N. Courty, R. Flamary, A. Habrard, A. Rakotomamonjy, Joint Distribution
        Optimal Transportation for Domain Adaptation, Neural Information Processing
        Systems (NIPS), 2017.

    """
    estimator = clone(base_estimator)

    # compute feature distance matrix
    Mf = ot.dist(Xs, Xt)
    Mf = Mf / Mf.mean()

    nt = Xt.shape[0]
    if ws is None:
        a = np.ones((len(ys),)) / len(ys)
    else:
        a = ws / ws.sum()
    if wt is None:
        b = np.ones((nt,)) / nt
    else:
        b = wt / wt.sum()
        kwargs["sample_weight"] = wt  # add it as sample_weight for fit

    lst_loss_ot = []
    lst_loss_tgt_labels = []
    y_pred = 0
    Ml = ot.dist(ys.reshape(-1, 1), np.zeros((nt, 1)))

    for i in range(n_iter_max):
        if i > 0:
            # update the cost matrix
            M = (1 - alpha) * Mf + alpha * Ml
        else:
            M = (1 - alpha) * Mf

        # sole OT problem
        sol = ot.solve(M, a, b)

        T = sol.plan
        loss_ot = sol.value

        if i == 0:
            loss_ot += alpha * np.sum(Ml * T)

        lst_loss_ot.append(loss_ot)

        # compute the transported labels
        yth = ys.T.dot(T) / b

        # fit the estimator
        estimator.fit(Xt, yth, **kwargs)
        y_pred = estimator.predict(Xt)

        Ml = ot.dist(ys.reshape(-1, 1), y_pred.reshape(-1, 1))

        # compute the loss
        loss_tgt_labels = np.mean((yth - y_pred) ** 2)
        lst_loss_tgt_labels.append(loss_tgt_labels)

        if verbose:
            print(f"iter={i}, loss_ot={loss_ot}, loss_tgt_labels={loss_tgt_labels}")

        # break on tol OT loss
        if i > 0 and abs(lst_loss_ot[-1] - lst_loss_ot[-2]) < tol:
            break

        # break on tol target loss
        if i > 0 and abs(lst_loss_tgt_labels[-1] - lst_loss_tgt_labels[-2]) < tol:
            break

        # update the cost matrix
        if i == n_iter_max - 1:
            warnings.warn("Maximum number of iterations reached.")

    return estimator, lst_loss_ot, lst_loss_tgt_labels, sol


def solve_jdot_classification(
    base_estimator,
    Xs,
    ys,
    Xt,
    alpha=0.5,
    ws=None,
    wt=None,
    metric="multinomial",
    n_iter_max=100,
    tol=1e-5,
    verbose=False,
    thr_weights=1e-6,
    **kwargs,
):
    """Solve the joint distribution optimal transport classification problem [10]

    .. warning::
        This estimator assumes that the loss function optimized by the base
        estimator is compatible with the given metric. For instance, if the
        metric is 'multinomial', the base estimator should optimize a
        cross-entropy loss (e.g. LogisticRegression with multi_class='multinomial')
        or a hinge loss (e.g. SVC with kernel='linear' and one versus rest) if the
        metric is 'hinge'. While any estimator providing the necessary prediction
        functions can be used, the convergence of the fixed point is not guaranteed
        and behavior can be unpredictable.


    Parameters
    ----------
    base_estimator : object
        The base estimator to be used for the classification task. This estimator
        should solve a classification problem to correspond to JDOT theoretical
        classification problem but other approaches can be used with the risk
        that the fixed point might not converge.
    Xs : array-like of shape (n_samples, n_features)
        Source domain samples.
    ys : array-like of shape (n_samples,)
        Source domain labels.
    Xt : array-like of shape (m_samples, n_features)
        Target domain samples.
    alpha : float, default=0.5
        The trade-off parameter between the feature and label loss in OT metric
    ws : array-like of shape (n_samples,)
        Source domain weights (will ne normalized to sum to 1).
    wt : array-like of shape (m_samples,)
        Target domain weights (will ne normalized to sum to 1).
    metric : str, default='multinomial'
        The metric to use for the cost matrix. Can be 'multinomial' for
        cross-entropy cost/ multinomial logistic regression or 'hinge' for
        hinge cost (SVM/SVC).
    n_iter_max: int
        Max number of JDOT alternate optimization iterations.
    tol: float>0
        Tolerance for loss variations (OT and mse) stopping iterations.
    verbose: bool
        Print loss along iterations if True.as_integer_ratio
    thr_weights : float, default=1e-6
        The relative threshold for the weights
    kwargs : dict
        Additional parameters to be passed to the base estimator.

    Returns
    -------
    estimator : object
        The fitted estimator.
    lst_loss_ot : list
        The list of OT losses at each iteration.
    lst_loss_tgt_labels : list
        The list of target labels losses at each iteration.
    sol : object
        The solution of the OT problem.

    References
    ----------
    [10] N. Courty, R. Flamary, A. Habrard, A. Rakotomamonjy, Joint Distribution
         Optimal Transportation for Domain Adaptation, Neural Information Processing
         Systems (NIPS), 2017.

    """
    estimator = clone(base_estimator)

    # compute feature distance matrix
    Mf = ot.dist(Xs, Xt)
    Mf = Mf / Mf.mean()

    nt = Xt.shape[0]
    if ws is None:
        a = np.ones((len(ys),)) / len(ys)
    else:
        a = ws / ws.sum()
    if wt is None:
        b = np.ones((nt,)) / nt
    else:
        b = wt / wt.sum()

    encoder = OneHotEncoder(sparse_output=False)
    Ys = encoder.fit_transform(ys.reshape(-1, 1))
    labels = encoder.categories_[0]

    lst_loss_ot = []
    lst_loss_tgt_labels = []
    Ml = get_jdot_class_cost_matrix(ys, Xt, None, metric=metric)

    for i in range(n_iter_max):
        if i > 0:
            # update the cost matrix
            M = (1 - alpha) * Mf + alpha * Ml
        else:
            M = (1 - alpha) * Mf

        # sole OT problem
        sol = ot.solve(M, a, b)

        T = sol.plan
        loss_ot = sol.value

        if i == 0:
            loss_ot += alpha * np.sum(Ml * T)

        lst_loss_ot.append(loss_ot)

        # compute the transported labels
        Yth = T.T.dot(Ys) * nt  # not normalized because weights used in fit

        # create reweighted taregt data for classification
        Xh, yh, wh = get_data_jdot_class(Xt, Yth, labels, thr_weights=thr_weights)

        # fit the estimator
        estimator.fit(Xh, yh, sample_weight=wh, **kwargs)

        Ml = get_jdot_class_cost_matrix(Ys, Xt, estimator, metric=metric)

        # compute the losses
        loss_tgt_labels = (
            get_tgt_loss_jdot_class(
                Xh, encoder.transform(yh[:, None]), wh, estimator, metric=metric
            )
            / nt
        )
        lst_loss_tgt_labels.append(loss_tgt_labels)

        if verbose:
            print(f"iter={i}, loss_ot={loss_ot}, loss_tgt_labels={loss_tgt_labels}")

        # break on tol OT loss
        if i > 0 and abs(lst_loss_ot[-1] - lst_loss_ot[-2]) < tol:
            break

        # break on tol target loss
        if i > 0 and abs(lst_loss_tgt_labels[-1] - lst_loss_tgt_labels[-2]) < tol:
            break

        # update the cost matrix
        if i == n_iter_max - 1:
            warnings.warn("Maximum number of iterations reached.")

    return estimator, lst_loss_ot, lst_loss_tgt_labels, sol


class JDOTRegressor(DAEstimator):
    """Joint Distribution Optimal Transport Regressor proposed in [10]

    .. warning::
        This estimator assumes that the loss function optimized by the base
        estimator is the quadratic loss. For instance, the base estimator should
        optimize and L2 loss (e.g. LinearRegression() or Ridge() or even
        MLPRegressor ()). While any estimator providing the necessary prediction
        functions can be used, the convergence of the fixed point is not guaranteed
        and behavior can be unpredictable.


    Parameters
    ----------
    base_estimator : object
        The base estimator to be used for the regression task. This estimator
        should solve a least squares regression problem (regularized or not)
        to correspond to JDOT theoretical regression problem but other
        approaches can be used with the risk that the fixed point might not
        converge. default value is LinearRegression() from scikit-learn.
    alpha : float, default=0.5
        The trade-off parameter between the feature and label loss in OT metric
    n_iter_max: int
        Max number of JDOT alternat optimization iterations.
    tol: float>0
        Tolerance for loss variations (OT and mse) stopping iterations.
    verbose: bool
        Print loss along iterations if True.as_integer_ratio

    Attributes
    ----------
    estimator_ : object
        The fitted estimator.
    lst_loss_ot_ : list
        The list of OT losses at each iteration.
    lst_loss_tgt_labels_ : list
        The list of target labels losses at each iteration.
    sol_ : object
        The solution of the OT problem.

    References
    ----------
    [10] N. Courty, R. Flamary, A. Habrard, A. Rakotomamonjy, Joint Distribution
         Optimal Transportation for Domain Adaptation, Neural Information
         Processing Systems (NIPS), 2017.

    """

    def __init__(
        self,
        base_estimator=None,
        alpha=0.5,
        n_iter_max=100,
        tol=1e-5,
        verbose=False,
        **kwargs,
    ):
        if base_estimator is None:
            base_estimator = LinearRegression()
        else:
            if not hasattr(base_estimator, "fit") or not hasattr(
                base_estimator, "predict"
            ):
                raise ValueError(
                    "base_estimator must be a regressor with" " fit and predict methods"
                )
        self.base_estimator = base_estimator
        self.kwargs = kwargs
        self.alpha = alpha
        self.n_iter_max = n_iter_max
        self.tol = tol
        self.verbose = verbose

    def fit(self, X, y=None, sample_domain=None, *, sample_weight=None):
        """Fit adaptation parameters"""
        Xs, Xt, ys, yt, ws, wt = source_target_split(
            X, y, sample_weight, sample_domain=sample_domain
        )

        res = solve_jdot_regression(
            self.base_estimator,
            Xs,
            ys,
            Xt,
            ws=ws,
            wt=wt,
            alpha=self.alpha,
            n_iter_max=self.n_iter_max,
            tol=self.tol,
            verbose=self.verbose,
            **self.kwargs,
        )

        self.estimator_, self.lst_loss_ot_, self.lst_loss_tgt_labels_, self.sol_ = res

    def predict(self, X, sample_domain=None, *, sample_weight=None):
        """Predict using the model"""
        check_is_fitted(self)
        if sample_domain is not None and np.any(sample_domain >= 0):
            warnings.warn(
                "Source domain detected. Predictor is trained on target"
                "and prediction might be biased."
            )
        return self.estimator_.predict(X)

    def score(self, X, y, sample_domain=None, *, sample_weight=None):
        """Return the coefficient of determination R^2 of the prediction"""
        check_is_fitted(self)
        if sample_domain is not None and np.any(sample_domain >= 0):
            warnings.warn(
                "Source domain detected. Predictor is trained on target"
                "and score might be biased."
            )
        return self.estimator_.score(X, y, sample_weight=sample_weight)


class JDOTClassifier(DAEstimator):
    """Joint Distribution Optimal Transport Classifier proposed in [10]

    .. warning::
        This estimator assumes that the loss function optimized by the base
        estimator is compatible with the given metric. For instance, if the
        metric is 'multinomial', the base estimator should optimize a
        cross-entropy loss (e.g. LogisticRegression with multi_class='multinomial')
        or a hinge loss (e.g. SVC with kernel='linear' and one versus rest) if the
        metric is 'hinge'. While any estimator providing the necessary prediction
        functions can be used, the convergence of the fixed point is not guaranteed
        and behavior can be unpredictable.


    Parameters
    ----------
    base_estimator : object
        The base estimator to be used for the classification task. This
        estimator should solve a classification problem to correspond to JDOT
        theoretical classification problem but other approaches can be used with
        the risk that the fixed point might not converge. default value is
        LogisticRegression() from scikit-learn.
    alpha : float, default=0.5
        The trade-off parameter between the feature and label loss in OT metric
    metric : str, default='multinomial'
        The metric to use for the cost matrix. Can be 'multinomial' for
        cross-entropy cost/ multinomial logistic regression or 'hinge' for hinge
        cost (SVM/SVC).
    n_iter_max: int
        Max number of JDOT alternat optimization iterations.
    tol: float>0
        Tolerance for loss variations (OT and mse) stopping iterations.
    verbose: bool
        Print loss along iterations if True.as_integer_ratio
    thr_weights : float, default=1e-6
        The relative threshold for the weights

    Attributes
    ----------
    estimator_ : object
        The fitted estimator.
    lst_loss_ot_ : list
        The list of OT losses at each iteration.
    lst_loss_tgt_labels_ : list
        The list of target labels losses at each iteration.
    sol_ : object
        The solution of the OT problem.

    References
    ----------
    [10] N. Courty, R. Flamary, A. Habrard, A. Rakotomamonjy, Joint Distribution
         Optimal Transportation for Domain Adaptation, Neural Information
         Processing Systems (NIPS), 2017.

    """

    def __init__(
        self,
        base_estimator=None,
        alpha=0.5,
        metric="multinomial",
        n_iter_max=100,
        tol=1e-5,
        verbose=False,
        thr_weights=1e-6,
        **kwargs,
    ):
        if base_estimator is None:
            base_estimator = LogisticRegression(multi_class="multinomial")
        else:
            if not hasattr(base_estimator, "fit") or not hasattr(
                base_estimator, "predict"
            ):
                raise ValueError(
                    "base_estimator must be a regressor with" " fit and predict methods"
                )
        self.base_estimator = base_estimator
        self.kwargs = kwargs
        self.alpha = alpha
        self.metric = metric
        self.n_iter_max = n_iter_max
        self.tol = tol
        self.verbose = verbose
        self.thr_weights = thr_weights

    def fit(self, X, y=None, sample_domain=None, *, sample_weight=None):
        """Fit adaptation parameters"""
        Xs, Xt, ys, yt, ws, wt = source_target_split(
            X, y, sample_weight, sample_domain=sample_domain
        )

        res = solve_jdot_classification(
            self.base_estimator,
            Xs,
            ys,
            Xt,
            ws=ws,
            wt=wt,
            alpha=self.alpha,
            metric=self.metric,
            n_iter_max=self.n_iter_max,
            tol=self.tol,
            verbose=self.verbose,
            thr_weights=self.thr_weights,
            **self.kwargs,
        )

        self.estimator_, self.lst_loss_ot_, self.lst_loss_tgt_labels_, self.sol_ = res

    def predict(self, X, sample_domain=None, *, sample_weight=None, allow_source=False):
        """Predict using the model"""
        check_is_fitted(self)
        if sample_domain is not None and np.any(sample_domain >= 0):
            warnings.warn(
                "Source domain detected. Predictor is trained on target"
                "and prediction might be biased."
            )
        return self.estimator_.predict(X)

    def _check_proba(self):
        if hasattr(self.base_estimator, "predict_proba"):
            return True
        else:
            raise AttributeError(
                "The base estimator does not have a predict_proba method"
            )

    @available_if(_check_proba)
    def predict_proba(
        self, X, sample_domain=None, *, sample_weight=None, allow_source=False
    ):
        """Predict using the model"""
        check_is_fitted(self)
        if sample_domain is not None and np.any(sample_domain >= 0):
            warnings.warn(
                "Source domain detected. Predictor is trained on target"
                "and prediction might be biased."
            )
        return self.estimator_.predict_proba(X)

    def score(self, X, y, sample_domain=None, *, sample_weight=None, **kwargs):
        """Return the scores of the prediction"""
        check_is_fitted(self)
        if sample_domain is not None and np.any(sample_domain >= 0):
            warnings.warn(
                "Source domain detected. Predictor is trained on target"
                "and score might be biased."
            )
        return self.estimator_.score(X, y, sample_weight=sample_weight)


class OTLabelPropAdapter(BaseAdapter):
    """Label propagation using optimal transport plan.

    This adapter uses the optimal transport plan to propagate labels from
    source to target domain. This was proposed originally in [28] for
    semi-supervised learning and can be used for domain adaptation.

    Parameters
    ----------
    metric : str, default='sqeuclidean'
        The metric to use for the cost matrix. Can be 'sqeuclidean' for
        squared euclidean distance, 'euclidean' for euclidean distance,
    reg : float, default=None
        The entropic  regularization parameter for the optimal transport
        problem. If None, the exact OT is solved, else it is used to weight
        the entropy regularizationof the coupling matrix.
    n_iter_max: int
        Maximum number of iterations for the OT solver.

    Attributes
    ----------
    G_ : array-like of shape (n_samples, m_samples)
        The optimal transport plan.
    Xt_ : array-like of shape (m_samples, n_features)
        The target domain samples.
    yht_ : array-like of shape (m_samples,)
        The transported source domain labels.

    References
    ----------
    [28] Solomon, J., Rustamov, R., Guibas, L., & Butscher, A. (2014, January).
     Wasserstein propagation for semi-supervised learning. In International
     Conference on Machine Learning (pp. 306-314). PMLR.
    """

    __metadata_request__fit = {"sample_weight": True}
    __metadata_request__fit_transform = {"sample_weight": True}

    def __init__(self, metric="sqeuclidean", reg=None, n_iter_max=200):
        super().__init__()
        self.metric = metric
        self.reg = reg
        self.n_iter_max = n_iter_max

    def fit_transform(self, X, y, sample_domain=None, *, sample_weight=None):
        """Fit adaptation parameters"""
        X, y, sample_domain = check_X_y_domain(X, y, sample_domain)
        if sample_weight is not None:
            Xs, Xt, ys, yt, ws, wt = source_target_split(
                X, y, sample_weight, sample_domain=sample_domain
            )
            ws = ws / ws.sum()
            wt = wt / wt.sum()
        else:
            Xs, Xt, ys, yt = source_target_split(X, y, sample_domain=sample_domain)
            ws = ot.unif(Xs.shape[0])
            wt = ot.unif(Xt.shape[0])

        M = ot.dist(Xs, Xt, metric=self.metric)
        G = ot.solve(M, ws, wt, reg=self.reg, max_iter=self.n_iter_max).plan

        self.discrete_ = discrete = _find_y_type(ys) == Y_Type.DISCRETE
        if discrete:
            self.classes_ = classes = np.unique(ys)
            Y = np.zeros((Xs.shape[0], len(classes)))
            for i, c in enumerate(classes):
                Y[:, i] = (ys == c).astype(int)
            yht = G.T.dot(Y)
            self.yht_continuous_ = yht
            yht = np.argmax(yht, axis=1)
            yht = classes[yht]
            yout = -np.ones_like(y)
        else:
            Y = ys
            yht = G.T.dot(Y) / wt
            self.yht_continuous_ = yht
            yout = np.ones_like(y) * np.nan

        self.G_ = G
        self.Xt_ = Xt
        self.yht_ = yht

        # set estimated labels
        yout[sample_domain < 0] = yht

        # return sample weight only if it was provided
        dico = dict()
        if sample_weight is not None:
            dico["sample_weight"] = sample_weight

        return X, yout, dico


def OTLabelProp(base_estimator=None, reg=0, metric="sqeuclidean", n_iter_max=200):
    """Label propagation using optimal transport plan.

    This adapter uses the optimal transport plan to propagate labels from
    source to target domain. This was proposed originally in [28] for
    semi-supervised learning and can be used for domain adaptation.

    Parameters
    ----------
    base_estimator : object
        The base estimator to be used for the classification task. This
        estimator should optimize a classification loss corresponding to the
        given metric and provide compatible predict method (decision_function of
        predict_proba).
    reg : float, default=0
        The entropic  regularization parameter for the optimal transport
        problem. If None, the exact OT is solved, else it is used to weight
        the entropy regularizationof the coupling matrix.
    metric : str, default='sqeuclidean'
        The metric to use for the cost matrix. Can be 'sqeuclidean' for
        squared euclidean distance, 'euclidean' for euclidean distance,
    n_iter_max: int
        Maximum number of iterations for the OT solver.

    Returns
    -------
    adapter : OTLabelPropAdapter
        The optimal transport label propagation adapter.

    References
    ----------
    [28] Solomon, J., Rustamov, R., Guibas, L., & Butscher, A. (2014, January).
     Wasserstein propagation for semi-supervised learning. In International
     Conference on Machine Learning (pp. 306-314). PMLR.

    """
    if base_estimator is None:
        base_estimator = SVC(kernel="rbf").set_fit_request(sample_weight=True)

    return make_da_pipeline(
        OTLabelPropAdapter(reg=reg, metric=metric, n_iter_max=n_iter_max),
        base_estimator,
    )


class JCPOTLabelPropAdapter(BaseAdapter):
    """JCPOT Label Propagation Adapter for multi source target shift

    This adapter uses the optimal transport plan to propagate labels from
    sources to target domain with target shift (change in proportion of
    classes). This was proposed in [31].

    Parameters
    ----------
    metric : str, default='sqeuclidean'
        The metric to use for the cost matrix. Can be 'sqeuclidean' for
        squared euclidean distance, 'euclidean' for euclidean distance,
    reg : float, default=1
        The entropic  regularization parameter for the optimal transport
        problem.
    max_iter : int, default=10
        Maximum number of iterations for the JCPOT solver.
    tol : float, default=1e-9
        Tolerance for loss variations (OT and mse) stopping iterations.
    verbose : bool, default=False
        Print loss along iterations if True.


    References
    ----------
    [31] Redko, Ievgen, Nicolas Courty, Rémi Flamary, and Devis Tuia. "Optimal
         transport for multi-source domain adaptation under target shift." In
         The 22nd International Conference on artificial intelligence and
         statistics, pp. 849-858. PMLR, 2019.

    """

    def __init__(
        self, metric="sqeuclidean", reg=1, max_iter=10, tol=1e-9, verbose=False
    ):
        super().__init__()
        self.metric = metric
        self.reg = reg
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        # we predict target labels in this function so we can't mask them
        self.predicts_target_labels = True

    def fit_transform(self, X, y, sample_domain=None, *, sample_weight=None):
        X, y, sample_domain = check_X_y_domain(X, y, sample_domain)

        sources, targets = per_domain_split(X, y, sample_domain=sample_domain)

        Xs = [X for X, y in sources.values()]
        ys = [y for X, y in sources.values()]

        if len(ys) == 1:
            Xs = Xs * 2
            ys = ys * 2

        Xt = [X for X, y in targets.values()]

        Xt = np.concatenate(Xt, axis=0)

        self.ot_adapter_ = ot.da.JCPOTTransport(
            reg_e=self.reg,
            metric=self.metric,
            max_iter=self.max_iter,
            tol=self.tol,
            log=True,
        )

        self.ot_adapter_.fit(Xs=Xs, ys=ys, Xt=Xt)

        yh = self.ot_adapter_.transform_labels(ys)

        self.yh_continuous_ = yh

        yh = np.argmax(yh, axis=1)

        yout = -np.ones_like(y)
        yout[sample_domain < 0] = yh

        return X, yout, {}


def JCPOTLabelProp(
    base_estimator=None,
    reg=1,
    metric="sqeuclidean",
    max_iter=10,
    tol=1e-9,
    verbose=False,
):
    """JCPOT Label Propagation Adapter for multi source target shift

    This adapter uses the optimal transport plan to propagate labels from
    sources to target domain with target shift (change in proportion of
    classes). This was proposed in [31].

    Parameters
    ----------
    base_estimator : object, default=LinearRegression()
        The base estimator to be used for the classification task. This
        estimator should optimize a classification loss corresponding to the
        given metric and provide compatible predict method (decision_function of
        predict_proba).
    reg : float, default=1
        The entropic  regularization parameter for the optimal transport
        problem.
    metric : str, default='sqeuclidean'
        The metric to use for the cost matrix. Can be 'sqeuclidean' for
        squared euclidean distance, 'euclidean' for euclidean distance,
    max_iter : int, default=10
        Maximum number of iterations for the JCPOT solver.
    tol : float, default=1e-9
        Tolerance for loss variations (OT and mse) stopping iterations.
    verbose : bool, default=False
        Print loss along iterations if True.

    Returns
    -------
    adapter : JCPOTLabelPropAdapter
        The optimal transport label propagation adapter.

    References
    ----------
    [31] Redko, Ievgen, Nicolas Courty, Rémi Flamary, and Devis Tuia. "Optimal
         transport for multi-source domain adaptation under target shift." In
         The 22nd International Conference on artificial intelligence and
         statistics, pp. 849-858. PMLR, 2019.

    """
    if base_estimator is None:
        base_estimator = LogisticRegression()

    return make_da_pipeline(
        JCPOTLabelPropAdapter(
            reg=reg, metric=metric, max_iter=max_iter, tol=tol, verbose=verbose
        ),
        base_estimator,
    )
