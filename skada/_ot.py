# Author: Remi Flamary <remi.flamary@polytechnique.edu>
#
# License: BSD 3-Clause


import numpy as np
from sklearn.base import clone
from sklearn.utils.validation import check_is_fitted
from sklearn.linear_model import LinearRegression
from .base import DAEstimator
from .utils import source_target_split
import ot
import warnings


def solve_jdot_regression(base_estimator, Xs, ys, Xt, alpha=0.5, ws=None, wt=None,
                          n_iter_max=100, tol=1e-5, verbose=False, **kwargs):
    """Solve the joint distribution optimal transport regression problem

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
    [1] N. Courty, R. Flamary, A. Habrard, A. Rakotomamonjy, Joint Distribution
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
    else :
        a = ws / ws.sum()
    if wt is None:
        b = np.ones((nt,)) / nt
    else:
        b = wt / wt.sum()
        kwargs['sample_weight'] = wt  # add it as sample_weight for fit

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
        loss_tgt_labels = np.mean((yth - y_pred)**2)
        lst_loss_tgt_labels.append(loss_tgt_labels)

        if verbose:
            print(f'iter={i}, loss_ot={loss_ot}, loss_tgt_labels={loss_tgt_labels}')

        # break on tol OT loss
        if i > 0 and abs(lst_loss_ot[-1] - lst_loss_ot[-2]) < tol:
            break

        # break on tol target loss
        if i > 0 and abs(lst_loss_tgt_labels[-1] - lst_loss_tgt_labels[-2]) < tol:
            break

        # update the cost matrix
        if i == n_iter_max - 1:
            warnings.warn('Maximum number of iterations reached.')

    return estimator, lst_loss_ot, lst_loss_tgt_labels, sol


class JDOTRegressor(DAEstimator):
    """Joint Distribution Optimal Transport Regressor

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
    [1] N. Courty, R. Flamary, A. Habrard, A. Rakotomamonjy, Joint Distribution
        Optimal Transportation for Domain Adaptation, Neural Information
        Processing Systems (NIPS), 2017.

    """

    def __init__(self, base_estimator=None, alpha=0.5, n_iter_max=100,
                 tol=1e-5, verbose=False, **kwargs):
        if base_estimator is None:
            base_estimator = LinearRegression()
        else:
            if not hasattr(base_estimator, 'fit'):
                raise ValueError('base_estimator must be a regressor with fit method')
            self.base_estimator = base_estimator
        self.kwargs = kwargs
        self.alpha = alpha
        self.n_iter_max = n_iter_max
        self.tol = tol
        self.verbose = verbose

    def fit(self, X, y=None, sample_domain=None, *, sample_weight=None):
        """Fit adaptation parameters"""

        Xs, Xt, ys, yt, ws, wt = source_target_split(
            X, y, sample_weight, sample_domain=sample_domain)

        res = solve_jdot_regression(self.base_estimator, Xs, ys, Xt,
                                    alpha=self.alpha, n_iter_max=self.n_iter_max,
                                    tol=self.tol, verbose=self.verbose, **self.kwargs)

        self.estimator_, self.lst_loss_ot_, self.lst_loss_tgt_labels_, self.sol_ = res

    def predict(self, X, sample_domain=None, *, sample_weight=None):
        """Predict using the model"""
        check_is_fitted(self)
        if sample_domain is not None and np.any(sample_domain >= 0):
            warnings.warn(
                'Source domain detected. Predictor is trained on target'
                'and prediction might be biased.')
        return self.estimator_.predict(X)

    def score(self, X, y, sample_domain=None, *, sample_weight=None):
        """Return the coefficient of determination R^2 of the prediction"""
        check_is_fitted(self)
        if sample_domain is not None and np.any(sample_domain >= 0):
            warnings.warn(
                'Source domain detected. Predictor is trained on target'
                'and score might be biased.')
        return self.estimator_.score(X, y, sample_weight=sample_weight)
