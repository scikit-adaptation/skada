# Author: Remi Flamary <remi.flamary@polytechnique.edu>
#
# License: BSD 3-Clause


import numpy as np
from sklearn.base import clone
from .base import DAEstimator
from .utils import source_target_split
import ot
import warnings


def solve_jdot_regression(Xs, ys, Xt, base_estimator, alpha=0.5,
                          n_iter_max=100, tol=1e-5, verbose=False, log=False, **kwargs):
    """Solve the joint distribution optimal transport regression problem

    Parameters
    ----------
    Xs : array-like of shape (n_samples, n_features)
        Source domain samples.

    ys : array-like of shape (n_samples,)
        Source domain labels.

    Xt : array-like of shape (m_samples, n_features)
        Target domain samples.

    base_estimator : object
        The base estimator to be used for the regression task. This estimator
        should solve a least squares regression problem (regularized or not)
        to correspond to JDOT theoretical regression problem but other
        approaches can be used with the risk that the fxed point might not converge.

    alpha : float, default=0.5
        The trade-off parameter between the feature and label loss in OT metric

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
    """

    estimator = clone(base_estimator)

    # compute feature distance matrix
    Mf = ot.dist(Xs, Xt)
    Mf = Mf / Mf.mean()

    nt = Xt.shape[0]

    lst_loss_ot = []
    lst_loss_tgt_labels = []
    y_pred = 0
    Ml = 0

    for i in range(n_iter_max):

        if i > 0:
            # update the cost matrix
            M = (1 - alpha) * Mf + alpha * Ml
        else:
            M = Mf

        # sole OT problem
        sol = ot.solve(M)

        T = sol.plan
        loss_ot = sol.value
        lst_loss_ot.append(loss_ot)

        # compute the transported labels
        yth = nt * ys.T.dot(T)

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
            if log:
                warnings.warn('Maximum number of iterations reached.')

    return estimator, lst_loss_ot, lst_loss_tgt_labels, sol


class JDOTRegressor(DAEstimator):
    """Joint Distribution Optimal Transport Regressor

    """

    def __init__(self, base_estimator, alpha=0.5, n_iter_max=100,
                 tol=1e-5, verbose=False, **kwargs):
        self.base_estimator = base_estimator
        self.kwargs = kwargs
        self.alpha = alpha
        self.n_iter_max = n_iter_max
        self.tol = tol
        self.verbose = verbose

    def fit(self, X, y=None, sample_domain=None, *, sample_weight=None):
        """Fit adaptation parameters"""

        Xs, Xt, ys, yt = source_target_split(X, y, sample_domain=sample_domain)

        res = solve_jdot_regression(Xs, ys, Xt, self.base_estimator,
                                    alpha=self.alpha, n_iter_max=self.n_iter_max,
                                    tol=self.tol, verbose=self.verbose, **self.kwargs)

        self.estimator_, self.lst_loss_ot_, self.lst_loss_tgt_labels_, self.sol_ = res

    def predict(self, X, sample_domain=None, *, sample_weight=None):
        """Predict using the model"""
        return self.estimator_.predict(X)
