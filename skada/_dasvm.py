# Author: Ruben Bueno <ruben.bueno@polytechnique.edu>
# dasvm implementation

import numpy as np
import math

from skada.base import BaseEstimator, BaseAdapter
from skada.utils import check_X_y_domain, source_target_split
from sklearn.base import clone

from sklearn.svm import SVC


class DASVMEstimator(BaseAdapter):
    """
    Simple dasvm estimator
    """
    def __init__(
            self, base_estimator: BaseEstimator = SVC(gamma='auto'),
            k=3, Stop=1_000, **kwargs
            ):

        super().__init__()
        self.base_estimator = base_estimator
        self.Stop = Stop
        self.k = k

    def _find_points_next_step(self, I_list, d, c):
        """
        This function allow us to find the next points to add/discard
        """
        I = np.array([], dtype=int)
        # We should take k points for each of the c classes,
        # depending on the values of d
        for j in range(min(self.k, math.ceil(d.shape[0]/c))):
            mask = np.ones(d.shape[0], dtype=bool)
            mask[I] = False
            I_ = np.unique(np.argmax(d[mask], axis=0))
            # We need to get all those indices to be take into account
            # the fact that the some previous points weren't in the list
            for l in I:
                I_[I_ >= l] += 1
            I = np.concatenate((I, I_))
            I = np.sort(I)
            # we sort I as it is an assumption for I
            # to be sorted for the previous algorithm

        # Again we need to get all those indices to be take into account the
        # fact that the some previous points weren't in the list
        for l in range(len(I_list[-1])):
            if I_list[-1][l]:
                I[I >= l] += 1

        I_list.append(np.copy(I_list[-1]))
        for l in I:
            if I_list[-1][l]:
                raise ValueError(f"problem here: {l}")
            I_list[-1][l] = True

    def fit(self, X, y=None, sample_domain=None):
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
        X, y, sample_domain = check_X_y_domain(X, y, sample_domain)
        Xs, Xt, ys, _ = source_target_split(
            X, y, sample_domain=sample_domain
        )

        n = Xs.shape[0]
        m = Xt.shape[0]
        c = np.unique(ys).shape[0]  # number of classes

        # This is the list of the indices from the
        # points from Xs that have been discarded
        Id = [np.array([False]*n)]
        # This is the list of the indices from the
        # points from Xt that have been added
        Ia = [np.array([False]*m)]

        # In the first step, the SVC is fitted on the source
        X = Xs
        y = ys
        self.Estimators = [self.base_estimator]
        self.Estimators[-1].fit(X, y)

        # We need to look at the decision function to select
        # the labaled data that we discard and the semi-labeled points that we will add

        # look at those that have not been discarded
        decisions_s = self.Estimators[-1].decision_function(X[~Id[-1]])
        if c == 2:
            decisions_s = np.array([-decisions_s, decisions_s]).T
        # look at those that haven't been added
        decisions_ta = self.Estimators[-1].decision_function(Xt[~Ia[-1]])
        if c == 2:
            decisions_ta = np.array([-decisions_ta, decisions_ta]).T

        # We want to take values that are unsure, meaning we want those that have
        # values the closest that we can to c-1 (to 0 when label='binary')
        decisions_ta = -np.abs(decisions_ta-c-1)

        # doing the selection on the labeled data
        self._find_points_next_step(Id, decisions_s, c)
        # doing the selection on the semi-labeled data
        self._find_points_next_step(Ia, decisions_ta, c)

        i = 0
        while (sum(Ia[-1]) < m or sum(Id[-1]) < n) and i < self.Stop:
            i += 1
            X, y = self._get_X_y(Ia, Id, Xs, Xt, ys)

            self.Estimators.append(clone(self.base_estimator))
            self.Estimators[-1].fit(X, y)

            for j in range(len(Ia[-1])):
                if Ia[-1][j]:
                    x = Xt[j]
                    if self.Estimators[-1].predict(
                            [x]) != self.Estimators[-2].predict([x]):
                        if not Ia[-1][j]:
                            raise ValueError("There is a problem here...")
                        Ia[-1][j] = False

            # look at those that have not been discarded
            X_ = Xs[~Id[-1]]
            decisions_s = (
                self.Estimators[-1].decision_function(X_) if X_.shape[0] else X_)
            if c == 2:
                decisions_s = np.array([-decisions_s, decisions_s]).T
            # look at those that haven't been added
            X_ = Xt[~Ia[-1]]
            decisions_ta = (
                self.Estimators[-1].decision_function(X_) if X_.shape[0] else X_)
            if c == 2:
                decisions_ta = np.array([-decisions_ta, decisions_ta]).T

            # We want to take values the estimator is unsure about, meaning that we
            # want those that have values the closest that we can to c-1
            # (to 0 when label='binary', or 4 when is its 'multiclass')
            decisions_ta = -np.abs(decisions_ta-c-1)

            # doing the selection on the labeled data
            self._find_points_next_step(Id, decisions_s, c)
            # doing the selection on the semi-labeled data
            self._find_points_next_step(Ia, decisions_ta, c)

        # On last fit only on semi-labeled data
        X, y = self._get_X_y(Ia, Id, Xs, Xt, ys)

        self.Estimators.append(self.base_estimator)
        self.Estimators[-1].fit(X, y)

        self.base_estimator_ = self.Estimators[-1]

        # it could be interesting to return multiple estimators,
        # or the list Ia and Id (this making them being an object attribute)
        return self

    def _get_X_y(self, Ia, Id, Xs, Xt, ys):
        mask = np.ones(Xs.shape[0], dtype=bool)
        mask[Id[-1]] = False
        X = np.concatenate((Xs[mask], Xt[Ia[-1]]))
        y = np.concatenate((ys[mask], self.Estimators[-1].predict(Xt[Ia[-1]])))
        return X, y

    def adapt(self, X, y=None, sample_domain=None):
        """
        I was thinking of returning Xt and \\hat{y}t
        """
        _, Xt, _, _ = source_target_split(
            X, y, sample_domain=sample_domain
        )
        return Xt, self.base_estimator_.predict(Xt)

    def get_estimator(self, *params) -> BaseEstimator:
        """Returns estimator associated with `params`.

        The set of available estimators and access to them has to be provided
        by specific implementations.
        """
        return self.base_estimator_

    def predict(self, X):
        """ return predicted value by the fitted estimator for `X`
        `predict` method from the estimator we fitted
        """
        return self.base_estimator_.predict(X)

    def decision_function(self, X):
        """ return values of the decision function of the
                fitted estimator for `X`
        `decision_function` method from the estimator we fitted
        """
        return self.base_estimator_.decision_function(X)
