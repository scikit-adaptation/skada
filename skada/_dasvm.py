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
            self, base_estimator=None,
            k=3, Stop=1_000, **kwargs
            ):

        super().__init__()
        if base_estimator == None:
            self.base_estimator = SVC(gamma='auto')
        else:
            self.base_estimator = base_estimator
        self.Stop = Stop
        self.k = k

    def _find_points_next_step(self, I_list, d):
        """
        This function allow us to find the next points to add/discard
        """
        # We should take k points for each of the c classes,
        # depending on the values of d
        for j in range(min(self.k, math.ceil(sum(~I_list)/self.c))):
            I = np.unique(np.argmax(d[~I_list], axis=0))
            # We need to get all those indices to be take into account
            # the fact that the some previous points weren't in the list
            for l in range(len(I_list)):
                if I_list[l]:
                    I[I >= l] += 1

            # We finally only need to change the list
            for l in I:
                if I_list[l]:
                    raise ValueError(f"problem here: {l}")
                I_list[l] = True

        # it is an inplace method, but it is more understandable when 
        # returning the list
        return I_list

    def _get_X_y(self, new_estimator, Index_target_added, Index_source_deleted, Xs, Xt, ys):
        mask = np.ones(Xs.shape[0], dtype=bool)
        mask[Index_source_deleted] = False
        X = np.concatenate((Xs[mask], Xt[Index_target_added]))
        y = np.concatenate((ys[mask], new_estimator.predict(Xt[Index_target_added])))
        return X, y

    def get_decision(self, new_estimator, X, I_list):
        # We look at the points that have either not been discarded or not been added
        decisions = np.ones(X.shape[0])
        if sum(~I_list)>0:
            decisions[~I_list] = new_estimator.predict(X[~I_list])
            if self.c == 2:
                decisions = np.array([-decisions, decisions]).T
        return decisions

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
        self.c = np.unique(ys).shape[0]  # number of classes

        # This is the list of the indices from the
        # points from Xs that have been discarded
        Index_source_deleted = np.array([False]*n)
        # This is the list of the indices from the
        # points from Xt that have been added
        Index_target_added = np.array([False]*m)

        # In the first step, the SVC is fitted on the source
        X = Xs
        y = ys
        new_estimator = self.base_estimator
        new_estimator.fit(Xs, ys)

        # We need to look at the decision function to select
        # the labaled data that we discard and the semi-labeled points that we will add

        # look at those that have not been discarded
        decisions_s = new_estimator.decision_function(Xs)
        if self.c == 2:
            decisions_s = np.array([-decisions_s, decisions_s]).T
        # look at those that haven't been added
        decisions_ta = new_estimator.decision_function(Xt)
        if self.c == 2:
            decisions_ta = np.array([-decisions_ta, decisions_ta]).T

        # We want to take values that are unsure, meaning we want those that have
        # values the closest that we can to c-1 (to 0 when label='binary')
        decisions_ta = -np.abs(decisions_ta-self.c-1)


        # doing the selection on the labeled data
        Index_source_deleted = self._find_points_next_step(Index_source_deleted, decisions_s)
        # doing the selection on the semi-labeled data
        Index_target_added = self._find_points_next_step(Index_target_added, decisions_ta)

        i = 0
        while (sum(Index_target_added) < m or sum(Index_source_deleted) < n) and i < self.Stop:
            i += 1
            old_estimator = new_estimator
            X, y = self._get_X_y(new_estimator, Index_target_added, Index_source_deleted, Xs, Xt, ys)

            new_estimator = clone(self.base_estimator)
            new_estimator.fit(X, y)

            for j in range(len(Index_target_added)):
                if Index_target_added[j]:
                    x = Xt[j]
                    if new_estimator.predict(
                            [x]) != old_estimator.predict([x]):
                        if not Index_target_added[j]:
                            raise ValueError("There is a problem here...")
                        Index_target_added[j] = False

            # look at those that have not been discarded
            decisions_s = self.get_decision(new_estimator, Xs, Index_source_deleted)
            # look at those that haven't been added
            decisions_ta = self.get_decision(new_estimator, Xt, Index_target_added)

            # We want to take values the estimator is unsure about, meaning that we
            # want those that have values the closest that we can to c-1
            # (to 0 when label='binary', or 4 when is its 'multiclass')
            decisions_ta = -np.abs(decisions_ta-self.c-1)

            # doing the selection on the labeled data
            self._find_points_next_step(Index_source_deleted, decisions_s)
            # doing the selection on the semi-labeled data
            self._find_points_next_step(Index_target_added, decisions_ta)

        old_estimator = new_estimator
        # On last fit only on semi-labeled data
        X, y = Xt, old_estimator.predict(Xt)

        new_estimator = clone(self.base_estimator)
        new_estimator.fit(X, y)

        self.base_estimator_ = new_estimator

        # it could be interesting to return multiple estimators,
        # or the list Index_target_added and Index_source_deleted (this making them being an object attribute)
        return self

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
