# Author: Ruben Bueno <ruben.bueno@polytechnique.edu>
# dasvm implementation

import numpy as np
import math

from skada.base import BaseEstimator
from skada.utils import check_X_y_domain, source_target_split
from sklearn.base import clone

from sklearn.svm import SVC


class DASVMEstimator(BaseEstimator):
    """
    Simple dasvm estimator
    """

    __metadata_request__fit = {'sample_domain': True}
    __metadata_request__transform = {'sample_domain': True, 'allow_source': True}

    def __init__(
            self, base_estimator=None,
            k=3, Stop=1_000, **kwargs
            ):
        """Fit adaptation parameters.

        Parameters
        ----------
        base_estimator : BaseEstimator
            The estimator that will be used in the algorithm,
            It is a SVC by default, but can use any classifier
            equiped with a `decision_function` method
        k: int>0
            The number of points per classes that will be discarded/added
            at each steps of the algorithm
        Stop : int
            The maximal number of iteration of the algorithm when using `fit`
        """

        super().__init__()
        if base_estimator is None:
            self.base_estimator = SVC(gamma='auto')
        else:
            self.base_estimator = base_estimator
        self.Stop = Stop
        self.k = k

    def _find_points_next_step(self, indices_list, d):
        """
        This function allow us to find the next points to add/discard,
        It is an inplace method, changing indices_list
        """
        # We should take k points for each of the c classes,
        # depending on the values of d
        for j in range(min(self.k, math.ceil(sum(~indices_list)/self.c))):
            I = np.unique(np.argmax(d[~indices_list], axis=0))
            # We need to get all those indices to be take into account
            # the fact that the some previous points weren't in the list
            for l in range(len(indices_list)):
                if indices_list[l]:
                    I[I >= l] += 1

            # We finally only need to change the list
            for l in I:
                # indices_list[l] is False at that point
                indices_list[l] = True

    def _get_X_y(
            self, new_estimator, index_target_added, index_source_deleted, Xs, Xt, ys
            ):
        # Allow to get the X and y arrays at a state of the algorithm
        # We take the source datapoints that have not been
        # deleted, and the target points
        # that have been added
        X = np.concatenate((Xs[~index_source_deleted], Xt[index_target_added]))
        y = np.concatenate((ys[~index_source_deleted], new_estimator.predict(
            Xt[index_target_added])))
        return X, y

    def get_decision(self, new_estimator, X, indices_list):
        # We look at the points that have either not been discarded or not been added
        # We are assuming that the `decision_function` from the base_estimator is:
        # giving c values between -1 and c-1, not having the same
        # integer parts, for the same datapoint x (the same as for SVC).
        # When c == 2, we assume we only get one value (as for SVC)
        # We take c to be the number of classes on which `new_estimator`
        # has been fitted
        if sum(~indices_list) > 0:
            df = new_estimator.decision_function(X[~indices_list])
            # df.ndim allows us to know if we are in the
            # `binary` case or the `multiclass` one
            if df.ndim == 1:
                decisions = np.ones(X.shape[0])
                decisions[~indices_list] = df
                decisions = np.array([-decisions, decisions]).T
            else:
                decisions = np.ones((X.shape[0], df.shape[1]))
                decisions[~indices_list] = df
        else:
            decisions = np.ones(X.shape[0])
        return decisions

    def fit(self, X, y=None, sample_domain=None):
        """Fit adaptation parameters.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The source and training data.
        y : array-like, shape (n_samples,)
            The source labels, followed by some labels that won't be looked at.
        sample_domain : array-like, shape (n_samples,)
            The domain labels.

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
        index_source_deleted = np.array([False]*n)
        # This is the list of the indices from the
        # points from Xt that have been added
        index_target_added = np.array([False]*m)

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
        self._find_points_next_step(
            index_source_deleted, decisions_s)
        # doing the selection on the semi-labeled data
        self._find_points_next_step(
            index_target_added, decisions_ta)

        i = 0
        while (
                sum(index_target_added) < m or sum(index_source_deleted) < n
                ) and i < self.Stop:
            i += 1
            old_estimator = new_estimator
            X, y = self._get_X_y(
                new_estimator, index_target_added, index_source_deleted, Xs, Xt, ys)

            new_estimator = clone(self.base_estimator)
            new_estimator.fit(X, y)

            for j in range(len(index_target_added)):
                if index_target_added[j]:
                    x = Xt[j]
                    if new_estimator.predict(
                            [x]) != old_estimator.predict([x]):
                        # index_target_added[j] should be True
                        index_target_added[j] = False

            # look at those that have not been discarded
            decisions_s = self.get_decision(new_estimator, Xs, index_source_deleted)
            # look at those that haven't been added
            decisions_ta = self.get_decision(new_estimator, Xt, index_target_added)

            # We want to take values the estimator is unsure about, meaning that we
            # want those that have values the closest that we can to c-1
            # (to 0 when label='binary', or 4 when is its 'multiclass')
            decisions_ta = -np.abs(decisions_ta-self.c-1)

            # doing the selection on the labeled data
            self._find_points_next_step(index_source_deleted, decisions_s)
            # doing the selection on the semi-labeled data
            self._find_points_next_step(index_target_added, decisions_ta)

        old_estimator = new_estimator
        # On last fit only on semi-labeled data
        X, y = Xt, old_estimator.predict(Xt)

        new_estimator = clone(self.base_estimator)
        new_estimator.fit(X, y)

        self.base_estimator_ = new_estimator

        # or the list of index_target_added and index_source_deleted (this making
        # them being an attribute if the object)
        return self

    def predict(self, X):
        """ return predicted value by the fitted estimator for `X`
        `predict` method from the estimator we fitted
        """
        return self.base_estimator_.predict(X)

    def decision_function(self, X):
        """ return values of the decision function of the
                fitted estimator for `X`
        `decision_function` method from the base_estimator_ we fitted
        """
        return self.base_estimator_.decision_function(X)
