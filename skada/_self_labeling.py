"""
DASVM estimator
===============

The DASVM method comes from [11].
.. [11] Bruzzone, L., & Marconcini, M. 'Domain adaptation problems: A DASVM
        classification technique and a circular validation strategy.'
        IEEE transactions on pattern analysis and machine intelligence, (2009).
"""
# Author: Ruben Bueno <ruben.bueno@polytechnique.edu>
#         Yanis Lalou <yanis.lalou@polytechnique.edu>
#
# License: BSD 3-Clause

import math
import warnings

import numpy as np
from sklearn.base import clone
from sklearn.svm import SVC
from sklearn.utils.metaestimators import available_if
from sklearn.utils.validation import check_is_fitted

from skada.utils import check_X_y_domain, source_target_split

from .base import DAEstimator


class DASVMClassifier(DAEstimator):
    """DASVM Estimator:

    Parameters
    ----------
    base_estimator : BaseEstimator
        The estimator that will be used in the algorithm,
        It is a SVC by default, but can use any classifier
        equipped with a `decision_function` method
    k: int>0
        The number of points per classes that will be discarded/added
        at each steps of the algorithm
    max_iter : int
        The maximal number of iteration of the algorithm when using `fit`
    save_estimators : Bool
        True if this object should remembers all the fitted estimators
    save_indices : Bool
        True if this object should remembers all the values of
            `index_source_deleted` and `index_target_added`

    References
    ----------
    .. [11] Bruzzone, L., & Marconcini, M. 'Domain adaptation problems: A DASVM
            classification technique and a circular validation strategy.'
            IEEE transactions on pattern analysis and machine intelligence, (2009).
    """

    __metadata_request__fit = {"sample_domain": True}
    __metadata_request__partial_fit = {"sample_domain": False}
    __metadata_request__predict = {"sample_domain": False, "allow_source": False}
    __metadata_request__predict_proba = {"sample_domain": False, "allow_source": False}
    __metadata_request__predict_log_proba = {
        "sample_domain": False,
        "allow_source": False,
    }
    __metadata_request__score = {"sample_domain": False, "allow_source": False}
    __metadata_request__decision_function = {
        "sample_domain": False,
        "allow_source": False,
    }

    def __init__(
        self,
        base_estimator=None,
        k=3,
        max_iter=1_000,
        save_estimators=False,
        save_indices=False,
        **kwargs,
    ):
        super().__init__()
        if base_estimator is None:
            self.base_estimator = SVC(probability=True)
        else:
            self.base_estimator = base_estimator
        self.max_iter = max_iter
        self.save_estimators = save_estimators
        self.save_indices = save_indices
        self.k = k

    def _find_points_next_step(self, indices_list, d, cond_array):
        """This function allow us to find the next points to add/discard.

        It is an inplace method, changing indices_list
        """
        # We should take k points for each of the c classes,
        # depending on the values of d
        condition = np.logical_and(~indices_list, cond_array)
        for _ in range(min(self.k, math.ceil(sum(condition) / self.n_class))):
            idx = np.unique(np.argmax(d[condition], axis=0))
            # We need to get all those indices to be take into account
            # the fact that the some previous points weren't in the list
            for ll in range(condition.shape[0]):
                if ~condition[ll]:
                    idx[idx >= ll] += 1

            # We finally only need to change the list
            for ll in idx:
                # indices_list[l] is False at that point
                indices_list[ll] = True

    def _get_X_y(
        self, new_estimator, index_target_added, index_source_deleted, Xs, Xt, ys
    ):
        """
        Allow to get the X and y arrays at a state of the algorithm
        We take the source datapoints that have not been
        deleted, and the target points
        that have been added
        """
        X = np.concatenate((Xs[~index_source_deleted], Xt[index_target_added]))
        semi_labels = new_estimator.predict(Xt[index_target_added])
        y = np.concatenate((ys[~index_source_deleted], semi_labels))
        return X, y

    def _get_decision(self, new_estimator, X, indices_list):
        """Look at the points that have either not been discarded or not been added."""
        if sum(~indices_list) > 0:
            df = new_estimator.decision_function(X[~indices_list])
            # df.ndim allows us to know if we are in the
            # `binary` case or the `multiclass` one
            decisions = np.ones(X.shape[0])
            decisions[~indices_list] = df
            decisions = np.array([-decisions + 1, decisions + 1]).T
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
        Xs, Xt, ys, _ = source_target_split(X, y, sample_domain=sample_domain)

        n = Xs.shape[0]
        m = Xt.shape[0]
        self.n_class = 2  # number of classes
        self.estimators = []
        self.indices_source_deleted = []
        self.indices_target_added = []

        index_source_deleted = np.array([False] * n)
        index_target_added = np.array([False] * m)
        if self.save_indices:
            self.indices_source_deleted.append(np.copy(index_source_deleted))
            self.indices_target_added.append(np.copy(index_target_added))

        X_train = Xs
        y_train = ys
        new_estimator = self.base_estimator
        new_estimator.fit(X_train, y_train)

        if self.save_estimators:
            self.estimators.append(new_estimator)

        decisions_source = new_estimator.decision_function(Xs)
        if self.n_class == 2:
            decisions_source = np.array([-decisions_source, decisions_source]).T
        decisions_target = new_estimator.decision_function(Xt)
        if self.n_class == 2:
            decisions_target = np.array([-decisions_target, decisions_target]).T

        decisions_target_ = -np.abs(decisions_target - self.n_class + 1)

        self._find_points_next_step(
            index_source_deleted,
            decisions_source,
            np.ones(index_source_deleted.shape[0], dtype=bool),
        )
        in_margin_target = (
            np.sum(
                np.logical_and(
                    decisions_target < self.n_class - 1,
                    decisions_target > self.n_class - 2,
                ),
                axis=1,
            )
            > 0
        )
        self._find_points_next_step(
            index_target_added, decisions_target_, in_margin_target
        )

        if self.save_indices:
            self.indices_source_deleted.append(np.copy(index_source_deleted))
            self.indices_target_added.append(np.copy(index_target_added))

        i = 0
        for i in range(1, self.max_iter):
            if sum(in_margin_target) == 0:
                break

            old_estimator = new_estimator
            X_train, y_train = self._get_X_y(
                new_estimator, index_target_added, index_source_deleted, Xs, Xt, ys
            )

            new_estimator = clone(self.base_estimator)
            new_estimator.fit(X_train, y_train)
            if self.save_estimators:
                self.estimators.append(new_estimator)

            for j in range(len(index_target_added)):
                if index_target_added[j]:
                    x = Xt[j]
                    if new_estimator.predict([x]) != old_estimator.predict([x]):
                        # index_target_added[j] should be True
                        index_target_added[j] = False

            decisions_source = self._get_decision(
                new_estimator, Xs, index_source_deleted
            )

            decisions_target = self._get_decision(new_estimator, Xt, index_target_added)

            if decisions_target.ndim > 1:
                decisions_target_ = -np.abs(decisions_target - 1)

            self._find_points_next_step(
                index_source_deleted,
                decisions_source,
                np.ones(index_source_deleted.shape[0], dtype=bool),
            )
            in_margin_target = (
                np.sum(
                    np.logical_and(decisions_target < 1, decisions_target > 0), axis=1
                )
                > 0
            )
            self._find_points_next_step(
                index_target_added, decisions_target, in_margin_target
            )

            if self.save_indices:
                self.indices_source_deleted.append(np.copy(index_source_deleted))
                self.indices_target_added.append(np.copy(index_target_added))

        old_estimator = new_estimator
        X_train, y_train = Xt, old_estimator.predict(Xt)

        new_estimator = clone(self.base_estimator)
        new_estimator.fit(X_train, y_train)
        if self.save_estimators:
            self.estimators.append(new_estimator)

        if self.save_indices:
            self.indices_source_deleted.append(np.ones(n, dtype=bool))
            self.indices_target_added.append(np.ones(m, dtype=bool))

        self.base_estimator_ = new_estimator

        return self

    def predict(self, X, **kwargs):
        """Return predicted value by the fitted estimator for `X`
        `predict` method from the estimator we fitted
        """
        return self.base_estimator_.predict(X, **kwargs)

    def _check_proba(self):
        if hasattr(self.base_estimator, "predict_proba"):
            return True
        else:
            raise AttributeError(
                "The base estimator does not have a predict_proba method"
            )

    @available_if(_check_proba)
    def predict_proba(self, X, **kwargs):
        """Return predicted probabilities by the fitted estimator for `X`
        `predict_proba` method from the estimator we fitted
        """
        check_is_fitted(self)
        return self.base_estimator.predict_proba(X, **kwargs)

    def decision_function(self, X):
        """Return values of the decision function of the
                fitted estimator for `X`
        `decision_function` method from the base_estimator_ we fitted
        """
        return self.base_estimator_.decision_function(X)

    def score(self, X, y, sample_domain=None, *, sample_weight=None, **kwargs):
        """Return the scores of the prediction"""
        check_is_fitted(self)
        if sample_domain is not None and np.any(sample_domain >= 0):
            warnings.warn(
                "Source domain detected. Predictor is trained on target"
                "and score might be biased."
            )
        return self.base_estimator_.score(X, y, sample_weight=sample_weight, **kwargs)
