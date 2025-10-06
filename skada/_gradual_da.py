# Authors: Julie Alberge <julie.alberge@inria.fr>
#          Felix Lefebvre <felix.lefebvre@inria.fr>
#
# License: BSD 3-Clause

from copy import deepcopy

import numpy as np
from ot import da
from sklearn.neural_network import MLPClassifier

from skada.base import DAEstimator


class GradualEstimator(DAEstimator):
    """Implementation of the GOAT algorithm [38].
    Gradually adapt a classifier from a source domain to a target domain using
    Optimal Transport.

    Parameters
    ----------
    alpha : float, default=1
        Parameter to control the number of samples generated at each step.
        More precisely, at each step, we generate alpha * (n + m) samples,
        where n and m are the number of source and target samples.
        Only used when `advanced_ot_plan_sampling=False`.
    T : int, default=10
        The number of adaptation steps.
    ot_method : ot.da.BaseTransport, default=SinkhornTransport
        The Optimal Transport method to use.
    base_estimator : BaseEstimator, default=None
        The classifier to use. If None, a MLPClassifier with default parameters is
        used. Note that the GOAT algorithm is designed for neural-networks methods.
    advanced_ot_plan_sampling : bool, default=False
        Whether to use the advanced OT plan sampling strategy.
        This strategy consists in sampling at least one point per column and row
        of the OT plan. This ensures a better coverage of the target domain.
        This strategy is not described in the original paper.
    save_estimators : bool, default=False
        Whether to store the intermediate estimators.
    save_intermediate_data : bool, default=False
        Whether to store the intermediate generated data.

    References
    ----------
    .. [38] Y. He, H. Wang, B. Li, H. Zhao
        Gradual Domain Adaptation: Theory and Algorithms in
        Journal of Machine Learning Research, 2024.
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
        alpha=1,
        T=10,
        ot_method=da.SinkhornTransport(
            reg_e=1.0,
            metric="sqeuclidean",
            norm=None,
            max_iter=1000,
            tol=1e-8,
        ),
        base_estimator=None,
        advanced_ot_plan_sampling=False,
        save_estimators=False,
        save_intermediate_data=False,
    ):
        self.alpha = alpha
        self.T = T
        self.ot_method = ot_method
        self.base_estimator = base_estimator
        self.advanced_ot_plan_sampling = advanced_ot_plan_sampling
        self.save_estimators = save_estimators
        self.save_intermediate_data = save_intermediate_data

    def fit(self, X, y=None, *, sample_domain=None):
        """Fit gradual adaptation parameters.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The data.
        y : array-like, shape (n_samples,)
            The labels.
        sample_domain : array-like, shape (n_samples,)
            The domain labels.

        Returns
        -------
        self : object
            Returns self.
        """
        self.estimators_ = []
        source = sample_domain >= 0
        target = sample_domain < 0

        # Compute the OT mapping between source and target
        self.ot_method.fit(Xs=X[source], ys=y[source], Xt=X[target], yt=y[target])
        gamma = self.ot_method.coupling_
        if self.advanced_ot_plan_sampling:
            mask_gamma = self._advanced_cut_off_gamma(gamma)
        else:
            mask_gamma = self._cut_off_gamma(gamma)
        source = sample_domain >= 0
        X_t = X.copy()
        if not self.base_estimator:
            default_params = {
                "hidden_layer_sizes": (100,),
                "activation": "relu",
                "solver": "adam",
                "batch_size": "auto",
                "learning_rate": "constant",
                "learning_rate_init": 0.001,
                "max_iter": 100,
            }
            self.base_estimator = MLPClassifier(**default_params)
        self.base_estimator.fit(X[source], y[source])
        if self.save_estimators:
            self.estimators_.append(deepcopy(self.base_estimator))
        if self.save_intermediate_data:
            self.intermediate_data_ = []
        for t in range(1, self.T + 1):
            X_t, y_t = self.generate_data_at_t(X, sample_domain, mask_gamma, t)

            self.base_estimator.max_iter += 100
            self.base_estimator.fit(X_t, y_t)
            if self.save_estimators:
                self.estimators_.append(deepcopy(self.base_estimator))
            if self.save_intermediate_data:
                self.intermediate_data_.append((X_t, y_t))
        return self

    def _cut_off_gamma(self, gamma):
        """Cut off the OT mapping to keep only the largest values."""
        n, m = gamma.shape
        # Get the self.alpha * (n + m) largest coefficients of gamma
        self.max_index = int(min(self.alpha * (n + m), gamma.size - 1))

        threshold = np.sort(gamma.flatten())[-self.max_index - 1]
        return gamma >= threshold

    def _advanced_cut_off_gamma(self, gamma):
        """Cut off the OT mapping to keep at least one value per row and column."""
        n, m = gamma.shape
        # Keep only the largest element of each row and column
        row_max = np.zeros((n, m), dtype=bool)
        col_max = np.zeros((n, m), dtype=bool)
        for i in range(n):
            row_max[i, np.argmax(gamma[i])] = True
        for j in range(m):
            col_max[np.argmax(gamma[:, j]), j] = True
        combined_mask = row_max | col_max
        self.max_index = int(combined_mask.sum()) - 1
        return combined_mask

    def generate_data_at_t(self, X, sample_domain, mask_gamma, t):
        """Generate data at a given time step t.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The data.
        sample_domain : array-like, shape (n_samples,)
            The domain labels.
        mask_gamma : array-like, shape (n_samples, n_samples)
            The pruned OT mapping.
        t : int
            The current step.

        Returns
        -------
        X_t : array-like, shape (n_samples_k, n_features)
            The generated data at step k.
        y_t : array-like, shape (n_samples_k,)
            The labels for the generated data.
        """
        source = sample_domain >= 0
        target = sample_domain < 0
        X_source = X[source]
        X_target = X[target]
        X_t = np.zeros((self.max_index + 1, X.shape[1]))
        for idx, (i, j) in enumerate(np.argwhere(mask_gamma)):
            X_t[idx] = (self.T - t) * X_source[i] / self.T + t * X_target[j] / self.T

        y_t = self.base_estimator.predict(X_t)  # Use the classifier to predict labels
        return X_t, y_t

    def get_intermediate_estimators(self):
        """Return the intermediate estimators.

        Returns
        -------
        estimators_ : list
            The list of intermediate estimators.
        """
        return self.estimators_

    def predict(self, X, **kwargs):
        """Predict labels for the input data.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input data.

        Returns
        -------
        y_pred : array-like, shape (n_samples,)
            The predicted labels.
        """
        if not hasattr(self, "base_estimator"):
            raise ValueError("The model has not been fitted yet.")

        # Use the last classifier to predict
        return self.base_estimator.predict(X, **kwargs)

    def predict_proba(self, X, **kwargs):
        """Predict class probabilities for the input data.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input data.

        Returns
        -------
        y_proba : array-like, shape (n_samples, n_classes)
            The predicted class probabilities.
        """
        if not hasattr(self, "base_estimator"):
            raise ValueError("The model has not been fitted yet.")
        if not hasattr(self.base_estimator, "predict_proba"):
            raise ValueError("The underlying estimator does not support predict_proba.")

        return self.base_estimator.predict_proba(X, **kwargs)

    def predict_log_proba(self, X, **kwargs):
        """Predict class log-probabilities for the input data.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input data.

        Returns
        -------
        y_log_proba : array-like, shape (n_samples, n_classes)
            The predicted class log-probabilities.
        """
        if not hasattr(self, "base_estimator"):
            raise ValueError("The model has not been fitted yet.")
        if not hasattr(self.base_estimator, "predict_log_proba"):
            raise ValueError(
                "The underlying estimator does not support predict_log_proba."
            )

        return self.base_estimator.predict_log_proba(X, **kwargs)

    def score(self, X, y):
        """Compute the accuracy of the model on the given data.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input data.
        y : array-like, shape (n_samples,)
            The true labels.

        Returns
        -------
        score : float
            The accuracy of the model.
        """
        if not hasattr(self, "base_estimator"):
            raise ValueError("The model has not been fitted yet.")

        else:
            return (self.base_estimator.predict(X) == y).sum() / len(y)
