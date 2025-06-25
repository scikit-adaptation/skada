# Authors: Julie Alberge <julie.alberge@inria.fr>
#          Felix Lefebvre <felix.lefebvre@inria.fr>
#
# License: BSD 3-Clause

import numpy as np
from ot import da
from sklearn.ensemble import HistGradientBoostingClassifier

from skada.base import BaseAdapter, DAEstimator


class GradualTreeAdapter(BaseAdapter):
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
    ):
        self.alpha = alpha
        self.T = T
        self.clf = []
        self.ot_method = ot_method

    def fit(self, X, y=None, *, sample_domain=None):
        """Fit gradual adaptation parameters.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The source data.
        y : array-like, shape (n_samples,)
            The source labels.
        T : int, default=10
            The number of adaptation steps.
        sample_domain : array-like, shape (n_samples,)
            The domain labels (same as sample_domain).

        Returns
        -------
        self : object
            Returns self.
        """
        source = sample_domain >= 0
        target = sample_domain < 0

        # Compute the OT mapping between source and target
        self.ot_method.fit(Xs=X[source], ys=y[source], Xt=X[target], yt=y[target])
        gamma = self.ot_method.coupling_
        mask_gamma = self.cut_off_gamma(gamma)
        source = sample_domain >= 0
        X_t = X.copy()
        self.estimator_ = self._build_base_estimator()
        self.estimator_.fit(X[source], y[source])
        for t in range(1, self.T):
            X_t, y_t = self.generate_data_at_t(X, sample_domain, mask_gamma, t)

            self.estimator_.max_iter += 10
            self.estimator_.fit(X_t, y_t)
            import ipdb

            ipdb.set_trace()
        return self

    def _build_base_estimator(self):
        return HistGradientBoostingClassifier(
            loss="log_loss",
            max_iter=30,
            warm_start=True,
            learning_rate=0.3,
        )

    def cut_off_gamma(self, gamma):
        n, m = gamma.shape
        # Get the self.alpha * (n + m) largest coefficients of gamma
        self.max_index = int(min(self.alpha * (n + m), gamma.size - 1))

        threshold = np.sort(gamma.flatten())[-self.max_index - 1]
        return gamma >= threshold

    def generate_data_at_t(self, X, sample_domain, mask_gamma, t):
        """Generate data at a given time step t.

        Parameters
        ----------
        t : int
            The current step.
        X : array-like, shape (n_samples, n_features)
            The source data.
        y : array-like, shape (n_samples,)
            The source labels.
        gamma : array-like, shape (n_samples, n_samples)
            The OT mapping.

        Returns
        -------
        X_t : array-like, shape (n_samples_k, n_features)
            The generated data at step k.
        y_t : array-like, shape (n_samples_k,)
            The labels for the generated data.
        sample_domain_t : array-like, shape (n_samples_k,)
            The domain labels for the generated data.
        """
        source = sample_domain >= 0
        target = sample_domain < 0
        X_source = X[source]
        X_target = X[target]
        X_t = np.zeros((self.max_index + 1, X.shape[1]))
        for idx, (i, j) in enumerate(np.argwhere(mask_gamma)):
            X_t[idx] = (self.T - t) * X_source[i] / self.T + t * X_target[j] / self.T

        # import ipdb; ipdb.set_trace()
        y_t = self.estimator_.predict(X_t)  # Use the classifier to predict labels
        return X_t, y_t


class GradualTreeEstimator(GradualTreeAdapter, DAEstimator):
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
    ):
        super().__init__(alpha=alpha, T=T, ot_method=ot_method)

    def predict(self, X):
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
        if not self.estimator_:
            raise ValueError("The model has not been fitted yet.")

        # Use the last classifier to predict
        return self.estimator_.predict(X) if self.estimator_ else None

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
        if not self.estimator_:
            raise ValueError("The model has not been fitted yet.")

        else:
            return (self.estimator_.predict(X) == y).sum() / len(y)
