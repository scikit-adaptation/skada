import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from skada._mapping import EntropicOTMappingAdapter


class GradualTreeAdapter:
    def __init__(
        self,
        da_adapter,
        da_clf,
        ot_method=EntropicOTMappingAdapter(),
        threshold=0.5,
        T=10,
    ):
        self.ot_method = ot_method
        self.da_adapter = da_adapter
        self.base_clf = da_clf
        self.threshold = threshold
        self.T = T
        self.clf = []

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
        # Compute the OT mapping between source and target
        self.ot_method.fit(X, y, sample_domain=sample_domain)
        gamma = self.ot_method.ot_transport_
        mask_gamma = self.cut_off_gamma(gamma)
        source = sample_domain >= 0
        X_t = X.copy()
        self.estimator_ = self._build_base_estimator()
        self.estimator_.fit(X[source], y[source])
        for t in range(1, self.T):
            X_t, y_t = self.generate_data_at_t(X, sample_domain, mask_gamma, t)

            self.estimator_.max_iter += 5
            self.estimator_.fit(X_t, y_t)

        return self

    def _build_base_estimator(self):
        return HistGradientBoostingClassifier(
            loss="log_loss",
            max_iter=50,
            warm_start=True,
            # learning_rate=self.learning_rate,
            # max_leaf_nodes=self.max_leaf_nodes,
            # max_depth=self.max_depth,
            # min_samples_leaf=self.min_samples_leaf,
        )

    def cut_off_gamma(self, gamma):
        return gamma > self.threshold

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

        for i, j in np.where(mask_gamma):
            X_t = (self.T - t) * X_source[i] / self.T + t * X_target[j] / self.T

        y_t = self.estimator_.predict(X_t)  # Use the classifier to predict labels
        return X_t, y_t


class GradualTreeEstimator(GradualTreeAdapter):
    def __init__(
        self,
        da_adapter,
        da_clf,
        ot_method=EntropicOTMappingAdapter(),
        threshold=0.5,
        T=10,
    ):
        super().__init__(da_adapter, da_clf, ot_method, threshold, T)

    def predict_proba(self, X):
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
        if not self.clf:
            raise ValueError("The model has not been fitted yet.")

        # Use the last classifier to predict
        return self.estimator_.predict_proba(X) if self.clf else None
