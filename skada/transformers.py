# Author: Remi Flamary <remi.flamary@polytechnique.edu>
#
# License: BSD 3-Clause

from sklearn.utils import check_random_state

from .base import BaseAdapter
from .utils import check_X_y_domain


class SubsampleTransformer(BaseAdapter):
    """Transformer that subsamples the data.

    Parameters
    ----------
    n_subsample : int
        Number of samples to keep (keep all if data smaller).
    random_state : int, RandomState instance or None, default=None
        Controls the random resampling of the data.
    """

    def __init__(self, n_subsample, random_state=None):
        self.n_subsample = n_subsample
        self.random_state = random_state

    def fit_transform(self, X, y=None, *, sample_domain=None, sample_weight=None):
        """Fit and transform the data."""
        X, y, sample_domain2 = check_X_y_domain(X, y, sample_domain)

        if self.n_subsample >= X.shape[0]:
            return X

        rng = check_random_state(self.random_state)
        idx = rng.choice(X.shape[0], self.n_subsample, replace=False)

        X_subsampled = X[idx]

        if y is not None:
            y_subsampled = y[idx]

        dic = {}

        if sample_domain is not None:
            sample_domain2_subsampled = sample_domain2[idx]
            dic["sample_domain"] = sample_domain2_subsampled

        if sample_weight is not None:
            sample_weight_subsampled = sample_weight[idx]
            dic["sample_weight"] = sample_weight_subsampled

        return X_subsampled, y if y is None else y_subsampled, dic

    def transform(self, X, y=None, *, sample_domain=None, sample_weight=None):
        """Transform the data."""
        dic = {}

        if sample_domain is not None:
            dic["sample_domain"] = sample_domain

        if sample_weight is not None:
            dic["sample_weight"] = sample_weight

        return X, y, dic
