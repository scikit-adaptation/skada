# Author: Remi Flamary <remi.flamary@polytechnique.edu>
#
# License: BSD 3-Clause

from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit
from sklearn.utils import check_random_state

from .base import BaseAdapter
from .model_selection import StratifiedDomainShuffleSplit
from .utils import check_X_y_domain


class Subsampler(BaseAdapter):
    """Transformer that subsamples the data.

    This transformer is useful to speed up computations when the data is too
    large. It randomly selects a subset of the data to work with during training
    but does not change the data during testing.

    .. note::
        This transformer should not be used as the last step of a pipeline
        because it returns non standard output.

    Parameters
    ----------
    train_size : int, float
        Number of samples to keep (keep all if data smaller) if integer, or
        proportion of train sample if float 0<= train_size <= 1.
    random_state : int, RandomState instance or None, default=None
        Controls the random resampling of the data.
    """

    def __init__(self, train_size, random_state=None):
        self.train_size = train_size
        self.random_state = random_state

    def _pack_params(self, idx, **params):
        return {
            k: (v[idx] if idx is not None else v)
            for k, v in params.items()
            if v is not None
        }

    def fit_transform(self, X, y=None, *, sample_domain=None, sample_weight=None):
        """Fit and transform the data."""
        X, y, sample_domain2 = check_X_y_domain(X, y, sample_domain)

        self.rng_ = check_random_state(self.random_state)

        if self.train_size >= X.shape[0]:
            return (
                X,
                y,
                self._pack_params(
                    None, sample_domain=sample_domain, sample_weight=sample_weight
                ),
            )

        splitter = ShuffleSplit(
            n_splits=1, train_size=self.train_size, random_state=self.rng_
        )

        idx = next(splitter.split(X))[0]
        X_subsampled = X[idx]
        y_subsampled = y[idx] if y is not None else None
        params = self._pack_params(
            idx, sample_domain=sample_domain2, sample_weight=sample_weight
        )
        return X_subsampled, y_subsampled, params

    def transform(
        self, X, y=None, *, sample_domain=None, sample_weight=None, allow_source=None
    ):
        """Transform the data."""
        return X


class DomainSubsampler(BaseAdapter):
    """Transformer that subsamples the data in a domain stratified way.

    This transformer is useful to speed up computations when the data is too
    large. It randomly selects a subset of the data to work with during training
    but does not change the data during testing.

    .. note::
        This transformer should not be used as the last step of a pipeline
        because it returns non standard output.

    Parameters
    ----------
    train_size : int, float
        Number of samples to keep (keep all if data smaller) if integer, or
        proportion of train sample if float 0<= train_size <= 1.
    random_state : int, RandomState instance or None, default=None
        Controls the random resampling of the data.
    """

    def __init__(self, train_size, random_state=None):
        self.train_size = train_size
        self.random_state = random_state

    def _pack_params(self, idx, **params):
        return {
            k: (v[idx] if idx is not None else v)
            for k, v in params.items()
            if v is not None
        }

    def fit_transform(self, X, y=None, *, sample_domain=None, sample_weight=None):
        """Fit and transform the data."""
        X, y, sample_domain2 = check_X_y_domain(X, y, sample_domain)

        self.rng_ = check_random_state(self.random_state)

        if self.train_size >= X.shape[0]:
            return (
                X,
                y,
                self._pack_params(
                    None, sample_domain=sample_domain, sample_weight=sample_weight
                ),
            )

        splitter = StratifiedShuffleSplit(
            n_splits=1, train_size=self.train_size, random_state=self.rng_
        )
        idx = next(splitter.split(X, sample_domain2))[0]
        X_subsampled = X[idx]
        y_subsampled = y[idx] if y is not None else None
        params = self._pack_params(
            idx, sample_domain=sample_domain2, sample_weight=sample_weight
        )
        return X_subsampled, y_subsampled, params

    def transform(
        self, X, y=None, *, sample_domain=None, sample_weight=None, allow_source=None
    ):
        """Transform the data."""
        return X


class StratifiedDomainSubsampler(BaseAdapter):
    """Transformer that subsamples the data in a domain and label stratified way.
    This transformer is useful to speed up computations when the data is too
    large. It randomly selects a subset of the data to work with during training
    but does not change the data during testing.

    .. note::
        This transformer should not be used as the last step of a pipeline
        because it returns non standard output.

    Parameters
    ----------
    train_size : int, float
        Number of samples to keep (keep all if data smaller) if integer, or
        proportion of train sample if float 0<= train_size <= 1.
    random_state : int, RandomState instance or None, default=None
        Controls the random resampling of the data.
    """

    def __init__(self, train_size, random_state=None):
        self.train_size = train_size
        self.random_state = random_state

    def _pack_params(self, idx, **params):
        return {
            k: (v[idx] if idx is not None else v)
            for k, v in params.items()
            if v is not None
        }

    def fit_transform(self, X, y=None, *, sample_domain=None, sample_weight=None):
        """Fit and transform the data."""
        X, y, sample_domain = check_X_y_domain(X, y, sample_domain)

        self.rng_ = check_random_state(self.random_state)

        if self.train_size >= X.shape[0]:
            return (
                X,
                y,
                self._pack_params(
                    None, sample_domain=sample_domain, sample_weight=sample_weight
                ),
            )

        splitter = StratifiedDomainShuffleSplit(
            n_splits=1, train_size=self.train_size, random_state=self.rng_
        )

        train_idx, _ = next(splitter.split(X, y, sample_domain))
        X_subsampled = X[train_idx]
        y_subsampled = y[train_idx] if y is not None else None
        params = self._pack_params(
            train_idx, sample_domain=sample_domain, sample_weight=sample_weight
        )
        return X_subsampled, y_subsampled, params

    def transform(
        self, X, y=None, *, sample_domain=None, sample_weight=None, allow_source=None
    ):
        """Transform the data."""
        return X
