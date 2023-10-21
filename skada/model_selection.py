from abc import ABCMeta, abstractmethod

from sklearn.model_selection._split import (
    _build_repr,
    _num_samples,
    _validate_shuffle_split
)
from sklearn.utils import check_random_state, indexable
from sklearn.utils.metadata_routing import _MetadataRequester

from ._utils import check_X_domain


class BaseDomainAwareShuffleSplit(_MetadataRequester, metaclass=ABCMeta):
    """Base class for ShuffleSplit and StratifiedShuffleSplit"""

    __metadata_request__split = {"groups": "sample_domain"}

    def __init__(
        self, n_splits=10, *, test_size=None, train_size=None, random_state=None
    ):
        self.n_splits = n_splits
        self.test_size = test_size
        self.train_size = train_size
        self.random_state = random_state
        self._default_test_size = 0.1

    def split(self, X, y=None, groups=None):
        """Generate indices to split data into training and test set.

        Domain-aware shuffle split re-uses 'groups' parameter to pass
        information in 'sample_domain' which is done to by-pass current
        limitation of sklearn API for metadata routing w.r.t. split calls
        (for example, `GridSearchCV` supports only 'groups' being passed
        to a downstream caller).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where `n_samples` is the number of samples
            and `n_features` is the number of features.

        y : array-like of shape (n_samples,)
            The target variable for supervised learning problems.

        groups : array-like of shape (n_samples,), default=None
            Group labels for the samples used while splitting the dataset into
            train/test set.

        Yields
        ------
        train : ndarray
            The training set indices for that split.

        test : ndarray
            The testing set indices for that split.

        Notes
        -----
        Randomized CV splitters may return different results for each call of
        split. You can make the results identical by setting `random_state`
        to an integer.
        """
        X, y, groups = indexable(X, y, groups)
        yield from self._iter_indices(X, y, sample_domain=groups)

    @abstractmethod
    def _iter_indices(self, X, y=None, sample_domain=None):
        """Generate (train, test) indices"""

    def get_n_splits(self, X=None, y=None, sample_domain=None):
        """Returns the number of splitting iterations in the cross-validator

        Parameters
        ----------
        X : object
            Always ignored, exists for compatibility.

        y : object
            Always ignored, exists for compatibility.

        groups : object
            Always ignored, exists for compatibility.

        Returns
        -------
        n_splits : int
            Returns the number of splitting iterations in the cross-validator.
        """
        return self.n_splits

    def __repr__(self):
        return _build_repr(self)


class DomainAwareShuffleSplit(BaseDomainAwareShuffleSplit):

    def __init__(
        self, n_splits=10, *, test_size=None, train_size=None, random_state=None
    ):
        super().__init__(
            n_splits=n_splits,
            test_size=test_size,
            train_size=train_size,
            random_state=random_state,
        )
        self._default_test_size = 0.1

    def _iter_indices(self, X, y=None, sample_domain=None):
        X_source, X_target = check_X_domain(X, sample_domain, return_joint=False)
        n_samples = _num_samples(X_source)
        n_train, n_test = _validate_shuffle_split(
            n_samples,
            self.test_size,
            self.train_size,
            default_test_size=self._default_test_size,
        )

        rng = check_random_state(self.random_state)
        for i in range(self.n_splits):
            # random partition
            permutation = rng.permutation(n_samples)
            ind_test = permutation[:n_test]
            ind_train = permutation[n_test : (n_test + n_train)]
            yield ind_train, ind_test
