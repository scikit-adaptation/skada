# Author: Theo Gnassounou <theo.gnassounou@inria.fr>
#         Remi Flamary <remi.flamary@polytechnique.edu>
#         Oleksii Kachaiev <kachayev@gmail.com>
#         Yanis Lalou <yanis.lalou@polytechnique.edu>
#
# License: BSD 3-Clause

from abc import ABCMeta, abstractmethod
from functools import reduce

import numpy as np
from sklearn.model_selection._split import (
    _build_repr,
    _num_samples,
    _validate_shuffle_split,
    GroupKFold
)
from sklearn.utils import check_random_state, indexable
from sklearn.utils.metadata_routing import _MetadataRequester

from .utils import check_X_domain, extract_source_indices, extract_domains_indices


class SplitSampleDomainRequesterMixin(_MetadataRequester):
    """Mixin for domain aware splitting that requires 'sample_domain' parameter."""

    __metadata_request__split = {"sample_domain": True}


class BaseDomainAwareShuffleSplit(SplitSampleDomainRequesterMixin, metaclass=ABCMeta):
    """Base class for domain aware implementation of the
    ShuffleSplit and StratifiedShuffleSplit.
    """

    def __init__(
        self, n_splits=10, *, test_size=None, train_size=None, random_state=None
    ):
        self.n_splits = n_splits
        self.test_size = test_size
        self.train_size = train_size
        self.random_state = random_state
        self._default_test_size = 0.1

    def split(self, X, y=None, sample_domain=None):
        """Generate indices to split data into training and test set.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where `n_samples` is the number of samples
            and `n_features` is the number of features.

        y : array-like of shape (n_samples,)
            The target variable for supervised learning problems.

        sample_domain : array-like of shape (n_samples,), default=None
            Domain labels for the samples used while splitting the dataset into
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
        # automatically derive sample_domain if it is not provided
        X, sample_domain = check_X_domain(
            X,
            sample_domain,
            allow_auto_sample_domain=True
        )
        X, y, sample_domain = indexable(X, y, sample_domain)
        yield from self._iter_indices(X, y, sample_domain=sample_domain)

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


class SourceTargetShuffleSplit(BaseDomainAwareShuffleSplit):
    """Source-Target-Shuffle-Split cross-validator.

    Provides train/test indices to split data in train/test sets.
    Each sample is used once as a test set (singleton) while the
    remaining samples form the training set.

    Default split is implemented hierarchically. If first designates
    a single domain as a target followed up by the single train/test
    shuffle split.
    """

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
        X, sample_domain = check_X_domain(X, sample_domain)
        indices = extract_source_indices(sample_domain)
        source_idx, = np.where(indices)
        target_idx, = np.where(~indices)
        n_source_samples = _num_samples(source_idx)
        n_source_train, n_source_test = _validate_shuffle_split(
            n_source_samples,
            self.test_size,
            self.train_size,
            default_test_size=self._default_test_size,
        )
        n_target_samples = _num_samples(target_idx)
        n_target_train, n_target_test = _validate_shuffle_split(
            n_target_samples,
            self.test_size,
            self.train_size,
            default_test_size=self._default_test_size,
        )

        rng = check_random_state(self.random_state)
        for i in range(self.n_splits):
            # random partition
            source_permutation = source_idx[rng.permutation(n_source_samples)]
            ind_source_train = source_permutation[
                n_source_test : (n_source_test + n_source_train)
            ]
            ind_source_test = source_permutation[:n_source_test]
            target_permutation = target_idx[rng.permutation(n_target_samples)]
            ind_target_train = target_permutation[
                n_target_test : (n_target_test + n_target_train)
            ]
            ind_target_test = target_permutation[:n_target_test]
            yield (
                np.concatenate([ind_source_train, ind_target_train]),
                np.concatenate([ind_source_test, ind_target_test]),
            )


class LeaveOneDomainOut(SplitSampleDomainRequesterMixin):
    """Leave-One-Domain-Out cross-validator.

    Provides train/test indices to split data in train/test sets.
    """

    def __init__(
        self, max_n_splits=10, *, test_size=None, train_size=None, random_state=None
    ):
        self.max_n_splits = max_n_splits
        self.test_size = test_size
        self.train_size = train_size
        self.random_state = random_state
        self._default_test_size = 0.1
        # so we can re-use existing implementation for shuffle split
        self._n_splits = 1

    def split(self, X, y=None, sample_domain=None):
        """Generate indices to split data into training and test set.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where `n_samples` is the number of samples
            and `n_features` is the number of features.

        y : array-like of shape (n_samples,)
            The target variable for supervised learning problems.

        sample_domain : array-like of shape (n_samples,), default=None
            Domain labels for the samples used while splitting the dataset into
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
        # automatically derive sample_domain if it is not provided
        X, sample_domain = check_X_domain(
            X,
            sample_domain,
            allow_auto_sample_domain=True
        )
        X, y, sample_domain = indexable(X, y, sample_domain)
        # xxx(okachaiev): make sure all domains are given both as sources and targets
        domains = self._get_domain_labels(sample_domain)
        n_domains = domains.shape[0]
        rng = check_random_state(self.random_state)
        domain_idx = rng.permutation(n_domains)
        if n_domains > self.max_n_splits:
            domain_idx = domain_idx[:self.max_n_splits]
        for target_domain_idx in domain_idx:
            target_domain = domains[target_domain_idx]
            split_idx = reduce(
                np.logical_or,
                (
                    sample_domain == (domain if domain != target_domain else -domain)
                    for domain
                    in domains
                )
            )
            split_idx, = np.where(split_idx)
            X_split = X[split_idx]
            split_sample_domain = sample_domain[split_idx]
            for train_idx, test_idx in self._iter_indices(
                X_split,
                y=None,
                sample_domain=split_sample_domain
            ):
                yield split_idx[train_idx], split_idx[test_idx]

    def _iter_indices(self, X, y=None, sample_domain=None):
        X, sample_domain = check_X_domain(X, sample_domain)
        indices = extract_source_indices(sample_domain)
        source_idx, = np.where(indices)
        target_idx, = np.where(~indices)
        n_source_samples = _num_samples(source_idx)
        n_source_train, n_source_test = _validate_shuffle_split(
            n_source_samples,
            self.test_size,
            self.train_size,
            default_test_size=self._default_test_size,
        )
        n_target_samples = _num_samples(target_idx)
        n_target_train, n_target_test = _validate_shuffle_split(
            n_target_samples,
            self.test_size,
            self.train_size,
            default_test_size=self._default_test_size,
        )

        rng = check_random_state(self.random_state)
        for i in range(self._n_splits):
            # random partition
            source_permutation = source_idx[rng.permutation(n_source_samples)]
            ind_source_train = source_permutation[
                n_source_test : (n_source_test + n_source_train)
            ]
            ind_source_test = source_permutation[:n_source_test]
            target_permutation = target_idx[rng.permutation(n_target_samples)]
            ind_target_train = target_permutation[
                n_target_test : (n_target_test + n_target_train)
            ]
            ind_target_test = target_permutation[:n_target_test]
            yield (
                np.concatenate([ind_source_train, ind_target_train]),
                np.concatenate([ind_source_test, ind_target_test]),
            )

    def _get_domain_labels(self, sample_domain: np.ndarray) -> np.ndarray:
        return np.unique(sample_domain[sample_domain >= 0])

    def get_n_splits(self, X=None, y=None, sample_domain=None):
        """Returns the number of splitting iterations in the cross-validator

        Parameters
        ----------
        X : object
            Always ignored, exists for compatibility.

        y : object
            Always ignored, exists for compatibility.

        sample_domain : np.ndarray
            Per-sample domain labels.

        Returns
        -------
        n_splits : int
            Returns the number of splitting iterations in the cross-validator.
        """
        domains = self._get_domain_labels(sample_domain)
        n_splits = domains.shape[0]
        return min(self.max_n_splits, n_splits)

    def __repr__(self):
        return _build_repr(self)


class GroupDomainAwareKFold(GroupKFold):
    """Group-DomainAware-KFold cross-validator.

    Each combinaison of sample_domain and labels form a group.
    Each group will appear exactly once in the test set across
    all folds (the number of distinct sample_domain * distinct
    labels has to be at least equal to the number of folds).

    The folds are approximately balanced in the sense that the number of
    distinct groups is approximately the same in each fold.

    Parameters
    ----------
    n_splits : int, default=5
        Number of folds. Must be at least 2.

    Examples
    --------
    >>> import numpy as np
    >>> from skada.model_selection import GroupDomainAwareKFold
    >>> X = np.ones((10, 2))
    >>> y = np.array([-1, 0, 1, -1, 0, 1, -1, 0, 1, -1])
    >>> sample_domain = np.array([-2, 1, 1, -2, 1, 1, -2, 1, 1, -2])
    >>> group_da_kfold = GroupDomainAwareKFold(n_splits=3)
    >>> group_da_kfold.get_n_splits(X, y, sample_domain)
    3
    >>> print(group_da_kfold)
    GroupDomainAwareKFold(n_splits=3)
    >>> for i, (train_index, test_index) in enumerate(
    ...    group_da_kfold.split(X, y, sample_domain)
    ... ):
    ...     print(f"Fold {i}:")
    ...     print(f"  Train: index={train_index}, "
    ...     f'''group={[str(b) + '_' +str(a)
    ...     for a, b in zip(y[train_index], sample_domain[train_index])
    ...     ]}''')
    ...     print(f"  Test: index={test_index}, "
    ...     f'''group={[str(b) + '_' +str(a)
    ...     for a, b in zip(y[test_index], sample_domain[test_index])
    ...     ]}''')
    Fold 0:
        Train: index=[1 2 4 5 7 8], group=['1_0', '1_1', '1_0', '1_1', '1_0', '1_1']
        Test:  index=[0 3 6 9], group=['-2_-1', '-2_-1', '-2_-1', '-2_-1']
    Fold 1:
        Train: index=[0 1 3 4 6 7 9],
        group=['-2_-1', '1_0', '-2_-1', '1_0', '-2_-1', '1_0', '-2_-1']
        Test:  index=[2 5 8], group=['1_1', '1_1', '1_1']
    Fold 2:
        Train: index=[0 2 3 5 6 8 9],
        group=['-2_-1', '1_1', '-2_-1', '1_1', '-2_-1', '1_1', '-2_-1']
        Test:  index=[1 4 7], group=['1_0', '1_0', '1_0']
    """

    __metadata_request__split = {"groups": "sample_domain"}

    def split(self, X, y=None, groups=None):
        if (groups is not None) and (y is not None):
            # might be a better way to do this
            # using [a, b] instead doesn't work for some reason
            groups = np.array([str(b) + '_' + str(a) for a, b in zip(y, groups)])

        return super().split(X, y, groups)


class DomainShuffleSplit(BaseDomainAwareShuffleSplit):
    """Random-Shuffle-DomainAware-Split cross-validator.
    Provides randomized train/test indices to split data depending
    on their sample_domain.
    Each fold is composed of samples coming from both source and target
    domains.
    The folds are made by preserving the percentage of samples for
    each sample domain.
    Parameters
    ----------
    n_splits : int, default=10
        Number of re-shuffling & splitting iterations.
    test_size : float or int, default=None
        If float, should be between 0.0 and 1.0 and represent the proportion
        of the dataset to include in the test split. If int, represents the
        absolute number of test samples. If None, the value is set to the
        complement of the train size. If train_size is also None, it will be
        set to 0.1.
    train_size : float or int, default=None
        If float, should be between 0.0 and 1.0 and represent the
        proportion of the dataset to include in the train split. If int,
        represents the absolute number of train samples. If None, the value
        is automatically set to the complement of the test size.
    random_state : int or RandomState instance, default=None
        Controls the randomness of the training and testing indices produced.
    Examples
    --------
    >>> import numpy as np
    >>> from skada.model_selection import DomainShuffleSplit
    >>> X = np.ones((10, 2))
    >>> y = np.ones((10, 1))
    >>> sample_domain = np.array([1, -2, 1, -2, 1, -2, 1, -2, 1, -2])
    >>> rsdas = DomainShuffleSplit(
    ...     n_splits=3,
    ...     random_state=0,
    ...     test_size=0.1,
    ...     )
    >>> for train_index, test_index in rsdas.split(X, y, sample_domain):
    ...     print("TRAIN:", train_index, "TEST:", test_index)
    TRAIN: [0 2 6 5 3 9] TEST: [4 1]
    TRAIN: [6 8 0 3 5 9] TEST: [2 7]
    TRAIN: [4 6 2 7 9 3] TEST: [8 5]
    """
    def __init__(
        self,
        n_splits=10,
        *,
        test_size=None,
        train_size=None,
        random_state=None,
        under_sampling=0.8
    ):
        super().__init__(
            n_splits=n_splits,
            test_size=test_size,
            train_size=train_size,
            random_state=random_state,
        )
        self.random_state = random_state
        self.under_sampling = under_sampling

        if not (0 <= under_sampling <= 1):
            raise ValueError("under_sampling should be between 0 and 1")

    def _iter_indices(self, X, y=None, sample_domain=None):
        X, sample_domain = check_X_domain(X, sample_domain)
        domain_source_idx_dict, domain_target_idx_dict = extract_domains_indices(
            sample_domain,
            split_source_target=True
        )
        rng = check_random_state(self.random_state)
        for _ in range(self.n_splits):
            source_idx = np.concatenate(
                [rng.choice(v, int(len(v)*self.under_sampling), replace=False)
                 for v in domain_source_idx_dict.values()]
                )
            target_idx = np.concatenate(
                [rng.choice(v, int(len(v)*self.under_sampling), replace=False)
                 for v in domain_target_idx_dict.values()]
                )
            n_source_samples = _num_samples(source_idx)
            n_source_train, n_source_test = _validate_shuffle_split(
                n_source_samples,
                self.test_size,
                self.train_size,
                default_test_size=self._default_test_size,
            )
            n_target_samples = _num_samples(target_idx)
            n_target_train, n_target_test = _validate_shuffle_split(
                n_target_samples,
                self.test_size,
                self.train_size,
                default_test_size=self._default_test_size,
            )
            ind_source_train = source_idx[
                n_source_test : (n_source_test + n_source_train)
            ]
            ind_source_test = source_idx[:n_source_test]
            ind_target_train = target_idx[
                n_target_test : (n_target_test + n_target_train)
            ]
            ind_target_test = target_idx[:n_target_test]
            yield (
                np.concatenate([ind_source_train, ind_target_train]),
                np.concatenate([ind_source_test, ind_target_test]),
            )
