# Author: Yanis Lalou <yanis.lalou@polytechnique.edu>
#
# License: BSD 3-Clause

from typing import Optional, Set, Sequence

import warnings

import numpy as np
from itertools import chain

from sklearn.utils import check_array, check_consistent_length
from sklearn.utils.multiclass import type_of_target

from skada._utils import _check_y_masking
from skada._utils import (
    _DEFAULT_SOURCE_DOMAIN_LABEL,
    _DEFAULT_TARGET_DOMAIN_LABEL,
    _DEFAULT_MASKED_TARGET_CLASSIFICATION_LABEL,
    _DEFAULT_TARGET_DOMAIN_ONLY_LABEL,
    _DEFAULT_MASKED_TARGET_REGRESSION_LABEL
)


def check_X_y_domain(
    X,
    y,
    sample_domain=None,
    allow_source: bool = True,
    allow_multi_source: bool = True,
    allow_target: bool = True,
    allow_multi_target: bool = True,
    allow_auto_sample_domain: bool = True,
    allow_nd: bool = False,
):
    """
    Input validation for domain adaptation (DA) estimator.
    If we work in single-source and single target mode, return source and target
    separately to avoid additional scan for 'sample_domain' array.

    Parameters:
    ----------
    X : array-like of shape (n_samples, n_features)
        Input features
    y : array-like of shape (n_samples,)
        Target variable
    sample_domain : array-like or None, optional (default=None)
        Array specifying the domain labels for each sample.
    allow_source : bool, optional (default=True)
        Allow the presence of source domains.
    allow_multi_source : bool, optional (default=True)
        Allow multiple source domains.
    allow_target : bool, optional (default=True)
        Allow the presence of target domains.
    allow_multi_target : bool, optional (default=True)
        Allow multiple target domains.
    allow_auto_sample_domain : bool, optional (default=True)
        Allow automatic generation of sample_domain if not provided.
    allow_nd : bool, optional (default=False)
        Allow X and y to be N-dimensional arrays.

    Returns:
    ----------
    X : array
        Input features
    y : array
        Target variable
    sample_domain : array
        Array specifying the domain labels for each sample.
    """

    X = check_array(X, input_name='X', allow_nd=allow_nd)
    y = check_array(y, force_all_finite=True, ensure_2d=False, input_name='y')
    check_consistent_length(X, y)

    if sample_domain is None and not allow_auto_sample_domain:
        raise ValueError("Either 'sample_domain' or 'allow_auto_sample_domain' "
                         "should be set")
    elif sample_domain is None and allow_auto_sample_domain:
        y_type = _check_y_masking(y)
        sample_domain = _DEFAULT_SOURCE_DOMAIN_LABEL*np.ones_like(y)
        # labels masked with -1 (for classification) are recognized as targets,
        # labels masked with nan (for regression) are recognized as targets,
        # the rest is treated as a source
        if y_type == 'classification':
            mask = (y == _DEFAULT_MASKED_TARGET_CLASSIFICATION_LABEL)
        else:
            mask = (np.isnan(y))
        sample_domain[mask] = _DEFAULT_TARGET_DOMAIN_LABEL

    source_idx = extract_source_indices(sample_domain)

    # xxx(okachaiev): this needs to be re-written to accommodate for a
    # a new domain labeling convention without "intersections"
    n_sources = np.unique(sample_domain[source_idx]).shape[0]
    n_targets = np.unique(sample_domain[~source_idx]).shape[0]

    if not allow_source and n_sources > 0:
        raise ValueError(f"Number of sources provided is {n_sources} "
                         "and 'allow_source' is set to False")
    if not allow_target and n_targets > 0:
        raise ValueError(f"Number of targets provided is {n_targets} "
                         "and 'allow_target' is set to False")
    if not allow_multi_source and n_sources > 1:
        raise ValueError(f"Number of sources provided is {n_sources} "
                         "and 'allow_multi_source' is set to False")
    if not allow_multi_target and n_sources > 1:
        raise ValueError(f"Number of targets provided is {n_targets} "
                         "and 'allow_multi_target' is set to False")

    return X, y, sample_domain


# xxx(okachaiev): code duplication, just for testing
def check_X_domain(
    X,
    sample_domain,
    *,
    allow_domains: Optional[Set[int]] = None,
    allow_source: bool = True,
    allow_multi_source: bool = True,
    allow_target: bool = True,
    allow_multi_target: bool = True,
    allow_auto_sample_domain: bool = True,
):
    """
    Input validation for domain adaptation (DA) estimator.
    If we work in single-source and single target mode, return source and target
    separately to avoid additional scan for 'sample_domain' array.

    Parameters:
    ----------
    X : array-like of shape (n_samples, n_features)
        Input features.
    sample_domain : array-like of shape (n_samples,)
        Domain labels for each sample.
    allow_domains : set of int, optional (default=None)
        Set of allowed domain labels. If provided, only these domain labels are allowed.
    allow_source : bool, optional (default=True)
        Allow the presence of source domains.
    allow_multi_source : bool, optional (default=True)
        Allow multiple source domains.
    allow_target : bool, optional (default=True)
        Allow the presence of target domains.
    allow_multi_target : bool, optional (default=True)
        Allow multiple target domains.
    allow_auto_sample_domain : bool, optional (default=True)
        Allow automatic generation of sample_domain if not provided.

    Returns:
    ----------
    X : array
        Input features.
    sample_domain : array
        Combined domain labels for source and target domains.
    """
    X = check_array(X, input_name='X')

    if sample_domain is None and not allow_auto_sample_domain:
        raise ValueError("Either 'sample_domain' or 'allow_auto_sample_domain' "
                         "should be set")
    elif sample_domain is None and allow_auto_sample_domain:
        # default target domain when sample_domain is not given
        # The idea is that with no labels we always assume
        # target domain (_DEFAULT_TARGET_DOMAIN_ONLY_LABEL)
        sample_domain = (
            _DEFAULT_TARGET_DOMAIN_ONLY_LABEL * np.ones(X.shape[0], dtype=np.int32)
        )

    source_idx = extract_source_indices(sample_domain)
    check_consistent_length(X, sample_domain)

    if allow_domains is not None:
        for domain in np.unique(sample_domain):
            # xxx(okachaiev): re-definition of the wildcards
            wildcard = np.inf if domain >= 0 else -np.inf
            if domain not in allow_domains and wildcard not in allow_domains:
                raise ValueError(f"Unknown domain label '{domain}' given")

    n_sources = np.unique(sample_domain[source_idx]).shape[0]
    n_targets = np.unique(sample_domain[~source_idx]).shape[0]

    if not allow_source and n_sources > 0:
        raise ValueError(f"Number of sources provided is {n_sources} "
                         "and 'allow_source' is set to False")
    if not allow_target and n_targets > 0:
        raise ValueError(f"Number of targets provided is {n_targets} "
                         "and 'allow_target' is set to False")
    if not allow_multi_source and n_sources > 1:
        raise ValueError(f"Number of sources provided is {n_sources} "
                         "and 'allow_multi_source' is set to False")
    if not allow_multi_target and n_sources > 1:
        raise ValueError(f"Number of targets provided is {n_targets} "
                         "and 'allow_multi_target' is set to False")

    return X, sample_domain


def extract_source_indices(sample_domain):
    """Extract the indices of the source samples.

    Parameters:
    ----------
    sample_domain : array-like of shape (n_samples,)
        Array specifying the domain labels for each sample.

    Returns:
    ----------
    source_idx : array
        Boolean array indicating source indices.
    """
    sample_domain = check_array(
        sample_domain,
        dtype=np.int32,
        ensure_2d=False,
        input_name='sample_domain'
    )

    source_idx = (sample_domain >= 0)
    return source_idx


def source_target_split(
    *arrays,
    sample_domain
):
    r""" Split data into source and target domains

    Parameters
    ----------
    *arrays : sequence of array-like of identical shape (n_samples, n_features)
        Input features and target variable(s), and or sample_weights to be
        split. All arrays should have the same length except if None is given
        then a couple of None variables are returned to allow for optional
        sample_weight.
    sample_domain : array-like of shape (n_samples,)
        Array specifying the domain labels for each sample.

    Returns
    -------
    splits : list, length=2 * len(arrays)
        List containing source-target split of inputs.
    """

    if len(arrays) == 0:
        raise ValueError("At least one array required as input")

    check_consistent_length(arrays)

    source_idx = extract_source_indices(sample_domain)

    return list(chain.from_iterable(
        (a[source_idx], a[~source_idx]) if a is not None else (None, None)
        for a in arrays
    ))


def source_target_merge(
    *arrays,
    sample_domain : Optional[np.ndarray] = None
) -> Sequence[np.ndarray]:
    f""" Merge source and target domain data based on sample domain labels.

    Parameters
    ----------
    *arrays : sequence of array-like
        (, n_features). The number of arrays must be even
        since we consider pairs of source-target arrays each time.
        Each pair should have at least one non-empty array.
        In each pair the first array is considered as the source and
        the second as the target. If one of the array is None or empty,
        it's value will be inferred from the other array and the sample_domain
        (depending on the type of the arrays, they'll have a value of
        {_DEFAULT_MASKED_TARGET_CLASSIFICATION_LABEL} or
        {_DEFAULT_MASKED_TARGET_REGRESSION_LABEL}).
    sample_domain : array-like of shape (n_samples,)
        Array specifying the domain labels for each sample. If None or empty
        the domain labels will be inferred from the the *arrays, being a
        default source and target domain (depending on the type of the arrays,
        they'll have a value of {_DEFAULT_SOURCE_DOMAIN_LABEL}
        or {_DEFAULT_TARGET_DOMAIN_LABEL}).

    Returns
    -------
    merges : list, length=len(arrays)/2
        List containing merged data based on the sample domain labels.
    sample_domain : array-like of shape (n_samples,)
        Array specifying the domain labels for each sample.

    Examples
    --------
    >>> X_source = np.array([[1, 2], [3, 4], [5, 6]])
    >>> X_target = np.array([[7, 8], [9, 10]])
    >>> sample_domain = np.array([0, 0, 1, 1])
    >>> X, _ = source_target_merge(X_source, X_target, sample_domain = sample_domain)
    >>> X
    np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])

    >>> X_source = np.array([[1, 2], [3, 4], [5, 6]])
    >>> X_target = np.array([[7, 8], [9, 10]])
    >>> y_source = np.array([0, 1, 1])
    >>> y_target = None
    >>> sample_domain = np.array([0, 0, 1, 1])
    >>> X, y, _ = source_target_merge(
        X_source,
        X_target,
        y_source,
        y_target,
        sample_domain = sample_domain
        )
    >>> X
    np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])

    >>> y
    np.array([0,
        1,
        1,
    {_DEFAULT_MASKED_TARGET_CLASSIFICATION_LABEL},
    {_DEFAULT_MASKED_TARGET_CLASSIFICATION_LABEL}
    ])
    """

    arrays = list(arrays)   # Convert to list to be able to modify it

    # ################ Check *arrays #################
    if len(arrays) < 2:
        raise ValueError("At least two array required as input")

    if len(arrays) % 2 != 0:
        raise ValueError("Even number of arrays required as input")

    for i in range(0, len(arrays), 2):
        if arrays[i] is None or arrays[i].shape[0] == 0:
            arrays[i] = np.array([])

        if arrays[i+1] is None or arrays[i+1].shape[0] == 0:
            arrays[i+1] = np.array([])

        # Check no pair is empty
        if (np.size(arrays[i]) == 0 and np.size(arrays[i+1]) == 0):
            raise ValueError("Only one array can be None or empty in each pair")

        # Check consistent dim of arrays
        if (np.size(arrays[i]) != 0 and np.size(arrays[i+1]) != 0):
            if arrays[i].shape[1:] != arrays[i+1].shape[1:]:
                raise ValueError(
                    "Inconsistent number of features in source-target arrays"
                )

    # ################ Check sample_domain #################
    # If sample_domain is None, we need to infer it from the target array
    if sample_domain is None or sample_domain.shape[0] == 0:
        # We need to infer the domain from the target array
        warnings.warn(
            "sample_domain is None or empty, it will be inferred from the arrays"
        )

        # By assuming that the first array is the source and the second the target
        source_assumed_index = 0
        target_assumed_index = 1
        sample_domain = np.concatenate((
            _DEFAULT_SOURCE_DOMAIN_LABEL*np.ones(
                arrays[source_assumed_index].shape[0]
            ),
            _DEFAULT_TARGET_DOMAIN_LABEL*np.ones(
                arrays[target_assumed_index].shape[0]
            )
        ))

    # To test afterward if the number of samples in source-target arrays
    # and the number infered in the sample_domain are consistent
    source_indices = extract_source_indices(sample_domain)
    source_samples = np.count_nonzero(source_indices)
    target_samples = np.count_nonzero(~source_indices)

    # ################ Merge arrays #################
    merges = []  # List of merged arrays

    for i in range(0, len(arrays), 2):

        # If one of the array is empty, we need to infer its values
        index_is_empty = None  # Index of the array that was None
        if np.size(arrays[i]) == 0:
            index_is_empty = i

        if np.size(arrays[i+1]) == 0:
            index_is_empty = i+1

        if index_is_empty is not None:
            # We need to infer the value of the empty array in the pair
            # warnings.warn(
            #    "One of the arrays in a pair is empty, it will be inferred"
            # )

            pair_index = i+1 if index_is_empty == i else i

            y_type = type_of_target(arrays[pair_index])
            if y_type == 'binary' or y_type == 'multiclass':
                default_masked_label = _DEFAULT_MASKED_TARGET_CLASSIFICATION_LABEL
            else:
                default_masked_label = _DEFAULT_MASKED_TARGET_REGRESSION_LABEL

            arrays[index_is_empty] = (
                default_masked_label *
                np.ones(
                    (sample_domain.shape[0] - arrays[pair_index].shape[0],) +
                    arrays[pair_index].shape[1:]
                )
            )

        # Check consistent number of samples in source-target arrays
        # and the number infered in the sample_domain
        if (sample_domain is not None) and (sample_domain.shape[0] != 0):
            if (
                (arrays[i].shape[0] != source_samples) or
                (arrays[i+1].shape[0] != target_samples)
            ):
                raise ValueError(
                    "Inconsistent number of samples in source-target arrays "
                    "and the number infered in the sample_domain"
                )

        merges.append(
            _merge_arrays(arrays[i], arrays[i+1], sample_domain=sample_domain)
        )

    return (*merges, sample_domain)


def _merge_arrays(
    array_source,
    array_target,
    sample_domain
):
    """Merge source and target domain data based on sample domain labels.

    Parameters
    ----------
    array_source : sequence of array-like of identical number of samples
        (n_samples_source, n_features).
    array_target : sequence of array-like of identical number of samples
        (n_samples_target, n_features).
    sample_domain : array-like of shape (n_samples_source + n_samples_target,)
        Array specifying the domain labels for each sample.

    Returns
    -------
    merge : sequence of array-like of identical number of samples
        (n_samples_source + n_samples_target, n_features)
    """

    if array_source.shape[0] > 0 and array_target.shape[0] > 0:
        output = np.zeros_like(
            np.concatenate((array_source, array_target)), dtype=array_source.dtype
        )
        output[sample_domain >= 0] = array_source
        output[sample_domain < 0] = array_target

    elif array_source.shape[0] > 0:
        output = np.zeros_like(
            array_source, dtype=array_source.dtype
        )
        output[sample_domain >= 0] = array_source

    else:
        output = np.zeros_like(
            array_target, dtype=array_target.dtype
        )
        output[sample_domain < 0] = array_target

    return output
