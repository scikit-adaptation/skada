# Author: Yanis Lalou <yanis.lalou@polytechnique.edu>
#
# License: BSD 3-Clause

from typing import Optional, Set

import numpy as np
from itertools import chain

from sklearn.utils import check_array, check_consistent_length

from skada._utils import _check_y_masking
from skada._utils import (
    _DEFAULT_SOURCE_DOMAIN_LABEL, _DEFAULT_TARGET_DOMAIN_LABEL,
    _DEFAULT_MASKED_TARGET_CLASSIFICATION_LABEL, _DEFAULT_TARGET_DOMAIN_ONLY_LABEL
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
        Input features
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
        (a[source_idx], a[~source_idx]) for a in arrays
    ))


def source_target_merge(
        X_source,
        X_target,
        sample_domain
    ) -> np.ndarray:
    """
    Merge source and target domain data based on sample domain labels.

    Parameters
    ----------
    X_source : array-like of shape (n_samples_source, n_features)
        Input features for the source domain.
    X_target : array-like of shape (n_samples_target, n_features)
        Input features for the target domain.
    sample_domain : array-like of shape (n_samples,)
        Array specifying the domain labels for each sample.

    Returns
    -------
    merged_data : np.ndarray
        Merged data based on the sample domain labels.
    """
    n_samples = X_source.shape[0] + X_target.shape[0]
    assert n_samples > 0

    check_consistent_length(
        np.concatenate((X_source, X_target)), sample_domain
    )

    if X_source.shape[0] > 0:
        output = np.zeros_like(
            np.concatenate((X_source, X_target)), dtype=X_source.dtype
        )

        output[sample_domain >= 0] = X_source
    else:
        output = np.zeros_like(
            np.concatenate((X_source, X_target)), dtype=X_target.dtype
        )
    output[sample_domain < 0] = X_target
    return output
