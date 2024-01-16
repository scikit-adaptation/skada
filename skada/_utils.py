# Author: Theo Gnassounou <theo.gnassounou@inria.fr>
#         Remi Flamary <remi.flamary@polytechnique.edu>
#         Oleksii Kachaiev <kachayev@gmail.com>
#
# License: BSD 3-Clause

import logging
from numbers import Real
from typing import Optional, Set

import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.covariance import (
    empirical_covariance,
    ledoit_wolf,
    shrunk_covariance,
)
from sklearn.utils import check_array, check_consistent_length
from sklearn.utils.multiclass import type_of_target

_logger = logging.getLogger('skada')
_logger.setLevel(logging.DEBUG)

# Default label for datasets with source and target domains
_DEFAULT_TARGET_DOMAIN_LABEL = -2
_DEFAULT_SOURCE_DOMAIN_LABEL = 1

# Default label for datasets without source domain
_DEFAULT_TARGET_DOMAIN_ONLY_LABEL = -1

# Default label for datasets with masked target labels
_DEFAULT_MASKED_TARGET_CLASSIFICATION_LABEL = -1
_DEFAULT_MASKED_TARGET_REGRESSION_LABEL = np.nan


def _estimate_covariance(X, shrinkage):
    if shrinkage is None:
        s = empirical_covariance(X)
    elif shrinkage == "auto":
        sc = StandardScaler()  # standardize features
        X = sc.fit_transform(X)
        s = ledoit_wolf(X)[0]
        # rescale
        s = sc.scale_[:, np.newaxis] * s * sc.scale_[np.newaxis, :]
    elif isinstance(shrinkage, Real):
        s = shrunk_covariance(empirical_covariance(X), shrinkage)
    return s


def check_X_y_domain(
    X,
    y,
    sample_domain=None,
    allow_source: bool = True,
    allow_multi_source: bool = True,
    allow_target: bool = True,
    allow_multi_target: bool = True,
    return_indices: bool = False,
    # xxx(okachaiev): most likely this needs to be removed as it doesn't fit new API
    return_joint: bool = False,
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
    return_indices : bool, optional (default=False)
        If True, return only the source indices.
    return_joint : bool, optional (default=False)
        If True, return X, y, and sample_domain without separation.
    allow_auto_sample_domain : bool, optional (default=True)
        Allow automatic generation of sample_domain if not provided.
    allow_nd : bool, optional (default=False)
        Allow X and y to be N-dimensional arrays.

    Returns:
    ----------
    If return_indices is True :
        source_idx : array
            Boolean array indicating source indices.
    If return_joint is False :
        X_source : array
            Input features for source domains.
        y_source : array
            Target variable for source domains.
        X_target : array
            Input features for target domains.
        y_target : array
            Target variable for target domains.
    If return_joint is True :
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
    else:
        sample_domain = check_array(
            sample_domain,
            dtype=np.int32,
            ensure_2d=False,
            input_name='sample_domain'
        )
        check_consistent_length(X, sample_domain)

    source_idx = (sample_domain >= 0)
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

    if return_indices:
        # only source indices are given, target indices are ~source_idx
        return source_idx
    elif not return_joint:
        # commonly used X, y, X_target, y_target format
        return X[source_idx], y[source_idx], X[~source_idx], y[~source_idx]
    else:
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
    return_indices: bool = False,
    # xxx(okachaiev): most likely this needs to be removed as it doesn't fit new API
    return_joint: bool = True,
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
    return_indices : bool, optional (default=False)
        If True, return only the source indices.
    return_joint : bool, optional (default=True)
        If True, return X and sample_domain without separation.
    allow_auto_sample_domain : bool, optional (default=True)
        Allow automatic generation of sample_domain if not provided.

    Returns:
    ----------
    If return_indices is True:
        source_idx : array
            Boolean array indicating source indices.
    If return_joint is False:
        X_source : array
            Input features for source domains.
        X_target : array
            Input features for target domains.
    If return_joint is True:
        X : array
            Combined input features for source and target domains.
        sample_domain : array
            Combined domain labels for source and target domains.
    """
    X = check_array(X, input_name='X')
    if sample_domain is None and not allow_auto_sample_domain:
        raise ValueError("Either 'sample_domain' or 'allow_auto_sample_domain' "
                         "should be set")
    elif sample_domain is None and allow_auto_sample_domain:
        # default target domain when sample_domain is not given
        # xxx(okachaiev): I guess this should be -inf instead of a number
        # The idea is that with no labels we always assume
        # target domain (_DEFAULT_TARGET_DOMAIN_ONLY_LABEL)
        sample_domain = (
            _DEFAULT_TARGET_DOMAIN_ONLY_LABEL * np.ones(X.shape[0], dtype=np.int32)
        )
    else:
        sample_domain = check_array(
            sample_domain,
            dtype=np.int32,
            ensure_2d=False,
            input_name='sample_domain'
        )
        check_consistent_length(X, sample_domain)
    if allow_domains is not None:
        for domain in np.unique(sample_domain):
            # xxx(okachaiev): re-definition of the wildcards
            wildcard = np.inf if domain >= 0 else -np.inf
            if domain not in allow_domains and wildcard not in allow_domains:
                raise ValueError(f"Unknown domain label '{domain}' given")

    source_idx = (sample_domain >= 0)
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

    if return_indices:
        # only source indices are given, target indices are ~source_idx
        return source_idx
    elif not return_joint:
        # commonly used X, y, X_target, y_target format
        return X[source_idx], X[~source_idx]
    else:
        return X, sample_domain


def _merge_source_target(X_source, X_target, sample_domain) -> np.ndarray:
    n_samples = X_source.shape[0] + X_target.shape[0]
    assert n_samples > 0
    if X_source.shape[0] > 0:
        output = np.zeros((n_samples, X_source.shape[1]), dtype=X_source.dtype)
        output[sample_domain >= 0] = X_source
    else:
        output = np.zeros((n_samples, X_target.shape[1]), dtype=X_target.dtype)
    output[sample_domain < 0] = X_target
    return output


def source_target_split(X,
                        y,
                        sample_domain=None,
                        sample_weight=None,
                        allow_auto_sample_domain=True,
                        return_domain=False):
    r""" Split data into source and target domains

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Data to be split
    y : array-like of shape (n_samples,)
        Labels for the data
    sample_domain : array-like of shape (n_samples,)
        Domain labels for the data. Positive values are treated as source
        domains, negative values are treated as target domains. If not given,
        all samples are treated as source domains except those with y==-1.
    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights
    allow_auto_sample_domain : bool, optional (default=True)
        Allow automatic generation of sample_domain if not provided.
    return_domain : bool, default=False
        Whether to return domain labels

    Returns
    -------
    X_s : array-like of shape (n_samples_s, n_features)
        Source data
    y_s : array-like of shape (n_samples_s,)
        Source labels
    domain_s : array-like of shape (n_samples_s,)
        Source domain labels (returned only if `return_domain` is True)
    sample_weight_s : array-like of shape (n_samples_s,), default=None
        Source sample weights (returned only if `sample_weight` is not None)
    X_t : array-like of shape (n_samples_t, n_features)
        Target data
    y_t : array-like of shape (n_samples_t,)
        Target labels
    domain_t : array-like of shape (n_samples_t,)
        Target domain labels (returned only if `return_domain` is True)
    sample_weight_t : array-like of shape (n_samples_t,),
        Target sample weights (returned only if `sample_weight` is not None)

    """

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

    X_s = X[sample_domain >= 0]
    y_s = y[sample_domain >= 0]
    domain_s = sample_domain[sample_domain >= 0]

    X_t = X[sample_domain < 0]
    y_t = y[sample_domain < 0]
    domain_t = sample_domain[sample_domain < 0]

    if sample_weight is not None:
        sample_weight_s = sample_weight[sample_domain >= 0]
        sample_weight_t = sample_weight[sample_domain < 0]

        if return_domain:
            return (
                X_s, y_s, domain_s, sample_weight_s,
                X_t, y_t, domain_t, sample_weight_t
            )
        else:
            return X_s, y_s, sample_weight_s, X_t, y_t, sample_weight_t
    else:

        if return_domain:
            return X_s, y_s, domain_s, X_t, y_t, domain_t
        else:
            return X_s, y_s, X_t, y_t


def _check_y_masking(y):
    """Check that labels are properly masked
    ie. labels are either -1 or >= 0


    Parameters
    ----------
    y : array-like of shape (n_samples,)
        Labels for the data
    """

    # We need to check for this case first because
    # type_of_target() doesnt handle nan values
    if np.any(np.isnan(y)):
        if y.ndim != 1:
            raise ValueError("For a regression task, "
                             "more than 1D labels are not supported")
        else:
            return 'continuous'

    # Check if the target is a classification or regression target.
    y_type = type_of_target(y)

    if y_type == 'continuous':
        raise ValueError("For a regression task, "
                         "masked labels should be, "
                         f"{_DEFAULT_MASKED_TARGET_REGRESSION_LABEL}")
    elif y_type == 'binary' or y_type == 'multiclass':
        if (np.any(y < _DEFAULT_MASKED_TARGET_CLASSIFICATION_LABEL) or
                not np.any(y == _DEFAULT_MASKED_TARGET_CLASSIFICATION_LABEL)):
            raise ValueError("For a classification task, "
                             "masked labels should be, "
                             f"{_DEFAULT_MASKED_TARGET_CLASSIFICATION_LABEL}")
        else:
            return 'classification'
    else:
        raise ValueError("Uncompatible label type: %r" % y_type)
