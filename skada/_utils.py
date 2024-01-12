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
    sample_domain = None,
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
    """Input validation for DA estimator.
    If we work in single-source and single target mode, return source and target
    separately to avoid additional scan for 'sample_domain' array.
    """

    X = check_array(X, input_name='X', allow_nd=allow_nd)
    y = check_array(y, force_all_finite=True, ensure_2d=False, input_name='y')
    check_consistent_length(X, y)
    if sample_domain is None and not allow_auto_sample_domain:
        raise ValueError("Either 'sample_domain' or 'allow_auto_sample_domain' "
                         "should be set")
    elif sample_domain is None and allow_auto_sample_domain:
        _check_y_masking(y)
        sample_domain = np.ones_like(y)
        # labels masked with -1 are recognized as targets,
        # the rest is treated as a source
        sample_domain[y == -1] = -2
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
    allow_auto_sample_domain: bool = False,
):
    X = check_array(X, input_name='X')
    if sample_domain is None and allow_auto_sample_domain:
        # default target domain when sample_domain is not given
        # xxx(okachaiev): I guess this should be -inf instead of a number
        sample_domain = -2*np.ones(X.shape[0], dtype=np.int32)
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


def source_target_split(X, y, sample_domain=None,
                        sample_weight=None, return_domain=False):
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

    if sample_domain is None:
        sample_domain = np.ones_like(y)
        # labels masked with -1 are recognized as targets,
        # the rest is treated as a source
        sample_domain[y == -1] = -1

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

    y_type = type_of_target(y) #Check if the target is a classification or regression target.

    if y_type == 'continuous':
        if not np.any(np.isnan(y)):
            raise ValueError("For a regression task, "
                             "masked labels should be NaN")
    elif y_type == 'binary' or y_type == 'multiclass':
        if not np.any(y < -1):
            raise ValueError("For a classification task, "
                             "masked labels should be -1")
    else:
        raise ValueError("Uncompatible label type: %r" % y_type)
    
