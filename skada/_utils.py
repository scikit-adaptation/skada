# Author: Theo Gnassounou <theo.gnassounou@inria.fr>
#         Remi Flamary <remi.flamary@polytechnique.edu>
#         Oleksii Kachaiev <kachayev@gmail.com>
#
# License: BSD 3-Clause

import logging
from numbers import Real

import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.covariance import (
    empirical_covariance,
    ledoit_wolf,
    shrunk_covariance,
)
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
