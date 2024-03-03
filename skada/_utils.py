# Author: Theo Gnassounou <theo.gnassounou@inria.fr>
#         Remi Flamary <remi.flamary@polytechnique.edu>
#         Oleksii Kachaiev <kachayev@gmail.com>
#         Yanis Lalou <yanis.lalou@polytechnique.edu>
#
# License: BSD 3-Clause

import logging
from enum import Enum
from numbers import Real

import numpy as np
from sklearn.covariance import (
    empirical_covariance,
    ledoit_wolf,
    shrunk_covariance,
)
from sklearn.preprocessing import StandardScaler
from sklearn.utils.multiclass import type_of_target

_logger = logging.getLogger("skada")
_logger.setLevel(logging.DEBUG)

# Default label for datasets with source and target domains
_DEFAULT_TARGET_DOMAIN_LABEL = -2
_DEFAULT_SOURCE_DOMAIN_LABEL = 1

# Default label for datasets without source domain
_DEFAULT_TARGET_DOMAIN_ONLY_LABEL = -1

# Default label for datasets with masked target labels
_DEFAULT_MASKED_TARGET_CLASSIFICATION_LABEL = -1
_DEFAULT_MASKED_TARGET_REGRESSION_LABEL = np.nan


class Y_Type(Enum):
    CONTINUOUS = "continuous"
    DISCRETE = "discrete"


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


def _check_y_masking(y):
    """Check that labels are properly masked ie. labels are either -1 or >= 0

    Parameters
    ----------
    y : array-like of shape (n_samples,)
        Labels for the data
    """
    # Find the type of the labels, continuous or classification
    y_type = _find_y_type(y)

    if y_type == Y_Type.CONTINUOUS:
        if np.any(np.isnan(y)):
            return y_type
        else:
            raise ValueError(
                "For a regression task, "
                "masked labels should be, "
                f"{_DEFAULT_MASKED_TARGET_REGRESSION_LABEL}"
            )
    elif y_type == Y_Type.DISCRETE:
        if np.any(y < _DEFAULT_MASKED_TARGET_CLASSIFICATION_LABEL) or not np.any(
            y == _DEFAULT_MASKED_TARGET_CLASSIFICATION_LABEL
        ):
            raise ValueError(
                "For a classification task, "
                "masked labels should be, "
                f"{_DEFAULT_MASKED_TARGET_CLASSIFICATION_LABEL}"
            )
        else:
            return y_type


def _find_y_type(y):
    """
    Find the type of the labels. They can either be continuous or
    classification.

    Parameters
    ----------
    y : array-like of shape (n_samples,)
        Labels for the data

    Returns
    -------
    y_type : str
        Type of labels between 'continuous' and 'classification'
    """
    # We need to check for this case first because
    # type_of_target() doesn't handle nan values
    if np.any(np.isnan(y)):
        if y.ndim != 1:
            raise ValueError(
                "For a regression task, " "more than 1D labels are not supported"
            )
        else:
            return Y_Type.CONTINUOUS

    # Check if the target is a classification or regression target.
    y_type = type_of_target(y)

    if y_type == "continuous":
        return Y_Type.CONTINUOUS
    elif y_type == "binary" or y_type == "multiclass":
        return Y_Type.DISCRETE
    else:
        # Here y_type is 'multilabel-indicator', 'continuous-multioutput',
        # 'multiclass-multioutput' or 'unknown'
        raise ValueError(f"Incompatible label type: {y_type}")


def _remove_masked(X, y, params):
    """Internal API for removing masked samples before passing them
    to the estimator that does not accept 'sample_domain' (e.g. any
    standard sklearn estimator).

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Input data
    y : array-like of shape (n_samples,)
        Labels for the data
    params : dict
        Additional parameters declared in the routing

    Returns
    -------
    X : array-like of shape (n_samples, n_features)
        Input data
    y : array-like of shape (n_samples,)
        Labels for the data
    params : dict
        Additional parameters declared in the routing
    """
    # technically, `y` is optional but if we have no
    # labels, - there are no masks
    if y is None:
        return X, y, params

    y_type = _find_y_type(y)
    if y_type == Y_Type.DISCRETE:
        unmasked_idx = y != _DEFAULT_MASKED_TARGET_CLASSIFICATION_LABEL
    elif y_type == Y_Type.CONTINUOUS:
        unmasked_idx = np.isfinite(y)

    X = X[unmasked_idx]
    y = y[unmasked_idx]
    params = {
        # this is somewhat crude way to test is `v` is indexable
        k: v[unmasked_idx]
        if (hasattr(v, "__len__") and len(v) == len(unmasked_idx))
        else v
        for k, v in params.items()
    }
    return X, y, params
