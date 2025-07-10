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
from sklearn.utils import check_random_state
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


def _estimate_covariance(X, shrinkage, assume_centered=False):
    if shrinkage is None:
        s = empirical_covariance(X, assume_centered=assume_centered)
    elif shrinkage == "auto":
        sc = StandardScaler(with_mean=not assume_centered)
        X = sc.fit_transform(X)
        s = ledoit_wolf(X)[0]
        # rescale
        s = sc.scale_[:, np.newaxis] * s * sc.scale_[np.newaxis, :]
    elif isinstance(shrinkage, Real):
        s = shrunk_covariance(
            empirical_covariance(X, assume_centered=assume_centered),
            shrinkage=shrinkage,
        )
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

    if y_type in ["continuous", "continuous-multioutput"]:
        return Y_Type.CONTINUOUS
    elif y_type in ["binary", "multiclass"]:
        return Y_Type.DISCRETE
    else:
        # Here y_type is 'multilabel-indicator', 'continuous-multioutput',
        # 'multiclass-multioutput' or 'unknown'
        raise ValueError(f"Incompatible label type: {y_type}")


def _apply_domain_masks(X, y, params, masks):
    X = X[masks]
    if y is not None:
        y = y[masks]
    params = {
        k: v[masks] if (hasattr(v, "__len__") and len(v) == len(masks)) else v
        for k, v in params.items()
    }
    return X, y, params


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
    y_type = _find_y_type(y)
    if y_type == Y_Type.DISCRETE:
        unmasked_idx = y != _DEFAULT_MASKED_TARGET_CLASSIFICATION_LABEL
    elif y_type == Y_Type.CONTINUOUS:
        unmasked_idx = np.isfinite(y)

    X, y, params = _apply_domain_masks(X, y, params, masks=unmasked_idx)
    return X, y, params


def _merge_domain_outputs(n_samples, domain_outputs, *, allow_containers=False):
    assert len(domain_outputs), "At least a single domain has to be given"
    _, first_output = next(iter(domain_outputs.values()))
    if isinstance(first_output, tuple):
        assert (
            allow_containers
        ), "Container output is given while `allow_containers` set to False"
        X_output, y_output, params_output = None, None, {}
        for idx, domain_output in domain_outputs.values():
            if len(domain_output) == 2:
                domain_X, domain_params = domain_output
                domain_y = None
            elif len(domain_output) == 3:
                domain_X, domain_y, domain_params = domain_output
            else:
                raise ValueError("Invalid container structure")
            if X_output is None:
                X_output = np.zeros(
                    (n_samples, *domain_X.shape[1:]), dtype=domain_X.dtype
                )
            if domain_y is not None and y_output is None:
                y_output = np.zeros(
                    (n_samples, *domain_y.shape[1:]), dtype=domain_y.dtype
                )
            X_output[idx] = domain_X
            if domain_y is not None:
                y_output[idx] = domain_y
            for k, v in domain_params.items():
                if k not in params_output:
                    params_output[k] = np.zeros(
                        (n_samples, *v.shape[1:]), dtype=v.dtype
                    )
                params_output[k][idx] = v
        output = X_output, y_output, params_output
    else:
        assert isinstance(first_output, np.ndarray)
        output = np.zeros(
            (n_samples, *first_output.shape[1:]), dtype=first_output.dtype
        )
        for idx, domain_output in domain_outputs.values():
            output[idx] = domain_output
    return output


def _shuffle_arrays(*arrays, random_state=None):
    """Function to shuffle multiple arrays in the same order."""
    random_state = check_random_state(random_state)

    indices = np.arange(arrays[0].shape[0])
    random_state.shuffle(indices)
    shuffled_arrays = []
    for arr in arrays:
        shuffled_arrays.append(arr[indices])
    return tuple(shuffled_arrays)


def _route_params(request, params, caller):
    return request._route_params(params=params, parent=caller, caller=caller)
