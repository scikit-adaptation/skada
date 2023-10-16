import logging
from numbers import Real

import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.covariance import (
    ledoit_wolf, empirical_covariance, shrunk_covariance
)


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


_logger = logging.getLogger('skada')
_logger.setLevel(logging.DEBUG)