# Author: Yanis Lalou <yanis.lalou@polytechnique.edu>
#
# License: BSD 3-Clause

import numpy as np
import pytest

from skada.base import BaseAdapter, DAEstimator


def test_BaseAdapter():
    rng = np.random.default_rng(42)
    X = rng.standard_normal((10, 2))

    cls = BaseAdapter()

    with pytest.raises(NotImplementedError):
        cls.fit(X=X, y=None, sample_domain=None)

    cls.fitted_ = 1  # set one attribute to show it is fitted
    cls.fit_transform(X=X, y=None, sample_domain=None)
    cls.transform(X=X, y=None, sample_domain=None)


def test_DAEstimator():
    rng = np.random.default_rng(42)
    X = rng.standard_normal((10, 2))

    cls = DAEstimator()

    cls.fit(X=X, y=None, sample_domain=None)
    cls.fitted_ = 1  # set one attribute to show it is fitted
    cls.predict(X=X, sample_domain=None)
