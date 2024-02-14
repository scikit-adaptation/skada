# Author: Yanis Lalou <yanis.lalou@polytechnique.edu>
#
# License: BSD 3-Clause

import numpy as np

from skada.base import BaseAdapter, DAEstimator

def test_BaseAdapter():

    X = np.random.rand(10, 2)

    cls = BaseAdapter()

    cls.fit(X=X, y=None, sample_domain=None)
    # set one attribute to shohat something fitted
    cls.something_ = 1
    cls.transform(X=X, y=None, sample_domain=None)
    cls.fit_transform(X=X, y=None, sample_domain=None)


def test_DAEstimator():

    X = np.random.rand(10, 2)

    cls = DAEstimator()

    cls.fit(X=X, y=None, sample_domain=None)
    # set one attribute to shohat something fitted
    cls.something_ = 1
    cls.predict(X=X, sample_domain=None)
