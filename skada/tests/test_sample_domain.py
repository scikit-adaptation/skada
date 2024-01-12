# Author:      
#
# License:

import numpy as np

from skada.datasets._base import DomainAwareDataset
from skada.datasets._samples_generator import make_shifted_datasets

import pytest


def test_sample_domain_type():
    X = np.array(list(range(10))).reshape((-1, 2))
    y = np.array([1]*X.shape[0])
    X, y, d = DomainAwareDataset(domains=[(X, y, 's'),(X, y, 't'),]).pack(as_sources= ['t'], as_targets = ['t'], return_X_y = True)
    assert type(d[0]) == np.int32

    X = np.array(list(range(10))).reshape((-1, 2))
    y = np.array([1.5]*X.shape[0])
    X, y, d = DomainAwareDataset(domains=[(X, y, 's'),(X, y, 't'),]).pack(as_sources= ['t'], as_targets = ['t'], return_X_y = True)
    assert type(d[0]) == np.int32
    
    X, y, sample_domain = make_shifted_datasets()
    assert type(sample_domain[0]) == np.int32
    
    X, y, sample_domain = make_shifted_datasets(label="regression")
    assert type(sample_domain[0]) == np.int32

