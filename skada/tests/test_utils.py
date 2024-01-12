# Author: Yanis Lalou <yanis.lalou@polytechnique.edu>
#
# License: BSD 3-Clause

import pytest

import numpy as np
from numpy.testing import assert_almost_equal
from skada.datasets import (
    make_dataset_from_moons_distribution
)
from skada._utils import check_X_y_domain, check_X_domain, _check_y_masking

def test_check_y_masking_classification():
    y_properly_masked = np.array([-1, 1, 2, -1, 2, 1, 1])
    y_wrongfuly_masked_1 = np.array([-1, -2, 2, -1, 2, 1, 1])
    y_wrongfuly_masked_2 = np.array([1, 2, 2, 1, 2, 1, 1])

    # Test that no ValueError is raised
    _check_y_masking(y_properly_masked)

    with pytest.raises(ValueError):
        _check_y_masking(y_wrongfuly_masked_1)

    with pytest.raises(ValueError):
        _check_y_masking(y_wrongfuly_masked_2)


def test_check_y_masking_regression():
    y_properly_masked = np.array([np.nan, 1, 2.5, -1, np.nan, 0, -1.5])
    y_wrongfuly_masked = np.array([-1, -2, 2.5, -1, 2, 0, 1])

    # Test that no ValueError is raised
    _check_y_masking(y_properly_masked)

    with pytest.raises(ValueError):
        _check_y_masking(y_wrongfuly_masked)


def test_check_2d_y_masking():
    y_wrong_dim = np.array([[-1, 2], [1, 2], [1, 2]])
    
    with pytest.raises(ValueError):
        _check_y_masking(y_wrong_dim)


def test_check_X_y_domain_exceptions():
    X, y, sample_domain = make_dataset_from_moons_distribution(
        pos_source=0.1,
        pos_target=0.9,
        n_samples_source=50,
        n_samples_target=20,
        random_state=0,
        return_X_y=True,
    )

     # Test that no ValueError is raised
    check_X_y_domain(X, y, sample_domain = sample_domain)

    with pytest.raises(ValueError):
        check_X_y_domain(X, y, sample_domain = None, allow_auto_sample_domain = False)


def test_check_X_domain_exceptions():
    X, y, sample_domain = make_dataset_from_moons_distribution(
        pos_source=0.1,
        pos_target=0.9,
        n_samples_source=50,
        n_samples_target=20,
        random_state=0,
        return_X_y=True,
    )

     # Test that no ValueError is raised
    check_X_domain(X, sample_domain = sample_domain)

    with pytest.raises(ValueError):
        check_X_domain(X, sample_domain = None, allow_auto_sample_domain = False)