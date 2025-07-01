# Author: Oleksii Kachaiev <kachayev@gmail.com>
#         Yanis Lalou <yanis.lalou@polytechnique.edu>
#
# License: BSD 3-Clause

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from skada.datasets import DomainAwareDataset

# mark all the test with the marker dataset
pytestmark = pytest.mark.dataset


def test_dataset_train_label_masking():
    dataset = DomainAwareDataset()
    dataset.add_domain(np.array([1.0, 2.0]), np.array([1, 2]), "s1")
    dataset.add_domain(np.array([10.0, 20.0, 30.0]), np.array([10, 20, 30]), "t1")
    X, y, sample_domain = dataset.pack(
        as_sources=["s1"], as_targets=["t1"], mask_target_labels=True
    )

    # test shape of the output
    assert X.shape == (5,)
    assert y.shape == (5,)
    assert sample_domain.shape == (5,)
    assert (sample_domain > 0).sum() == 2
    assert (sample_domain < 0).sum() == 3

    # test label masking
    assert_array_equal(y[sample_domain < 0], np.array([-1, -1, -1]))
    assert np.all(y[sample_domain > 0] > 0)

    # custom mask
    X, y, sample_domain = dataset.pack(
        as_sources=["s1"], as_targets=["t1"], mask_target_labels=True, mask=-10
    )
    assert_array_equal(y[sample_domain < 0], np.array([-10, -10, -10]))

    # test packing does not perform masking
    X, y, sample_domain = dataset.pack(
        as_sources=[], as_targets=["t1"], mask_target_labels=False
    )
    assert X.shape == (3,)
    assert y.shape == (3,)
    assert sample_domain.shape == (3,)
    assert np.all(y > 0)


def test_dataset_repr():
    dataset = DomainAwareDataset()
    dataset.add_domain(np.array([1.0, 2.0]), np.array([1, 2]), "s1")
    dataset.add_domain(np.array([10.0, 20.0, 30.0]), np.array([10, 20, 30]), "s2")
    dataset.add_domain(np.array([10.0, 20.0, 30.0]), np.array([10, 20, 30]), "t1")
    dataset.add_domain(np.array([10.0, 20.0, 30.0]), np.array([10, 20, 30]), "t2")

    assert str(dataset) == "DomainAwareDataset(domains=['s1', 's2', 't1', 't2'])"
    assert repr(dataset) == (
        "DomainAwareDataset(domains=['s1', 's2', 't1', 't2'])\n"
        "Number of domains: 4\nTotal size: 11"
    )

    dataset.add_domain(np.array([10.0, 20.0, 30.0]), np.array([10, 20, 30]), "s3")
    dataset.add_domain(np.array([10.0, 20.0, 30.0]), np.array([10, 20, 30]), "t3")

    assert str(dataset) == (
        "DomainAwareDataset(domains=['s1', 's2', 't1', 't2', 's3', ...])"
    )
    assert repr(dataset) == (
        "DomainAwareDataset(domains=['s1', 's2', 't1', 't2', 's3', ...])\n"
        "Number of domains: 6\nTotal size: 17"
    )


def test_dataset_y_string():
    dataset = DomainAwareDataset()
    dataset.add_domain(np.array([1.0, 2.0]), np.array(["a", "b"]), "s1")
    dataset.add_domain(np.array([10.0, 20.0, 30.0]), np.array(["a", "a", "b"]), "t1")

    X, y, sample_domain = dataset.pack(
        as_sources=["s1"], as_targets=["t1"], mask_target_labels=True
    )

    X, y, sample_domain = dataset.pack(
        as_sources=[], as_targets=["t1"], mask_target_labels=False
    )
