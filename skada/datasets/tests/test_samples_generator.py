# Author: Theo Gnassounou <theo.gnassounou@inria.fr>
#         Remi Flamary <remi.flamary@polytechnique.edu>
#         Oleksii Kachaiev <kachayev@gmail.com>
#
# License: BSD 3-Clause

import pytest

import numpy as np
from numpy.testing import assert_almost_equal

from skada.datasets import (
    make_dataset_from_moons_distribution,
    make_shifted_blobs,
    make_shifted_datasets,
    make_variable_frequency_dataset
)
from skada._utils import check_X_y_domain


def test_make_dataset_from_moons_distribution():
    X, y, sample_domain = make_dataset_from_moons_distribution(
        pos_source=0.1,
        pos_target=0.9,
        n_samples_source=50,
        n_samples_target=20,
        random_state=0,
        return_X_y=True,
    )
    X_source, y_source, X_target, y_target = check_X_y_domain(
        X,
        y=y,
        sample_domain=sample_domain,
        return_joint=False,
    )

    assert X_source.shape == (2 * 50, 2), "X source shape mismatch"
    assert y_source.shape == (2 * 50,), "y source shape mismatch"
    assert np.unique(y_source).shape == (2,), "Unexpected number of cluster"
    assert X_target.shape == (2 * 20, 2), "X target shape mismatch"
    assert y_target.shape == (2 * 20,), "y target shape mismatch"
    assert np.unique(y_target).shape == (2,), "Unexpected number of cluster"


# xxx(okachaiev): find out why this one doesn't work
def test_make_dataset_from_multi_moons_distribution():
    # Test for multi source and multi target
    X, y, sample_domain = make_dataset_from_moons_distribution(
        pos_source=[0.1, 0.2, 0.3],
        pos_target=[0.8, 0.9],
        n_samples_source=50,
        n_samples_target=20,
        random_state=0,
        return_X_y=True,
    )
    X_source, y_source, X_target, y_target = check_X_y_domain(
        X,
        y=y,
        sample_domain=sample_domain,
        return_joint=False,
    )

    assert X_source.shape == (3 * 2 * 50, 2), "X source shape mismatch"
    assert y_source.shape == (3 * 2 * 50,), "y source shape mismatch"
    assert np.unique(y_source).shape == (2,), "Unexpected number of cluster"
    assert np.unique(sample_domain[sample_domain > 0]).shape == (
        3,
    ), "Unexpected number of source and target"
    assert X_target.shape == (2 * 2 * 20, 2), "X target shape mismatch"
    assert y_target.shape == (2 * 2 * 20,), "y target shape mismatch"
    assert np.unique(y_target).shape == (2,), "Unexpected number of cluster"
    assert np.unique(sample_domain[sample_domain < 0]).shape == (
        2,
    ), "Unexpected number of source and target"
    assert np.unique(sample_domain).shape == (
        5,
    ), "Unexpected number of source and target"


def test_make_shifted_blobs():
    cluster_stds = np.array([0.05, 0.2, 0.4])
    cluster_centers = np.array([[0.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    X, y, sample_domain = make_shifted_blobs(
        n_samples=50,
        n_features=2,
        shift=0.10,
        noise=None,
        centers=cluster_centers,
        cluster_std=cluster_stds,
        random_state=None,
    )
    X_source, y_source, X_target, y_target = check_X_y_domain(
        X,
        y=y,
        sample_domain=sample_domain,
        return_joint=False,
    )

    assert X_source.shape == (50, 2), "X source shape mismatch"
    assert y_source.shape == (50,), "y source shape mismatch"
    assert np.unique(y_source).shape == (3,), "Unexpected number of cluster"
    assert X_target.shape == (50, 2), "X target shape mismatch"
    assert y_target.shape == (50,), "y target shape mismatch"
    assert np.unique(y_target).shape == (3,), "Unexpected number of cluster"
    assert_almost_equal((X_target - X_source), 0.10, 1, "Unexpected std")


@pytest.mark.parametrize(
    "shift",
    ["covariate_shift", "target_shift", "concept_drift"],
)
def test_make_shifted_datasets(shift):
    X, y, sample_domain = make_shifted_datasets(
        n_samples_source=10,
        n_samples_target=10,
        shift=shift,
        noise=None,
        label="binary",
    )
    X_source, y_source, X_target, y_target = check_X_y_domain(
        X,
        y=y,
        sample_domain=sample_domain,
        return_joint=False,
    )

    assert X_source.shape == (10 * 8, 2), "X source shape mismatch"
    assert y_source.shape == (10 * 8,), "y source shape mismatch"
    assert np.unique(y_source).shape == (2,), "Unexpected number of cluster"
    assert X_target.shape == (10 * 8, 2), "X target shape mismatch"
    assert y_target.shape == (10 * 8,), "y target shape mismatch"
    assert np.unique(y_target).shape == (2,), "Unexpected number of cluster"


@pytest.mark.parametrize(
    "shift",
    ["covariate_shift", "target_shift", "concept_drift"],
)
def test_make_multi_source_shifted_datasets(shift):
    # test for multi-source
    X, y, sample_domain = make_shifted_datasets(
        n_samples_source=10,
        n_samples_target=10,
        shift=shift,
        noise=None,
        label="multiclass",
    )
    X_source, y_source, X_target, y_target = check_X_y_domain(
        X,
        y=y,
        sample_domain=sample_domain,
        return_joint=False,
    )

    assert X_source.shape == (10 * 8, 2), "X source shape mismatch"
    assert y_source.shape == (10 * 8,), "y source shape mismatch"
    assert np.unique(y_source).shape == (5,), "Unexpected number of cluster"
    assert X_target.shape == (10 * 8, 2), "X target shape mismatch"
    assert y_target.shape == (10 * 8,), "y target shape mismatch"
    assert np.unique(y_target).shape[0] <= 5, "Unexpected number of cluster"


def test_make_subspace_datasets():
    X, y, sample_domain = make_shifted_datasets(
        n_samples_source=10,
        n_samples_target=10,
        shift="subspace",
        noise=None,
        label="binary",
    )
    X_source, y_source, X_target, y_target = check_X_y_domain(
        X,
        y=y,
        sample_domain=sample_domain,
        return_joint=False,
    )

    assert X_source.shape == (10 * 4, 2), "X source shape mismatch"
    assert y_source.shape == (10 * 4,), "y source shape mismatch"
    assert np.unique(y_source).shape == (2,), "Unexpected number of cluster"
    assert X_target.shape == (10 * 4, 2), "X target shape mismatch"
    assert y_target.shape == (10 * 4,), "y target shape mismatch"
    assert np.unique(y_target).shape == (2,), "Unexpected number of cluster"


def test_make_variable_frequency_dataset():
    X, y, sample_domain = make_variable_frequency_dataset(
        n_samples_source=10,
        n_samples_target=5,
        n_channels=1,
        n_classes=3,
        delta_f=1,
        band_size=1,
        noise=None,
        random_state=None
    )
    X_source, y_source, X_target, y_target = check_X_y_domain(
        X,
        y=y,
        sample_domain=sample_domain,
        return_joint=False,
        allow_nd=True,
    )

    assert X_source.shape == (3 * 10, 1, 3000), "X source shape mismatch"
    assert y_source.shape == (3 * 10,), "y source shape mismatch"
    assert np.unique(y_source).shape == (3,), "Unexpected number of cluster"
    assert X_target.shape == (3 * 5, 1, 3000), "X target shape mismatch"
    assert y_target.shape == (3 * 5,), "y target shape mismatch"
    assert np.unique(y_target).shape == (3,), "Unexpected number of cluster"


def test_invalid_shift_value():
    invalid_shift = "invalid_shift_value"

    with pytest.raises(ValueError):
        make_shifted_datasets(
            n_samples_source=10,
            n_samples_target=10,
            shift=invalid_shift,
            noise=None,
            label="binary",
        )


def test_invalid_label_value():
    invalid_label = "invalid_label_value"

    with pytest.raises(ValueError):
        make_shifted_datasets(
            n_samples_source=10,
            n_samples_target=10,
            noise=None,
            label=invalid_label,
        )
