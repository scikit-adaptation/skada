import pytest

import numpy as np
from numpy.testing import assert_almost_equal

from skada.datasets import make_dataset_from_moons_distribution
from skada.datasets import make_shifted_blobs
from skada.datasets import make_shifted_datasets
from skada.datasets import make_variable_frequency_dataset


def test_make_dataset_from_moons_distribution():
    X_source, y_source, X_target, y_target = make_dataset_from_moons_distribution(
        pos_source=0.1,
        pos_target=0.9,
        n_samples_source=50,
        n_samples_target=20,
        random_state=0
    )

    assert X_source.shape == (2 * 50, 2), "X source shape mismatch"
    assert y_source.shape == (2 * 50,), "y source shape mismatch"
    assert np.unique(y_source).shape == (2,), "Unexpected number of cluster"
    assert X_target.shape == (2 * 20, 2), "X target shape mismatch"
    assert y_target.shape == (2 * 20,), "y target shape mismatch"
    assert np.unique(y_target).shape == (2,), "Unexpected number of cluster"

    # Test for multisource and multitarget
    X_source, y_source, X_target, y_target = make_dataset_from_moons_distribution(
        pos_source=[0.1, 0.2, 0.3],
        pos_target=[0.8, 0.9],
        n_samples_source=50,
        n_samples_target=20,
        random_state=0
    )

    assert X_source.shape == (3, 2 * 50, 2), "X source shape mismatch"
    assert y_source.shape == (3, 2 * 50,), "y source shape mismatch"
    assert np.unique(y_source).shape == (2,), "Unexpected number of cluster"
    assert X_target.shape == (2, 2 * 20, 2), "X target shape mismatch"
    assert y_target.shape == (2, 2 * 20,), "y target shape mismatch"
    assert np.unique(y_target).shape == (2,), "Unexpected number of cluster"


def test_make_shifted_blobs():
    cluster_stds = np.array([0.05, 0.2, 0.4])
    cluster_centers = np.array([[0.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    X_source, y_source, X_target, y_target = make_shifted_blobs(
        n_samples=50,
        n_features=2,
        shift=0.10,
        noise=None,
        centers=cluster_centers,
        cluster_std=cluster_stds,
        random_state=None,
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
    X_source, y_source, X_target, y_target = make_shifted_datasets(
        n_samples_source=10,
        n_samples_target=10,
        shift=shift,
        noise=None,
        label='binary'
    )

    assert X_source.shape == (10 * 8, 2), "X source shape mismatch"
    assert y_source.shape == (10 * 8,), "y source shape mismatch"
    assert np.unique(y_source).shape == (2,), "Unexpected number of cluster"
    assert X_target.shape == (10 * 8, 2), "X target shape mismatch"
    assert y_target.shape == (10 * 8,), "y target shape mismatch"
    assert np.unique(y_target).shape == (2,), "Unexpected number of cluster"

    # test for multisource
    X_source, y_source, X_target, y_target = make_shifted_datasets(
        n_samples_source=10,
        n_samples_target=10,
        shift=shift,
        noise=None,
        label='multiclass'
    )

    assert X_source.shape == (10 * 8, 2), "X source shape mismatch"
    assert y_source.shape == (10 * 8,), "y source shape mismatch"
    assert np.unique(y_source).shape == (5,), "Unexpected number of cluster"
    assert X_target.shape == (10 * 8, 2), "X target shape mismatch"
    assert y_target.shape == (10 * 8,), "y target shape mismatch"
    assert np.unique(y_target).shape[0] <= 5, "Unexpected number of cluster"


def test_make_subspace_datasets():
    X_source, y_source, X_target, y_target = make_shifted_datasets(
        n_samples_source=10,
        n_samples_target=10,
        shift="subspace",
        noise=None,
        label='binary'
    )

    assert X_source.shape == (10 * 4, 2), "X source shape mismatch"
    assert y_source.shape == (10 * 4,), "y source shape mismatch"
    assert np.unique(y_source).shape == (2,), "Unexpected number of cluster"
    assert X_target.shape == (10 * 4, 2), "X target shape mismatch"
    assert y_target.shape == (10 * 4,), "y target shape mismatch"
    assert np.unique(y_target).shape == (2,), "Unexpected number of cluster"


def test_make_variable_frequency_dataset():
    X_source, y_source, X_target, y_target = make_variable_frequency_dataset(
        n_samples_source=10,
        n_samples_target=5,
        n_channels=1,
        n_classes=3,
        delta_f=1,
        band_size=1,
        noise=None,
        random_state=None
    )

    assert X_source.shape == (3 * 10, 1, 3000), "X source shape mismatch"
    assert y_source.shape == (3 * 10,), "y source shape mismatch"
    assert np.unique(y_source).shape == (3,), "Unexpected number of cluster"
    assert X_target.shape == (3 * 5, 1, 3000), "X target shape mismatch"
    assert y_target.shape == (3 * 5,), "y target shape mismatch"
    assert np.unique(y_target).shape == (3,), "Unexpected number of cluster"
