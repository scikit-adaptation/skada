# Author: Theo Gnassounou <theo.gnassounou@inria.fr>
#         Remi Flamary <remi.flamary@polytechnique.edu>
#         Oleksii Kachaiev <kachayev@gmail.com>
#         Bueno Ruben <ruben.bueno@polytechnique.edu>
#
# License: BSD 3-Clause

import numbers

import numpy as np
import pytest
from numpy.testing import assert_almost_equal

from skada.datasets import (
    DomainAwareDataset,
    make_dataset_from_moons_distribution,
    make_shifted_blobs,
    make_shifted_datasets,
    make_variable_frequency_dataset,
)
from skada.utils import check_X_y_domain, source_target_split

# mark all the test with the marker dataset
pytestmark = pytest.mark.dataset


@pytest.mark.parametrize(
    "pos_source, pos_target, noise",
    [
        (pos_source, pos_target, noise)
        for pos_source in [0.1, [0.1, 0.3]]
        for pos_target in [0.9, [0.7, 0.9]]
        for noise in [None, 1, [0, 1]]
    ],
)
def test_make_dataset_from_moons_distribution(pos_source, pos_target, noise):
    X, y, sample_domain = make_dataset_from_moons_distribution(
        n_samples_source=50,
        n_samples_target=20,
        noise=noise,
        pos_source=pos_source,
        pos_target=pos_target,
        random_state=0,
        return_X_y=True,
        return_dataset=False,
    )
    X, y, sample_domain = check_X_y_domain(X, y, sample_domain)
    X_source, X_target, y_source, y_target = source_target_split(
        X, y, sample_domain=sample_domain
    )

    pos_source_size = 1 if isinstance(pos_source, numbers.Real) else len(pos_source)
    pos_target_size = 1 if isinstance(pos_target, numbers.Real) else len(pos_target)
    assert X_source.shape == (pos_source_size * 2 * 50, 2), "X source shape mismatch"
    assert y_source.shape == (pos_source_size * 2 * 50,), "y source shape mismatch"
    assert np.unique(y_source).shape == (2,), "Unexpected number of cluster"
    assert X_target.shape == (pos_target_size * 2 * 20, 2), "X target shape mismatch"
    assert y_target.shape == (pos_target_size * 2 * 20,), "y target shape mismatch"
    assert np.unique(y_target).shape == (2,), "Unexpected number of cluster"

    dataset = make_dataset_from_moons_distribution(
        n_samples_source=50,
        n_samples_target=20,
        noise=noise,
        pos_source=0.1,
        pos_target=0.9,
        random_state=0,
        return_X_y=True,
        return_dataset=True,
    )
    assert isinstance(
        dataset, DomainAwareDataset
    ), "return_dataset=True but a dataset has not been returned"


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
        return_dataset=False,
    )
    X, y, sample_domain = check_X_y_domain(X, y, sample_domain)
    X_source, X_target, y_source, y_target = source_target_split(
        X, y, sample_domain=sample_domain
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

    dataset = make_dataset_from_moons_distribution(
        pos_source=[0.1, 0.2, 0.3],
        pos_target=[0.8, 0.9],
        n_samples_source=50,
        n_samples_target=20,
        random_state=0,
        return_X_y=True,
        return_dataset=True,
    )
    assert isinstance(
        dataset, DomainAwareDataset
    ), "return_dataset=True but a dataset has not been returned"


@pytest.mark.parametrize(
    "noise",
    [None, 1, [0, 1]],
)
def test_make_shifted_blobs(noise):
    cluster_stds = np.array([0.05, 0.2, 0.4])
    cluster_centers = np.array([[0.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    X, y, sample_domain = make_shifted_blobs(
        n_samples=50,
        n_features=2,
        shift=0.10,
        noise=noise,
        centers=cluster_centers,
        cluster_std=cluster_stds,
        random_state=None,
        return_dataset=False,
    )
    X, y, sample_domain = check_X_y_domain(X, y, sample_domain)
    X_source, X_target, y_source, y_target = source_target_split(
        X, y, sample_domain=sample_domain
    )

    assert X_source.shape == (50, 2), "X source shape mismatch"
    assert y_source.shape == (50,), "y source shape mismatch"
    assert np.unique(y_source).shape == (3,), "Unexpected number of cluster"
    assert X_target.shape == (50, 2), "X target shape mismatch"
    assert y_target.shape == (50,), "y target shape mismatch"
    assert np.unique(y_target).shape == (3,), "Unexpected number of cluster"
    if noise is None:
        assert_almost_equal((X_target - X_source), 0.10, 1, "Unexpected std")
    # There are no tests concerning std when there is noise

    dataset = make_shifted_blobs(
        n_samples=50,
        n_features=2,
        shift=0.10,
        noise=noise,
        centers=cluster_centers,
        cluster_std=cluster_stds,
        random_state=None,
        return_dataset=True,
    )
    assert isinstance(
        dataset, DomainAwareDataset
    ), "return_dataset=True but a dataset has not been returned"


@pytest.mark.parametrize(
    "shift, noise",
    [
        (shift, noise)
        for shift in [
            "covariate_shift",
            "target_shift",
            "conditional_shift",
            "subspace",
        ]
        for noise in [None, 1, [0, 1]]
    ],
)
def test_make_shifted_datasets(shift, noise):
    X, y, sample_domain = make_shifted_datasets(
        n_samples_source=10,
        n_samples_target=10,
        shift=shift,
        noise=noise,
        label="binary",
        return_dataset=False,
    )
    X, y, sample_domain = check_X_y_domain(X, y, sample_domain)
    X_source, X_target, y_source, y_target = source_target_split(
        X, y, sample_domain=sample_domain
    )
    if shift == "subspace":
        assert X_source.shape == (10 * 8 // 2, 2), "X source shape mismatch"
        assert y_source.shape == (10 * 8 // 2,), "y source shape mismatch"
    else:
        assert X_source.shape == (10 * 8, 2), "X source shape mismatch"
        assert y_source.shape == (10 * 8,), "y source shape mismatch"
    assert np.unique(y_source).shape == (2,), "Unexpected number of cluster"
    if shift == "subspace":
        assert X_target.shape == (10 * 8 // 2, 2), "X target shape mismatch"
        assert y_target.shape == (10 * 8 // 2,), "y target shape mismatch"
    else:
        assert X_target.shape == (10 * 8, 2), "X target shape mismatch"
        assert y_target.shape == (10 * 8,), "y target shape mismatch"
    assert np.unique(y_target).shape == (2,), "Unexpected number of cluster"

    dataset = make_shifted_datasets(
        n_samples_source=10,
        n_samples_target=10,
        shift=shift,
        noise=noise,
        label="binary",
        return_dataset=True,
    )
    assert isinstance(
        dataset, DomainAwareDataset
    ), "return_dataset=True but a dataset has not been returned"


@pytest.mark.parametrize(
    "shift",
    ["covariate_shift", "target_shift", "conditional_shift", "subspace"],
)
def test_make_multi_source_shifted_datasets(shift):
    # test for multi-source
    X, y, sample_domain = make_shifted_datasets(
        n_samples_source=10,
        n_samples_target=10,
        shift=shift,
        noise=None,
        label="multiclass",
        return_dataset=False,
    )
    X, y, sample_domain = check_X_y_domain(X, y, sample_domain)
    X_source, X_target, y_source, y_target = source_target_split(
        X, y, sample_domain=sample_domain
    )

    if shift == "subspace":
        assert X_source.shape == (10 * 8 // 2, 2), "X source shape mismatch"
        assert y_source.shape == (10 * 8 // 2,), "y source shape mismatch"
    else:
        assert X_source.shape == (10 * 8, 2), "X source shape mismatch"
        assert y_source.shape == (10 * 8,), "y source shape mismatch"
    assert np.unique(y_source).shape == (5,), "Unexpected number of cluster"
    if shift == "subspace":
        assert X_target.shape == (10 * 8 // 2, 2), "X target shape mismatch"
        assert y_target.shape == (10 * 8 // 2,), "y target shape mismatch"
    else:
        assert X_target.shape == (10 * 8, 2), "X target shape mismatch"
        assert y_target.shape == (10 * 8,), "y target shape mismatch"
    assert np.unique(y_target).shape[0] <= 5, "Unexpected number of cluster"

    dataset = make_shifted_datasets(
        n_samples_source=10,
        n_samples_target=10,
        shift=shift,
        noise=None,
        label="multiclass",
        return_dataset=True,
    )
    assert isinstance(
        dataset, DomainAwareDataset
    ), "return_dataset=True but a dataset has not been returned"


@pytest.mark.parametrize(
    "shift",
    ["covariate_shift", "target_shift", "conditional_shift", "subspace"],
)
def test_make_shifted_datasets_regression(shift):
    X, y, sample_domain = make_shifted_datasets(
        n_samples_source=10,
        n_samples_target=10,
        shift=shift,
        noise=None,
        label="regression",
    )
    X, y, sample_domain = check_X_y_domain(X, y, sample_domain)
    X_source, X_target, y_source, y_target = source_target_split(
        X, y, sample_domain=sample_domain
    )

    if shift == "subspace":
        assert X_source.shape == (10 * 8 // 2, 2), "X source shape mismatch"
        assert y_source.shape == (10 * 8 // 2,), "y source shape mismatch"
    else:
        assert X_source.shape == (10 * 8, 2), "X source shape mismatch"
        assert y_source.shape == (10 * 8,), "y source shape mismatch"
    if shift == "subspace":
        assert X_target.shape == (10 * 8 // 2, 2), "X target shape mismatch"
        assert y_target.shape == (10 * 8 // 2,), "y target shape mismatch"
    else:
        assert X_target.shape == (10 * 8, 2), "X target shape mismatch"
        assert y_target.shape == (10 * 8,), "y target shape mismatch"

    dataset = make_shifted_datasets(
        n_samples_source=10,
        n_samples_target=10,
        shift=shift,
        noise=None,
        label="regression",
        return_dataset=True,
    )
    assert isinstance(
        dataset, DomainAwareDataset
    ), "return_dataset=True but a dataset has not been returned"


def test_make_subspace_datasets():
    X, y, sample_domain = make_shifted_datasets(
        n_samples_source=10,
        n_samples_target=10,
        shift="subspace",
        noise=None,
        label="binary",
        return_dataset=False,
    )
    X, y, sample_domain = check_X_y_domain(X, y, sample_domain)
    X_source, X_target, y_source, y_target = source_target_split(
        X, y, sample_domain=sample_domain
    )

    assert X_source.shape == (10 * 4, 2), "X source shape mismatch"
    assert y_source.shape == (10 * 4,), "y source shape mismatch"
    assert np.unique(y_source).shape == (2,), "Unexpected number of cluster"
    assert X_target.shape == (10 * 4, 2), "X target shape mismatch"
    assert y_target.shape == (10 * 4,), "y target shape mismatch"
    assert np.unique(y_target).shape == (2,), "Unexpected number of cluster"

    dataset = make_shifted_datasets(
        n_samples_source=10,
        n_samples_target=10,
        shift="subspace",
        noise=None,
        label="binary",
        return_dataset=True,
    )
    assert isinstance(
        dataset, DomainAwareDataset
    ), "return_dataset=True but a dataset has not been returned"


@pytest.mark.parametrize(
    "noise",
    [None, 1, [0, 1]],
)
def test_make_variable_frequency_dataset(noise):
    X, y, sample_domain = make_variable_frequency_dataset(
        n_samples_source=10,
        n_samples_target=5,
        n_channels=1,
        n_classes=3,
        delta_f=1,
        band_size=1,
        noise=noise,
        random_state=None,
        return_dataset=False,
    )
    X, y, sample_domain = check_X_y_domain(X, y, sample_domain, allow_nd=True)
    X_source, X_target, y_source, y_target = source_target_split(
        X, y, sample_domain=sample_domain
    )

    assert X_source.shape == (3 * 10, 1, 3000), "X source shape mismatch"
    assert y_source.shape == (3 * 10,), "y source shape mismatch"
    assert np.unique(y_source).shape == (3,), "Unexpected number of cluster"
    assert X_target.shape == (3 * 5, 1, 3000), "X target shape mismatch"
    assert y_target.shape == (3 * 5,), "y target shape mismatch"
    assert np.unique(y_target).shape == (3,), "Unexpected number of cluster"

    dataset = make_variable_frequency_dataset(
        n_samples_source=10,
        n_samples_target=5,
        n_channels=1,
        n_classes=3,
        delta_f=1,
        band_size=1,
        noise=noise,
        random_state=None,
        return_dataset=True,
    )
    assert isinstance(
        dataset, DomainAwareDataset
    ), "return_dataset=True but a dataset has not been returned"


@pytest.mark.parametrize(
    "label",
    ["binary", "multiclass", "regression"],
)
def test_invalid_shift_value(label):
    invalid_shift = "invalid_shift_value"

    with pytest.raises(ValueError):
        make_shifted_datasets(
            n_samples_source=10,
            n_samples_target=10,
            shift=invalid_shift,
            noise=None,
            label=label,
        )


@pytest.mark.parametrize(
    "shift",
    ["covariate_shift", "subspace"],
)
def test_invalid_label_value(shift):
    invalid_label = "invalid_label_value"

    with pytest.raises(ValueError):
        make_shifted_datasets(
            n_samples_source=10,
            n_samples_target=10,
            noise=None,
            label=invalid_label,
            shift=shift,
        )
