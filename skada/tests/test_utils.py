# Author: Yanis Lalou <yanis.lalou@polytechnique.edu>
#
# License: BSD 3-Clause

import pytest

import numpy as np
from skada.datasets import (
    make_dataset_from_moons_distribution
)

from skada.utils import (
    check_X_y_domain, check_X_domain,
    extract_source_indices, split_source_target_X_y,
    split_source_target_X
)
from skada._utils import source_target_split
from skada._utils import _check_y_masking


def test_check_y_masking_classification():
    y_properly_masked = np.array([-1, 1, 2, -1, 2, 1, 1])
    y_wrongfuly_masked_1 = np.array([-1, -2, 2, -1, 2, 1, 1])
    y_not_masked = np.array([1, 2, 2, 1, 2, 1, 1])

    # Test that no ValueError is raised
    _check_y_masking(y_properly_masked)

    with pytest.raises(ValueError):
        _check_y_masking(y_wrongfuly_masked_1)

    with pytest.raises(ValueError):
        _check_y_masking(y_not_masked)


def test_check_y_masking_regression():
    y_properly_masked = np.array([np.nan, 1, 2.5, -1, np.nan, 0, -1.5])
    y_not_masked = np.array([-1, -2, 2.5, -1, 2, 0, 1])

    # Test that no ValueError is raised
    _check_y_masking(y_properly_masked)

    with pytest.raises(ValueError):
        _check_y_masking(y_not_masked)


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
    check_X_y_domain(X, y, sample_domain=sample_domain)

    with pytest.raises(ValueError):
        check_X_y_domain(X, y, sample_domain=None, allow_auto_sample_domain=False)


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
    check_X_domain(X, sample_domain=sample_domain)

    with pytest.raises(ValueError):
        check_X_domain(X, sample_domain=None, allow_auto_sample_domain=False)


def test_source_target_split():
    X, y, sample_domain = make_dataset_from_moons_distribution(
        pos_source=0.1,
        pos_target=0.9,
        n_samples_source=50,
        n_samples_target=20,
        random_state=0,
        return_X_y=True,
    )

    # Test that no ValueError is raised
    source_target_split(X, y, sample_domain=sample_domain)

    with pytest.raises(ValueError):
        source_target_split(X, y, sample_domain=None, allow_auto_sample_domain=False)


def test_check_X_y_allow_exceptions():
    X, y, sample_domain = make_dataset_from_moons_distribution(
        pos_source=0.1,
        pos_target=0.9,
        n_samples_source=50,
        n_samples_target=20,
        random_state=0,
        return_X_y=True,
    )

    # Generate a random_sample_domain of size len(y)
    # with random integers between -5 and 5 (excluding 0)
    random_sample_domain = np.random.choice(
        np.concatenate((np.arange(-5, 0), np.arange(1, 6))), size=len(y)
    )
    allow_source = False
    allow_target = False
    allow_multi_source = False
    allow_multi_target = False

    positive_numbers = random_sample_domain[random_sample_domain > 0]
    negative_numbers = random_sample_domain[random_sample_domain < 0]
    # Count unique positive numbers
    n_sources = len(np.unique(positive_numbers))
    n_targets = len(np.unique(negative_numbers))

    with pytest.raises(
        ValueError,
        match=(
            f"Number of sources provided is {n_sources} "
            f"and 'allow_source' is set to {allow_source}"
        )
    ):
        check_X_y_domain(
            X, y, sample_domain=random_sample_domain,
            allow_auto_sample_domain=False, allow_source=allow_source
        )

    with pytest.raises(
        ValueError,
        match=(
            f"Number of targets provided is {n_targets} "
            f"and 'allow_target' is set to {allow_target}"
            )
    ):
        check_X_y_domain(
            X, y, sample_domain=random_sample_domain,
            allow_auto_sample_domain=False, allow_target=allow_target
        )

    with pytest.raises(
        ValueError,
        match=(
            f"Number of sources provided is {n_sources} "
            f"and 'allow_multi_source' is set to {allow_multi_source}"
        )
    ):
        check_X_y_domain(
            X, y, sample_domain=random_sample_domain,
            allow_auto_sample_domain=False, allow_multi_source=allow_multi_source
        )

    with pytest.raises(
        ValueError,
        match=(
            f"Number of targets provided is {n_targets} "
            f"and 'allow_multi_target' is set to {allow_multi_target}"
        )
    ):
        check_X_y_domain(
            X, y, sample_domain=random_sample_domain,
            allow_auto_sample_domain=False, allow_multi_target=allow_multi_target
        )


def test_check_X_allow_exceptions():
    X, y, sample_domain = make_dataset_from_moons_distribution(
        pos_source=0.1,
        pos_target=0.9,
        n_samples_source=50,
        n_samples_target=20,
        random_state=0,
        return_X_y=True,
    )

    # Generate a random_sample_domain of size len(y)
    # with random integers between -5 and 5 (excluding 0)
    random_sample_domain = np.random.choice(
        np.concatenate((np.arange(-5, 0), np.arange(1, 6))), size=len(y)
    )
    allow_source = False
    allow_target = False
    allow_multi_source = False
    allow_multi_target = False

    positive_numbers = random_sample_domain[random_sample_domain > 0]
    negative_numbers = random_sample_domain[random_sample_domain < 0]

    # Count unique positive numbers
    n_sources = len(np.unique(positive_numbers))
    n_targets = len(np.unique(negative_numbers))

    with pytest.raises(
        ValueError,
        match=(
            f"Number of sources provided is {n_sources} "
            f"and 'allow_source' is set to {allow_source}"
        )
    ):
        check_X_domain(
            X, sample_domain=random_sample_domain,
            allow_auto_sample_domain=False, allow_source=allow_source
        )

    with pytest.raises(
        ValueError,
        match=(
            f"Number of targets provided is {n_targets} "
            f"and 'allow_target' is set to {allow_target}"
            )
    ):
        check_X_domain(
            X, sample_domain=random_sample_domain,
            allow_auto_sample_domain=False, allow_target=allow_target
        )

    with pytest.raises(
        ValueError,
        match=(
            f"Number of sources provided is {n_sources} "
            f"and 'allow_multi_source' is set to {allow_multi_source}"
        )
    ):
        check_X_domain(
            X, sample_domain=random_sample_domain,
            allow_auto_sample_domain=False, allow_multi_source=allow_multi_source
        )

    with pytest.raises(
        ValueError,
        match=(
            f"Number of targets provided is {n_targets} "
            f"and 'allow_multi_target' is set to {allow_multi_target}"
        )
    ):
        check_X_domain(
            X, sample_domain=random_sample_domain,
            allow_auto_sample_domain=False, allow_multi_target=allow_multi_target
        )

def test_extract_source_indices():
    n_samples_source = 50
    n_samples_target = 20
    X, y, sample_domain = make_dataset_from_moons_distribution(
        pos_source=0.1,
        pos_target=0.9,
        n_samples_source=n_samples_source,
        n_samples_target=n_samples_target,
        random_state=0,
        return_X_y=True,
    )
    source_idx = extract_source_indices(sample_domain)

    assert len(source_idx) == (len(sample_domain)), "source_idx shape mismatch"
    assert np.sum(source_idx) == 2 * n_samples_source, "source_idx sum mismatch"
    assert np.sum(~source_idx) == 2 * n_samples_target, "target_idx sum mismatch"


def test_split_source_target_X_y():
    n_samples_source = 50
    n_samples_target = 20
    X, y, sample_domain = make_dataset_from_moons_distribution(
        pos_source=0.1,
        pos_target=0.9,
        n_samples_source=n_samples_source,
        n_samples_target=n_samples_target,
        random_state=0,
        return_X_y=True,
    )
    X_source, y_source, X_target, y_target = split_source_target_X_y(
        X, y, sample_domain
    )

    assert len(X_source) == 2 * n_samples_source, "X_source shape mismatch"
    assert len(y_source) == 2 * n_samples_source, "y_source shape mismatch"
    assert len(X_target) == 2 * n_samples_target, "X_target shape mismatch"
    assert len(y_target) == 2 * n_samples_target, "y_target shape mismatch"


def split_source_target_X():
    n_samples_source = 50
    n_samples_target = 20
    X, y, sample_domain = make_dataset_from_moons_distribution(
        pos_source=0.1,
        pos_target=0.9,
        n_samples_source=n_samples_source,
        n_samples_target=n_samples_target,
        random_state=0,
        return_X_y=True,
    )
    X_source, X_target = split_source_target_X(X, sample_domain)

    assert len(X_source) == 2 * n_samples_source, "X_source shape mismatch"
    assert len(X_target) == 2 * n_samples_target, "X_target shape mismatch"