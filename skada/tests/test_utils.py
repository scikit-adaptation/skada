# Author: Yanis Lalou <yanis.lalou@polytechnique.edu>
#
# License: BSD 3-Clause

import pytest

import numpy as np

from skada.datasets import (
    make_dataset_from_moons_distribution
)
from skada.utils import (
    check_sample_domain,
    check_X_y_domain,
    check_X_domain,
    extract_source_indices,
    source_target_split,
    extract_domains_indices,
    source_target_merge
)
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
    y_wrong_dim_not_masked = np.array([[-1, 2], [1, 2], [1, 2]])
    y_wrong_dim_properly_masked = np.array([[-1, 2], [1, 2], [np.nan, np.nan]])

    with pytest.raises(ValueError):
        _check_y_masking(y_wrong_dim_not_masked)

    with pytest.raises(ValueError):
        _check_y_masking(y_wrong_dim_properly_masked)


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

    # Test that no ValueError is raised
    _, _ = source_target_split(X, sample_domain=sample_domain)

    X_source, X_target, y_source, y_target = source_target_split(
        X, y, sample_domain=sample_domain
    )

    assert X_source.shape == (2 * n_samples_source, 2), "X_source shape mismatch"
    assert y_source.shape == (2 * n_samples_source, ), "y_source shape mismatch"
    assert X_target.shape == (2 * n_samples_target, 2), "X_target shape mismatch"
    assert y_target.shape == (2 * n_samples_target, ), "y_target shape mismatch"

    with pytest.raises(IndexError):
        source_target_split(X, y[:-2], sample_domain=sample_domain)

    X_source, X_target, weights_source, weights_target = source_target_split(
        X, None, sample_domain=sample_domain)

    assert X_source.shape == (2 * n_samples_source, 2), "X_source shape mismatch"
    assert X_target.shape == (2 * n_samples_target, 2), "X_target shape mismatch"
    assert weights_source is None, "weights_source should be None"
    assert weights_target is None, "weights_target should be None"


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

    # Count number of sources and targets
    n_sources = len(np.unique(positive_numbers))
    n_targets = len(np.unique(negative_numbers))

    # Adjust targets to avoid sample_domain checker raise issues
    random_sample_domain[random_sample_domain < 0] -= (n_sources + 1)

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

    # Count number of sources and targets
    n_sources = len(np.unique(positive_numbers))
    n_targets = len(np.unique(negative_numbers))

    # Adjust targets to avoid sample_domain checker raise issues
    random_sample_domain[random_sample_domain < 0] -= (n_sources + 1)

    with pytest.raises(
        ValueError,
        match=(
            f"Number of sources provided is {n_sources} "
            f"and 'allow_source' is set to {allow_source}"
        )
    ):
        check_X_domain(
            X,
            sample_domain=random_sample_domain,
            allow_auto_sample_domain=False,
            allow_source=allow_source
        )

    with pytest.raises(
        ValueError,
        match=(
            f"Number of targets provided is {n_targets} "
            f"and 'allow_target' is set to {allow_target}"
        )
    ):
        check_X_domain(
            X,
            sample_domain=random_sample_domain,
            allow_auto_sample_domain=False,
            allow_target=allow_target
        )

    with pytest.raises(
        ValueError,
        match=(
            f"Number of sources provided is {n_sources} "
            f"and 'allow_multi_source' is set to {allow_multi_source}"
        )
    ):
        check_X_domain(
            X,
            sample_domain=random_sample_domain,
            allow_auto_sample_domain=False,
            allow_multi_source=allow_multi_source
        )

    with pytest.raises(
        ValueError,
        match=(
            f"Number of targets provided is {n_targets} "
            f"and 'allow_multi_target' is set to {allow_multi_target}"
        )
    ):
        check_X_domain(
            X,
            sample_domain=random_sample_domain,
            allow_auto_sample_domain=False,
            allow_multi_target=allow_multi_target
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


def test_extract_domains_indices():
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

    domain_idx_dict = extract_domains_indices(sample_domain)
    domain_source_idx_dict, domain_target_idx_dict = extract_domains_indices(
            sample_domain,
            split_source_target=True
        )

    assert (
        sum(len(values) for values in domain_idx_dict.values()) ==
        len(sample_domain)
    ), "domain_idx_dict shape mismatch"

    assert (
        sum(len(values) for values in domain_source_idx_dict.values()) ==
        2 * n_samples_source
    ), "domain_source_idx_dict shape mismatch"

    assert (
        sum(len(values) for values in domain_target_idx_dict.values()) ==
        2 * n_samples_target
    ), "domain_target_idx_dict shape mismatch"


def test_source_target_merge():
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

    X_source, X_target, y_source, y_target = source_target_split(
        X, y, sample_domain=sample_domain
    )

    # Test that no Error is raised for a 2D array
    samples, _ = source_target_merge(X_source, X_target, sample_domain=sample_domain)
    assert (
        samples.shape[0] == X_source.shape[0] + X_target.shape[0]
    ), "target_samples shape mismatch"

    # Test that no Error is raised for a 1D array
    labels, _ = source_target_merge(y_source, y_target, sample_domain=sample_domain)
    assert (
        labels.shape[0] == y_source.shape[0] + y_target.shape[0]
    ), "labels shape mismatch"

    # Test with empty samples
    with pytest.raises(ValueError,
                       match="Only one array can be None or empty in each pair"
                       ):
        _ = source_target_merge(np.array([]), np.array([]), sample_domain=np.array([]))

    # Test that no Error with one empty sample
    _ = source_target_merge(
        X_source,
        np.array([]),
        sample_domain=np.array([1]*X_source.shape[0])
    )
    _ = source_target_merge(
        np.array([]),
        X_target,
        sample_domain=np.array([-1]*X_target.shape[0])
    )

    # Test consistent length
    with pytest.raises(
        ValueError,
        match="Inconsistent number of samples in source-target arrays "
              "and the number infered in the sample_domain"
    ):
        _ = source_target_merge(X_source[0], X_target[1], sample_domain=sample_domain)

    # Test no sample domain
    _ = source_target_merge(X_source, X_target, sample_domain=None)

    # Test odd number of array
    with pytest.raises(
        ValueError,
        match="Even number of arrays required as input"
    ):
        _ = source_target_merge(
            X_source,
            X_target,
            y_source,
            sample_domain=sample_domain
        )

    # Test >2 arrays
    _ = source_target_merge(
        X_source,
        X_target,
        y_source,
        y_target,
        sample_domain=sample_domain
    )

    # Test one array
    with pytest.raises(
        ValueError,
        match="At least two array required as input"
    ):
        _ = source_target_merge(X_source, sample_domain=sample_domain)

    # Test y_target = None + Inconsistent number of samples in source-target
    with pytest.raises(
        ValueError,
        match="Inconsistent number of samples in source-target arrays "
              "and the number infered in the sample_domain"
    ):
        _ = source_target_merge(
            X_source,
            X_target,
            np.ones_like(sample_domain),
            None,
            sample_domain=sample_domain
        )

    # Test y_target = None + Consistent number of samples in source-target
    _ = source_target_merge(
        X_source,
        X_target,
        y_source,
        None,
        sample_domain=sample_domain
    )

    # Test 2 None in a pair of arrays
    with pytest.raises(
        ValueError,
        match="Only one array can be None or empty in each pair"
    ):
        _ = source_target_merge(None, None, sample_domain=sample_domain)

    # Test 1 None in 2 pair of arrays
    _ = source_target_merge(X_source, None, y_source, None, sample_domain=sample_domain)

    # Test inconsistent number of features
    with pytest.raises(
        ValueError,
        match="Inconsistent number of features in source-target arrays"
    ):
        _ = source_target_merge(
            X_source[:, :-1],
            X_target,
            sample_domain=sample_domain
        )


def test_check_sample_domain_lodo():
    # same domain label added twice (as a source and as a target)
    check_sample_domain(np.array([1, 1, 2, 2, 2, -1, -1, -2, -2, -2]))

    # 'lodo' packing but counters are off
    with pytest.raises(ValueError):
        check_sample_domain(np.array([1, 1, -1, 2, 2, 2, -2, -2]))
