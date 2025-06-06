# Author: Yanis Lalou <yanis.lalou@polytechnique.edu>
#         Antoine Collas <contact@antoinecollas.fr>
#
# License: BSD 3-Clause

import numpy as np
import pytest
from sklearn.utils import check_random_state

from skada._utils import (
    _DEFAULT_MASKED_TARGET_CLASSIFICATION_LABEL,
    _check_y_masking,
    _merge_domain_outputs,
)
from skada.datasets import make_dataset_from_moons_distribution
from skada.utils import (
    check_X_domain,
    check_X_y_domain,
    extract_domains_indices,
    extract_source_indices,
    frank_wolfe,
    per_domain_split,
    qp_solve,
    source_target_merge,
    source_target_split,
    torch_minimize,
)

try:
    import torch
except ImportError:
    torch = False


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


def test_check_X_domain_scalar():
    X, _, sample_domain = make_dataset_from_moons_distribution(
        pos_source=0.1,
        pos_target=0.9,
        n_samples_source=50,
        n_samples_target=20,
        random_state=0,
        return_X_y=True,
    )

    # Test scalar sample_domain
    X_target = X[sample_domain == -2]
    returned_X_target, sample_domain_target = check_X_domain(X_target, sample_domain=-2)

    assert sample_domain_target.shape[0] == X_target.shape[0]
    assert np.array_equal(returned_X_target, X_target)
    assert np.array_equal(sample_domain_target, -2 * np.ones(X_target.shape[0]))


def test_check_X_y_domain_scalar():
    X, y, sample_domain = make_dataset_from_moons_distribution(
        pos_source=0.1,
        pos_target=0.9,
        n_samples_source=50,
        n_samples_target=20,
        random_state=0,
        return_X_y=True,
    )

    # Test scalar sample_domain
    X_target = X[sample_domain == -2]
    y_target = y[sample_domain == -2]
    returned_X_target, returned_y_target, sample_domain_target = check_X_y_domain(
        X_target, y_target, sample_domain=-2
    )

    assert sample_domain_target.shape[0] == X_target.shape[0]
    assert np.array_equal(returned_X_target, X_target)
    assert np.array_equal(returned_y_target, y_target)
    assert np.array_equal(sample_domain_target, -2 * np.ones(X_target.shape[0]))


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
    assert y_source.shape == (2 * n_samples_source,), "y_source shape mismatch"
    assert X_target.shape == (2 * n_samples_target, 2), "X_target shape mismatch"
    assert y_target.shape == (2 * n_samples_target,), "y_target shape mismatch"

    with pytest.raises(IndexError):
        source_target_split(X, y[:-2], sample_domain=sample_domain)

    X_source, X_target, weights_source, weights_target = source_target_split(
        X, None, sample_domain=sample_domain
    )

    assert X_source.shape == (2 * n_samples_source, 2), "X_source shape mismatch"
    assert X_target.shape == (2 * n_samples_target, 2), "X_target shape mismatch"
    assert weights_source is None, "weights_source should be None"
    assert weights_target is None, "weights_target should be None"


def test_per_domain_split():
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

    sources, targets = per_domain_split(X, sample_domain=sample_domain)

    assert len(sources) == 1, "sources length mismatch"
    assert len(targets) == 1, "targets length mismatch"
    assert -2 in targets, "targets should contain -1"
    assert 1 in sources, "sources should contain 1"

    sample_domain_2 = np.arange(X.shape[0])
    sample_domain_2[n_samples_source:] *= -1

    sources, targets = per_domain_split(X, sample_domain=sample_domain_2)

    assert len(sources) == n_samples_source, "sources length mismatch"
    assert len(targets) == X.shape[0] - n_samples_source, "targets length mismatch"


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
    rng = check_random_state(42)
    random_sample_domain = rng.choice(
        np.concatenate((np.arange(-5, 0), np.arange(1, 6))), size=len(y)
    )
    common_idx_sample_domain = rng.choice(np.array([1, -1]), size=len(y))
    allow_source = False
    allow_target = False
    allow_multi_source = False
    allow_multi_target = False
    allow_common_domain_idx = False

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
        ),
    ):
        check_X_y_domain(
            X,
            y,
            sample_domain=random_sample_domain,
            allow_auto_sample_domain=False,
            allow_source=allow_source,
        )

    with pytest.raises(
        ValueError,
        match=(
            f"Number of targets provided is {n_targets} "
            f"and 'allow_target' is set to {allow_target}"
        ),
    ):
        check_X_y_domain(
            X,
            y,
            sample_domain=random_sample_domain,
            allow_auto_sample_domain=False,
            allow_target=allow_target,
        )

    with pytest.raises(
        ValueError,
        match=(
            f"Number of sources provided is {n_sources} "
            f"and 'allow_multi_source' is set to {allow_multi_source}"
        ),
    ):
        check_X_y_domain(
            X,
            y,
            sample_domain=random_sample_domain,
            allow_auto_sample_domain=False,
            allow_multi_source=allow_multi_source,
        )

    with pytest.raises(
        ValueError,
        match=(
            f"Number of targets provided is {n_targets} "
            f"and 'allow_multi_target' is set to {allow_multi_target}"
        ),
    ):
        check_X_y_domain(
            X,
            y,
            sample_domain=random_sample_domain,
            allow_auto_sample_domain=False,
            allow_multi_target=allow_multi_target,
        )
    with pytest.raises(
        ValueError,
        match=(
            "Domain labels should be unique: the same domain "
            "index should not be used both for source and target"
        ),
    ):
        check_X_domain(
            X,
            sample_domain=common_idx_sample_domain,
            allow_common_domain_idx=allow_common_domain_idx,
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
    rng = check_random_state(42)
    random_sample_domain = rng.choice(
        np.concatenate((np.arange(-5, 0), np.arange(1, 6))), size=len(y)
    )
    common_idx_sample_domain = rng.choice(np.array([1, -1]), size=len(y))
    allow_source = False
    allow_target = False
    allow_multi_source = False
    allow_multi_target = False
    allow_common_domain_idx = False

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
        ),
    ):
        check_X_domain(
            X,
            sample_domain=random_sample_domain,
            allow_auto_sample_domain=False,
            allow_source=allow_source,
        )

    with pytest.raises(
        ValueError,
        match=(
            f"Number of targets provided is {n_targets} "
            f"and 'allow_target' is set to {allow_target}"
        ),
    ):
        check_X_domain(
            X,
            sample_domain=random_sample_domain,
            allow_auto_sample_domain=False,
            allow_target=allow_target,
        )

    with pytest.raises(
        ValueError,
        match=(
            f"Number of sources provided is {n_sources} "
            f"and 'allow_multi_source' is set to {allow_multi_source}"
        ),
    ):
        check_X_domain(
            X,
            sample_domain=random_sample_domain,
            allow_auto_sample_domain=False,
            allow_multi_source=allow_multi_source,
        )

    with pytest.raises(
        ValueError,
        match=(
            f"Number of targets provided is {n_targets} "
            f"and 'allow_multi_target' is set to {allow_multi_target}"
        ),
    ):
        check_X_domain(
            X,
            sample_domain=random_sample_domain,
            allow_auto_sample_domain=False,
            allow_multi_target=allow_multi_target,
        )

    with pytest.raises(
        ValueError,
        match=(
            "Domain labels should be unique: the same domain "
            "index should not be used both for source and target"
        ),
    ):
        check_X_domain(
            X,
            sample_domain=common_idx_sample_domain,
            allow_common_domain_idx=allow_common_domain_idx,
        )


def test_check_X_domain_multi_nd():
    # Create a 3D array (10 samples, 2 features, 3 channels)
    X = np.random.rand(10, 2, 3)
    sample_domain = np.array([1] * 5 + [-1] * 5)

    # Test with allow_nd=True
    check_X_domain(X, sample_domain=sample_domain, allow_nd=True)

    # Test with allow_nd=False (should raise an error)
    with pytest.raises(
        ValueError, match="Found array with dim 3, while dim <= 2 is required."
    ):
        check_X_domain(X, sample_domain=sample_domain, allow_nd=False)


def test_check_X_y_domain_multi_nd():
    # Create a 3D array for X (10 samples, 2 features, 3 channels)
    X = np.random.rand(10, 2, 3)

    # Create a 2D array for y (10 samples, 2 outputs)
    y = np.random.rand(10, 2)
    sample_domain = np.array([1] * 5 + [-1] * 5)

    # Test with allow_nd=True
    check_X_y_domain(X, y, sample_domain=sample_domain, allow_nd=True)

    # Test with allow_nd=False (should raise an error for X)
    with pytest.raises(
        ValueError, match="Found array with dim 3, while dim <= 2 is required."
    ):
        check_X_y_domain(X, y, sample_domain=sample_domain, allow_nd=False)


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
        sample_domain, split_source_target=True
    )

    assert sum(len(values) for values in domain_idx_dict.values()) == len(
        sample_domain
    ), "domain_idx_dict shape mismatch"

    assert (
        sum(len(values) for values in domain_source_idx_dict.values())
        == 2 * n_samples_source
    ), "domain_source_idx_dict shape mismatch"

    assert (
        sum(len(values) for values in domain_target_idx_dict.values())
        == 2 * n_samples_target
    ), "domain_target_idx_dict shape mismatch"


def test_source_target_merge():
    # Test simple source-target merge with 2 domains
    X_source = np.array([[1, 2], [3, 4], [5, 6]])
    X_target = np.array([[7, 8], [9, 10]])
    y_source = np.array([0, 1, 1])
    y_target = None
    sample_domain = np.array([1, 1, 1, -2, -2])

    X, y, _ = source_target_merge(
        X_source, X_target, y_source, y_target, sample_domain=sample_domain
    )

    np.testing.assert_array_equal(
        X, np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
    )
    np.testing.assert_array_equal(
        y,
        np.array(
            [
                0,
                1,
                1,
                _DEFAULT_MASKED_TARGET_CLASSIFICATION_LABEL,
                _DEFAULT_MASKED_TARGET_CLASSIFICATION_LABEL,
            ]
        ),
    )

    # Test moons dataset with 2 domains
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

    # Test that no Error is raised for empty X_source and X_target in {-1, 0, 1}
    X_target_discrete = np.array([[-1, -1], [0, 1], [1, -1]])
    samples, _ = source_target_merge(np.array([]), X_target_discrete)
    assert (
        samples.shape[0] == X_target_discrete.shape[0]
    ), "target_samples shape mismatch"
    sample_domain_discrete = np.array([-1] * X_target_discrete.shape[0])
    samples, _ = source_target_merge(
        np.array([]), X_target_discrete, sample_domain=sample_domain_discrete
    )

    # Test with empty samples
    with pytest.raises(
        ValueError, match="Only one array can be None or empty in each pair"
    ):
        _ = source_target_merge(np.array([]), np.array([]), sample_domain=np.array([]))

    # Test that no Error with one empty sample
    _ = source_target_merge(
        X_source, np.array([]), sample_domain=np.array([1] * X_source.shape[0])
    )
    _ = source_target_merge(
        np.array([]), X_target, sample_domain=np.array([-1] * X_target.shape[0])
    )

    # Test consistent length
    with pytest.raises(
        ValueError,
        match="Inconsistent number of samples in source-target arrays "
        "and the number inferred in the sample_domain",
    ):
        _ = source_target_merge(X_source[0], X_target[1], sample_domain=sample_domain)

    # Test no sample domain
    _ = source_target_merge(X_source, X_target, sample_domain=None)

    # Test odd number of array
    with pytest.raises(ValueError, match="Even number of arrays required as input"):
        _ = source_target_merge(
            X_source, X_target, y_source, sample_domain=sample_domain
        )

    # Test >2 arrays
    _ = source_target_merge(
        X_source, X_target, y_source, y_target, sample_domain=sample_domain
    )

    # Test one array
    with pytest.raises(ValueError, match="At least two array required as input"):
        _ = source_target_merge(X_source, sample_domain=sample_domain)

    # Test y_target = None + Inconsistent number of samples in source-target
    with pytest.raises(
        ValueError,
        match="Inconsistent number of samples in source-target arrays "
        "and the number inferred in the sample_domain",
    ):
        _ = source_target_merge(
            X_source,
            X_target,
            np.ones_like(sample_domain),
            None,
            sample_domain=sample_domain,
        )

    # Test y_target = None + Consistent number of samples in source-target
    _ = source_target_merge(
        X_source, X_target, y_source, None, sample_domain=sample_domain
    )

    # Test 2 None in a pair of arrays
    with pytest.raises(
        ValueError, match="Only one array can be None or empty in each pair"
    ):
        _ = source_target_merge(None, None, sample_domain=sample_domain)

    # Test 1 None in 2 pair of arrays
    _ = source_target_merge(X_source, None, y_source, None, sample_domain=sample_domain)

    # Test inconsistent number of features
    with pytest.raises(
        ValueError, match="Inconsistent number of features in source-target arrays"
    ):
        _ = source_target_merge(X_source[:, :-1], X_target, sample_domain=sample_domain)


def test_qp_solve():
    Q = np.array([[2.0, 0.5], [0.5, 1.0]])
    c = np.array([1.0, 1.0])

    Aeq = np.array([[1.0, 1.0]])
    beq = np.array([1.0])

    A = -np.eye(2)
    b = np.zeros(2)

    lb = np.array([0.0, 0.0])
    ub = np.array([0.0, 0.0])

    x0 = 2.0 * np.ones(2)

    sol1 = np.array([0.25, 0.75])
    sol2 = -np.linalg.inv(Q) @ c

    res = qp_solve(Q, c, Aeq=Aeq, beq=beq, lb=lb)
    assert np.allclose(res[0], sol1)

    res = qp_solve(Q, c, Aeq=Aeq, beq=beq, lb=lb, x0=x0)
    assert np.allclose(res[0], sol1)

    res = qp_solve(Q, c, A, b, Aeq=Aeq, beq=beq)
    assert np.allclose(res[0], sol1)

    res = qp_solve(Q, c)
    assert np.allclose(res[0], sol2)

    res = qp_solve(Q, c, ub=ub)
    assert np.allclose(res[0], sol2)

    res = qp_solve(Q, c, ub=ub)
    assert np.allclose(res[0], sol2)

    res = qp_solve(Q)
    assert np.allclose(res[0], np.zeros(2))

    res = qp_solve(Q, c, lb=lb, ub=ub)
    assert np.allclose(res[0], np.zeros(2))

    res = qp_solve(Q, c, log=True)
    assert isinstance(res[2], dict)

    with pytest.warns(UserWarning, match="Iteration limit reached"):
        qp_solve(Q, c, Aeq=Aeq, beq=beq, lb=lb, max_iter=1)


def test_qp_solve_frank_wolfe():
    Q = np.array([[2.0, 0.5], [0.5, 1.0]])
    c = np.array([1.0, 1.0])

    Aeq = np.array([[1.0, 1.0]])
    beq = np.array([1.0])

    A = np.array([[1.0, 1.0]])
    b = np.array([1.0])

    A2 = np.array([[1.0, 1.0], [-1.0, -1.0]])
    b2 = np.array([1.0, -1.0])

    b3 = np.array([1.0, 0.0])

    lb = np.array([0.0, 0.0])

    sol1 = np.array([0.25, 0.75])

    res = qp_solve(Q, c, Aeq=Aeq, beq=beq, lb=lb, solver="frank-wolfe")
    assert np.allclose(res[0], sol1, atol=1e-3)

    res = qp_solve(Q, c, A=A, b=b, solver="frank-wolfe")
    assert np.allclose(res[0], sol1, atol=1e-3)

    res = qp_solve(Q, c, A=A2, b=b2, solver="frank-wolfe")
    assert np.allclose(res[0], sol1, atol=1e-3)

    res = qp_solve(Q, A=A2, b=b3, solver="frank-wolfe")
    assert np.allclose(res[0], np.zeros(2), atol=1e-3)

    res_scipy = qp_solve(Q, c, Aeq=Aeq, beq=beq, lb=lb, max_iter=10000, solver="scipy")
    res_fw = qp_solve(
        Q, c, Aeq=Aeq, beq=beq, lb=lb, max_iter=10000, solver="frank-wolfe"
    )
    assert np.allclose(res_scipy[0], res_fw[0], atol=1e-4)

    with pytest.raises(ValueError, match="`A` or `Aeq` must be given"):
        qp_solve(Q, c, solver="frank-wolfe")

    with pytest.raises(ValueError, match="must be equal to 1"):
        qp_solve(Q, c, Aeq=np.eye(2), beq=np.ones(2), solver="frank-wolfe")

    with pytest.raises(ValueError, match="must be lower than 2"):
        qp_solve(Q, c, A=np.eye(3), b=np.ones(3), solver="frank-wolfe")

    with pytest.raises(ValueError, match="must be equal to `-A"):
        qp_solve(Q, c, A=np.eye(2), b=np.ones(2), solver="frank-wolfe")

    with pytest.raises(ValueError, match="must be greater or equal to"):
        qp_solve(Q, c, A=A2, b=np.array([1.0, -2.0]), solver="frank-wolfe")


def test_frank_wolfe():
    Q = np.array([[2.0, 0.5], [0.5, 1.0]])
    c = np.array([1.0, 1.0])

    Aeq = np.array([1.0, 1.0])

    sol = np.array([0.25, 0.75])

    def jac(x):
        return Q @ x + c

    x1 = frank_wolfe(jac, Aeq, clb=1.0, cub=1.0, max_iter=1000)
    assert np.allclose(x1, sol, atol=1e-3)

    x2 = frank_wolfe(jac, Aeq, clb=1.0, cub=1.0, max_iter=10000)
    assert np.abs(x2 - sol).sum() < np.abs(x1 - sol).sum()

    x1 = frank_wolfe(jac, Aeq, clb=0.5, cub=1.0, max_iter=1000)
    assert np.allclose(x1, sol / 2, atol=1e-3)


@pytest.mark.skipif(not torch, reason="PyTorch not installed")
def test_torch_minimize():
    A = torch.tensor([[5.0, 2.0], [2.0, 5.0]], dtype=torch.float64)
    true_eigval, true_eigvec = np.linalg.eigh(A)
    true_eigval = true_eigval[0]
    true_eigvec = true_eigvec[:, 0].reshape(-1, 1)

    # Rayleigh quotient
    def loss(x):
        return (x.T @ A @ x) / (x.T @ x)

    # test optimization
    x0 = np.array([[-0.2], [0.1]])

    eigvec, min_eigval = torch_minimize(loss, x0, max_iter=100, tol=1e-6)
    eigvec = eigvec / np.linalg.norm(eigvec)

    assert np.allclose(min_eigval, true_eigval)
    assert np.allclose(eigvec, true_eigvec)

    # test optimization with torch tensor as initial guess
    x0_torch = torch.tensor(x0)

    eigvec, min_eigval = torch_minimize(loss, x0_torch, max_iter=100, tol=1e-6)
    eigvec = eigvec / np.linalg.norm(eigvec)

    assert np.allclose(min_eigval, true_eigval)
    assert np.allclose(eigvec, true_eigvec)

    # test optimization with list of variables as initial guess
    x0 = [x0]
    (eigvec), min_eigval = torch_minimize(loss, x0, max_iter=100, tol=1e-6)
    eigvec = eigvec / np.linalg.norm(eigvec)

    assert np.allclose(min_eigval, true_eigval)
    assert np.allclose(eigvec, true_eigvec)

    # test warning when convergence is not reached
    with pytest.warns(UserWarning):
        torch_minimize(loss, x0, max_iter=1, tol=0)


def test_merge_output_without_containers():
    X = _merge_domain_outputs(
        20,
        {
            "s": (np.arange(10), np.zeros(10)),
            "t": (np.arange(10, 20), np.ones(10)),
        },
    )
    assert X.shape[0] == 20
    assert X.sum() == 10


def test_merge_output_with_containers():
    # containers are not allowed by default
    with pytest.raises(AssertionError):
        _merge_domain_outputs(
            20,
            {
                "s": (np.arange(10), (np.zeros(10), None, {})),
                "t": (np.arange(10, 20), (np.zeros(10), None, {})),
            },
        )

    X, y, params = _merge_domain_outputs(
        20,
        {
            "s": (
                np.arange(10),
                (
                    np.zeros(10),
                    np.ones(10),
                    {"sample_weight": np.ones(10, dtype=np.float32)},
                ),
            ),
            "t": (
                np.arange(10, 20),
                (
                    np.zeros(10),
                    np.ones(10),
                    {"sample_weight": np.ones(10, dtype=np.float32) * 2},
                ),
            ),
        },
        allow_containers=True,
    )
    assert X.shape[0] == 20
    assert y.shape[0] == 20
    assert y.sum() == 20
    assert len(params) == 1
    assert params["sample_weight"].dtype == np.float32
    assert params["sample_weight"].shape[0] == 20
    assert params["sample_weight"].sum() == 30


def test_merge_output_with_optional_label():
    # empty y's
    _, y, _ = _merge_domain_outputs(
        20,
        {
            "s": (np.arange(10), (np.zeros(10), None, {"sample_weight": np.ones(10)})),
            "t": (
                np.arange(10, 20),
                (np.zeros(10), None, {"sample_weight": np.ones(10) * 2}),
            ),
        },
        allow_containers=True,
    )
    assert y is None

    # partially missing
    _, y, _ = _merge_domain_outputs(
        20,
        {
            "s": (np.arange(10), (np.zeros(10), None, {"sample_weight": np.ones(10)})),
            "t": (
                np.arange(10, 20),
                (np.zeros(10), np.ones(10), {"sample_weight": np.ones(10) * 2}),
            ),
        },
        allow_containers=True,
    )
    assert y.shape[0] == 20
    assert y.sum() == 10


def test_merge_output_invalid_container():
    with pytest.raises(ValueError):
        _merge_domain_outputs(
            20,
            {
                "s": (np.arange(10), (np.zeros(10), np.ones(10), {}, None, None)),
                "t": (
                    np.arange(10, 20),
                    (
                        np.zeros(10),
                        np.ones(10),
                        {"sample_weight": np.ones(10, dtype=np.float32) * 2},
                    ),
                ),
            },
            allow_containers=True,
        )
