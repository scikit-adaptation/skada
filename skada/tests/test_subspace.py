# Author: Theo Gnassounou <theo.gnassounou@inria.fr>
#         Remi Flamary <remi.flamary@polytechnique.edu>
#         Oleksii Kachaiev <kachayev@gmail.com>
#         Antoine Collas <contact@antoinecollas.fr>
#
# License: BSD 3-Clause

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

try:
    import torch
except ImportError:
    torch = False

from skada import (
    SubspaceAlignment,
    SubspaceAlignmentAdapter,
    TransferComponentAnalysis,
    TransferComponentAnalysisAdapter,
    TransferJointMatching,
    TransferJointMatchingAdapter,
    TransferSubspaceLearning,
    TransferSubspaceLearningAdapter,
    make_da_pipeline,
)
from skada.datasets import DomainAwareDataset


@pytest.mark.parametrize(
    "estimator",
    [
        make_da_pipeline(
            SubspaceAlignmentAdapter(n_components=1), LogisticRegression()
        ),
        SubspaceAlignment(n_components=1),
        make_da_pipeline(
            TransferComponentAnalysisAdapter(n_components=1), LogisticRegression()
        ),
        TransferComponentAnalysis(n_components=1),
        TransferJointMatching(n_components=1, kernel="linear"),
        make_da_pipeline(
            TransferJointMatchingAdapter(n_components=1, kernel="linear", verbose=True),
            LogisticRegression(),
        ),
        pytest.param(
            TransferSubspaceLearning(n_components=1),
            marks=pytest.mark.skipif(not torch, reason="PyTorch not installed"),
        ),
        pytest.param(
            TransferSubspaceLearning(n_components=1, base_method="pca"),
            marks=pytest.mark.skipif(not torch, reason="PyTorch not installed"),
        ),
        pytest.param(
            TransferSubspaceLearning(n_components=1, base_method="flda"),
            marks=pytest.mark.skipif(not torch, reason="PyTorch not installed"),
        ),
        pytest.param(
            TransferSubspaceLearning(n_components=1, base_method="lpp"),
            marks=pytest.mark.skipif(not torch, reason="PyTorch not installed"),
        ),
        pytest.param(
            make_da_pipeline(
                TransferSubspaceLearningAdapter(n_components=1),
                KNeighborsClassifier(n_neighbors=1),
            ),
            marks=pytest.mark.skipif(not torch, reason="PyTorch not installed"),
        ),
    ],
)
def test_subspace_estimator(estimator, da_dataset):
    X_train, y_train, sample_domain = da_dataset.pack(
        as_sources=["s"], as_targets=["t"], mask_target_labels=True
    )
    estimator.fit(X_train, y_train, sample_domain=sample_domain)
    X_test, y_test, sample_domain = da_dataset.pack(
        as_sources=[], as_targets=["t"], mask_target_labels=False
    )
    y_pred = estimator.predict(X_test, sample_domain=sample_domain)
    assert np.mean(y_pred == y_test) > 0.75
    score = estimator.score(X_test, y_test, sample_domain=sample_domain)
    assert score > 0.75


@pytest.mark.parametrize(
    "adapter, n_samples, n_features, n_components",
    [
        (SubspaceAlignmentAdapter(), 5, 3, 3),
        (SubspaceAlignmentAdapter(), 3, 3, 3),
        (TransferComponentAnalysisAdapter(), 5, 3, 3),
        (TransferComponentAnalysisAdapter(), 2, 3, 3),
        (TransferComponentAnalysisAdapter(), 2, 5, 4),
        (TransferJointMatchingAdapter(), 5, 3, 3),
        (TransferJointMatchingAdapter(), 2, 3, 3),
        (TransferJointMatchingAdapter(), 2, 5, 4),
        pytest.param(
            TransferSubspaceLearningAdapter(),
            5,
            3,
            3,
            marks=pytest.mark.skipif(not torch, reason="PyTorch not installed"),
        ),
        pytest.param(
            TransferSubspaceLearningAdapter(),
            2,
            3,
            3,
            marks=pytest.mark.skipif(not torch, reason="PyTorch not installed"),
        ),
        pytest.param(
            TransferSubspaceLearningAdapter(),
            2,
            5,
            4,
            marks=pytest.mark.skipif(not torch, reason="PyTorch not installed"),
        ),
    ],
)
def test_subspace_default_n_components(adapter, n_samples, n_features, n_components):
    rng = np.random.default_rng(42)
    X_source, y_source, X_target, y_target = (
        rng.standard_normal((n_samples, n_features)),
        np.eye(n_samples, dtype="int32")[0],
        rng.standard_normal((n_samples, n_features)),
        np.eye(n_samples, dtype="int32")[0],
    )
    dataset = DomainAwareDataset(
        [
            (X_source, y_source, "s"),
            (X_target, y_target, "t"),
        ]
    )

    X_train, y_train, sample_domain = dataset.pack(
        as_sources=["s"], as_targets=["t"], mask_target_labels=True
    )
    adapter.fit_transform(X_train, y_train, sample_domain=sample_domain)
    X_test, _, sample_domain = dataset.pack(
        as_sources=[], as_targets=["t"], mask_target_labels=False
    )
    output = adapter.transform(X_test, sample_domain=sample_domain)
    assert output.shape[1] == n_components


@pytest.mark.parametrize(
    "adapter, param_name, param_value",
    [
        pytest.param(
            TransferSubspaceLearning,
            "base_method",
            "wrong_method",
            marks=pytest.mark.skipif(not torch, reason="PyTorch not installed"),
        ),
        pytest.param(
            TransferSubspaceLearning,
            "reg",
            -0.1,
            marks=pytest.mark.skipif(not torch, reason="PyTorch not installed"),
        ),
        pytest.param(
            TransferSubspaceLearning,
            "reg",
            1.1,
            marks=pytest.mark.skipif(not torch, reason="PyTorch not installed"),
        ),
    ],
)
def test_instantiation_wrong_params(adapter, param_name, param_value):
    with pytest.raises(ValueError):
        adapter(**{param_name: param_value})


@pytest.mark.parametrize("kernel", ["linear", "rbf"])
@pytest.mark.parametrize("mu", [0.0, 0.1, 2.0])
def test_tca_solves_generalized_eigenproblem(kernel, mu):
    """TCA components satisfy B w = lambda A w, without assuming A^-1 B is symmetric."""
    X = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 2.0], [1.5, 1.0], [2.0, 0.5]])
    domain = np.array([1, 1, -1, -1, -1])
    adapter = TransferComponentAnalysisAdapter(kernel=kernel, n_components=2, mu=mu)
    adapter.fit(X, sample_domain=domain)
    K = adapter.K_
    # The domain-mean contrast gives L = e e^T; centering gives H.
    contrast = np.array([0.5, 0.5, -1 / 3, -1 / 3, -1 / 3])
    A = np.eye(5) + mu * np.outer(K @ contrast, K @ contrast)
    centered = K - K.mean(axis=0, keepdims=True)
    B = centered.T @ centered
    W = adapter.eigvects_
    # Test the fitted components through Rayleigh quotients and residuals,
    # rather than a mocked solver call.
    eigenvalues = np.sum(W * (B @ W), axis=0) / np.sum(W * (A @ W), axis=0)
    np.testing.assert_allclose(B @ W, (A @ W) * eigenvalues, atol=1e-10, rtol=1e-10)
    # Whiten with Cholesky for an independent ordinary symmetric
    # eigensolver reference.
    C = np.linalg.cholesky(A)
    whitened = np.linalg.solve(C, np.linalg.solve(C, B).T).T
    expected = np.linalg.eigvalsh(whitened)[-2:][::-1]
    np.testing.assert_allclose(eigenvalues, expected, atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(np.linalg.norm(W, axis=0), 1.0)
    transformed = adapter.transform(X, sample_domain=domain, allow_source=True)
    np.testing.assert_allclose(transformed, K @ W)
