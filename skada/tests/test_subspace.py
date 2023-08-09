import numpy as np
from sklearn.linear_model import LogisticRegression

from skada import SubspaceAlignment, TransferComponentAnalysis

import pytest


@pytest.mark.parametrize(
    "estimator", [
        SubspaceAlignment(base_estimator=LogisticRegression(), n_components=2),
        TransferComponentAnalysis(base_estimator=LogisticRegression(), n_components=2),
    ]
)
def test_subspace_alignment(estimator, tmp_da_dataset):
    X, y, X_target, y_target = tmp_da_dataset

    estimator.fit(X, y, X_target)
    y_pred = estimator.predict(X_target)
    assert np.mean(y_pred == y_target) > 0.9
    score = estimator.score(X_target, y_target)
    assert score > 0.9


@pytest.mark.parametrize(
    "estimator, n_samples, n_features, n_components", [
        (SubspaceAlignment(base_estimator=LogisticRegression()), 5, 3, 3),
        (SubspaceAlignment(base_estimator=LogisticRegression()), 2, 5, 2),
        (TransferComponentAnalysis(base_estimator=LogisticRegression()), 5, 3, 3),
        (TransferComponentAnalysis(base_estimator=LogisticRegression()), 2, 3, 3),
        (TransferComponentAnalysis(base_estimator=LogisticRegression()), 2, 5, 4),
    ]
)
def test_subspace_default_n_components(estimator, n_samples, n_features, n_components):
    rng = np.random.RandomState(42)
    X, y, X_target = (
        rng.rand(n_samples, n_features),
        np.eye(n_samples, dtype='int32')[0],
        rng.rand(n_samples, n_features)
    )

    estimator.fit(X, y, X_target)
    X_transform = estimator.transform(X_target)
    assert X_transform.shape[1] == n_components
