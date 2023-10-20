import numpy as np
from sklearn.linear_model import LogisticRegression

from skada import (
    SubspaceAlignment,
    SubspaceAlignmentAdapter,
    TransferComponentAnalysis,
    TransferComponentAnalysisAdapter,
)
from skada.datasets import DomainAwareDataset

import pytest


@pytest.mark.parametrize(
    "estimator", [
        SubspaceAlignment(base_estimator=LogisticRegression(), n_components=2),
        TransferComponentAnalysis(base_estimator=LogisticRegression(), n_components=2),
    ]
)
def test_subspace_alignment(estimator, tmp_da_dataset):
    X, y, X_target, y_target = tmp_da_dataset
    # xxx(okachaiev): make a special fixture for DA dataset object
    dataset = DomainAwareDataset([
        (X, y, 's'),
        (X_target, y_target, 't'),
    ])

    X_train, y_train, sample_domain = dataset.pack_for_train(as_sources=['s'], as_targets=['t'])
    estimator.fit(X_train, y_train, sample_domain=sample_domain)
    X_test, y_test, sample_domain = dataset.pack_for_test(as_targets=['t'])
    y_pred = estimator.predict(X_test, sample_domain=sample_domain)
    # xxx(okachaiev): this should be like 0.9
    assert np.mean(y_pred == y_test) > 0.
    # xxx(okachaiev): see the comment about 'score'
    # score = estimator.score(X_test, y_test, sample_domain=sample_domain)
    # assert score > 0.


@pytest.mark.parametrize(
    "adapter, n_samples, n_features, n_components", [
        (SubspaceAlignmentAdapter(), 5, 3, 3),
        (SubspaceAlignmentAdapter(), 2, 5, 2),
        (TransferComponentAnalysisAdapter(), 5, 3, 3),
        (TransferComponentAnalysisAdapter(), 2, 3, 3),
        (TransferComponentAnalysisAdapter(), 2, 5, 4),
    ]
)
def test_subspace_default_n_components(adapter, n_samples, n_features, n_components):
    rng = np.random.RandomState(42)
    X_source, y_source, X_target, y_target = (
        rng.rand(n_samples, n_features),
        np.eye(n_samples, dtype='int32')[0],
        rng.rand(n_samples, n_features),
        np.eye(n_samples, dtype='int32')[0],
    )
    dataset = DomainAwareDataset([
        (X_source, y_source, 's'),
        (X_target, y_target, 't'),
    ])

    X_train, y_train, sample_domain = dataset.pack_for_train(as_sources=['s'], as_targets=['t'])
    adapter.fit(X_train, y_train, sample_domain=sample_domain)
    X_test, y_test, sample_domain = dataset.pack_for_test(as_targets=['t'])
    X_transform, _, _, _ = adapter.adapt(X_test, y_test, sample_domain)
    assert X_transform.shape[1] == n_components
