import numpy as np
from sklearn.linear_model import LogisticRegression

try:
    import torch
except ImportError:
    torch = False

import pytest

from skada import (
    OTLabelPropAdapter,
    make_da_pipeline,
    source_target_split,
)
from skada.datasets import DomainAwareDataset


@pytest.mark.parametrize(
    "estimator",
    [
        make_da_pipeline(OTLabelPropAdapter(), LogisticRegression()),
    ],
)
def test_label_prop_estimator(estimator, da_blobs_dataset):
    X, y, sample_domain = da_blobs_dataset
    X_source, X_target, y_source, y_target = source_target_split(
        X, y, sample_domain=sample_domain
    )

    # Just scale some feature to avoid having an identity cov matrix
    X_scaled = np.copy(X_source)
    X_target_scaled = np.copy(X_target)
    X_scaled[:, 0] *= 2
    X_target_scaled[:, 1] *= 3
    dataset = DomainAwareDataset(
        [
            (X_scaled, y_source, "s"),
            (X_target_scaled, y_target, "t"),
        ]
    )

    X_train, y_train, sample_domain = dataset.pack_train(
        as_sources=["s"], as_targets=["t"]
    )
    estimator.fit(X_train, y_train, sample_domain=sample_domain)
    X_test, y_test, sample_domain = dataset.pack_test(as_targets=["t"])
    y_pred = estimator.predict(X_test, sample_domain=sample_domain)
    assert np.mean(y_pred == y_test) > 0.9
    score = estimator.score(X_test, y_test, sample_domain=sample_domain)
    assert score > 0.9
