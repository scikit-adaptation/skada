import numpy as np
from sklearn.linear_model import LogisticRegression

from skada.datasets import make_shifted_blobs
from skada import (
    ReweightDensity, GaussianReweightDensity, DiscriminatorReweightDensity, KLIEP
)

import pytest


@pytest.mark.parametrize(
    "estimator", [
        ReweightDensity(base_estimator=LogisticRegression()),
        GaussianReweightDensity(base_estimator=LogisticRegression()),
        DiscriminatorReweightDensity(base_estimator=LogisticRegression()),
        KLIEP(base_estimator=LogisticRegression(), kparam=[0.1, 1])
    ]
)
def test_reweight_estimator(estimator):
    centers = np.array([
        [0, 0],
        [1, 1],
    ])
    _, n_features = centers.shape

    X, y, X_target, y_target = make_shifted_blobs(
        n_samples=500,
        centers=centers,
        n_features=n_features,
        shift=0.13,
        random_state=42,
        cluster_std=0.05,
    )

    estimator.fit(X, y, X_target)
    y_pred = estimator.predict(X_target)
    assert np.mean(y_pred == y_target) > 0.9
    score = estimator.score(X_target, y_target)
    assert score > 0.9
