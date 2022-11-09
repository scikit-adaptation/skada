import numpy as np
from sklearn.linear_model import LogisticRegression

from skada.datasets import make_shifted_blobs
from skada import (
    OTmapping, EntropicOTmapping, ClassRegularizerOTmapping, LinearOTmapping
)
from skada import CORAL

import pytest


@pytest.mark.parametrize(
    "estimator", [
        OTmapping(base_estimator=LogisticRegression()),
        EntropicOTmapping(base_estimator=LogisticRegression()),
        ClassRegularizerOTmapping(base_estimator=LogisticRegression(), norm="lpl1"),
        ClassRegularizerOTmapping(base_estimator=LogisticRegression(), norm="l1l2"),
        LinearOTmapping(base_estimator=LogisticRegression()),
        CORAL(base_estimator=LogisticRegression())
    ]
)
def test_mapping_estimator(estimator):
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
