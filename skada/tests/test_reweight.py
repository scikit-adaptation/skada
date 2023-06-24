import numpy as np
from sklearn.linear_model import LogisticRegression

from skada import (
    ReweightDensity,
    GaussianReweightDensity,
    DiscriminatorReweightDensity,
    KLIEP,
    KMM,
)

import pytest


@pytest.mark.parametrize(
    "estimator",
    [
        ReweightDensity(base_estimator=LogisticRegression()),
        GaussianReweightDensity(base_estimator=LogisticRegression()),
        DiscriminatorReweightDensity(base_estimator=LogisticRegression()),
        KLIEP(base_estimator=LogisticRegression(), gamma=[0.1, 1], random_state=42),
        KMM(base_estimator=LogisticRegression())
    ],
)
def test_reweight_estimator(estimator, tmp_da_dataset):
    X, y, X_target, y_target = tmp_da_dataset

    estimator.fit(X, y, X_target)
    y_pred = estimator.predict(X_target)
    assert np.mean(y_pred == y_target) > 0.9
    score = estimator.score(X_target, y_target)
    assert score > 0.9
