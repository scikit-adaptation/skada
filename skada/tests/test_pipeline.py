import numpy as np
from numpy.testing import assert_almost_equal

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from skada.datasets import make_shifted_blobs
from skada import ReweightDensity, DAPipeline

import pytest


def test_pipeline():
    centers = np.array(
        [
            [0, 0],
            [1, 1],
        ]
    )
    _, n_features = centers.shape

    X, y, X_target, y_target = make_shifted_blobs(
        n_samples=500,
        centers=centers,
        n_features=n_features,
        shift=0.13,
        random_state=42,
        cluster_std=0.05,
    )
    estimator = ReweightDensity(base_estimator=LogisticRegression())
    pipe = DAPipeline([("scaler", StandardScaler()), ("estimator", estimator)])
    pipe.fit(X, y, X_target)
    y_pred = pipe.predict(X_target)
    assert np.mean(y_pred == y_target) > 0.9
    score = pipe.score(X_target, y_target)
    assert score > 0.9
    y_pred_2 = pipe.fit_predict(X, y, X_target)
    assert_almost_equal((y_pred - y_pred_2), 0.10, 1, "Unexpected prediction")
    with pytest.raises(NotImplementedError):
        pipe.fit_transform(X, y)
