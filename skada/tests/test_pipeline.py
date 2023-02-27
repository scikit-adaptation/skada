import numpy as np
from numpy.testing import assert_almost_equal

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from skada import ReweightDensity, DAPipeline

import pytest


def test_pipeline(tmp_da_dataset):
    X, y, X_target, y_target = tmp_da_dataset
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
