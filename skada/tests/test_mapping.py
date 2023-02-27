import numpy as np
from sklearn.linear_model import LogisticRegression

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
        CORAL(base_estimator=LogisticRegression()),
        pytest.param(CORAL(base_estimator=LogisticRegression(), reg=None),
                     marks=pytest.mark.xfail(reason='Fails without regularization')),
        CORAL(base_estimator=LogisticRegression(), reg=0.1),
    ]
)
def test_mapping_estimator(estimator, tmp_da_dataset):
    X, y, X_target, y_target = tmp_da_dataset

    # Just scale some feature to avoid having an identity cov matrix
    X_scaled = np.copy(X)
    X_target_scaled = np.copy(X_target)
    X_scaled[:, 0] *= 2
    X_target_scaled[:, 1] *= 3

    estimator.fit(X, y, X_target)
    y_pred = estimator.predict(X_target)
    assert np.mean(y_pred == y_target) > 0.9
    score = estimator.score(X_target, y_target)
    assert score > 0.9
