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
