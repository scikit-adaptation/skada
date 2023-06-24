import numpy as np

from skada.metrics import (
    MaximumMeanDiscrepancy,
    LinearDiscrepancy,
    KernelDiscrepancy,
    CorrelationDifference,
    HDivergence,
)

import pytest


@pytest.mark.parametrize(
    "metric",
    [
        MaximumMeanDiscrepancy(),
        LinearDiscrepancy(),
        KernelDiscrepancy(),
        CorrelationDifference(),
        HDivergence(),
    ],
)

def test_metric(metric, tmp_da_dataset):
    X, y, X_target, y_target = tmp_da_dataset
    
    W = np.ones(X.shape[0])
    
    metric.fit(X, X_target, W)
    eval1 = metric.eval()
    metric.gradient()
    metric.fit(X, X_target+2., W)
    eval2 = metric.eval()
    assert eval1 < eval2
