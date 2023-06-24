from skada.params import DirectWeighting
from skada.metrics import MaximumMeanDiscrepancy
from skada.optims import (
    gradient_descent,
    projected_gradient_descent,
    frank_wolfe
)

import pytest


@pytest.mark.parametrize(
    "optim_func",
    [
        gradient_descent,
        projected_gradient_descent,
        frank_wolfe
    ],
)
def test_reweight_estimator(optim_func, tmp_da_dataset):
    X, y, X_target, y_target = tmp_da_dataset
    
    weight_param = DirectWeighting()
    weight_param.fit(X, X, X_target)
    metric = MaximumMeanDiscrepancy()
    metric.fit(X, X_target, weight_param.predict())
    
    eval1 = metric.eval()
    optim_func(weight_param, metric, max_iter=100)
    eval2 = metric.eval()
    
    assert eval2 < eval1