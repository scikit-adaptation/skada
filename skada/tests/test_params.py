from skada.params import (
    KernelWeighting,
    LinearWeighting,
    DirectWeighting
)

import pytest


@pytest.mark.parametrize(
    "weight_param",
    [
        KernelWeighting(),
        LinearWeighting(),
        DirectWeighting()
    ],
)
def test_reweight_estimator(weight_param, tmp_da_dataset):
    X, y, X_target, y_target = tmp_da_dataset

    weight_param.fit(X, X, X_target)
    weight_param.predict()
    weight_param.gradient()