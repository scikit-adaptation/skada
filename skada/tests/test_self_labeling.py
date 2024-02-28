# Author: Ruben Bueno <ruben.bueno@polytechnique.edu>

from skada.datasets import make_shifted_datasets
from skada._self_labeling import DASVMClassifier
from skada.utils import check_X_y_domain, source_target_split
from skada._pipeline import make_da_pipeline

from sklearn.preprocessing import StandardScaler

import pytest


@pytest.mark.parametrize(
    "label, n, m",
    [("binary", 15, 5), ("binary", 10, 10), ("binary", 5, 15)],
)
def test_dasvm_estimator(label, n, m):
    X, y, sample_domain = make_shifted_datasets(
        n_samples_source=n,
        n_samples_target=m,
        shift="covariate_shift",
        noise=None,
        label=label,
    )
    X, y, sample_domain = check_X_y_domain(X, y, sample_domain)
    Xs, Xt, ys, yt = source_target_split(
        X, y, sample_domain=sample_domain)

    clf_dasvm = DASVMClassifier(
        k=3, save_estimators=True, save_indices=True).fit(
        X, y, sample_domain=sample_domain)

    assert clf_dasvm.predict(X).shape == y.shape, (
            "Wrong shape of the predicted y-values (labels) when using `predict` method"
            )

    assert clf_dasvm.decision_function(X).shape[0] == y.shape[0], (
            "Wrong lenght of the decision function's values "
            "when using `decision_function` method"
            )

    # The `DASVMClassifier` should be usable with `make_da_pipeline`
    manage_pipeline = False
    try:
        clf_dasvm = make_da_pipeline(
            StandardScaler(), DASVMClassifier(k=5)).fit(
            X, y, sample_domain=sample_domain)
        manage_pipeline = True
    finally:
        assert manage_pipeline, (
            "Couldn't use make_da_pipeline with DASVMClassifier"
            )
