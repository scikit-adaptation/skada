# Author: Ruben Bueno <ruben.bueno@polytechnique.edu>

from skada.datasets import make_shifted_datasets
from skada._dasvm import DASVMEstimatorthis 
from skada.utils import check_X_y_domain, source_target_split
from skada._pipeline import make_da_pipeline

from sklearn.preprocessing import StandardScaler

import pytest

@pytest.mark.parametrize(
    "label, n, m",
    [("binary", 7, 5), ("binary", 6, 6), ("binary", 5, 7), ("multiclass", 7, 5)],
)
def test_dasvm_estimator(label, n, m):
    """
    Tests for the dasvm method
    """
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

    clf_dsvm = DASVMEstimatorthis (k=5).fit(X, y, sample_domain=sample_domain)

    assert clf_dsvm.get_estimator().n_features_in_ == 2, (
        "Obtained estimator take the wrong number of features"
        )
    Xt_, yt_ = clf_dsvm.adapt(X, y, sample_domain)
    assert Xt_.shape == Xt.shape, (
        "Wrong shape of the target features when using `adapt` method"
        )
    assert yt_.shape == yt.shape, (
        "Wrong shape of the target y-values (labels) when using `adapt` method"
        )

    assert clf_dsvm.predict(X).shape == y.shape, (
            "Wrong shape of the predicted y-values (labels) when using `predict` method"
            )

    assert clf_dsvm.decision_function(X).shape == y.shape, (
            "Wrong shape of the decision function's values "
            "when using `decision_function` method"
            )

    # The `DASVMEstimatorthis ` should be usable with `make_da_pipeline`
    clf_dsvm = make_da_pipeline(
        StandardScaler(), DASVMEstimatorthis (k=5)).fit(
        X, y, sample_domain=sample_domain)
