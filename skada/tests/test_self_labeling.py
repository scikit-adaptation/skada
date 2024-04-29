# Author: Ruben Bueno <ruben.bueno@polytechnique.edu>
#         Yanis Lalou <yanis.lalou@polytechnique.edu>
#
# License: BSD 3-Clause

import numpy as np
import pytest
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from skada._pipeline import make_da_pipeline
from skada._self_labeling import DASVMClassifier
from skada.datasets import make_shifted_datasets
from skada.utils import check_X_y_domain


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

    clf_dasvm = DASVMClassifier(k=3, save_estimators=True, save_indices=True).fit(
        X, y, sample_domain=sample_domain
    )

    assert (
        clf_dasvm.predict(X).shape == y.shape
    ), "Wrong shape of the predicted y-values (labels) when using `predict` method"

    assert clf_dasvm.decision_function(X).shape[0] == y.shape[0], (
        "Wrong length of the decision function's values "
        "when using `decision_function` method"
    )

    assert clf_dasvm.score(X, y) >= 0, "The score should be non-negative"

    # The `DASVMClassifier` should be usable with `make_da_pipeline`
    manage_pipeline = False
    try:
        clf_dasvm = make_da_pipeline(StandardScaler(), DASVMClassifier(k=5)).fit(
            X, y, sample_domain=sample_domain
        )
        manage_pipeline = True
    finally:
        assert manage_pipeline, "Couldn't use make_da_pipeline with DASVMClassifier"

    # The `DASVMClassifier` should accept sample_domain as an argument
    if manage_pipeline:
        clf_dasvm.fit(X, y, sample_domain=sample_domain)
        clf_dasvm.predict(X, sample_domain=sample_domain)


def test_dasvm_estimator_predict_proba():
    X, y, sample_domain = make_shifted_datasets(
        n_samples_source=5,
        n_samples_target=5,
        shift="covariate_shift",
        noise=None,
        label="binary",
        random_state=0,
    )
    X, y, sample_domain = check_X_y_domain(X, y, sample_domain)

    # Test with a base estimator that does support predict_proba
    clf_dasvm = DASVMClassifier(k=3, base_estimator=SVC(probability=True)).fit(
        X, y, sample_domain=sample_domain
    )

    assert (
        clf_dasvm.predict_proba(X).shape[0] == y.shape[0]
        and clf_dasvm.predict_proba(X).shape[1] == np.unique(y).shape[0]
    ), "Wrong length of the predicted probabilities when using `predict_proba` method"

    # Test with a base estimator that does not support predict_proba
    clf_dasvm = DASVMClassifier(k=3, base_estimator=SVC(probability=False)).fit(
        X, y, sample_domain=sample_domain
    )

    with pytest.raises(AttributeError):
        clf_dasvm.predict_proba(X)
