import numpy as np
from sklearn.linear_model import LogisticRegression

from skada.base import DomainAwareEstimator
from skada.datasets import DomainAwareDataset
from skada import (
    CORALAdapter,
    ClassRegularizerOTMappingAdapter,
    EntropicOTMappingAdapter,
    LinearOTMappingAdapter,
    OTMappingAdapter,
)

import pytest


@pytest.mark.parametrize(
    "estimator", [
        DomainAwareEstimator(OTMappingAdapter(), LogisticRegression()),
        DomainAwareEstimator(EntropicOTMappingAdapter(), LogisticRegression()),
        DomainAwareEstimator(
            ClassRegularizerOTMappingAdapter(norm="lpl1"),
            LogisticRegression()
        ),
        DomainAwareEstimator(
            ClassRegularizerOTMappingAdapter(norm="l1l2"),
            LogisticRegression()
        ),
        DomainAwareEstimator(LinearOTMappingAdapter(), LogisticRegression()),
        DomainAwareEstimator(CORALAdapter(), LogisticRegression()),
        pytest.param(CORALAdapter(reg=None), marks=pytest.mark.xfail(reason='Fails without regularization')),
        DomainAwareEstimator(CORALAdapter(reg=0.1), LogisticRegression()),
    ]
)
def test_mapping_estimator(estimator, tmp_da_dataset):
    X_source, y_source, X_target, y_target = tmp_da_dataset

    # Just scale some feature to avoid having an identity cov matrix
    X_scaled = np.copy(X_source)
    X_target_scaled = np.copy(X_target)
    X_scaled[:, 0] *= 2
    X_target_scaled[:, 1] *= 3
    dataset = DomainAwareDataset([
        (X_scaled, y_source, 's'),
        (X_target_scaled, y_target, 't'),
    ])

    X_train, y_train, sample_domain = dataset.pack_for_train(as_sources=['s'], as_targets=['t'])
    estimator.fit(X_train, y_train, sample_domain=sample_domain)
    X_test, y_test, sample_domain = dataset.pack_for_test(as_targets=['t'])
    y_pred = estimator.predict(X_test, sample_domain=sample_domain)
    # xxx(okachaiev): this should be like 0.9
    assert np.mean(y_pred == y_test) > 0.
    score = estimator.score(X_test, y_test, sample_domain=sample_domain)
    # assert score > 0.
