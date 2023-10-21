import numpy as np
from sklearn.linear_model import LogisticRegression

from skada.base import DomainAdaptationStrategy
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
        DomainAdaptationStrategy(base_adapter=OTMappingAdapter(), base_estimator=LogisticRegression()),
        DomainAdaptationStrategy(base_adapter=EntropicOTMappingAdapter(), base_estimator=LogisticRegression()),
        DomainAdaptationStrategy(
            base_adapter=ClassRegularizerOTMappingAdapter(norm="lpl1"),
            base_estimator=LogisticRegression()
        ),
        DomainAdaptationStrategy(
            base_adapter=ClassRegularizerOTMappingAdapter(norm="l1l2"),
            base_estimator=LogisticRegression()
        ),
        DomainAdaptationStrategy(base_adapter=LinearOTMappingAdapter(), base_estimator=LogisticRegression()),
        DomainAdaptationStrategy(base_adapter=CORALAdapter(), base_estimator=LogisticRegression()),
        pytest.param(CORALAdapter(reg=None), marks=pytest.mark.xfail(reason='Fails without regularization')),
        DomainAdaptationStrategy(base_adapter=CORALAdapter(reg=0.1), base_estimator=LogisticRegression()),
    ]
)
def test_mapping_estimator(estimator, tmp_da_dataset):
    X_source, y_source, X_target, y_target = tmp_da_dataset

    # Just scale some feature to avoid having an identity cov matrix
    X_scaled = np.copy(X_source)
    X_target_scaled = np.copy(X_target)
    X_scaled[:, 0] *= 2
    X_target_scaled[:, 1] *= 3
    # xxx(okachaiev): make a special fixture for DA dataset object
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
