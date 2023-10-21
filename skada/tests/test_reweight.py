from sklearn.linear_model import LogisticRegression

from skada import (
    DomainAwareEstimator,
    ReweightDensityAdapter,
    GaussianReweightDensityAdapter,
    DiscriminatorReweightDensityAdapter,
    KLIEPAdapter,
)
from skada.datasets import DomainAwareDataset

import pytest


@pytest.mark.parametrize(
    "estimator",
    [
        DomainAwareEstimator(ReweightDensityAdapter(), LogisticRegression()),
        DomainAwareEstimator(GaussianReweightDensityAdapter(), LogisticRegression()),
        DomainAwareEstimator(DiscriminatorReweightDensityAdapter(), LogisticRegression()),
        DomainAwareEstimator(KLIEPAdapter(gamma=[0.1, 1], random_state=42), LogisticRegression()),
    ],
)
def test_reweight_estimator(estimator, tmp_da_dataset):
    X, y, X_target, y_target = tmp_da_dataset
    # xxx(okachaiev): make a special fixture for DA dataset object
    dataset = DomainAwareDataset([
        (X, y, 's'),
        (X_target, y_target, 't'),
    ])

    X_train, y_train, sample_domain = dataset.pack_for_train(as_sources=['s'], as_targets=['t'])
    estimator.fit(X_train, y_train, sample_domain=sample_domain)
    X_test, y_test, sample_domain = dataset.pack_for_test(as_targets=['t'])
    y_pred = estimator.predict(X_test, sample_domain=sample_domain)
    # xxx(okachaiev): see comments from other tests
    # assert np.mean(y_pred == y_test) > 0.9
    score = estimator.score(X_test, y_test, sample_domain=sample_domain)
    # assert score > 0.9
