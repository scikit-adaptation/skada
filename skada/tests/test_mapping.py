# Author: Theo Gnassounou <theo.gnassounou@inria.fr>
#         Remi Flamary <remi.flamary@polytechnique.edu>
#         Oleksii Kachaiev <kachayev@gmail.com>
#
# License: BSD 3-Clause

import numpy as np
from sklearn.linear_model import LogisticRegression

from skada import (
    CORALAdapter,
    CORAL,
    ClassRegularizerOTMappingAdapter,
    ClassRegularizerOTMapping,
    EntropicOTMappingAdapter,
    EntropicOTMapping,
    LinearOTMappingAdapter,
    LinearOTMapping,
    OTMappingAdapter,
    OTMapping,
    make_da_pipeline,
)
from skada.datasets import DomainAwareDataset
from skada.utils import source_target_split

import pytest


@pytest.mark.parametrize(
    "estimator", [
        make_da_pipeline(OTMappingAdapter(), LogisticRegression()),
        OTMapping(),
        make_da_pipeline(EntropicOTMappingAdapter(), LogisticRegression()),
        EntropicOTMapping(),
        make_da_pipeline(
            ClassRegularizerOTMappingAdapter(norm="lpl1"),
            LogisticRegression()
        ),
        ClassRegularizerOTMapping(),
        make_da_pipeline(
            ClassRegularizerOTMappingAdapter(norm="l1l2"),
            LogisticRegression()
        ),
        ClassRegularizerOTMapping(norm="l1l2"),
        make_da_pipeline(LinearOTMappingAdapter(), LogisticRegression()),
        LinearOTMapping(),
        make_da_pipeline(CORALAdapter(), LogisticRegression()),
        pytest.param(
            CORALAdapter(reg=None),
            marks=pytest.mark.xfail(reason='Fails without regularization')
        ),
        make_da_pipeline(CORALAdapter(reg=0.1), LogisticRegression()),
        CORAL(),
    ]
)
def test_mapping_estimator(estimator, da_blobs_dataset):
    X, y, sample_domain = da_blobs_dataset
    X_source, X_target, y_source, y_target = source_target_split(
        X, y, sample_domain=sample_domain
    )

    # Just scale some feature to avoid having an identity cov matrix
    X_scaled = np.copy(X_source)
    X_target_scaled = np.copy(X_target)
    X_scaled[:, 0] *= 2
    X_target_scaled[:, 1] *= 3
    dataset = DomainAwareDataset([
        (X_scaled, y_source, 's'),
        (X_target_scaled, y_target, 't'),
    ])

    X_train, y_train, sample_domain = dataset.pack_train(
        as_sources=['s'],
        as_targets=['t']
    )
    estimator.fit(X_train, y_train, sample_domain=sample_domain)
    X_test, y_test, sample_domain = dataset.pack_test(as_targets=['t'])
    y_pred = estimator.predict(X_test, sample_domain=sample_domain)
    assert np.mean(y_pred == y_test) > 0.9
    score = estimator.score(X_test, y_test, sample_domain=sample_domain)
    assert score > 0.9
