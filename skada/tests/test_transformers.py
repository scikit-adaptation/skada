# Author: Remi Flamary <remi.flamary@polytechnique.edu>
#
# License: BSD 3-Clause

import numpy as np
from sklearn.preprocessing import StandardScaler

from skada import CORAL, make_da_pipeline
from skada.transformers import (
    DomainStratifiedSubsampleTransformer,
    SubsampleTransformer,
)


def test_SubsampleTransformer(da_dataset):
    X, y, sample_domain = da_dataset.pack_train(as_sources=["s"], as_targets=["t"])
    sample_weight = np.ones_like(y)

    train_size = 10

    # test size of output on fit_transform
    transformer = SubsampleTransformer(train_size=train_size, random_state=42)

    X_subsampled, y_subsampled, params = transformer.fit_transform(
        X, y, sample_domain=sample_domain, sample_weight=sample_weight
    )

    assert X_subsampled.shape == (train_size, X.shape[1])
    assert y_subsampled.shape[0] == train_size
    assert params["sample_domain"].shape[0] == train_size
    assert params["sample_weight"].shape[0] == train_size

    # test size of output on transform
    X_target, y_target, sample_domain_target = da_dataset.pack_test(as_targets=["t"])

    X_target_subsampled = transformer.transform(
        X_target, y_target, sample_domain=sample_domain_target
    )

    assert X_target_subsampled.shape[0] == X_target.shape[0]

    # now with a pipeline with end task
    transformer = SubsampleTransformer(train_size=train_size)
    pipeline = make_da_pipeline(StandardScaler(), transformer, CORAL())

    pipeline.fit(X, y, sample_domain=sample_domain)

    ypred = pipeline.predict(X_target, sample_domain=sample_domain_target)
    assert ypred.shape[0] == X_target.shape[0]
    assert ypred.shape[0] == X_target.shape[0]


def test_DomainStratifiedSubsampleTransformer(da_dataset):
    X, y, sample_domain = da_dataset.pack_train(as_sources=["s"], as_targets=["t"])
    sample_weight = np.ones_like(y)

    train_size = 10

    # test size of output on fit_transform
    transformer = DomainStratifiedSubsampleTransformer(
        train_size=train_size, random_state=42
    )

    X_subsampled, y_subsampled, params = transformer.fit_transform(
        X, y, sample_domain=sample_domain, sample_weight=sample_weight
    )

    assert X_subsampled.shape == (train_size, X.shape[1])
    assert y_subsampled.shape[0] == train_size
    assert params["sample_domain"].shape[0] == train_size
    assert params["sample_weight"].shape[0] == train_size
    # check stratification
    assert sum(params["sample_domain"] == 1) == train_size // 2

    # test size of output on transform
    X_target, y_target, sample_domain_target = da_dataset.pack_test(as_targets=["t"])

    X_target_subsampled = transformer.transform(
        X_target, y_target, sample_domain=sample_domain_target
    )

    assert X_target_subsampled.shape[0] == X_target.shape[0]

    # now with a pipeline with end task
    transformer = DomainStratifiedSubsampleTransformer(train_size=train_size)
    pipeline = make_da_pipeline(StandardScaler(), transformer, CORAL())

    pipeline.fit(X, y, sample_domain=sample_domain)

    ypred = pipeline.predict(X_target, sample_domain=sample_domain_target)
    assert ypred.shape[0] == X_target.shape[0]
    assert ypred.shape[0] == X_target.shape[0]
