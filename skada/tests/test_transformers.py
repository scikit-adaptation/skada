# Author: Remi Flamary <remi.flamary@polytechnique.edu>
#
# License: BSD 3-Clause

from collections import Counter

import numpy as np
from sklearn.preprocessing import StandardScaler

from skada import CORAL, make_da_pipeline
from skada.transformers import (
    DomainSubsampler,
    StratifiedDomainSubsampler,
    Subsampler,
)


def test_Subsampler(da_dataset):
    X, y, sample_domain = da_dataset.pack(
        as_sources=["s"], as_targets=["t"], mask_target_labels=True
    )
    sample_weight = np.ones_like(y)

    train_size = 10

    # test size of output on fit_transform
    transformer = Subsampler(train_size=train_size, random_state=42)

    X_subsampled, y_subsampled, params = transformer.fit_transform(
        X, y, sample_domain=sample_domain, sample_weight=sample_weight
    )

    assert X_subsampled.shape == (train_size, X.shape[1])
    assert y_subsampled.shape[0] == train_size
    assert params["sample_domain"].shape[0] == train_size
    assert params["sample_weight"].shape[0] == train_size

    # test size of output on transform
    X_target, y_target, sample_domain_target = da_dataset.pack(
        as_sources=[], as_targets=["t"], mask_target_labels=False
    )

    X_target_subsampled = transformer.transform(
        X_target, y_target, sample_domain=sample_domain_target
    )

    assert X_target_subsampled.shape[0] == X_target.shape[0]

    # now with a pipeline with end task
    transformer = Subsampler(train_size=train_size)
    pipeline = make_da_pipeline(StandardScaler(), transformer, CORAL())

    pipeline.fit(X, y, sample_domain=sample_domain)

    ypred = pipeline.predict(X_target, sample_domain=sample_domain_target)
    assert ypred.shape[0] == X_target.shape[0]

    ypred = pipeline.predict(X, sample_domain=sample_domain, allow_source=True)
    assert ypred.shape[0] == X.shape[0]


def test_DomainSubsampler(da_dataset):
    X, y, sample_domain = da_dataset.pack(
        as_sources=["s"], as_targets=["t"], mask_target_labels=True
    )
    sample_weight = np.ones_like(y)

    train_size = 10

    # test size of output on fit_transform
    transformer = DomainSubsampler(train_size=train_size, random_state=42)

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
    X_target, y_target, sample_domain_target = da_dataset.pack(
        as_sources=[], as_targets=["t"], mask_target_labels=False
    )

    X_target_subsampled = transformer.transform(
        X_target, y_target, sample_domain=sample_domain_target
    )

    assert X_target_subsampled.shape[0] == X_target.shape[0]

    # now with a pipeline with end task
    transformer = DomainSubsampler(train_size=train_size)
    pipeline = make_da_pipeline(StandardScaler(), transformer, CORAL())

    pipeline.fit(X, y, sample_domain=sample_domain)

    ypred = pipeline.predict(X_target, sample_domain=sample_domain_target)
    assert ypred.shape[0] == X_target.shape[0]

    ypred = pipeline.predict(X, sample_domain=sample_domain, allow_source=True)
    assert ypred.shape[0] == X.shape[0]


def test_StratifiedDomainSubsampler(da_dataset):
    X, y, sample_domain = da_dataset.pack(
        as_sources=["s"], as_targets=["t"], mask_target_labels=True
    )
    sample_weight = np.ones_like(y)

    train_size = 10

    # test size of output on fit_transform
    transformer = StratifiedDomainSubsampler(train_size=train_size, random_state=42)

    X_subsampled, y_subsampled, params = transformer.fit_transform(
        X, y, sample_domain=sample_domain, sample_weight=sample_weight
    )

    assert X_subsampled.shape == (train_size, X.shape[1])
    assert y_subsampled.shape[0] == train_size
    assert params["sample_domain"].shape[0] == train_size
    assert params["sample_weight"].shape[0] == train_size

    # Check stratification proportions
    original_freq = Counter(zip(sample_domain, y))
    subsampled_freq = Counter(zip(params["sample_domain"], y_subsampled))

    for key in original_freq:
        original_ratio = original_freq[key] / len(y)
        subsampled_ratio = subsampled_freq[key] / train_size
        assert np.isclose(
            original_ratio, subsampled_ratio, atol=0.1
        ), f"Stratification not preserved for {key}"

    # test size of output on transform
    X_target, y_target, sample_domain_target = da_dataset.pack(
        as_sources=[], as_targets=["t"], mask_target_labels=False
    )

    X_target_subsampled = transformer.transform(
        X_target, y_target, sample_domain=sample_domain_target
    )

    assert X_target_subsampled.shape[0] == X_target.shape[0]

    # now with a pipeline with end task
    transformer = StratifiedDomainSubsampler(train_size=train_size)
    pipeline = make_da_pipeline(StandardScaler(), transformer, CORAL())

    pipeline.fit(X, y, sample_domain=sample_domain)

    ypred = pipeline.predict(X_target, sample_domain=sample_domain_target)
    assert ypred.shape[0] == X_target.shape[0]

    ypred = pipeline.predict(X, sample_domain=sample_domain, allow_source=True)
    assert ypred.shape[0] == X.shape[0]
