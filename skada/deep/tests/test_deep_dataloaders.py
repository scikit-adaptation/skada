# Author: Theo Gnassounou <theo.gnassounou@inria.fr>
#         Oleksii Kachaiev <kachayev@gmail.com>
#         Yanis Lalou <yanis.lalou@polytechnique.edu>
#         Antoine Collas <contact@antoinecollas.fr>
#
# License: BSD 3-Clause
import pytest

torch = pytest.importorskip("torch")

import numpy as np
from skorch.dataset import Dataset

from skada.datasets import make_dataset_from_moons_distribution, make_shifted_datasets
from skada.deep.dataloaders import (
    DomainBalancedDataLoader,
    DomainBalancedSampler,
    MultiSourceDomainBalancedDataLoader,
    MultiSourceDomainBalancedSampler,
)


def test_domain_balanced_sampler():
    n_samples = 20
    dataset = make_shifted_datasets(
        n_samples_source=n_samples,
        n_samples_target=n_samples,
        shift="concept_drift",
        noise=0.1,
        random_state=42,
        return_dataset=True,
    )
    X, y, sample_domain = dataset.pack_train(as_sources=["s"], as_targets=["t"])
    X_dict = {"X": X.astype(np.float32), "sample_domain": sample_domain}
    dataset = Dataset(X_dict, y)
    sampler = DomainBalancedSampler(dataset, 10)
    assert len(sampler) == len(dataset)


def test_domain_balanced_dataloader():
    n_samples = 20
    dataset = make_shifted_datasets(
        n_samples_source=n_samples,
        n_samples_target=n_samples,
        shift="concept_drift",
        noise=0.1,
        random_state=42,
        return_dataset=True,
    )
    X, y, sample_domain = dataset.pack_train(as_sources=["s"], as_targets=["t"])
    X_dict = {"X": X.astype(np.float32), "sample_domain": sample_domain}

    dataset = Dataset(X_dict, y)

    dataloader = DomainBalancedDataLoader(dataset, batch_size=10)

    for batch in dataloader:
        X, y = batch
        sample_domain = X["sample_domain"]
        assert len(sample_domain > 0) == len(sample_domain < 0)

    # with more source than target
    dataset = make_shifted_datasets(
        n_samples_source=2 * n_samples,
        n_samples_target=n_samples,
        shift="concept_drift",
        noise=0.1,
        random_state=42,
        return_dataset=True,
    )
    X, y, sample_domain = dataset.pack_train(as_sources=["s"], as_targets=["t"])
    X_dict = {"X": X.astype(np.float32), "sample_domain": sample_domain}

    dataset = Dataset(X_dict, y)

    dataloader = DomainBalancedDataLoader(dataset, batch_size=10)

    for batch in dataloader:
        X, y = batch
        sample_domain = X["sample_domain"]
        assert len(sample_domain > 0) == len(sample_domain < 0)

    # with more target than source
    dataset = make_shifted_datasets(
        n_samples_source=n_samples,
        n_samples_target=2 * n_samples,
        shift="concept_drift",
        noise=0.1,
        random_state=42,
        return_dataset=True,
    )
    X, y, sample_domain = dataset.pack_train(as_sources=["s"], as_targets=["t"])
    X_dict = {"X": X.astype(np.float32), "sample_domain": sample_domain}

    dataset = Dataset(X_dict, y)

    dataloader = DomainBalancedDataLoader(dataset, batch_size=10)

    for batch in dataloader:
        X, y = batch
        sample_domain = X["sample_domain"]
        assert len(sample_domain > 0) == len(sample_domain < 0)


def test_multi_source_domain_balanced_sampler():
    n_samples = 20
    dataset = make_dataset_from_moons_distribution(
        n_samples_source=n_samples,
        n_samples_target=n_samples,
        pos_source=[0.1, 0.2, 0.3],
        noise=0.1,
        random_state=42,
        return_dataset=True,
    )
    X, y, sample_domain = dataset.pack_train(
        as_sources=["s0", "s1", "s2"], as_targets=["t"]
    )
    X_dict = {"X": X.astype(np.float32), "sample_domain": sample_domain}

    dataset = Dataset(X_dict, y)
    source_domains = np.unique(sample_domain[sample_domain > 0])
    sampler = MultiSourceDomainBalancedSampler(
        dataset,
        10,
        source_domains=source_domains,
    )
    positive_indices = sampler.positive_indices
    for idx in positive_indices:
        assert np.unique(sample_domain[idx]).shape[0] == 1
    lenght_sources = [
        len(sample_domain[sample_domain == i])
        for i in np.unique(sample_domain[sample_domain > 0])
    ]
    max_length = np.max(lenght_sources)
    assert len(sampler) == max_length * len(source_domains) * 2

    # assert error if domain not in dataset
    with pytest.raises(AssertionError):
        sampler = MultiSourceDomainBalancedSampler(
            dataset,
            10,
            source_domains=[0, 1000],
        )


def test_multi_soure_domain_balanced_dataloader():
    n_samples = 30
    dataset = make_dataset_from_moons_distribution(
        n_samples_source=n_samples,
        n_samples_target=n_samples,
        pos_source=[0.1, 0.2, 0.3],
        noise=0.1,
        random_state=42,
        return_dataset=True,
    )
    X, y, sample_domain = dataset.pack_train(
        as_sources=["s0", "s1", "s2"], as_targets=["t"]
    )
    X_dict = {"X": X.astype(np.float32), "sample_domain": sample_domain}

    dataset = Dataset(X_dict, y)
    source_domains = np.unique(sample_domain[sample_domain > 0])
    dataloader = MultiSourceDomainBalancedDataLoader(
        dataset,
        batch_size=9,
        source_domains=source_domains,
    )

    for batch in dataloader:
        X, y = batch
        sample_domain_batch = X["sample_domain"]
        assert len(sample_domain_batch > 0) == len(sample_domain_batch < 0)
        assert (
            np.unique(sample_domain_batch[sample_domain_batch > 0]).shape[0]
            == np.unique(sample_domain[sample_domain > 0]).shape[0]
        )
        for domain in source_domains[1:]:
            assert torch.sum(sample_domain_batch == domain) == torch.sum(
                sample_domain_batch == source_domains[0]
            )

    # with more source than target
    dataset = make_dataset_from_moons_distribution(
        n_samples_source=2 * n_samples,
        n_samples_target=n_samples,
        pos_source=[0.1, 0.2, 0.3],
        noise=0.1,
        random_state=42,
        return_dataset=True,
    )
    X, y, sample_domain = dataset.pack_train(
        as_sources=["s0", "s1", "s2"], as_targets=["t"]
    )
    X_dict = {"X": X.astype(np.float32), "sample_domain": sample_domain}

    dataset = Dataset(X_dict, y)
    source_domains = np.unique(sample_domain[sample_domain > 0])
    dataloader = MultiSourceDomainBalancedDataLoader(
        dataset,
        batch_size=9,
        source_domains=source_domains,
    )

    for batch in dataloader:
        X, y = batch
        sample_domain_batch = X["sample_domain"]
        assert len(sample_domain_batch > 0) == len(sample_domain_batch < 0)
        assert (
            np.unique(sample_domain_batch[sample_domain_batch > 0]).shape[0]
            == np.unique(sample_domain[sample_domain > 0]).shape[0]
        )
        for domain in source_domains[1:]:
            assert torch.sum(sample_domain_batch == domain) == torch.sum(
                sample_domain_batch == source_domains[0]
            )

    # with more target than source
    dataset = make_dataset_from_moons_distribution(
        n_samples_source=n_samples,
        n_samples_target=2 * n_samples,
        pos_source=[0.1, 0.2, 0.3],
        noise=0.1,
        random_state=42,
        return_dataset=True,
    )
    X, y, sample_domain = dataset.pack_train(
        as_sources=["s0", "s1", "s2"], as_targets=["t"]
    )
    X_dict = {"X": X.astype(np.float32), "sample_domain": sample_domain}

    dataset = Dataset(X_dict, y)
    source_domains = np.unique(sample_domain[sample_domain > 0])
    dataloader = MultiSourceDomainBalancedDataLoader(
        dataset,
        batch_size=9,
        source_domains=source_domains,
    )

    for batch in dataloader:
        X, y = batch
        sample_domain_batch = X["sample_domain"]
        assert len(sample_domain_batch > 0) == len(sample_domain_batch < 0)
        assert (
            np.unique(sample_domain_batch[sample_domain_batch > 0]).shape[0]
            == np.unique(sample_domain[sample_domain > 0]).shape[0]
        )
        for domain in source_domains[1:]:
            assert torch.sum(sample_domain_batch == domain) == torch.sum(
                sample_domain_batch == source_domains[0]
            )
