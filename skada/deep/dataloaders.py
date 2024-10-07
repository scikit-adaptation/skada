# Author: Theo Gnassounou <theo.gnassounou@inria.fr>
#         Remi Flamary <remi.flamary@polytechnique.edu>
#         Yanis Lalou <yanis.lalou@polytechnique.edu>
#         Antoine Collas <contact@antoinecollas.fr>
#
# License: BSD 3-Clause

import torch
from torch.utils.data import DataLoader, Sampler


class DomainBalancedSampler(Sampler):
    """Domain balanced sampler

    A sampler to have as much as source and target in the batch.

    Parameters
    ----------
    dataset : torch dataset
        The dataset to sample from.
    batch_size : int
        The batch size.
    max_samples : str, default='max'
        The maximum number of samples to use.
        It can be 'max', 'min', 'source', or 'target'.
    """

    def __init__(self, dataset, batch_size, max_samples="max"):
        self.dataset = dataset
        self.positive_indices = [
            idx for idx, sample in enumerate(dataset) if sample[0]["sample_domain"] >= 0
        ]
        self.negative_indices = [
            idx for idx, sample in enumerate(dataset) if sample[0]["sample_domain"] < 0
        ]
        self.num_samples_source = (
            len(self.positive_indices) - len(self.positive_indices) % batch_size
        )
        self.num_samples_target = (
            len(self.negative_indices) - len(self.negative_indices) % batch_size
        )
        if max_samples == "max":
            self.num_samples = max(self.num_samples_source, self.num_samples_target)
        elif max_samples == "min":
            self.num_samples = min(self.num_samples_source, self.num_samples_target)
        elif max_samples == "source":
            self.num_samples = self.num_samples_source
        elif max_samples == "target":
            self.num_samples = self.num_samples_target

    def __iter__(self):
        positive_sampler = torch.utils.data.sampler.RandomSampler(self.positive_indices)
        negative_sampler = torch.utils.data.sampler.RandomSampler(self.negative_indices)

        positive_iter = iter(positive_sampler)
        negative_iter = iter(negative_sampler)

        for _ in range(self.num_samples):
            try:
                pos_idx = self.positive_indices[next(positive_iter)]
            except StopIteration:
                positive_iter = iter(positive_sampler)
                pos_idx = self.positive_indices[next(positive_iter)]
            try:
                neg_idx = self.negative_indices[next(negative_iter)]
            except StopIteration:
                negative_iter = iter(negative_sampler)
                neg_idx = self.negative_indices[next(negative_iter)]
            yield pos_idx
            yield neg_idx

    def __len__(self):
        return 2 * self.num_samples


class DomainBalancedDataLoader(DataLoader):
    """Domain balanced data loader

    A data loader to have as much as source and target in the batch.

    Parameters
    ----------
    dataset : torch dataset
        The dataset to sample from.
    batch_size : int
        The batch size.
    max_samples : str, default='max'
        The maximum number of samples to use.
        It can be 'max', 'min', 'source', or 'target'.
    """

    def __init__(
        self,
        dataset,
        batch_size,
        max_samples="max",
        shuffle=False,
        sampler=None,
        batch_sampler=None,
        num_workers=0,
        collate_fn=None,
        pin_memory=False,
        drop_last=False,
        timeout=0,
        worker_init_fn=None,
        multiprocessing_context=None,
    ):
        sampler = DomainBalancedSampler(dataset, batch_size, max_samples=max_samples)
        super().__init__(
            dataset,
            2 * batch_size,
            shuffle,
            sampler,
            batch_sampler,
            num_workers,
            collate_fn,
            pin_memory,
            drop_last,
            timeout,
            worker_init_fn,
            multiprocessing_context,
        )


class DomainOnlySampler(Sampler):
    """Domain balanced sampler

    A sampler to have only source or target domain in the batch.

    Parameters
    ----------
    dataset : torch dataset
        The dataset to sample from.
    """

    def __init__(self, dataset, batch_size, domain_used="source"):
        self.dataset = dataset
        if domain_used == "source":
            self.indices = [
                idx
                for idx, sample in enumerate(dataset)
                if sample[0]["sample_domain"] >= 0
            ]
        elif domain_used == "target":
            self.indices = [
                idx
                for idx, sample in enumerate(dataset)
                if sample[0]["sample_domain"] < 0
            ]
        else:
            raise ValueError(f"Unknown domain_used: {domain_used}")
        self.num_samples = len(self.indices) - len(self.indices) % batch_size

    def __iter__(self):
        sampler = torch.utils.data.sampler.RandomSampler(self.indices)

        iterator = iter(sampler)

        for _ in range(self.num_samples):
            idx = self.indices[next(iterator)]
            yield idx

    def __len__(self):
        return self.num_samples


class DomainOnlyDataLoader(DataLoader):
    """Domain balanced data loader

    A data loader to have either source or target domain in the batch.

    Parameters
    ----------
    dataset : torch dataset
        The dataset to sample from.
    batch_size : int
        The batch size.
    domain_used : str, default='source'
        The domain to use for the batch.
    """

    def __init__(
        self,
        dataset,
        batch_size,
        domain_used="source",
        shuffle=False,
        sampler=None,
        batch_sampler=None,
        num_workers=0,
        collate_fn=None,
        pin_memory=False,
        drop_last=False,
        timeout=0,
        worker_init_fn=None,
        multiprocessing_context=None,
    ):
        sampler = DomainOnlySampler(dataset, batch_size, domain_used=domain_used)
        super().__init__(
            dataset,
            batch_size,
            shuffle,
            sampler,
            batch_sampler,
            num_workers,
            collate_fn,
            pin_memory,
            drop_last,
            timeout,
            worker_init_fn,
            multiprocessing_context,
        )


class MultiSourceDomainBalancedSampler(Sampler):
    """Domain balanced sampler

    A sampler to have as much as source and target in the batch.

    Parameters
    ----------
    dataset : torch dataset
        The dataset to sample from.
    batch_size : int
        The batch size. It should be a multiple of the number of source domains.
    source_domains : list of int
        The list of source domains.
    """

    def __init__(self, dataset, batch_size, source_domains):
        self.dataset = dataset
        self.batch_size = batch_size
        self.source_domains = source_domains
        self.positive_indices = []
        for domain in source_domains:
            assert domain >= 0
            self.positive_indices.append(
                [
                    idx
                    for idx, sample in enumerate(dataset)
                    if sample[0]["sample_domain"] == domain
                ]
            )
        assert all(
            [len(self.positive_indices[i]) > 0 for i in range(len(self.source_domains))]
        ), "Some source domains are not in the dataset."
        self.negative_indices = [
            idx for idx, sample in enumerate(dataset) if sample[0]["sample_domain"] < 0
        ]
        max_len = max([len(indices) for indices in self.positive_indices])
        self.num_samples = max_len - max_len % batch_size

    def __iter__(self):
        positive_samplers = [
            torch.utils.data.sampler.RandomSampler(self.positive_indices[i])
            for i in range(len(self.source_domains))
        ]
        negative_sampler = torch.utils.data.sampler.RandomSampler(self.negative_indices)

        positive_iters = [
            iter(positive_samplers[i]) for i in range(len(self.source_domains))
        ]
        negative_iter = iter(negative_sampler)
        for _ in range(
            self.num_samples // (self.batch_size // len(self.source_domains))
        ):
            for i in range(len(self.source_domains)):
                for _ in range(self.batch_size // len(self.source_domains)):
                    try:
                        pos_idx = self.positive_indices[i][next(positive_iters[i])]
                    except StopIteration:
                        positive_iters[i] = iter(positive_samplers[i])
                        pos_idx = self.positive_indices[i][next(positive_iters[i])]
                    try:
                        neg_idx = self.negative_indices[next(negative_iter)]
                    except StopIteration:
                        negative_iter = iter(negative_sampler)
                        neg_idx = self.negative_indices[next(negative_iter)]
                    yield pos_idx
                    yield neg_idx

    def __len__(self):
        return self.num_samples * len(self.source_domains) * 2


class MultiSourceDomainBalancedDataLoader(DataLoader):
    """Domain balanced data loader

    A data loader to have as much as source and target in the batch.

    Parameters
    ----------
    dataset : torch dataset
        The dataset to sample from.
    batch_size : int
        The batch size. It should be a multiple of the number of source domains.
        The final batch size will be 2 * len(source_domains) * batch_size.
    source_domains : list of int
        The list of source domains.
    """

    def __init__(
        self,
        dataset,
        batch_size,
        source_domains,
        shuffle=False,
        sampler=None,
        batch_sampler=None,
        num_workers=0,
        collate_fn=None,
        pin_memory=False,
        drop_last=False,
        timeout=0,
        worker_init_fn=None,
        multiprocessing_context=None,
    ):
        sampler = MultiSourceDomainBalancedSampler(dataset, batch_size, source_domains)
        super().__init__(
            dataset,
            2 * batch_size,
            shuffle,
            sampler,
            batch_sampler,
            num_workers,
            collate_fn,
            pin_memory,
            drop_last,
            timeout,
            worker_init_fn,
            multiprocessing_context,
        )
