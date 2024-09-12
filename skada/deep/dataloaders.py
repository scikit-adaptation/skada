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
    """

    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.positive_indices = [
            idx for idx, sample in enumerate(dataset) if sample[0]["sample_domain"] >= 0
        ]
        self.negative_indices = [
            idx for idx, sample in enumerate(dataset) if sample[0]["sample_domain"] < 0
        ]
        self.num_samples = (
            len(self.positive_indices) - len(self.positive_indices) % batch_size
        )

    def __iter__(self):
        positive_sampler = torch.utils.data.sampler.RandomSampler(self.positive_indices)
        negative_sampler = torch.utils.data.sampler.RandomSampler(self.negative_indices)

        positive_iter = iter(positive_sampler)
        negative_iter = iter(negative_sampler)

        for _ in range(self.num_samples):
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
    """

    def __init__(
        self,
        dataset,
        batch_size,
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
        sampler = DomainBalancedSampler(dataset, batch_size)
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


class MultiSourceDomainBalancedSampler(Sampler):
    """Domain balanced sampler

    A sampler to have as much as source and target in the batch.

    Parameters
    ----------
    dataset : torch dataset
        The dataset to sample from.
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
        for _ in range(self.num_samples // self.batch_size):
            for i in range(len(self.source_domains)):
                for _ in range(self.batch_size):
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
                    print(pos_idx, neg_idx)
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
