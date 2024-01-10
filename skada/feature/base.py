# Author: Theo Gnassounou <theo.gnassounou@inria.fr>
#         Remi Flamary <remi.flamary@polytechnique.edu>
#
# License: BSD 3-Clause


import torch
from torch.utils.data import DataLoader, Sampler

from .utils import _register_forwards_hook


class DomainAwareCriterion(torch.nn.Module):
    def __init__(self, criterion, dacriterion):
        super(DomainAwareCriterion, self).__init__()
        self.criterion = criterion
        self.dacriterion = dacriterion

    def forward(
        self,
        y_pred,
        y_true,
    ):
        # choose source or target domain
        y_pred, features, sample_domain = y_pred
        ys = y_pred[sample_domain > 0]
        yt = y_pred[sample_domain < 0]

        features_s = features[sample_domain > 0]
        features_t = features[sample_domain < 0]

        # predict
        return self.criterion(ys, y_true[sample_domain > 0]) + self.dacriterion(
            yt, features_s, features_t
        )


class DomainBalancedSampler(Sampler):
    def __init__(self, dataset):
        self.dataset = dataset
        self.positive_indices = [
            idx for idx, sample in enumerate(dataset) if sample[0]["sample_domain"] >= 0
        ]
        self.negative_indices = [
            idx for idx, sample in enumerate(dataset) if sample[0]["sample_domain"] < 0
        ]
        self.num_samples = (
            min(len(self.positive_indices), len(self.negative_indices)) * 2
        )

    def __iter__(self):
        positive_sampler = torch.utils.data.sampler.RandomSampler(self.positive_indices)
        negative_sampler = torch.utils.data.sampler.RandomSampler(self.negative_indices)

        positive_iter = iter(positive_sampler)
        negative_iter = iter(negative_sampler)

        for _ in range(self.num_samples // 2):
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
        return self.num_samples


class DomainBalancedDataLoader(DataLoader):
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
        sampler = DomainBalancedSampler(dataset)
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


class DomainAwareModule(torch.nn.Module):
    def __init__(self, module, layer_name):
        super(DomainAwareModule, self).__init__()
        self.module_ = module
        self.layer_name = layer_name
        self.intermediate_layers = {}
        _register_forwards_hook(
            self.module_, self.intermediate_layers, [self.layer_name]
        )

    def forward(self, X, sample_domain):
        # choose source or target domain
        Xs = X[sample_domain > 0]
        Xt = X[sample_domain < 0]
        # predict
        ys = self.module_(Xs)
        features_s = self.intermediate_layers[self.layer_name]
        yt = self.module_(Xt)
        features_t = self.intermediate_layers[self.layer_name]
        return (
            torch.cat((ys, yt), dim=0),  # prediction
            torch.cat((features_s, features_t), dim=0),  # feature
            sample_domain,  # domain
        )
