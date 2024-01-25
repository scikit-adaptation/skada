# Author: Theo Gnassounou <theo.gnassounou@inria.fr>
#         Remi Flamary <remi.flamary@polytechnique.edu>
#
# License: BSD 3-Clause


from abc import abstractmethod

import torch
from torch.utils.data import DataLoader, Sampler
from skorch import NeuralNetClassifier

from .utils import _register_forwards_hook


class DomainAwareCriterion(torch.nn.Module):
    """Criterion for domain aware loss

    Parameters
    ----------
    criterion : torch criterion (class)
        The initialized criterion (loss) used to optimize the
        module with prediction on source.
    dacriterion : torch criterion (class)
        The initialized criterion (loss) used to compute the
        loss to reduce the divergence between domains.
    """

    def __init__(self, criterion, dacriterion):
        super(DomainAwareCriterion, self).__init__()
        self.criterion = criterion
        self.dacriterion = dacriterion

    def forward(
        self,
        y_pred,
        y_true,
    ):
        y_pred, y_pred_domain, features, sample_domain = y_pred  # unpack
        y_pred_s = y_pred[sample_domain > 0]
        y_pred_t = y_pred[sample_domain < 0]

        if y_pred_domain is not None:
            y_pred_domain_s = y_pred_domain[sample_domain > 0]
            y_pred_domain_t = y_pred_domain[sample_domain < 0]

        features_s = features[sample_domain > 0]
        features_t = features[sample_domain < 0]

        # predict
        return self.criterion(y_pred_s, y_true[sample_domain > 0]) + self.dacriterion(
            y_pred_s, y_pred_t, y_pred_domain_s, y_pred_domain_t, features_s, features_t
        )


class BaseDALoss(torch.nn.Module):
    def __init__(
        self,
    ):
        super(BaseDALoss, self).__init__()

    @abstractmethod
    def forward(
        self,
        y_pred_s,
        y_pred_t,
        y_pred_domain_s,
        y_pred_domain_t,
        features_s,
        features_t,
    ):
        """Compute the domain adaptation loss"""
        pass


class DomainBalancedSampler(Sampler):
    """Domain balanced sampler

    A sampler to have as much as source and target in the batch.

    Parameters
    ----------
    dataset : torch dataset
        The dataset to sample from.
    """

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
    """Domain aware module

    A domain aware module allowing to separe the source and target and
    compute their respective prediction and feaures.

    Parameters
    ----------
    module : torch module
        The module to use.
    layer_name : str
        The name of the layer to extract the features from.
    """

    def __init__(self, module, layer_name, domain_classifier=None):
        super(DomainAwareModule, self).__init__()
        self.module_ = module
        self.domain_classifier_ = domain_classifier
        self.layer_name = layer_name
        self.intermediate_layers = {}
        _register_forwards_hook(
            self.module_, self.intermediate_layers, [self.layer_name]
        )

    def forward(self, X, sample_domain):
        # choose source or target domain
        X_s = X[sample_domain > 0]
        X_t = X[sample_domain < 0]
        # predict
        y_pred_s = self.module_(X_s)
        features_s = self.intermediate_layers[self.layer_name]
        y_pred_t = self.module_(X_t)
        features_t = self.intermediate_layers[self.layer_name]

        if self.domain_classifier_ is not None:
            y_pred_domain_s = self.domain_classifier_(features_s)
            y_pred_domain_t = self.domain_classifier_(features_t)
            y_pred_domain = torch.cat((y_pred_domain_s, y_pred_domain_t), dim=0)
        else:
            y_pred_domain = None
        return (
            torch.cat((y_pred_s, y_pred_t), dim=0),  # predictions
            y_pred_domain,  # domain predictions
            torch.cat((features_s, features_t), dim=0),  # features
            sample_domain,  # domains
        )


class DomainAwareNet(NeuralNetClassifier):
    """Domain aware net

    A wrapper for NeuralNetClassifier to give a dict as input to the fit.

    Parameters
    ----------
    module : torch module
        The module to use.
    layer_name : str
        The name of the layer to extract the features from.
    domain_classifier : torch module
        The domain classifier to use.
    **kwargs : dict
        Keyword arguments passed to the skorch NeuralNetClassifier class.
    """

    def __init__(self, module, layer_name, domain_classifier=None, **kwargs):
        super().__init__(
            DomainAwareModule(module, layer_name, domain_classifier), **kwargs
        )

    def fit(self, X, y, sample_domain=None, **fit_params):
        """Fit the model

        Parameters
        ----------
        X : dict or torch tensor
            The input data. If a dict, it should contain a key 'X' with the
            input data and a key 'sample_domain' with the domain of each
            sample.
        y : torch tensor
            The target data.
        sample_domain : torch tensor
            The domain of each sample.
        """
        if not isinstance(X, dict):
            X = {"X": X}
            X["sample_domain"] = sample_domain

        return super().fit(X, y, **fit_params)
