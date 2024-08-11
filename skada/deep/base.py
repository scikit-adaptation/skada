# Author: Theo Gnassounou <theo.gnassounou@inria.fr>
#         Remi Flamary <remi.flamary@polytechnique.edu>
#         Yanis Lalou <yanis.lalou@polytechnique.edu>
#
# License: BSD 3-Clause

from abc import abstractmethod
from typing import Dict, Any, Union

import torch
from torch.utils.data import DataLoader, Sampler, Dataset
from sklearn.base import _clone_parametrized
from skorch import NeuralNetClassifier

from .utils import _register_forwards_hook

from skada.base import _DAMetadataRequesterMixin
from skada.utils import check_X_domain, check_X_y_domain

import numpy as np
from skorch.utils import to_device
from skorch.utils import to_numpy
from skorch.utils import to_tensor
from skorch.dataset import unpack_data
from collections.abc import Mapping

class DomainAwareCriterion(torch.nn.Module):
    """Criterion for domain aware loss

    Parameters
    ----------
    base_criterion : torch criterion (class)
        The initialized criterion (loss) used to optimize the
        module with prediction on source.
    adapt_criterion : torch criterion (class)
        The initialized criterion (loss) used to compute the
        loss to reduce the divergence between domains.
    reg: float, default=1
        Regularization parameter.
    """

    def __init__(self, base_criterion, adapt_criterion, reg=1, reduce=None):
        super(DomainAwareCriterion, self).__init__()
        self.base_criterion = base_criterion
        self.adapt_criterion = adapt_criterion
        self.reg = reg
        self.reduce = reduce

    def forward(
        self,
        y_pred,
        y_true,
    ):
        """
        Parameters
        ----------
        y_pred : tuple
            This tuple comprises all the different data
            needed to compute DA loss:
                - y_pred : prediction of the source and target domains
                - domain_pred : prediction of domain classifier if given
                - features :  features of the chosen layer
                  of source and target domains
                - sample_domain : giving the domain of each samples
        y_true :
            The true labels. Available for source, masked for target.
        """
        y_pred, domain_pred, features, sample_domain = y_pred  # unpack
        source_idx = (sample_domain >= 0)
        y_pred_s = y_pred[source_idx]
        y_pred_t = y_pred[~source_idx]

        if domain_pred is not None:
            domain_pred_s = domain_pred[source_idx]
            domain_pred_t = domain_pred[~source_idx]
        else:
            domain_pred_s = None
            domain_pred_t = None

        features_s = features[source_idx]
        features_t = features[~source_idx]

        # predict
        return self.base_criterion(
            y_pred_s, y_true[source_idx],
        ) + self.reg * self.adapt_criterion(
            y_true[source_idx],
            y_pred_s,
            y_pred_t,
            domain_pred_s,
            domain_pred_t,
            features_s,
            features_t,
        )


class BaseDALoss(torch.nn.Module):
    def __init__(
        self,
    ):
        super(BaseDALoss, self).__init__()

    @abstractmethod
    def forward(
        self,
        y_s,
        y_pred_s,
        y_pred_t,
        domain_pred_s,
        domain_pred_t,
        features_s,
        features_t,
    ):
        """Compute the domain adaptation loss

        Parameters
        ----------
        y_s :
            The true labels for source.
        y_pred_s :
            Predictions of the source domain.
        y_pred_t :
            Predictions of the target domain.
        domain_pred_s :
            Domain predictions of the source domain.
        domain_pred_t :
            Domain predictions of the source domain.
        features_s :
            Features of the chosen layer of source domain.
        features_t :
            Features of the chosen layer of target domain.
        """
        pass


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


class DomainAwareModule(torch.nn.Module):
    """Domain aware module

    A domain aware module allowing to separate the source and target and
    compute their respective prediction and feaures.

    Parameters
    ----------
    module : torch module (class or instance)
        A PyTorch :class:`~torch.nn.Module`.
    layer_name : str
        The name of the module's layer whose outputs are
        collected during the training for the adaptation.
    domain_classifier : torch module, default=None
        A PyTorch :class:`~torch.nn.Module` used to classify the
        domain. Could be None.
    """

    def __init__(self, base_module, layer_name, domain_classifier=None):
        super(DomainAwareModule, self).__init__()
        self.base_module_ = base_module
        self.domain_classifier_ = domain_classifier
        self.layer_name = layer_name
        self.intermediate_layers = {}
        self._setup_hooks()

    def _setup_hooks(self):
        _register_forwards_hook(
            self.base_module_, self.intermediate_layers, [self.layer_name]
        )

    def get_params(self, deep=True) -> Dict[str, Any]:
        return {
            'base_module': self.base_module_,
            'layer_name': self.layer_name,
            'domain_classifier': self.domain_classifier_,
        }

    def __sklearn_clone__(self) -> torch.nn.Module:
        estimator = _clone_parametrized(self, safe=True)
        estimator._setup_hooks()
        return estimator
    

    def forward(self, X, sample_domain=None, sample_weight=None, is_fit=False, return_features=False):
        if is_fit:
            source_idx = (sample_domain >= 0)

            X_s = X[source_idx]
            X_t = X[~source_idx]

            # Pass sample_weight to base_module_
            if sample_weight is not None:
                sample_weight_s = sample_weight[source_idx]
                sample_weight_t = sample_weight[~source_idx]

                y_pred_s = self.base_module_(X_s, sample_weight=sample_weight_s)
                features_s = self.intermediate_layers[self.layer_name]

                y_pred_t = self.base_module_(X_t, sample_weight=sample_weight_t)
                features_t = self.intermediate_layers[self.layer_name]
            else:
                y_pred_s = self.base_module_(X_s)
                features_s = self.intermediate_layers[self.layer_name]

                y_pred_t = self.base_module_(X_t)
                features_t = self.intermediate_layers[self.layer_name]

            if self.domain_classifier_ is not None:
                domain_pred_s = self.domain_classifier_(features_s)
                domain_pred_t = self.domain_classifier_(features_t)
                domain_pred = torch.empty(
                    (len(sample_domain)), 
                    device=domain_pred_s.device
                )
                domain_pred[source_idx] = domain_pred_s
                domain_pred[~source_idx] = domain_pred_t
            else:
                domain_pred = None

            y_pred = torch.empty(
                (len(sample_domain), y_pred_s.shape[1]),
                device=y_pred_s.device
            )

            y_pred[source_idx] = y_pred_s
            y_pred[~source_idx] = y_pred_t

            features = torch.empty(
                (len(sample_domain), features_s.shape[1]),
                device=features_s.device
            )

            features[source_idx] = features_s
            features[~source_idx] = features_t

            return (
                y_pred,
                domain_pred,
                features,
                sample_domain,
            )
        else:
            if return_features:
                return self.base_module_(X, sample_weight=sample_weight), self.intermediate_layers[self.layer_name]
            else:
                return self.base_module_(X, sample_weight=sample_weight)


class DomainAwareNet(NeuralNetClassifier, _DAMetadataRequesterMixin):
    __metadata_request__fit = {"sample_weight": True}
    __metadata_request__score = {"sample_weight": True}
    """
    A domain-aware neural network classifier with sample weight support.

    This class extends NeuralNetClassifier to handle domain-specific input data
    and sample weights. It supports various input formats and provides methods 
    for training, prediction, and feature extraction while considering domain 
    information and sample weights.

    Parameters:
    -----------
    module : torch.nn.Module
        The PyTorch module to be used as the core of the classifier.
    iterator_train : torch.utils.data.DataLoader, optional
        Custom data loader for training. If None, DomainBalancedDataLoader is used.
    criterion__reduce : bool, optional
        Whether to reduce the loss in the criterion. Default is False.
    **kwargs : dict
        Additional keyword arguments passed to the skorch NeuralNetClassifier.
    """

    def __init__(self, module, iterator_train=None, criterion__reduce=False, **kwargs):
        iterator_train = DomainBalancedDataLoader if iterator_train is None else iterator_train
        super().__init__(module, iterator_train=iterator_train, criterion__reduce=criterion__reduce, **kwargs)

    def fit(self, X: Union[Dict, torch.Tensor, np.ndarray, Dataset],
            y: Union[torch.Tensor, np.ndarray], 
            sample_domain: Union[torch.Tensor, np.ndarray] = None,
            sample_weight: Union[torch.Tensor, np.ndarray] = None, **fit_params):
        """
        Fit the model to the provided data.

        Parameters:
        -----------
        X : dict, torch.Tensor, np.ndarray, or torch.utils.data.Dataset
            The input data. If dict, it should contain 'X' and 'sample_domain' keys.
        y : torch.Tensor or np.ndarray
            The target data.
        sample_domain : torch.Tensor or np.ndarray, optional
            The domain of each sample (if not provided in X).
        sample_weight : torch.Tensor or np.ndarray, optional
            The weight of each sample.
        **fit_params : dict
            Additional parameters passed to the fit method of the base class.

        Returns:
        --------
        self : DomainAwareNet
            The fitted model.
        """
        # TODO val is not working
        # if train_split is None:
        # iterator_valid = None
        # else:
        # iterator_valid = (
        # DomainBalancedDataLoader if iterator_valid is None else iterator_valid
        # )
        X = self._prepare_input(X, sample_domain, sample_weight)
        X['X'], y, X['sample_domain'] = check_X_y_domain(X['X'], y, X['sample_domain'], allow_source=True)
        return super().fit(X, y, is_fit=True, **fit_params)

    def predict(self, X: Union[Dict, torch.Tensor, np.ndarray, Dataset], 
                sample_domain: Union[torch.Tensor, np.ndarray] = None, 
                sample_weight: Union[torch.Tensor, np.ndarray] = None, **predict_params):
        """
        Make predictions on the provided data.

        Parameters:
        -----------
        X : dict, torch.Tensor, np.ndarray, or torch.utils.data.Dataset
            The input data for prediction.
        sample_domain : torch.Tensor or np.ndarray, optional
            The domain of each sample (if not provided in X).
        sample_weight : torch.Tensor or np.ndarray, optional
            The weight of each sample (not used in prediction, but included for consistency).
        **predict_params : dict
            Additional parameters passed to the predict method of the base class.

        Returns:
        --------
        np.ndarray
            The predicted classes.
        """
        X = self._prepare_input(X, sample_domain, sample_weight)
        X['X'], X['sample_domain'] = check_X_domain(X['X'], X['sample_domain'], 
                                                    allow_source=True, allow_target=True,
                                                    allow_multi_source=True, allow_multi_target=True)
        return super().predict(X, **predict_params)

    def predict_proba(self, X: Union[Dict, torch.Tensor, np.ndarray, Dataset], 
                      sample_domain: Union[torch.Tensor, np.ndarray] = None, 
                      sample_weight: Union[torch.Tensor, np.ndarray] = None, **predict_params):
        """
        Predict class probabilities for the provided data.

        Parameters:
        -----------
        X : dict, torch.Tensor, np.ndarray, or torch.utils.data.Dataset
            The input data for prediction.
        sample_domain : torch.Tensor or np.ndarray, optional
            The domain of each sample (if not provided in X).
        sample_weight : torch.Tensor or np.ndarray, optional
            The weight of each sample (not used in prediction, but included for consistency).
        **predict_params : dict
            Additional parameters passed to the predict_proba method of the base class.

        Returns:
        --------
        np.ndarray
            The predicted class probabilities.
        """
        X = self._prepare_input(X, sample_domain, sample_weight)
        X['X'], X['sample_domain'] = check_X_domain(X['X'], X['sample_domain'], 
                                                    allow_source=True, allow_target=True,
                                                    allow_multi_source=True, allow_multi_target=True)
        return super().predict_proba(X, **predict_params)
    
    def predict_features(self, X: Union[Dict, torch.Tensor, np.ndarray, Dataset]):
        """
        Extract features from the input data using the trained model.

        Parameters:
        -----------
        X : dict, torch.Tensor, np.ndarray, or torch.utils.data.Dataset
            The input data for feature extraction.

        Returns:
        --------
        np.ndarray
            The extracted features.
        """
        if not self.initialized_:
            self.initialize()

        X = self._prepare_input(X, None)
        X, sample_domain = check_X_domain(X['X'], X['sample_domain'], 
                                          allow_source=True, allow_target=True,
                                          allow_multi_source=True, allow_multi_target=True)

        X = torch.tensor(X) if not torch.is_tensor(X) else X

        features_list = []
        for features in self.feature_iter(X, training=False, return_features=True):
            features = features[0] if isinstance(features, tuple) else features
            features_list.append(to_numpy(features))
        return np.concatenate(features_list, 0)

    def score(self, X: Union[Dict, torch.Tensor, np.ndarray], y: Union[torch.Tensor, np.ndarray], 
              sample_domain: Union[torch.Tensor, np.ndarray] = None, 
              sample_weight: Union[torch.Tensor, np.ndarray] = None, **score_params):
        """
        Compute the mean accuracy on the provided data and labels.

        Parameters:
        -----------
        X : dict, torch.Tensor, or np.ndarray
            The input data for scoring.
        y : torch.Tensor or np.ndarray
            The true labels.
        sample_domain : torch.Tensor or np.ndarray, optional
            The domain of each sample (if not provided in X).
        sample_weight : torch.Tensor or np.ndarray, optional
            The weight of each sample (not used in scoring, but included for consistency).
        **score_params : dict
            Additional parameters passed to the score method of the base class.

        Returns:
        --------
        float
            The mean accuracy score.
        """
        X = self._prepare_input(X, sample_domain, sample_weight)
        X['X'], X['sample_domain'] = check_X_domain(X['X'], X['sample_domain'], 
                                                    allow_source=True, allow_target=True,
                                                    allow_multi_source=True, allow_multi_target=True)
        return super().score(X, y, **score_params)
    
    def feature_iter(self, X: torch.Tensor, training: bool = False, device: str = 'cpu', return_features: bool = True):
        """
        Iterate over the input data and yield features.

        Parameters:
        -----------
        X : torch.Tensor
            The input data.
        training : bool, optional
            Whether to use training mode (default is False).
        device : str, optional
            The device to use for computation (default is 'cpu').
        return_features : bool, optional
            Whether to return features (default is True).

        Yields:
        -------
        torch.Tensor
            The extracted features for each batch.
        """
        dataset = self.get_dataset(X)
        iterator = self.get_iterator(dataset, training=training)
        for batch in iterator:
            _, features = self.feature_eval_step(batch, training=training, return_features=return_features)
            yield to_device(features, device=device)

    def feature_eval_step(self, batch: Any, training: bool = False, return_features: bool = True):
        """
        Perform a single feature evaluation step.

        Parameters:
        -----------
        batch : Any
            The input batch data.
        training : bool, optional
            Whether to use training mode (default is False).
        return_features : bool, optional
            Whether to return features (default is True).

        Returns:
        --------
        tuple
            A tuple containing the output and features.
        """
        self.check_is_fitted()
        Xi, _ = unpack_data(batch)
        with torch.set_grad_enabled(training):
            self._set_training(training)
            return self.feature_infer(Xi, return_features)
        
    def feature_infer(self, x: Union[torch.Tensor, Dict[str, Any]], return_features: bool):
        """
        Perform inference to extract features.

        Parameters:
        -----------
        x : torch.Tensor or dict
            The input data.
        return_features : bool
            Whether to return features.

        Returns:
        --------
        torch.Tensor or tuple
            The output of the module, potentially including extracted features.
        """
        x = to_tensor(x, device=self.device)
        if isinstance(x, dict):
            x_dict = self._merge_x_and_fit_params(x, {"return_features": return_features})
            return self.module_(**x_dict)
        return self.module_(x, return_features=return_features)

    def _prepare_input(self, X: Union[Dict, torch.Tensor, np.ndarray, Dataset], 
                       sample_domain: Union[torch.Tensor, np.ndarray] = None,
                       sample_weight: Union[torch.Tensor, np.ndarray] = None) -> Dict[str, Any]:
        """
        Prepare the input data for processing, including sample weights if provided.

        Parameters:
        -----------
        X : dict, torch.Tensor, np.ndarray, or torch.utils.data.Dataset
            The input data.
        sample_domain : torch.Tensor or np.ndarray, optional
            The domain of each sample.
        sample_weight : torch.Tensor or np.ndarray, optional
            The weight of each sample.

        Returns:
        --------
        dict
            A dictionary containing 'X', 'sample_domain', and optionally 'sample_weight' keys.

        Raises:
        -------
        ValueError
            If the input format is invalid or missing required information.
        """
        if isinstance(X, dict):
            if "X" not in X or "sample_domain" not in X:
                raise ValueError("X should contain both 'X' and 'sample_domain' keys.")
            if sample_weight is not None:
                X['sample_weight'] = sample_weight
            return X
        elif isinstance(X, Dataset):
            return self._process_dataset(X)
        else:
            result = {"X": X, "sample_domain": sample_domain}
            if sample_weight is not None:
                result['sample_weight'] = sample_weight
            return result

    def _process_dataset(self, dataset: Dataset) -> Dict[str, np.ndarray]:
        """
        Process a PyTorch Dataset into a dictionary format.

        Parameters:
        -----------
        dataset : torch.utils.data.Dataset
            The input dataset to process.

        Returns:
        --------
        dict
            A dictionary containing 'X', 'sample_domain', and optionally 'sample_weight' as numpy arrays.

        Raises:
        -------
        ValueError
            If the dataset samples are not in the expected format.
        """
        X, sample_domain, sample_weight = [], [], []
        has_sample_weight = False
        for sample in dataset:
            if not isinstance(sample, dict) or "X" not in sample or "sample_domain" not in sample:
                raise ValueError("Dataset samples should be dictionaries with 'X' and 'sample_domain' keys.")
            X.append(sample['X'])
            sample_domain.append(sample['sample_domain'])
            if 'sample_weight' in sample and sample['sample_weight'] is not None:
                sample_weight.append(sample['sample_weight'])
                has_sample_weight = True
        result = {"X": np.array(X), "sample_domain": np.array(sample_domain)}
        if has_sample_weight:
            result['sample_weight'] = np.array(sample_weight)
        return result

    def get_loss(self, y_pred, y_true, X, *args, **kwargs):
        """
        Calculate the weighted loss using sample weights.

        Parameters:
        -----------
        y_pred : torch.Tensor
            The predicted values.
        y_true : torch.Tensor
            The true values.
        X : dict
            The input data dictionary, which may contain 'sample_weight'.
        *args : tuple
            Additional positional arguments.
        **kwargs : dict
            Additional keyword arguments.

        Returns:
        --------
        torch.Tensor
            The calculated loss, weighted by sample weights if provided.
        """
        loss_unreduced = super().get_loss(y_pred, y_true, X, *args, **kwargs)
        if 'sample_weight' in X and X['sample_weight'] is not None:
            sample_weight = to_tensor(X['sample_weight'], device=self.device)
            loss_reduced = (sample_weight * loss_unreduced).mean()
        else:
            loss_reduced = loss_unreduced.mean()
        return loss_reduced
