# Author: Theo Gnassounou <theo.gnassounou@inria.fr>
#         Remi Flamary <remi.flamary@polytechnique.edu>
#         Yanis Lalou <yanis.lalou@polytechnique.edu>
#         Antoine Collas <contact@antoinecollas.fr>
#         Maxence Barneche <maxence.barneche@etu-upsaclay.fr>
#
# License: BSD 3-Clause

from abc import abstractmethod
from typing import Dict, Any, Union

import torch
from torch.utils.data import DataLoader, Sampler
from sklearn.base import _clone_parametrized
from sklearn.metrics import accuracy_score
from skorch import NeuralNetClassifier, NeuralNetRegressor, NeuralNet, NeuralNetBinaryClassifier

from .utils import _register_forwards_hook, _infer_predict_nonlinearity

from skada.base import _DAMetadataRequesterMixin
from sklearn.utils.validation import check_array

import numpy as np
from skorch.utils import to_device
from skorch.utils import to_numpy, to_tensor
from skorch.dataset import unpack_data, Dataset
from collections.abc import Mapping

from pandas import DataFrame

_EMPTY_ = torch.Tensor()
_EMPTY_INT_ = torch.tensor([],dtype=torch.int64)
_DEFAULT_SAMPLE_DOMAIN_ = 0
_NO_LABEL_ = -1

class DomainAwareCriterion(torch.nn.Module):
    """Criterion for domain aware loss

    Parameters
    ----------
    base_criterion : torch criterion (instance)
        The initialized criterion (loss) used to optimize the
        module with prediction on source.
    adapt_criterion : torch criterion (instance)
        The initialized criterion (loss) used to compute the
        loss to reduce the divergence between domains.
    reg: float, default=1
        Regularization parameter for DA loss.
    reduction: str, default='mean'
        Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'.
        'none': no reduction will be applied,
        'mean': the sum of the output will be divided by the number of elements in the output,
        'sum': the output will be summed.
    train_on_target: bool, default=False
        If True, the loss is computed on target domain.
    """

    def __init__(
        self,
        base_criterion,
        adapt_criterion,
        reg=1,
        reduction="mean",
        train_on_target=False,
    ):
        super(DomainAwareCriterion, self).__init__()
        self.base_criterion = base_criterion
        self.adapt_criterion = adapt_criterion
        self.reg = reg
        self.train_on_target = train_on_target

        # Update the reduce parameter for both criteria if specified
        if hasattr(self.base_criterion, "reduction"):
            self.base_criterion.reduction = reduction

        # TODO: implement losses between source and target
        # that are sum of the losses of each sample
        # if hasattr(self.adapt_criterion, 'reduction'):
        #     self.adapt_criterion.reduction = reduction

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
        y_pred, domain_pred, features, sample_domain, sample_idx = y_pred  # unpack
        source_idx = sample_domain >= 0
        y_pred_s = y_pred[source_idx]
        y_pred_t = y_pred[~source_idx]

        if domain_pred is not None:
            domain_pred_s = domain_pred[source_idx]
            domain_pred_t = domain_pred[~source_idx]
        else:
            domain_pred_s = None
            domain_pred_t = None

        if features is not None:
            features_s = features[source_idx]
            features_t = features[~source_idx]
        else:
            features_s = None
            features_t = None

        if sample_idx is not None:
            sample_idx_s = sample_idx[source_idx]
            sample_idx_t = sample_idx[~source_idx]
        else:
            sample_idx_s = None
            sample_idx_t = None

        if self.train_on_target:
            base_loss = self.base_criterion(y_pred_t, y_true[~source_idx])
        else:
            base_loss = self.base_criterion(y_pred_s, y_true[source_idx])

        # predict
        return base_loss + self.reg * self.adapt_criterion(
            y_s=y_true[source_idx],
            y_pred_s=y_pred_s,
            y_pred_t=y_pred_t,
            domain_pred_s=domain_pred_s,
            domain_pred_t=domain_pred_t,
            features_s=features_s,
            features_t=features_t,
            sample_idx_s=sample_idx_s,
            sample_idx_t=sample_idx_t,
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
        sample_idx,
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
    batch_size : int
        The batch size.
    max_samples : str, default='max'
        The maximum number of samples to use. It can be 'max', 'min', 'source', or 'target'.
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
        The maximum number of samples to use. It can be 'max', 'min', 'source', or 'target'.
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
                idx for idx, sample in enumerate(dataset) if sample[0]["sample_domain"] >= 0
            ]
        elif domain_used == "target":
            self.indices = [
                idx for idx, sample in enumerate(dataset) if sample[0]["sample_domain"] < 0
            ]
        else:
            raise ValueError(f"Unknown domain_used: {domain_used}")
        self.num_samples = (
            len(self.indices) - len(self.indices) % batch_size
        )

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
            "base_module": self.base_module_,
            "layer_name": self.layer_name,
            "domain_classifier": self.domain_classifier_,
        }

    def __sklearn_clone__(self) -> torch.nn.Module:
        estimator = _clone_parametrized(self, safe=True)
        estimator._setup_hooks()
        return estimator

    def forward(
        self,
        X,
        sample_domain=None,
        sample_weight=None,
        is_fit=False,
        return_features=False,
        sample_idx=None,
    ):
        if is_fit:
            if sample_weight is not None:
                y_pred = self.base_module_(X, sample_weight=sample_weight)
            else:
                y_pred = self.base_module_(X)

            if self.layer_name is not None:
                features = self.intermediate_layers[self.layer_name]
            else:
                features = None

            if self.domain_classifier_ is not None:
                domain_pred = self.domain_classifier_(features)
            else:
                domain_pred = None

            return (
                y_pred,
                domain_pred,
                features,
                sample_domain,
                sample_idx
            )
        else:
            if return_features:
                return (
                    self.base_module_(X, sample_weight=sample_weight),
                    self.intermediate_layers[self.layer_name],
                )
            else:
                return self.base_module_(X, sample_weight=sample_weight)


class _DomainAwareNet(NeuralNet, _DAMetadataRequesterMixin):
    __metadata_request__fit = {"sample_weight": True}
    __metadata_request__score = {'sample_weight': True, 'sample_domain': True, 'allow_source': True}
    """
    Base class for a domain-aware neural network with sample weight support.

    This class extends NeuralNet to handle domain-specific input data
    and sample weights. It supports various input formats and provides methods
    for training, prediction, and feature extraction while considering domain
    information and sample weights.

    This class should hardly, if ever, be used directly. Instead, use
    DomainAwareNetClassifier, DomainAwareNetDualClassifier or DomainAwareNetRegressor,
    which are specialized for classification, binary classification and
    regression tasks, respectively.

    Parameters:
    -----------
    module : torch.nn.Module
        The PyTorch module to be used as the core of the classifier.
    iterator_train : torch.utils.data.DataLoader, optional
        Custom data loader for training. If None, DomainBalancedDataLoader is used.
    **kwargs : dict
        Additional keyword arguments passed to the skorch NeuralNetClassifier.
    """

    def __init__(self, module, iterator_train=None, **kwargs):
        # TODO val is not working
        # if train_split is None:
        #     iterator_valid = None
        # else:
        #     iterator_valid = (
        #         DomainBalancedDataLoader if iterator_valid is None else iterator_valid
        #     )
        iterator_train = (
            DomainBalancedDataLoader if iterator_train is None else iterator_train
        )
        super().__init__(module, iterator_train=iterator_train, **kwargs)

    def fit(
        self,
        X: Union[Dict, torch.Tensor, np.ndarray, Dataset],
        y: Union[torch.Tensor, np.ndarray],
        sample_domain: Union[torch.Tensor, np.ndarray] = None,
        sample_weight: Union[torch.Tensor, np.ndarray] = None,
        **fit_params
    ):
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
        X = self._prepare_input(X, y, sample_domain, sample_weight)
        return super().fit(X, None, is_fit=True, **fit_params)

    def _get_predict_nonlinearity(self):
        """Return the nonlinearity to be applied to the prediction

        This can be useful, e.g., when
        :func:`~skada.DomainAwareNet.predict_proba`
        should return probabilities but a criterion is used that does
        not expect probabilities. In that case, the module can return
        whatever is required by the criterion and the
        ``predict_nonlinearity`` transforms this output into
        probabilities.

        The nonlinearity is applied only when calling
        :func:`~skada.DomainAwareNet.predict` or
        :func:`~skada.DomainAwareNet.predict_proba`
        but not anywhere else -- notably, the loss is unaffected by
        this nonlinearity.

        Raises
        ------
        TypeError
          Raise a TypeError if the return value is not callable.

        Returns
        -------
        nonlin : callable
          A callable that takes a single argument, which is a PyTorch
          tensor, and returns a PyTorch tensor.

        """
        self.check_is_fitted()
        nonlin = self.predict_nonlinearity
        if nonlin is None:
            nonlin = _identity
        elif nonlin == 'auto':
            nonlin = _infer_predict_nonlinearity(self)
        if not callable(nonlin):
            raise TypeError("predict_nonlinearity has to be a callable, 'auto' or None")
        return nonlin

    def predict_proba(
        self,
        X: Union[Dict, torch.Tensor, np.ndarray, Dataset],
        sample_domain: Union[torch.Tensor, np.ndarray] = None,
        sample_weight: Union[torch.Tensor, np.ndarray] = None,
        allow_source: bool = False,
        **predict_params
    ):
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
        allow_source: bool = False,
            Allow the presence of source domains. 
            If False, only the target are selected for prediction.
        **predict_params : dict
            Additional parameters passed to the predict_proba method of the base class.

        Returns:
        --------
        np.ndarray
            The predicted class probabilities.
        """
        X = self._prepare_input(X, None, sample_domain, sample_weight)
        if not allow_source:
            X = X.select_target()
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

        X = self._prepare_input(X)
        
        features_list = []
        for features in self.feature_iter(X, training=False):
            features = features[0] if isinstance(features, tuple) else features
            features_list.append(to_numpy(features))
        return np.concatenate(features_list, 0)

    def feature_iter(
        self, X: torch.Tensor, training: bool = False, device: str = "cpu"
    ):
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

        Yields:
        -------
        torch.Tensor
            The extracted features for each batch.
        """
        dataset = self.get_dataset(X)
        iterator = self.get_iterator(dataset, training=training)
        for batch in iterator:
            _, features = self.feature_eval_step(batch, training=training)
            yield to_device(features, device=device)

    def feature_eval_step(self, batch: Any, training: bool = False):
        """
        Perform a single feature evaluation step.

        Parameters:
        -----------
        batch : Any
            The input batch data.
        training : bool, optional
            Whether to use training mode (default is False).

        Returns:
        --------
        tuple
            A tuple containing the output and features.
        """
        self.check_is_fitted()
        Xi, _ = unpack_data(batch)
        with torch.set_grad_enabled(training):
            self._set_training(training)
            return self.feature_infer(Xi)

    def feature_infer(self, x: Union[torch.Tensor, Dict[str, Any]], **fit_params):
        """
        Perform inference to extract features.

        Parameters:
        -----------
        x : torch.Tensor or dict
            The input data.
        **fit_params : dict
            Additional parameters passed to the ``forward`` method of
            the module and to the ``self.train_split`` call.

        Returns:
        --------
        torch.Tensor or tuple
            The output of the module, potentially including extracted features.
        """
        x = to_tensor(x, device=self.device)
        if isinstance(x, Mapping):
            x_dict = self._merge_x_and_fit_params(x, fit_params)
            return self.module_(return_features=True, **x_dict)
        return self.module_(x, return_features=True, **fit_params)

    def _prepare_input(
        self,
        X: Union[Dict, torch.Tensor, np.ndarray, 'DeepDADataset', Dataset],
        y: Union[torch.Tensor, np.ndarray] = None,
        sample_domain: Union[torch.Tensor, np.ndarray] = None,
        sample_weight: Union[torch.Tensor, np.ndarray] = None
    ) -> 'DeepDADataset':
        f"""
        Prepare the input data for processing, including sample weights if provided.

        Parameters:
        -----------
        X : dict, torch.Tensor, np.ndarray, or torch.utils.data.Dataset
            The input data.
        y : torch.Tensor or np.ndarray, optional
            The target labels. If not provided, the dataset will have default
            labels {_NO_LABEL_}.
        sample_domain : torch.Tensor or np.ndarray, optional
            The domain of each sample.
        sample_weight : torch.Tensor or np.ndarray, optional
            The weight of each sample.

        Returns:
        --------
        DeepDADataset: the input converted if needed

        """
        if isinstance(X, DeepDADataset):
            return X
        elif isinstance(X, Dataset):
            X, y = self._process_dataset(X)
            return DeepDADataset(X, y, sample_domain, sample_weight, self.device)
        else:
            dataset = DeepDADataset(X, y, sample_domain, sample_weight, self.device)
            return dataset
        
    def _process_dataset(
        self, dataset: Dataset
    ) -> Union[Dict[str, np.ndarray], np.ndarray]:
        """
        Process a PyTorch Dataset into a dictionary format.

        Parameters:
        -----------
        dataset : torch.utils.data.Dataset
            The input dataset to process.

        Returns:
        --------
        dict
            A dictionary containing 'X', 'sample_domain', and optionally 'y' and 'sample_weight' as numpy arrays.
        np.ndarray
            y as a numpy array.

        Raises:
        -------
        ValueError
            If the dataset samples are not in the expected format.
        """
        X, y, sample_domain, sample_weight = [], [], [], []
        # TODO: remove has_y or add in l744?
        has_y, has_sample_weight = False, False
        for sample in dataset:
            # Sample is a tuple (X, y) from skorch.dataset.Dataset
            x, y_ = sample
            if isinstance(x, dict) and "X" in x and "sample_domain" in x:
                X.append(x["X"])
                sample_domain.append(x["sample_domain"])
                y.append(y_)
                if "sample_weight" in x and x["sample_weight"] is not None:
                    sample_weight.append(x["sample_weight"])
                    has_sample_weight = True
            else:
                raise ValueError(
                    "For tuple samples, X should be a dictionary with 'X' and 'sample_domain' keys."
                )

        result = {"X": np.array(X), "sample_domain": np.array(sample_domain)}
        y = np.array(y)
        if has_sample_weight:
            result["sample_weight"] = np.array(sample_weight)
        return result, y

class DomainAwareNetClassifier(_DomainAwareNet, NeuralNetClassifier):
    __metadata_request__fit = {"sample_weight": True}
    __metadata_request__score = {'sample_weight': True, 'sample_domain': True, 'allow_source': True}
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
    **kwargs : dict
        Additional keyword arguments passed to the skorch NeuralNetClassifier.
    """

    def predict(
        self,
        X: Union[Dict, torch.Tensor, np.ndarray, Dataset],
        sample_domain: Union[torch.Tensor, np.ndarray] = None,
        sample_weight: Union[torch.Tensor, np.ndarray] = None,
        allow_source: bool = False,
        **predict_params
    ):
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
        allow_source: bool = False,
            Allow the presence of source domains.
        **predict_params : dict
            Additional parameters passed to the predict method of the base class.

        Returns:
        --------
        np.ndarray
            The predicted classes.
        """
        return self.predict_proba(X, sample_domain, sample_weight, allow_source, **predict_params).argmax(axis=1)

    def score(
        self,
        X: Union[Dict, torch.Tensor, np.ndarray],
        y: Union[torch.Tensor, np.ndarray],
        sample_domain: Union[torch.Tensor, np.ndarray] = None,
        sample_weight: Union[torch.Tensor, np.ndarray] = None,
        allow_source: bool = False,
        **score_params
    ):
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
        allow_source: bool = False,
            Allow the presence of source domains.
        **score_params : dict
            Additional parameters passed to the score method of the base class.

        Returns:
        --------
        float
            The mean accuracy score.
        """
        X = self._prepare_input(X, y, sample_domain, sample_weight)
        if not allow_source:
            X = X.select_target()
        
        return accuracy_score(y, self.predict(X, sample_domain, allow_source=allow_source), sample_weight=sample_weight)

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
        loss = super().get_loss(y_pred, y_true, X, *args, **kwargs)

        if "sample_weight" in X and X["sample_weight"] is not None:
            sample_weight = to_tensor(X["sample_weight"], device=self.device)
            sample_weight = sample_weight[X["sample_domain"] > 0]
            if loss.dim() == 0 and len(sample_weight) > 1:
                raise ValueError(
                    "You are using a criterion function that returns a scalar loss value, but sample weights are provided."
                )

            loss = sample_weight * loss

        return loss.mean() 

class DomainAwareNetBinaryClassifier(_DomainAwareNet, NeuralNetBinaryClassifier):
    """
    A domain-aware neural network binary classifier with sample weight support.

    This class extends NeuralNetBinaryClassifier to handle domain-specific input data
    and sample weights. It supports various input formats and provides methods
    for training, prediction, and feature extraction while considering domain
    information and sample weights.

    Parameters
    ----------
    module : torch.nn.Module
        The PyTorch module to be used as the core of the classifier.
    iterator_train : torch.utils.data.DataLoader, optional
        Custom data loader for training. If None, DomainBalancedDataLoader is used.
    **kwargs : dict
        Additional keyword arguments passed to the skorch NeuralNetBinaryClassifier.
    """
    __metadata_request__fit = {"sample_weight": True}
    __metadata_request__score = {'sample_weight': True, 'sample_domain': True, 'allow_source': True}

    def predict(
        self,
        X,
        sample_domain=None,
        sample_weight=None,
        allow_source=False,
        **predict_params
    ):
        """
        Make binary predictions on the provided data.

        Parameters
        ----------
        X : dict, torch.Tensor, np.ndarray, or torch.utils.data.Dataset
            The input data for prediction.
        sample_domain : torch.Tensor or np.ndarray, optional
            The domain of each sample (if not provided in X).
        sample_weight : torch.Tensor or np.ndarray, optional
            The weight of each sample (not used in prediction, but included for consistency).
        allow_source : bool, default=False
            Allow the presence of source domains.
        **predict_params : dict
            Additional parameters passed to the predict method of the base class.

        Returns
        -------
        np.ndarray
            The predicted binary classes (0 or 1).
        """
        return self.predict_proba(X, sample_domain, sample_weight, allow_source, **predict_params)[:, 1] > 0.5

    def score(
        self,
        X,
        y,
        sample_domain=None,
        sample_weight=None,
        allow_source=False,
        **score_params
    ):
        """
        Compute the mean accuracy on the provided data and labels.

        Parameters
        ----------
        X : dict, torch.Tensor, or np.ndarray
            The input data for scoring.
        y : torch.Tensor or np.ndarray
            The true labels.
        sample_domain : torch.Tensor or np.ndarray, optional
            The domain of each sample (if not provided in X).
        sample_weight : torch.Tensor or np.ndarray, optional
            The weight of each sample (not used in scoring, but included for consistency).
        allow_source : bool, default=False
            Allow the presence of source domains.
        **score_params : dict
            Additional parameters passed to the score method of the base class.

        Returns
        -------
        float
            The mean accuracy score.
        """
        X = self._prepare_input(X, y, sample_domain, sample_weight)
        if not allow_source:
            X = X.select_target()
        return accuracy_score(y, self.predict(X, sample_domain, allow_source=allow_source), sample_weight=sample_weight)

    def get_loss(self, y_pred, y_true, X, *args, **kwargs):
        """
        Calculate the weighted loss using sample weights.

        Parameters
        ----------
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

        Returns
        -------
        torch.Tensor
            The calculated loss, weighted by sample weights if provided.
        """
        loss = super().get_loss(y_pred, y_true, X, *args, **kwargs)
        
        if "sample_weight" in X and X["sample_weight"] is not None:
            sample_weight = to_tensor(X["sample_weight"], device=self.device)
            sample_weight = sample_weight[X["sample_domain"] > 0]
            if loss.dim() == 0 and len(sample_weight) > 1:
                raise ValueError(
                    "You are using a criterion function that returns a scalar loss value, but sample weights are provided."
                )
            loss = sample_weight * loss
        return loss.mean()

class DomainAwareNetRegressor(_DomainAwareNet, NeuralNetRegressor):
    __metadata_request__fit = {"sample_weight": True}
    __metadata_request__score = {'sample_weight': True, 'sample_domain': True, 'allow_source': True}
    """
    A domain-aware neural network regressor with sample weight support.

    This class extends NeuralNetRegressor to handle domain-specific input data
    and sample weights. It supports various input formats and provides methods
    for training, prediction, and feature extraction while considering domain
    information and sample weights.

    Parameters:
    -----------
    module : torch.nn.Module
        The PyTorch module to be used as the core of the regressor.
    iterator_train : torch.utils.data.DataLoader, optional
        Custom data loader for training. If None, DomainBalancedDataLoader is used.
    **kwargs : dict
        Additional keyword arguments passed to the skorch NeuralNetRegressor.
    """

    def predict(
        self,
        X: Union[Dict, torch.Tensor, np.ndarray, Dataset],
        sample_domain: Union[torch.Tensor, np.ndarray] = None,
        sample_weight: Union[torch.Tensor, np.ndarray] = None,
        allow_source: bool = False,
        **predict_params
    ):
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
        allow_source: bool, default=False
            Allow the presence of source domains.
        **predict_params : dict
            Additional parameters passed to the predict method of the base class.

        Returns:
        --------
        np.ndarray
            The predicted classes.
        """
    # TODO

    def score(
        self,
        X: Union[Dict, torch.Tensor, np.ndarray],
        y: Union[torch.Tensor, np.ndarray],
        sample_domain: Union[torch.Tensor, np.ndarray] = None,
        sample_weight: Union[torch.Tensor, np.ndarray] = None,
        allow_source: bool = False,
        **score_params
    ):
    # TODO 
        X = self._prepare_input(X, y, sample_domain, sample_weight)
        if not allow_source:
            X = X.select_target()
        
        return ## accuracy_score(y, self.predict(X, sample_domain, allow_source=allow_source), sample_weight=sample_weight)

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
        loss = super().get_loss(y_pred, y_true, X, *args, **kwargs)

        if "sample_weight" in X and X["sample_weight"] is not None:
            sample_weight = to_tensor(X["sample_weight"], device=self.device)
            sample_weight = sample_weight[X["sample_domain"] > 0]
            if loss.dim() == 0 and len(sample_weight) > 1:
                raise ValueError(
                    "You are using a criterion function that returns a scalar loss value, but sample weights are provided."
                )

            loss = sample_weight * loss

        return loss.mean()


class DeepDADataset(Dataset):
    f"""The Domain Aware Dataset class for deep learning.
    This class fills the gap between dictionary representation and array like
    representation by combining them into a single object.

    All passed data will be converted to :code:`torch.Tensor`. Dict representation
    should at least contain an 'X' key containing the input data or be completely empty

    If no sample domain is provided, domain :code:`{_DEFAULT_SAMPLE_DOMAIN_}` is attributed
    to the given data.

    If no label is provided, :code:`{_NO_LABEL_}` is attributed to the given data.

    When accessing an item from this dataset, returned is a dict representation
    of the element and its associated label.
    """

    def __init__(
        self,
        X=None,
        y=None,
        sample_domain=None,
        sample_weight=None,
        device="cpu",
    ):
        """
        The Domain Aware Dataset class for deep learning.

        Parameters:
        -----------
        X : torch.Tensor, numpy.ndarray, list, tuple, dict, DataFrame, DeepDADataset
            The input data can be of various types:
            - non-scalar :code:`torch.Tensor`
            - non-scalar :code:`numpy.ndarray`
            - :code:`list`
            - :code:`tuple`
            - :code:`dict` with keys 'X', 'y' (optional), 'sample_domain' (optional)
            and 'sample_weight' (optional) with values of one of the above types
            - :code:`pandas.DataFrame` with same structure as the dictionary
            - another :code:`~skada.DeepDADataset` instance

        y : 
            The target labels.
            If not provided, the dataset will have default labels :code:`{_NO_LABEL_}`.

        sample_domain : 
            The sample domain information.
            If not provided, the dataset will have default domain 
            :code:`{_DEFAULT_SAMPLE_DOMAIN_}`.
            If provided, it can be of the following types:
                - :code:`int` (single domain for all samples)
                - non-scalar :code:`torch.Tensor`
                - non-scalar :code:`numpy.ndarray`
                - :code:`list`
                - :code:`tuple`

        sample_weight :
            The sample weights.
        device : 
            The device to which the tensors should be moved. Defaults to "cpu".

        Raises:
        -------
        TypeError
            If the input data is not of the expected type.
        """
        self.X = _EMPTY_
        self.y = _EMPTY_INT_
        self.sample_domain = _EMPTY_INT_
        self.sample_weight = _EMPTY_
        self.has_y = self.y != _NO_LABEL_
        self.has_weights = True
        self.device = device

        if X is not None:
            if isinstance(X, (dict, DataFrame)):
                if len(X):
                    self._if_given_dict(X, y, sample_domain, sample_weight)
            elif isinstance(X, DeepDADataset):
                self.merge(X, keep_weights=True, out=False)
            else:
                try:
                    if hasattr(X, 'shape') and len(X.shape) == 0:
                        X = torch.tensor([X])
                    X = check_array(X, 
                                    ensure_2d=False,
                                    allow_nd=True,
                                    ensure_min_samples=0,
                                    ensure_min_features=0,
                                    )
                    X = to_tensor(X, device=device)

                except ValueError:
                    raise TypeError(
                        "Invalid dataset representation. Expected a dict of the form "
                        "{'X', 'y'(optional), 'sample_domain'(optional), "
                        "'sample_weight'(optional)},"
                        " another DeepDADataset instance or "
                        "a data type convertible to a torch Tensor"
                        f" but found {type(X)} instead."
                    )
                self._initialize(X, y, sample_domain, sample_weight)
        
        self.has_weights = bool(len(self.sample_weight))
        assert self._is_correct(), (
            "Every input data X should have a domain associated, a label (if there is "
            f"no label, it must be represented by {_NO_LABEL_}) and, optionally, a weight."
        )
        self._sample_idx = self._infer_sample_idx()

    def _if_given_dict(self, d: dict, y, sample_domain, sample_weight):
        if "X" not in d:
            raise ValueError("dataset represented as dict or pandas dataframe " \
            "should contain 'X' key associated to a data type convertible " \
            "to torch Tensor.")

        X = d["X"]
        y = d.get("y", y)
        sample_domain = d.get("sample_domain", sample_domain)
        sample_weight = d.get("sample_weight", sample_weight)
        if isinstance(d, DataFrame):
            X = X.to_list()
            if "y" in d.columns:
                y = y.to_list()
            if "sample_domain" in d.columns:
                sample_domain = sample_domain.to_list()
            if "sample_weight" in d.columns:
                sample_weight = sample_weight.to_list()

        X = check_array(X, 
                ensure_2d=False,
                allow_nd=True,
                ensure_min_samples=0,
                ensure_min_features=0,
                )
        X = to_tensor(X, self.device)
        self._initialize(X, y, sample_domain, sample_weight)

    def _initialize(self, X:torch.Tensor, y, sample_domain, sample_weight):
        if sample_domain is None or len(sample_domain) == 0:
            sample_domain = _DEFAULT_SAMPLE_DOMAIN_
        if isinstance(sample_domain, int):
            sample_domain = torch.full((X.shape[0],), sample_domain)
        else:
            sample_domain = check_array(sample_domain, 
                ensure_2d=False,
                allow_nd=True,
                ensure_min_samples=0,
                ensure_min_features=0,
                )
            sample_domain = to_tensor(sample_domain, self.device)
        
        if y is None or len(y) == 0:
            y = torch.full((len(X),), _NO_LABEL_, dtype=torch.float)
            has_y = torch.full((len(X),), False, dtype=torch.bool)
        else:
            y = check_array(y, 
                ensure_2d=False,
                allow_nd=True,
                ensure_min_samples=0,
                ensure_min_features=0,
                )
            y = to_tensor(y, self.device)
            has_y = y != _NO_LABEL_

        if sample_weight is None or len(sample_weight) == 0:
            sample_weight = _EMPTY_
            has_weights = False
        else:
            sample_weight = check_array(sample_weight, 
                ensure_2d=False,
                allow_nd=True,
                ensure_min_samples=0,
                ensure_min_features=0,
                )
            sample_weight = to_tensor(sample_weight, device=self.device)
            has_weights = bool(sample_weight.size()[0])

        self.X = X
        self.y = y
        self.sample_domain = sample_domain
        self.sample_weight = sample_weight
        self.has_y = has_y
        self.has_weights = has_weights

    def merge(self, dataset: "DeepDADataset", keep_weights=False, out=True):
        """Merges to instances of DeepDADataset and either returns the result
        or updates the first one. The merging is done by concatenation of the data.

        Parameters:
        -----------
        dataset : DeepDADataset
          The second dataset to merge.
        keep_weights : bool, optional
          Whether to keep the weights (if any). If False, weights become empty. If True, weights are concatenated. Defaults to False.

          ..WARNING::
              There is no check to ensure weights still form
              a probability distribution after merging.

        out : bool, optional
          Whether to return the result instead of updating first dataset. Defaults to True.

        Returns:
        --------
        DeepDADataset
          the concatenation of the datasets.

        Raises:
        -------
        TypeError
          raises if the other dataset is not a DeepDADataset instance.
        """
        if isinstance(dataset, DeepDADataset):
            X = torch.cat((self.X, dataset.X))
            y = torch.cat((self.y, dataset.y))
            sample_domain = torch.cat((self.sample_domain, dataset.sample_domain))
            if keep_weights:
                sample_weight = torch.cat((self.sample_weight, dataset.sample_weight))
            else:
                sample_weight = _EMPTY_
            if out:
                return DeepDADataset(X, y, sample_domain, sample_weight, str(X.device))
            else:
                self.X = X
                self.y = y
                self.sample_domain = sample_domain
                self.sample_weight = sample_weight
                self.has_y = torch.cat((self.has_y, dataset.has_y))
                self.has_weights = bool(len(self.sample_weight))
                self.device = str(X.device)
                assert self._is_correct(), (
            "Every input data X should have a domain associated, a label (if there is "
            f"no label, it must be represented by {_NO_LABEL_}) and, "
            "optionally, a weight."
        )
        else:
            raise TypeError("Can only merge two instances of DeepDADataset")

    def is_empty(self):
        return not bool(len(self))

    def _is_correct(self):
        """Validates the dataset. Fails if there are not as many labels, domain,
        or weights (if any) as data samples or if the dimensions are not compatible.
        """
        return (len(self.X) == len(self.y) == len(self.sample_domain)) and (
            len(self.X) == len(self.sample_weight) or not self.has_weights) and (
            self.y.ndim == self.sample_domain.ndim == 1) and (
            not self.has_weights or self.sample_weight.ndim == 1)

    def __len__(self):
        assert self._is_correct(), (
            "Every input data X should have a domain associated, a label "
            f"(if there is no label, it must be represented by {_NO_LABEL_}) and,"
            " optionally, a weight."
        )
        return len(self.X)

    def __getitem__(self, index):
        X = {"X": self.X[index], "sample_domain": self.sample_domain[index], "sample_idx": self._sample_idx[index]}
        
        if self.has_weights:
            X["sample_weight"] = self.sample_weight[index]
        
        return X, self.y[index]

    def add_domain(self, dataset):
        """Adds a domain to the dataset.

        Parameters:
        -----------
        dataset : DeepDADataset, dict, list, tuple(torch.Tensor, np.array)
            The domain to add. Should be a data type convertible to DeepDADataset.
        
        Returns:
        --------
        DeepDADataset
          the original dataset with the new domain added.
        """
        if isinstance(dataset, dict):    
            dataset = DeepDADataset(dataset)
        elif not isinstance(dataset, DeepDADataset):
            dataset = DeepDADataset(*dataset)
        return self.merge(dataset)

    def __repr__(self):
        if self.is_empty():
            return "DeepDADataset(data[], labels[], domains[], weights[])"
        rep = "DeepDADataset("
        xrep = "\nx[" + str(self.X) + "],"
        yrep = "\n\ny[" + str(self.y) + "],"
        sdrep = "\n\ndomains[" + str(self.sample_domain) + "],"
        if self.has_weights:
            wrep = "\n\nweights[" + str(self.sample_weight) + "],"
        else:
            wrep = "\n\nweights["
        wrep += "]"
        rep += xrep + yrep + sdrep + wrep + "\n    )"
        return rep


    def as_dict(self, sample_indices=True):
        """Switches to dict representation of the dataset.
        Dictionary representation is of the form
        {'X': input data (torch tensor),
        'y': corresponding label (torch tensor),
        'sample_domain': corresponding domains (torch tensor),
        'sample_weight': corresponding weights if any (torch tensor),
        'sample_idx': the indices of the data relative to their domain 
            (torch tensor, optional)}

        Parameters:
        -----------
        sample_indices : bool
            whether to return the sample indices as key 'sample_idx' within the dictionary. Defaults to True.

        Returns:
        --------
        dict
            dictionary representation of the dataset
        """
        dataset = {"X": self.X, 'y':self.y, "sample_domain": self.sample_domain}
        if self.has_weights:
            dataset["sample_weight"] = self.sample_weight
        if sample_indices:
            dataset["sample_idx"] = self._sample_idx
        return dataset

    def as_arrays(self, return_weights=True) -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
        ]:
        f"""switches to array representation of the dataset.

        Parameters:
        -----------
        return_weights : bool, optional
            whether to return the sample weights (if any). Defaults to False.

        Returns:
        --------
        torch.Tensor
            input data
        torch.Tensor
            corresponding labels (if a sample has no label, {_NO_LABEL_} at its index)
        torch.Tensor
            corresponding domain of each sample
        torch.Tensor (optional)
            corresponding weight of each sample (if any)
        """
        if self.has_weights and return_weights:
            return self.X, self.y, self.sample_domain, self.sample_weight
        else:
            return self.X, self.y, self.sample_domain

    def domains(self):
        """return all the domains comprising the dataset

        Returns:
        --------
        tuple
            the domains of the dataset, sorted by domain id
        """
        return tuple(int(domain_id) for domain_id in self.sample_domain.unique())
    
    def _infer_sample_idx(self):
        """Returns the indices of each sample relative to their own domain.

        if self.sample domain is [1, 1, 1, -2, -2, 1, 3, 3, 3, -2, -2],
        result would be [0, 1, 2, 0, 1, 3, 0, 1, 2, 2, 3]

        Returns:
        --------
        torch.Tensor
            the indices of each sample relative to their domain
        """
        result = torch.zeros_like(self.sample_domain, dtype=torch.int64)
        for domain_id in self.domains():
            mask = self.sample_domain == domain_id
            indices = torch.arange(mask.sum(), device=self.device)
            result[mask] = indices
        
        return result
    
    def select(self, condition, on, return_weights=True):
        """Selects the data samples validating the condition. 
        The condition must be applicable to a torch tensor so that it returns a mask
        of True or False.

        Where the condition is applied depends on the `on` argument.

        Parameters:
        -----------
            condition : callable
               the validation condition.
            on : str, optional
                where the condition is applied. 
                Either X, y, sample_domain or sample_weight.
            return_weights : bool, optional
                whether to return the weights (if any). Defaults to True.

        Raises:
        -------
            ValueError: raises when `on` argument is invalid.

        Returns:
        --------
        DeepDADataset
            the subset of the dataset validating the condition.
        """
        if on == 'X':
            mask = condition(self.X)
        elif on == 'y':
            mask = self.has_y & condition(self.y)
        elif on == 'sample_domain':
            mask = condition(self.sample_domain)
        elif on == 'sample_weight':
            if self.has_weights:
                mask = condition(self.sample_weight)
        else:
            raise ValueError(
                "'on' argument must be one of "
                "('X', 'y', 'sample_domain', 'sample_weight')"
                )
        return self._select_from_mask(mask, return_weights=return_weights)
    
    def _select_from_mask(self, mask, return_weights=True):
        """Returns a DeepDADataset instance of the data corresponding to the mask
        The mask must be a boolean array like of True or False corresponding to PyTorch
        boolean indexing methods.

        Parameters:
        -----------
        mask : array_like of bools
            the mask of the data to get.
        return_weights : bool, optional
            whether to return the sample weights (if any). Defaults to False.

        Returns:
        --------
        DeepDADataset
            the dataset corresponding to the input mask.
        """
        mask = check_array(
                mask,
                allow_nd=True,
                ensure_2d=False,
                ensure_min_samples=False,
                dtype=bool
            )
        mask = to_tensor(mask, device=self.device)
        dataset = self.as_arrays(return_weights=return_weights)
        dataset = (data[mask] for data in dataset)
        return DeepDADataset(*dataset, device=self.device)

    def select_source(self, return_weights=True):
        """Returns a DeepDADataset composed only of the source (marked with 
        sample_domain >= 0)

        Parameters:
        -----------
        return_weights : bool, optional
            whether to return the sample weights (if any). Defaults to False.

        Returns:
        --------
        DeepDADataset
            the source domain from the dataset.
        """
        mask = self.sample_domain >= 0
        return self._select_from_mask(mask, return_weights)

    def select_target(self, return_weights=True):
        """Returns a DeepDADataset composed only of the target (marked with 
        sample_domain < 0)

        Parameters:
        -----------
        return_weights : bool, optional
            whether to return the sample weights (if any). Defaults to False.

        Returns:
        --------
        DeepDADataset
            the target domain from the dataset.
        """
        mask = self.sample_domain < 0
        return self._select_from_mask(mask, return_weights)

    def select_domain(self, domain_id, return_weights=True):
        """Returns a DeepDADataset composed only of the selected domain

        Parameters:
        -----------
        return_weights : bool, optional
            whether to return the sample weights (if any). Defaults to False.

        Returns:
        --------
        DeepDADataset
            the selected domain from the dataset.
        """
        mask = self.sample_domain == domain_id
        return self._select_from_mask(mask, return_weights)

    def select_with_labels(self, return_weights=True):
        f"""Returns a DeepDADataset instance composed of the data that is labelled
        (that is, data with a y associated that is not {_NO_LABEL_}).

        Parameters:
        -----------
        return_weights : bool, optional
            whether to return the sample weights (if any). Defaults to False.

        Returns:
        --------
        DeepDADataset
            the data with labels.
        """
        mask = self.has_y
        return self._select_from_mask(mask, return_weights)

    def per_domain_split(self, return_weights=True):
        """Splits the data per domain, returning a dict where each key is a domain id
        and value is a DeepDADataset composed of said domain.

        Parameters:
        -----------
        return_weights : bool, optional
            whether to return the sample weights (if any). Defaults to False.

        Returns:
        --------
        dict
            The dataset split per domain id
        """
        dataset = {}
        for domain_id in self.domains():
            dataset[domain_id] = self.select_domain(domain_id, return_weights)
        return dataset

    def add_weights(self, sample_weight):
        """Adds weights to the dataset.

        Parameters:
        -----------
        sample_weight : torch.Tensor or np.ndarray
            the weights to add to the dataset. Must be convertible to torch Tensor.

        Returns:
        --------
        DeepDADataset
            the weighted dataset.
        """
        sample_weight = check_array(
            sample_weight,
            allow_nd=True,
            ensure_2d=False,
            ensure_min_samples=False,
            dtype=float
        )
        self.sample_weight = to_tensor(sample_weight, device=self.device)
        self.has_weights = bool(len(self.sample_weight))
        assert self._is_correct(), "There must be a weight for every sample."
        return self
        
    def remove_weights(self):
        """Removes the weight of the dataset Changes the dataset in place
        and returns a new dataset without weights.

        Returns:
        --------
        DeepDADataset
            The unweighted dataset. 
        """
        self.sample_weight = _EMPTY_
        self.has_weights = False
        return self



