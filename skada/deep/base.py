# Author: Theo Gnassounou <theo.gnassounou@inria.fr>
#         Remi Flamary <remi.flamary@polytechnique.edu>
#         Yanis Lalou <yanis.lalou@polytechnique.edu>
#         Antoine Collas <contact@antoinecollas.fr>
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
from skada.deep.dataloaders import DomainBalancedDataLoader

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
    base_criterion : torch criterion (instance)
        The initialized criterion (loss) used to optimize the
        module with prediction on source.
    adapt_criterion : torch criterion (instance)
        The initialized criterion (loss) used to compute the
        loss to reduce the divergence between domains.
    reg: float, default=1
        Regularization parameter.
    reduction: str, default='mean'
        Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'.
        'none': no reduction will be applied,
        'mean': the sum of the output will be divided by the number of elements in the output,
        'sum': the output will be summed.
    """

    def __init__(
        self,
        base_criterion,
        adapt_criterion,
        reg=1,
        reduction="mean",
        is_multi_source=False,
    ):
        super(DomainAwareCriterion, self).__init__()
        self.base_criterion = base_criterion
        self.adapt_criterion = adapt_criterion
        self.reg = reg
        self.is_multi_source = is_multi_source

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

        (
            (y_pred_s, y_pred_t),
            (domain_pred_s, domain_pred_t),
            (features_s, features_t),
            sample_domain,
        ) = y_pred

        source_idx = sample_domain >= 0

        # predict
        if self.is_multi_source:
            base_loss = 0
            for n_domain in sample_domain[source_idx].unique():
                idx = sample_domain[source_idx] == n_domain
                base_loss += self.base_criterion(y_pred_s[idx], y_true[source_idx][idx])
        else:
            base_loss = self.base_criterion(y_pred_s, y_true[source_idx])
        return base_loss + self.reg * self.adapt_criterion(
            y_true[source_idx],
            y_pred_s,
            y_pred_t,
            domain_pred_s,
            domain_pred_t,
            features_s,
            features_t,
            sample_domain,
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
        sample_domain,
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
        sample_domain :
            The domain of each sample.
        """
        pass


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

    def __init__(
        self, base_module, layer_name, domain_classifier=None, is_multi_source=False
    ):
        super(DomainAwareModule, self).__init__()
        self.base_module_ = base_module
        self.domain_classifier_ = domain_classifier
        self.is_multi_source = is_multi_source
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
    ):
        if is_fit:
            source_idx = sample_domain >= 0

            X_s = X[source_idx]
            X_t = X[~source_idx]

            args_s = {"X": X_s}
            if sample_weight is not None:
                args_s["sample_weight"] = sample_weight[source_idx]
            if self.is_multi_source:
                args_s["sample_domain"] = sample_domain[source_idx]
            y_pred_s = self.base_module_(**args_s)
            features_s = self.intermediate_layers[self.layer_name]

            args_t = {"X": X_t}
            if sample_weight is not None:
                args_t["sample_weight"] = sample_weight[~source_idx]
            if self.is_multi_source:
                args_t["sample_domain"] = sample_domain[~source_idx]
                args_t["is_source"] = False
            y_pred_t = self.base_module_(**args_t)
            features_t = self.intermediate_layers[self.layer_name]

            if self.domain_classifier_ is not None:
                domain_pred_s = self.domain_classifier_(features_s)
                domain_pred_t = self.domain_classifier_(features_t)
            else:
                domain_pred_s = None
                domain_pred_t = None

            return (
                (y_pred_s, y_pred_t),
                (domain_pred_s, domain_pred_t),
                (features_s, features_t),
                sample_domain,
            )
        else:
            if return_features:
                return (
                    self.base_module_(X, sample_weight=sample_weight),
                    self.intermediate_layers[self.layer_name],
                )
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
        X, y_ = self._prepare_input(X, sample_domain, sample_weight)
        y = y_ if y is None else y

        # TODO: check X and y
        # but it requires to adapt skada.utils.check_X_y_domain
        # to handle dict, Dataset, torch.Tensor, ...

        return super().fit(X, y, is_fit=True, **fit_params)

    def predict(
        self,
        X: Union[Dict, torch.Tensor, np.ndarray, Dataset],
        sample_domain: Union[torch.Tensor, np.ndarray] = None,
        sample_weight: Union[torch.Tensor, np.ndarray] = None,
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
        **predict_params : dict
            Additional parameters passed to the predict method of the base class.

        Returns:
        --------
        np.ndarray
            The predicted classes.
        """
        X, _ = self._prepare_input(X, sample_domain, sample_weight)
        return super().predict(X, **predict_params)

    def predict_proba(
        self,
        X: Union[Dict, torch.Tensor, np.ndarray, Dataset],
        sample_domain: Union[torch.Tensor, np.ndarray] = None,
        sample_weight: Union[torch.Tensor, np.ndarray] = None,
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
        **predict_params : dict
            Additional parameters passed to the predict_proba method of the base class.

        Returns:
        --------
        np.ndarray
            The predicted class probabilities.
        """
        X, _ = self._prepare_input(X, sample_domain, sample_weight)
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

        X, _ = self._prepare_input(X, None)
        X, sample_domain = X["X"], X["sample_domain"]
        X = torch.tensor(X) if not torch.is_tensor(X) else X

        features_list = []
        for features in self.feature_iter(X, training=False):
            features = features[0] if isinstance(features, tuple) else features
            features_list.append(to_numpy(features))
        return np.concatenate(features_list, 0)

    def score(
        self,
        X: Union[Dict, torch.Tensor, np.ndarray],
        y: Union[torch.Tensor, np.ndarray],
        sample_domain: Union[torch.Tensor, np.ndarray] = None,
        sample_weight: Union[torch.Tensor, np.ndarray] = None,
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
        **score_params : dict
            Additional parameters passed to the score method of the base class.

        Returns:
        --------
        float
            The mean accuracy score.
        """
        X, _ = self._prepare_input(X, sample_domain, sample_weight)
        return super().score(X, y, **score_params)

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
        X: Union[Dict, torch.Tensor, np.ndarray, Dataset],
        sample_domain: Union[torch.Tensor, np.ndarray] = None,
        sample_weight: Union[torch.Tensor, np.ndarray] = None,
    ) -> Union[Dict[str, Any], np.ndarray]:
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
            A dictionary containing 'X', 'sample_domain', and optionally 'y' and 'sample_weight' keys.
        np.ndarray
            y as a numpy array.

        Raises:
        -------
        ValueError
            If the input format is invalid or missing required information.
        """
        if isinstance(X, dict):
            if "X" not in X or "sample_domain" not in X:
                raise ValueError("X should contain both 'X' and 'sample_domain' keys.")
            return X, None
        elif isinstance(X, Dataset):
            return self._process_dataset(X)
        else:
            result = {"X": X, "sample_domain": sample_domain}
            if sample_weight is not None:
                result["sample_weight"] = sample_weight
            return result, None

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
