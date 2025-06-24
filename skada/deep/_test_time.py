# Author : Maxence Barneche
#
# License: BSD-3-Clause

import torch

from skada.deep.base import BaseDALoss, DomainAwareModule, DomainAwareNet
from skada.deep.losses import softmax_entropy


class TestTimeCriterion(torch.nn.Module):
    def __init__(
        self,
        base_criterion,
        adapt_criterion,
        reg=1,
        reduction="mean",
        train_on_target=False,
    ):
        super().__init__()
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
        y_pred :
            Prediction of the labels.
        y_true :
            The true labels. Available for source, masked for target.
        """
        if self.train_on_target:
            # In finetune mode, we only compute the base loss
            return self.adapt_criterion(
                y_s=y_true,
                y_pred=y_pred,
            )
        else:
            # In adapt mode, we compute the adaptation loss
            return self.base_criterion(y_pred, y_true)


class TestTimeNet(DomainAwareNet):
    def __init__(
        self,
        module,
        criterion: "TestTimeCriterion",
        optimizer_adapt=None,
        epochs_adapt=None,
        params_to_adapt=None,
        layers_to_adapt=None,
        **kwargs,
    ):
        super().__init__(module, criterion=criterion, **kwargs)
        self.optimizer_adapt = optimizer_adapt
        self.epochs_adapt = epochs_adapt
        self.params_to_adapt = params_to_adapt
        self.layers_to_adapt = layers_to_adapt

    def fit(self, X, y=None, sample_domain=None, sample_weight=None, **fit_params):
        print("Training model on source domain...")
        X = self._prepare_input(X, y, sample_domain, sample_weight)
        X = X.select_source()
        self.criterion.train_on_target = False
        self.partial_fit(X, None, **fit_params)
        return self

    def fit_adapt(self, X, sample_domain, sample_weight=None, **fit_params):
        print("Adapting model to target domain...")
        if self.optimizer_adapt is not None:
            self.initialize_adapt_optimizer()
        X = self._prepare_input(X, None, sample_domain, sample_weight)
        X = X.select_target()
        self.criterion.train_on_target = True
        self.freeze_all_params()
        self.parameters_to_adapt(
            param_name=self.params_to_adapt, layer_name=self.layers_to_adapt
        )
        self.partial_fit(X, None, epochs=self.epochs_adapt, **fit_params)
        return self

    def initialize_adapt_optimizer(self):
        named_parameters = self.get_all_learnable_params()
        args, kwargs = self.get_params_for_optimizer(
            "adapt_optimizer", named_parameters
        )
        self.optimizer_ = self.optimizer_adapt(*args, **kwargs)
        return self

    def unfreeze_params(self, module, param_name=None):
        """
        Unfreeze the parameters of a given module from the net.

        Parameters
        ----------
        param_name : str, list(str), optional
            The name of the parameter to unfreeze.
            If 'all', unfreeze all parameters of the module.
            If None, unfreeze no parameter of the module.
            If a list, unfreeze all parameters from the list.
        """
        # Pseudocode for freezing parameters of a specific layer
        if param_name is None:
            param_name = []
        if param_name == "all":
            param_name = [name for name, _ in module.named_parameters()]
        elif isinstance(param_name, str):
            param_name = [param_name]

        for name, param in module.named_parameters():
            if name in param_name:
                param.requires_grad = False

        return self

    def parameters_to_adapt(self, param_name=None, layer_name=None):
        """
        Unfreeze the parameters of the module.
        Choose the parameters to adapt during the adaptation phase based on
        the layer names and parameter names.

        Parameters
        ----------
        layer_name : str, list(str), optional
            The name of the layer to unfreeze.
            If 'all', unfreeze all layers of the module.
            If None, unfreeze params from no layers.
            If a list, unfreeze all layers in the list.
        param_name : str, list(str), list(list(str)) optional
            The name of the parameter to unfreeze.
            If None, unfreeze all parameters of the layer.
            If a list, unfreeze all parameters from the list in every layer given.
            If a list of lists, unfreeze the parameters with the given names in
            the corresponding layers.
        """
        # if None, freeze parameters from all layers of the net
        if layer_name is None:
            layer_name = []
        elif layer_name == "all":
            layer_name = [name for name, _ in self.module.named_modules()]
        elif isinstance(layer_name, str):
            layer_name = [layer_name]

        if isinstance(param_name, str):
            param_name = [param_name] * len(layer_name)

        for module_name, module in self.module.named_modules():
            if module_name in layer_name:
                self.unfreeze_params(module=module, param_name=param_name)
        return self

    def freeze_all_params(self):
        """Freeze all parameters of the module."""
        for _, mod in self.module.named_modules():
            for _, param in mod.named_parameters():
                param.requires_grad = False
        return self


class TentLoss(BaseDALoss):
    def __init__(self):
        super().__init__()

    def forward(self, y):
        loss = softmax_entropy(y)
        return loss


def Tent(
    module,
    layer_name,
    base_criterion=None,
    **kwargs,
):
    if base_criterion is None:
        base_criterion = torch.nn.CrossEntropyLoss()

    net = TestTimeNet(
        module=DomainAwareModule(module, layer_name),
        params_to_adapt=["weight", "bias"],
        layers_to_adapt="all",
        criterion=TestTimeCriterion,
        criterion__base_criterion=base_criterion,
        criterion__adapt_criterion=TentLoss(),
        criterion__reg=1,
        **kwargs,
    )

    return net
