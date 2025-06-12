# Author : Maxence Barneche
#
# License: BSD-3-Clause

import torch

from skada.deep.base import DomainAwareNet


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
        **kwargs,
    ):
        super().__init__(module, criterion=criterion, **kwargs)
        self.optimizer_adapt = optimizer_adapt
        self.epochs_adapt = epochs_adapt

    def fit_source(
        self, X, y=None, sample_domain=None, sample_weight=None, **fit_params
    ):
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
        self.partial_fit(X, None, epochs=self.epochs_adapt, **fit_params)
        return self

    def fit(self, X, y=None, sample_domain=None, sample_weight=None, **fit_params):
        self.fit_source(X, y, sample_domain, sample_weight, **fit_params)
        self.fit_adapt(X, sample_domain, sample_weight, **fit_params)
        return self

    def initialize_adapt_optimizer(self):
        named_parameters = self.get_all_learnable_params()
        args, kwargs = self.get_params_for_optimizer(
            "adapt_optimizer", named_parameters
        )
        self.optimizer_ = self.optimizer_adapt(*args, **kwargs)
        return self
