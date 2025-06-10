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
        mode="finetune",
    ):
        super().__init__()
        self.base_criterion = base_criterion
        self.adapt_criterion = adapt_criterion
        self.reg = reg
        self.train_on_target = train_on_target
        self.mode = mode

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
        y_pred, domain_pred, features, sample_domain, sample_idx = y_pred
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

        if self.mode == "finetune":
            # In finetune mode, we only compute the base loss
            return base_loss
        elif self.mode == "adapt":
            self.adapt_criterion(
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


class TestTimeNet(DomainAwareNet):
    def __init__(self, module, criterion: "TestTimeCriterion", **kwargs):
        super().__init__(module, criterion=criterion, **kwargs)

    def fit_prepare(
        self, X, y=None, sample_domain=None, sample_weight=None, **fit_params
    ):
        X = self._prepare_input(X, y, sample_domain, sample_weight)
        X = X.select_source()
        self.partial_fit(X, None, **fit_params)
        return self

    def fit_adapt(self, X, sample_domain=None, sample_weight=None, **fit_params):
        if sample_domain is None:
            sample_domain = -1
        X = self._prepare_input(X, None, sample_domain, sample_weight)
        X = X.select_target()
        self.criterion.mode = "adapt"
        self.partial_fit(X, None, **fit_params)
        # here should be a partial fit call with criterion_adapt

    def fit(self, X, y=None, sample_domain=None, sample_weight=None, **fit_params):
        X = self._prepare_input(X, y, sample_domain, sample_weight)
        self.fit_prepare(X, **fit_params)
        self.fit_adapt(X, **fit_params)
        return self
