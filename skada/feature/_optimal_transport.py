# Author: Theo Gnassounou <theo.gnassounou@inria.fr>
#         Remi Flamary <remi.flamary@polytechnique.edu>
#
# License: BSD 3-Clause
from skorch.utils import to_tensor
from torch import nn

from . import deepjdot_loss
from .base import BaseDANetwork


class DeepJDOT(BaseDANetwork):
    """Loss DeepJDOT.

    See [1]_.

    Parameters
    ----------
    module : torch module (class or instance)
        A PyTorch :class:`~torch.nn.Module`. In general, the
        uninstantiated class should be passed, although instantiated
        modules will also work.
    criterion : torch criterion (class)
        The uninitialized criterion (loss) used to optimize the
        module.
    layer_names : list of tuples
        The names of the module's layers whose outputs are
        collected during the training.
    reg_d : float, default=1
        Distance term regularization parameter.
    reg_cl : float, default=1
        Class distance term regularization parameter.
    target_criterion : torch criterion (class)
        The uninitialized criterion (loss) used to compute the
        DeepJDOT loss. The criterion should support reduction='none'.
    **kwargs : dict
        Keyword arguments passed to the skorch Model class.

    References
    ----------
    .. [1]  Bharath Bhushan Damodaran, Benjamin Kellenberger,
            Remi Flamary, Devis Tuia, and Nicolas Courty.
            DeepJDOT: Deep Joint Distribution Optimal Transport
            for Unsupervised Domain Adaptation. In ECCV 2018
            15th European Conference on Computer Vision,
            September 2018. Springer.
    """

    def __init__(
        self,
        module,
        criterion,
        layer_names,
        reg_d=1,
        reg_cl=1,
        target_criterion=nn.CrossEntropyLoss,
        **kwargs,
    ):
        super().__init__(module, criterion, layer_names, **kwargs)
        self.reg_d = reg_d
        self.reg_cl = reg_cl
        self.target_criterion = target_criterion

    def initialize_criterion(self):
        """Initializes the criterion.

        If the criterion is already initialized and no parameter was changed, it
        will be left as is.

        """
        kwargs = self.get_params_for("target_criterion")
        kwargs["reduction"] = "none"
        target_criterion = self.initialized_instance(self.target_criterion, kwargs)
        self.target_criterion_ = target_criterion
        return super().initialize_criterion()

    def _get_loss_da(
        self,
        y_pred,
        y_true,
        embedd,
        embedd_target,
        X=None,
        y_pred_target=None,
        training=True,
    ):
        """Compute the domain adaptation loss"""
        y_true = to_tensor(y_true, device=self.device)
        loss_deepjdot = 0
        for i in range(len(embedd)):
            loss_deepjdot += deepjdot_loss(
                embedd[i],
                embedd_target[i],
                y_true,
                y_pred_target,
                self.reg_d,
                self.reg_cl,
                criterion=self.target_criterion_,
            )
        loss_classif = self.criterion_(y_pred, y_true)

        return loss_classif + loss_deepjdot, loss_classif, loss_deepjdot
