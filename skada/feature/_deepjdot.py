import torch
from skorch.utils import to_tensor

from ..utils import distance_matrix, ot_solve
from .base import BaseDANetwork


class DeepJDOT(BaseDANetwork):
    """Loss DeepJDOT.

    See [1]_.

    Parameters
    ----------
    XXX

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
        alpha=1,
        beta=1,
        class_weights=None,
        **kwargs
    ):
        super().__init__(
            module, criterion, layer_names, **kwargs
        )
        self.alpha = alpha
        self.beta = beta
        self.class_weights = class_weights

    def get_loss_da(
        self,
        y_pred,
        y_true,
        embedd,
        embedd_target,
        X=None,
        y_pred_target=None,
        training=True
    ):
        y_true = to_tensor(y_true, device=self.device)

        loss_deepjdot = 0
        for i in range(len(embedd)):

            a = torch.full(
                (len(embedd[i]),),
                1.0 / len(embedd[i]),
                device=self.device
            )
            b = torch.full(
                (len(embedd_target[i]),),
                1.0 / len(embedd_target[i]),
                device=self.device
            )
            M = distance_matrix(
                embedd[i],
                embedd_target[i],
                y_true,
                y_pred_target,
                self.alpha,
                self.beta,
                self.class_weights,
            )
            gamma = ot_solve(a, b, M)

            loss_classif = self.criterion(y_pred, y_true)
            loss_deepjdot += torch.sum(gamma * M)

        return loss_classif + loss_deepjdot
