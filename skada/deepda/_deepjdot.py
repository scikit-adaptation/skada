import torch

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
        base_model,
        layer_names,
        n_classes,
        optimizer=None,
        criterion=None,
        n_epochs=100,
        batch_size=16,
        alpha=1,
        beta=1,
        class_weights=None
    ):
        super().__init__(
            base_model,
            layer_names,
            optimizer,
            criterion,
            n_epochs,
            batch_size
        )
        self.n_classes = n_classes
        self.alpha = alpha
        self.beta = beta
        self.class_weights = class_weights

    def _loss_da(self):
        # Update gamma
        # Uniform Distributions
        loss_deepjdot = 0
        for i in range(len(self.embedd)):

            a = torch.full(
                (len(self.embedd[i]),),
                1.0 / len(self.embedd[i]),
                device=self.device
            )
            b = torch.full(
                (len(self.embedd_target[i]),),
                1.0 / len(self.embedd_target[i]),
                device=self.device
            )
            M = distance_matrix(
                self.embedd[i],
                self.embedd_target[i],
                self.batch_y,
                self.output_target,
                self.alpha,
                self.beta,
                self.class_weights,
                self.n_classes
            )
            gamma = ot_solve(a, b, M)

            loss_deepjdot += torch.sum(gamma * M)
            loss_classif = self.criterion(self.output, self.batch_y)
        return loss_classif + loss_deepjdot
