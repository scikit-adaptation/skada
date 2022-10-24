from ..utils import cov, norm_coral
from . import BaseDANetwork


class DeepCORAL(BaseDANetwork):
    """Loss DeepCORAL

    Parameters
    ----------
    XXX

    References
    ----------
    .. [2]  Baochen Sun and Kate Saenko. Deep coral:
            Correlation alignment for deep domain
            adaptation. In ECCV Workshops, 2016.
    """

    def __init__(
        self,
        base_model,
        layer_names,
        optimizer=None,
        criterion=None,
        n_epochs=100,
        batch_size=16,
        alpha=0.5
    ):
        super().__init__(
            base_model,
            layer_names,
            optimizer,
            criterion,
            n_epochs,
            batch_size
            )
        self.alpha = alpha

    def _loss_da(self):
        loss_coral = 0
        for i in range(len(self.embedd)):
            Cs = cov(self.embedd[i])
            Ct = cov(self.embedd_target[i])

            loss_coral += self.alpha * norm_coral(Cs, Ct)
        loss_classif = self.criterion(self.output, self.batch_y)
        return loss_classif + loss_coral
