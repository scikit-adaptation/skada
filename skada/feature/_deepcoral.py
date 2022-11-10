from skorch.utils import to_tensor

from ..utils import cov, norm_coral
from .base import BaseDANetwork


class DeepCORAL(BaseDANetwork):
    """Loss DeepCORAL

    From [2]_.

    Parameters
    ----------
    base_model: torch model
        model used for training and prediction
    layer_names: list of tuples
        list storing the name of the layers
        from which we want to get the output.
    optimizer:  torch optimizer or None
        Optimizer to use for training,
        if None use Adam optimizer.
    criterion:  torch criterion or None
        criterion to use for training,
        if None use CrossEntropy.
    n_epochs: int
        number of the epoch during training.
    batch_size: int
        batch size used to create the dataloader.
    alpha: float
        parameter for DeepCoral method.

    References
    ----------
    .. [2]  Baochen Sun and Kate Saenko. Deep coral:
            Correlation alignment for deep domain
            adaptation. In ECCV Workshops, 2016.
    """

    def __init__(
        self,
        module,
        criterion,
        layer_names,
        alpha=1,
        **kwargs
    ):
        super().__init__(
            module, criterion, layer_names, **kwargs
        )
        self.alpha = alpha

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

        loss_coral = 0
        for i in range(len(embedd)):
            Cs = cov(embedd[i])
            Ct = cov(embedd_target[i])
            loss_coral += self.alpha * norm_coral(Cs, Ct)

        loss_classif = self.criterion_(y_pred, y_true)
        return loss_classif + loss_coral
