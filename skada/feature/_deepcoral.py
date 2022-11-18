from skorch.utils import to_tensor

from ..utils import cov, norm_coral
from .base import BaseDANetwork


class DeepCORAL(BaseDANetwork):
    """Loss DeepCORAL

    From [2]_.

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
    reg: float, optional (default=1)
        The regularization parameter of the covariance estimator.
    **kwargs : dict
        Keyword arguments passed to the skorch Model class.

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
        reg=1,
        **kwargs
    ):
        super().__init__(
            module, criterion, layer_names, **kwargs
        )
        self.reg = reg

    def _get_loss_da(
        self,
        y_pred,
        y_true,
        embedd,
        embedd_target,
        X=None,
        y_pred_target=None,
        training=True
    ):
        """Compute the domain adaptation loss"""
        y_true = to_tensor(y_true, device=self.device)

        loss_coral = 0
        for i in range(len(embedd)):
            Cs = cov(embedd[i])
            Ct = cov(embedd_target[i])
            loss_coral += self.reg * norm_coral(Cs, Ct)

        loss_classif = self.criterion_(y_pred, y_true)
        return loss_classif + loss_coral
