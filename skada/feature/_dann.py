import torch

from skorch.utils import to_tensor

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
    domain_classifier : torch module (class or instance)
        A PyTorch :class:`~torch.nn.Module` used for classying domain.
        In general, the uninstantiated class should be passed, although
        instantiated modules will also work.
    domain_criterion : torch criterion (class)
        The uninitialized criterion (loss) used to optimize the
        domain classifier.
    reg: float, optional (default=1)
        The regularization parameter of the covariance estimator.

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
        domain_classifier,
        domain_criterion,
        reg=1,
        **kwargs
    ):
        super().__init__(
            module, criterion, layer_names, **kwargs
        )
        self.reg = reg
        self.domain_classifier = domain_classifier
        self.domain_criterion = domain_criterion

    def _initialize_domain_criterion(self):
        """Initializes the domain criterion.

        If the domain criterion is already initialized and no parameter was changed, it
        will be left as is.

        """
        kwargs = self.get_params_for('domain_criterion')
        domain_criterion = self.initialized_instance(self.domain_criterion, kwargs)
        self.domain_criterion_ = domain_criterion
        return self

    def _initialize_domain_classifier(self):
        """Initializes the domain classifier.
        If the domain classifier is already initialized and no parameter was changed, it
        will be left as is.
        """
        kwargs = self.get_params_for('domain_classifier')
        domain_classifier = self.initialized_instance(self.domain_classifier, kwargs)
        self.domain_classifier_ = domain_classifier
        return

    def initialize(self):
        """Initializes all of its components and returns self."""
        self.check_training_readiness()

        self._initialize_virtual_params()
        self._initialize_callbacks()
        self._initialize_module()
        self._initialize_criterion()
        self._initialize_optimizer()
        self._initialize_history()
        self._initialize_domain_classifier()
        self._initialize_domain_criterion()

        self._validate_params()

        self.initialized_ = True
        return self

    def _get_loss_da(
        self,
        y_pred,
        y_true,
        embedd,
        embedd_target,
        X=None,
        X_target=None,
        y_pred_target=None,
        training=True
    ):
        """Compute the domain adaptation loss"""
        y_true = to_tensor(y_true, device=self.device)

        # create domain label
        domain_label = torch.zeros(
            (embedd.size()[0]), device=self.device, dtype=torch.int64
        )
        domain_label_target = torch.ones(
            (embedd_target.size()[0]), device=self.device, dtype=torch.int64
        )

        # update classification function
        output_domain = self.domain_classifier_.forward(embedd)
        output_domain_target = self.domain_classifier_.forward(embedd_target)

        loss_DANN = (
            self.domain_criterion_(output_domain, domain_label) +
            self.domain_criterion_(output_domain_target, domain_label_target)
        )

        loss_classif = self.criterion_(y_pred, y_true)
        return loss_classif + loss_DANN
