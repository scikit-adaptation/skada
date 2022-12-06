import torch
from torch import nn
from torch.autograd import Function


class toyCNN(nn.Module):
    """Toy CNN for examples and tests.

    Parameters
    ----------
    n_channels : int
        Number of channels of the input.
    input_size : int
        Size of the input.
    kernel_size : int, default=64
        Kernel size of the convolutional layer.
    out_channels : int, default=10
        Number of output channels in convolutional layer.
    """
    def __init__(
        self, n_channels, input_size, n_classes, kernel_size=64, out_channels=10
    ):
        super(toyCNN, self).__init__()

        self.feature_extractor = nn.Sequential(
            nn.Conv1d(n_channels, out_channels, kernel_size),
            nn.ReLU(),
            nn.AvgPool1d(kernel_size)
        )
        self.len_last_layer = self._len_last_layer(n_channels, input_size)
        self.fc = nn.Linear(self.len_last_layer, n_classes)

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.fc(x.flatten(start_dim=1))
        return x

    def _len_last_layer(self, n_channels, input_size):
        self.feature_extractor.eval()
        with torch.no_grad():
            out = self.feature_extractor(
                torch.Tensor(1, n_channels, input_size))
        self.feature_extractor.train()
        return len(out.flatten())


class GradientReversalLayer(Function):
    """Leaves the input unchanged during forward propagation
       and reverses the gradient by multiplying it by a
       negative scalar during the backpropagation.
    """
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None


class DomainClassifier(nn.Module):
    """Classifier Architecture from DANN paper [1]_.

    Parameters
    ----------
    len_last_layer : int
        Size of the input, e.g size of the last layer of
        the feature extractor
    n_classes : int, default=1
        Number of classes

    References
    ----------
    .. [1]  Yaroslav Ganin et. al. Domain-Adversarial Training
            of Neural Networks  In Journal of Machine Learning
            Research, 2016.
    """
    def __init__(
        self,
        len_last_layer,
        n_classes=1
    ):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(len_last_layer, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(),
            nn.Linear(100, n_classes),
            nn.Sigmoid()
        )

    def forward(self, x, alpha=None):
        """Forward pass.

        Parameters
        ---------
        x: torch.Tensor
            Batch of EEG windows of shape (batch_size, n_channels, n_times).
        alpha: float
            Parameter for the reverse layer.
        """
        reverse_x = GradientReversalLayer.apply(x, alpha)
        return self.classifier(reverse_x)
