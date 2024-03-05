# Author: Theo Gnassounou <theo.gnassounou@inria.fr>
#         Remi Flamary <remi.flamary@polytechnique.edu>
#         Remi Flamary <remi.flamary@polytechnique.edu>
#
# License: BSD 3-Clause
import torch
from torch import nn
from torch.autograd import Function


class ToyModule2D(torch.nn.Module):
    """XXX add docstring here."""

    def __init__(self, n_classes=2, num_features=10, nonlin=torch.nn.ReLU()):
        super().__init__()

        self.dense0 = torch.nn.Linear(2, num_features)
        self.nonlin = nonlin
        self.dropout = torch.nn.Dropout(0.5)
        self.dense1 = torch.nn.Linear(num_features, n_classes)

    def forward(
        self,
        X,
    ):
        """XXX add docstring here."""
        X = self.nonlin(self.dense0(X))
        X = self.dropout(X)
        X = self.dense1(X)
        return X


class ToyModuleClassification2D(torch.nn.Module):
    """XXX add docstring here."""

    # Last layer is a softmax to get probabilities
    # Mandatory to test the PredictionEntropyScorer scorer

    def __init__(self, n_classes=2, num_features=10, nonlin=torch.nn.ReLU()):
        super().__init__()

        self.dense0 = torch.nn.Linear(2, num_features)
        self.nonlin = nonlin
        self.dropout = torch.nn.Dropout(0.5)
        self.dense1 = torch.nn.Linear(num_features, n_classes)
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(
        self,
        X,
    ):
        """XXX add docstring here."""
        X = self.nonlin(self.dense0(X))
        X = self.dropout(X)
        X = self.dense1(X)
        X = self.softmax(X)
        return X


class ToyCNN(nn.Module):
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
        super().__init__()

        self.feature_extractor = nn.Sequential(
            nn.Conv1d(n_channels, out_channels, kernel_size),
            nn.ReLU(),
            nn.AvgPool1d(kernel_size),
        )
        self.num_features = self._num_features(n_channels, input_size)
        self.fc = nn.Linear(self.num_features, n_classes)

    def forward(self, x):
        """XXX add docstring here."""
        x = self.feature_extractor(x)
        x = self.fc(x.flatten(start_dim=1))
        return x

    def _num_features(self, n_channels, input_size):
        self.feature_extractor.eval()
        with torch.no_grad():
            out = self.feature_extractor(torch.Tensor(1, n_channels, input_size))
        self.feature_extractor.train()
        return len(out.flatten())


class GradientReversalLayer(Function):
    """Leaves the input unchanged during forward propagation
    and reverses the gradient by multiplying it by a
    negative scalar during the backpropagation.
    """

    @staticmethod
    def forward(ctx, x, alpha):
        """XXX add docstring here."""
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        """XXX add docstring here."""
        output = grad_output.neg() * ctx.alpha
        return output, None


class DomainClassifier(nn.Module):
    """Classifier Architecture from DANN paper [1]_.

    Parameters
    ----------
    num_features : int
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

    def __init__(self, num_features, n_classes=1, alpha=1):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(num_features, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(),
            nn.Linear(100, n_classes),
            nn.Softmax(),
        )
        self.alpha = alpha

    def forward(self, x):
        """Forward pass.

        Parameters
        ----------
        x: torch.Tensor
            Batch of EEG windows of shape (batch_size, n_channels, n_times).
        alpha: float
            Parameter for the reverse layer.
        """
        reverse_x = GradientReversalLayer.apply(x, self.alpha)
        return self.classifier(reverse_x).flatten()


class MNISTtoUSPSNet(nn.Module):
    """XXX add docstring here."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.relu2 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)
        self.maxpool = nn.MaxPool2d(2)

    def forward(self, x):
        """XXX add docstring here."""
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool(x)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.dropout2(x)
        output = self.fc2(x)
        return output
