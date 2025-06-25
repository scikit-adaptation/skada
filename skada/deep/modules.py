# Author: Theo Gnassounou <theo.gnassounou@inria.fr>
#         Remi Flamary <remi.flamary@polytechnique.edu>
#         Yanis Lalou <yanis.lalou@polytechnique.edu>
#         Antoine Collas <contact@antoinecollas.fr>
#         Ambroise Odonnat <ambroiseodonnattechnologie@gmail.com>
#
# License: BSD 3-Clause
import torch
import torch.nn.utils.weight_norm as weightNorm
from torch import nn
from torch.autograd import Function


class ToyModule2D(torch.nn.Module):
    """A simple fully connected module for data with 2 features, i.e. (n_samples, 2).

    This module contains two dense layers with a non-linearity and dropout.
    It can output either logits or probabilities, depending on the
    `proba` parameter.

    Parameters
    ----------
    n_classes : int, default=2
        The number of output classes.
    num_features : int, default=10
        The number of features in the hidden layer.
    nonlin : torch.nn.Module, default=torch.nn.ReLU()
        The non-linear activation function used after the first dense layer.
    proba : bool, default=False
        If True, the output will be probabilities (using softmax).
        Otherwise, raw logits will be returned.
    """

    def __init__(
        self, n_classes=2, num_features=10, nonlin=torch.nn.ReLU(), proba=False
    ):
        super().__init__()

        self.proba = proba

        self.dense0 = torch.nn.Linear(2, num_features)
        self.nonlin = nonlin
        self.dropout = torch.nn.Dropout(0.5)
        self.dense1 = torch.nn.Linear(num_features, n_classes)
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(
        self,
        X,
        sample_weight=None,
    ):
        """Forward pass of the ToyModule2D.

        Parameters
        ----------
        X : torch.Tensor
            Input tensor of shape (batch_size, 2).
        sample_weight : torch.Tensor, optional
            Unused in this module.

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, n_classes),
            either raw logits or probabilities based on the `proba` parameter.
        """
        X = self.nonlin(self.dense0(X))
        X = self.dropout(X)
        X = self.dense1(X)
        if self.proba:
            # Last layer is a softmax to get probabilities
            # Mandatory to test the PredictionEntropyScorer scorer
            X = self.softmax(X)
        return X


class ToyCNN(nn.Module):
    """Toy CNN for examples and tests on classification tasks.

    Made for 2D data (e.g. time series) with shape (batch_size, n_channels, input_size).

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
        )
        self.num_features = self._num_features(n_channels, input_size)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(start_dim=1),
            nn.Linear(out_channels, n_classes),
        )

    def forward(self, x, sample_weight=None):
        """Forward pass of the network.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, n_channels, input_size).
        sample_weight : torch.Tensor, optional
            Sample weights for the loss computation of shape (batch_size,).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, n_classes).
        """
        x = self.feature_extractor(x)
        x = self.fc(x)
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
    def forward(ctx, x, alpha, sample_weight=None):
        """XXX add docstring here."""
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        """XXX add docstring here."""
        output = grad_output.neg() * ctx.alpha
        return output, None


class DomainClassifier(nn.Module):
    """Classifier Architecture from DANN paper [15]_.

    Parameters
    ----------
    num_features : int
        Size of the input, e.g size of the last layer of
        the feature extractor
    n_classes : int, default=1
        Number of classes

    References
    ----------
    .. [15]  Yaroslav Ganin et. al. Domain-Adversarial Training
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

    def forward(self, x, sample_weight=None):
        """Forward pass.

        Parameters
        ----------
        x: torch.Tensor
            Batch of EEG windows of shape (batch_size, n_channels, n_times).
        alpha: float
            Parameter for the reverse layer.
        """
        reverse_x = GradientReversalLayer.apply(x, self.alpha)
        return self.classifier(reverse_x).squeeze()


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

    def forward(self, x, sample_weight=None):
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


class SHOTNet(nn.Module):
    """
    SHOT Network.
    This network consists of a feature extractor, a bottleneck,
    and a classifier. The feature extractor is a LeNetBase architecture.
    """

    def __init__(self, class_num, feature_dim=256, type="ori"):
        super().__init__()
        self.feature_extractor = ShotFeatureExtractor()
        self.bottleneck = FeatBottleneck(
            feature_dim=self.feature_extractor.in_features,
            bottleneck_dim=feature_dim,
            type=type,
        )
        self.classifier = FeatClassifier(
            class_num=class_num, bottleneck_dim=feature_dim, type="linear"
        )

    def forward(self, x):
        """
        Forward pass of the SHOTNet.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Output tensor after feature extraction, bottleneck, and classification.
        """
        x = self.feature_extractor(x)
        x = self.bottleneck(x)
        x = self.classifier(x)
        return x


class ShotFeatureExtractor(nn.Module):
    """
    Feature extractor for SHOT.
    LeNetBase architecture with two convolutional layers,
    followed by two max pooling layers and a dropout layer.
    The output is flattened to a vector.
    """

    def __init__(self):
        super().__init__()
        self.conv_params = nn.Sequential(
            nn.Conv2d(1, 20, kernel_size=5),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(20, 50, kernel_size=5),
            nn.Dropout2d(p=0.5),
            nn.MaxPool2d(2),
            nn.ReLU(),
        )
        self.in_features = 50 * 4 * 4

    def forward(self, x):
        """Forward pass through the bottleneck layer."""
        x = self.conv_params(x)
        x = x.view(x.size(0), -1)
        return x


def init_weights(m):
    """Initialize weights of layers using appropriate initialization schemes.

    Parameters
    ----------
    m : nn.Module
        The layer to initialize.
    """
    classname = m.__class__.__name__
    if classname.find("Conv2d") != -1 or classname.find("ConvTranspose2d") != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find("Linear") != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)


class FeatBottleneck(nn.Module):
    """Bottleneck layer for SHOT architecture."""

    def __init__(self, feature_dim, bottleneck_dim=256, type="ori"):
        super().__init__()
        self.bn = nn.BatchNorm1d(bottleneck_dim, affine=True)
        self.dropout = nn.Dropout(p=0.5)
        self.bottleneck = nn.Linear(feature_dim, bottleneck_dim)
        self.bottleneck.apply(init_weights)
        self.type = type

    def forward(self, x):
        """Forward pass through the bottleneck layer."""
        x = self.bottleneck(x)
        if self.type == "bn":
            x = self.bn(x)
            x = self.dropout(x)
        return x


class FeatClassifier(nn.Module):
    """Classifier layer for SHOT architecture."""

    def __init__(self, class_num, bottleneck_dim=256, type="linear"):
        super().__init__()
        if type == "linear":
            self.fc = nn.Linear(bottleneck_dim, class_num)
        else:
            self.fc = weightNorm(nn.Linear(bottleneck_dim, class_num), name="weight")
        self.fc.apply(init_weights)

    def forward(self, x):
        """Forward pass through the classifier layer."""
        x = self.fc(x)
        return x
