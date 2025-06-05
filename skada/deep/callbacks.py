# Author: Yanis Lalou <yanis.lalou@polytechnique.edu>
#
# License: BSD 3-Clause

import torch
import torch.nn.functional as F
from skorch.callbacks import Callback
from skorch.utils import to_tensor

from skada.deep.base import DomainAwareNet
from skada.deep.utils import SphericalKMeans


class ComputeSourceCentroids(Callback):
    """Callback to compute centroids of source domain features for each class.

    This callback computes the centroids of the normalized features for each class
    in the source domain at the beginning of each epoch. The centroids are stored
    in the adaptation criterion of the network for later use.
    """

    def on_epoch_begin(self, net: DomainAwareNet, dataset_train=None, **kwargs):
        """Compute source centroids at the beginning of each epoch.

        Parameters
        ----------
        net : NeuralNet
            The neural network being trained.
        dataset_train : Dataset, optional
            The training dataset.
        **kwargs : dict
            Additional arguments passed to the callback.
        """
        X = net._prepare_input(dataset_train)

        # Keep only source samples
        source = X.select_source()
        X_s = source.X
        y_s = source.y

        X_t = X.select_target().X

        # Disable gradient computation for feature extraction
        with torch.no_grad():
            features_s = net.predict_features(X_s)
            features_t = net.predict_features(X_t)

            features_s = torch.tensor(features_s, device=net.device)
            y_s = torch.tensor(y_s, device=net.device)

            features_t = torch.tensor(features_t, device=net.device)

            n_classes = len(y_s.unique())
            source_centroids = []

            for c in range(n_classes):
                mask = y_s == c
                if mask.sum() > 0:
                    class_features = features_s[mask]
                    normalized_features = F.normalize(class_features, p=2, dim=1)
                    centroid = normalized_features.sum(dim=0)
                    source_centroids.append(centroid)

            source_centroids = torch.stack(source_centroids)

            # Use source centroids to initialize target clustering
            target_kmeans = SphericalKMeans(
                n_clusters=n_classes,
                random_state=0,
                initial_centroids=source_centroids,
                device=features_t.device,
            )
            target_kmeans.fit(features_t)

        net.criterion__adapt_criterion.target_kmeans = target_kmeans


class MemoryBank(Callback):
    """Callback to compute memory features and outputs of target domain.

    This callback computes the memory features of target domain to be able
    to compute pseudo label during training.
    """

    def __init__(self, momentum=0.5):
        super().__init__()
        self.momentum = momentum

    def on_train_begin(self, net: DomainAwareNet, X=None, y=None):
        """This method is called at the beginning of training.

        Parameters
        ----------
        net (NeuralNet): The Skorch NeuralNet instance.
        X (numpy.ndarray or None): The input data used for training. Only passed if
            the `iterator_train` callback is not set.
        y (numpy.ndarray or None): The target data used for training. Only passed if
            the `iterator_train` callback is not set.
        """
        X = net._prepare_input(X, y)

        # Take only first sample without its label
        X_sample = X[0:1][0]

        # Disable gradient computation for feature extraction
        with torch.no_grad():
            features_sample = net.predict_features(X_sample)
            pred_sample = net.predict_proba(X_sample, allow_source=True)

        n_target_samples = len(X.select_target())
        n_features = features_sample.shape[1]

        n_classes = pred_sample.shape[1]

        memory_features = torch.rand(
            (n_target_samples, n_features),
            device=net.device,
            dtype=torch.float32,
        )
        memory_features = memory_features / torch.linalg.norm(
            memory_features, dim=1, keepdim=True
        )
        net.criterion__adapt_criterion.memory_features = memory_features

        memory_outputs = (
            torch.ones(
                (n_target_samples, n_classes),
                device=net.device,
                dtype=torch.float32,
            )
            / n_classes
        )
        net.criterion__adapt_criterion.memory_outputs = memory_outputs

    def on_batch_end(self, net, batch, **kwargs):
        """Compute memory bank at the end of each epoch.

        Parameters
        ----------
        net : NeuralNet
            The neural network being trained.
        dataset_train : Dataset, optional
            The training dataset.
        **kwargs : dict
            Additional arguments passed to the callback.
        """
        X, _ = batch
        X_t = X["X"][X["sample_domain"] < 0]
        batch_idx = X["sample_idx"][X["sample_domain"] < 0]

        net.module_.eval()
        with torch.no_grad():
            X_t = to_tensor(X_t, device=net.device)
            output_t, features_t = net.module_(X_t, return_features=True)
            features_t = F.normalize(features_t, p=2, dim=1)
            softmax_out = F.softmax(output_t, dim=1)
            outputs_target = softmax_out**2 / ((softmax_out**2).sum(dim=0))

            new_memory_features = (
                1.0 - self.momentum
            ) * net.criterion__adapt_criterion.memory_features[batch_idx]
            +self.momentum * features_t.clone()

            new_memory_outputs = (
                1.0 - self.momentum
            ) * net.criterion__adapt_criterion.memory_outputs[batch_idx]
            +self.momentum * outputs_target.clone()

        net.criterion__adapt_criterion.memory_features[batch_idx] = new_memory_features
        net.criterion__adapt_criterion.memory_outputs[batch_idx] = new_memory_outputs


class CountEpochs(Callback):
    """Callback to count the number of epochs."""

    def on_train_begin(self, *args, **kwargs):
        """Initialize the epoch counter."""
        self.n_epochs = 0

    def on_epoch_begin(self, net, **kwargs):
        """Increment the number of epochs."""
        net.criterion__adapt_criterion.n_epochs += 1
