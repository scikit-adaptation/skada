# Author: Yanis Lalou <yanis.lalou@polytechnique.edu>
#
# License: BSD 3-Clause

import torch
import torch.nn.functional as F
from skorch.callbacks import Callback


class ComputeSourceCentroids(Callback):
    """Callback to compute centroids of source domain features for each class.

    This callback computes the centroids of the normalized features for each class
    in the source domain at the beginning of each epoch. The centroids are stored
    in the adaptation criterion of the network for later use.
    """

    def on_epoch_begin(self, net, dataset_train=None, **kwargs):
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
        X, y = dataset_train.X, dataset_train.y

        X, y_ = net._prepare_input(X)
        y = y_ if y is None else y

        # Keep only source samples
        X_s = X["X"][X["sample_domain"] >= 0]
        y_s = y[X["sample_domain"] >= 0]

        features_s = net.predict_features(X_s)

        features_s = torch.tensor(features_s, device=net.device)
        y_s = torch.tensor(y_s, device=net.device)

        n_classes = len(y_s.unique())
        source_centroids = []

        for c in range(n_classes):
            mask = y_s == c
            if mask.sum() > 0:
                class_features = features_s[mask]
                normalized_features = F.normalize(class_features, p=2, dim=1)
                centroid = normalized_features.mean(dim=0)
                source_centroids.append(centroid)

        net.criterion__adapt_criterion.source_centroids = torch.stack(source_centroids)
