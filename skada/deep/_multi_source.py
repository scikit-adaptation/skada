# Author: Christopher Engoang <christopherengoangperrat@gmail.com>
#
# License: BSD 3-Clause
"""Deep multi-source domain adaptation methods."""

from itertools import combinations

import numpy as np
import torch
from sklearn.base import ClassifierMixin
from sklearn.metrics import accuracy_score
from sklearn.utils.validation import _check_sample_weight, check_is_fitted
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from skada.base import DAEstimator
from skada.utils import check_X_domain, check_X_y_domain


class M3SDALoss(nn.Module):
    """Moment distance used by M3SDA.

    The loss aligns every source distribution with the target distribution and
    every pair of source distributions. Raw element-wise moments from order one
    through ``moment_order`` are matched.

    Parameters
    ----------
    moment_order : int, default=2
        Highest raw moment to align. The M3SDA paper uses moments one and two.
    """

    def __init__(self, moment_order=2):
        super().__init__()
        if moment_order < 1:
            raise ValueError("moment_order must be greater than or equal to 1")
        self.moment_order = moment_order

    def forward(self, source_features, target_features):
        """Compute the multi-source moment distance.

        Parameters
        ----------
        source_features : sequence of torch.Tensor
            One feature tensor of shape ``(n_samples, n_features)`` per source.
        target_features : torch.Tensor
            Target feature tensor of shape ``(n_samples, n_features)``.

        Returns
        -------
        loss : torch.Tensor
            Scalar moment distance.
        """
        n_sources = len(source_features)
        if n_sources < 2:
            raise ValueError("M3SDA requires at least two source domains")
        if target_features.shape[0] == 0:
            raise ValueError("target_features must contain at least one sample")

        loss = target_features.new_zeros(())
        source_pair_weight = 2.0 / (n_sources * (n_sources - 1))

        for order in range(1, self.moment_order + 1):
            source_moments = [
                features.pow(order).mean(dim=0) for features in source_features
            ]
            target_moment = target_features.pow(order).mean(dim=0)

            source_target = (
                sum(
                    torch.linalg.vector_norm(moment - target_moment)
                    for moment in source_moments
                )
                / n_sources
            )
            source_source = sum(
                torch.linalg.vector_norm(first - second)
                for first, second in combinations(source_moments, 2)
            )
            loss = loss + source_target + source_pair_weight * source_source

        return loss


class _M3SDAModule(nn.Module):
    """Shared feature extractor with one classification head per source."""

    def __init__(self, input_dim, hidden_dim, n_classes, n_sources):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.classifiers = nn.ModuleList(
            nn.Linear(hidden_dim, n_classes) for _ in range(n_sources)
        )

    def extract_features(self, X):
        return self.feature_extractor(X)

    def source_logits(self, features, source_index):
        return self.classifiers[source_index](features)

    def target_proba(self, features):
        probabilities = [
            torch.softmax(classifier(features), dim=1)
            for classifier in self.classifiers
        ]
        return torch.stack(probabilities, dim=0).mean(dim=0)


class M3SDA(ClassifierMixin, DAEstimator):
    """Moment Matching for Multi-Source Domain Adaptation (M3SDA).

    This estimator implements the base M3SDA method: one shared feature
    extractor, one classifier per labeled source domain, and alignment of both
    source-target and source-source feature moments. Target predictions are the
    mean of the source-classifier probabilities.

    Parameters
    ----------
    input_dim : int or None, default=None
        Number of input features. If None, infer it during :meth:`fit`.
    hidden_dim : int, default=128
        Size of the shared latent representation.
    num_classes : int or None, default=None
        Expected number of source classes. If None, infer it during :meth:`fit`.
    reg : float, default=1.0
        Weight of the moment-matching term.
    moment_order : int, default=2
        Highest raw moment to align.
    max_epochs : int, default=10
        Number of training epochs.
    batch_size : int, default=64
        Number of samples drawn from each domain at every optimization step.
    lr : float, default=1e-3
        Adam learning rate.
    device : str or torch.device or None, default=None
        Training device. If None, use CUDA when available and CPU otherwise.
    random_state : int or None, default=None
        Seed controlling module initialization and data shuffling.

    References
    ----------
    .. [1] Peng, X., Bai, Q., Xia, X., Huang, Z., Saenko, K., & Wang, B.
           Moment Matching for Multi-Source Domain Adaptation. ICCV, 2019.
    """

    __metadata_request__fit = {"sample_domain": True, "sample_weight": True}
    __metadata_request__predict = {"sample_domain": True, "allow_source": True}
    __metadata_request__predict_proba = {
        "sample_domain": True,
        "allow_source": True,
    }
    __metadata_request__score = {
        "sample_domain": True,
        "sample_weight": True,
        "allow_source": True,
    }
    __metadata_request__transform = {"sample_domain": True, "allow_source": True}

    def __init__(
        self,
        input_dim=None,
        hidden_dim=128,
        num_classes=None,
        reg=1.0,
        moment_order=2,
        max_epochs=10,
        batch_size=64,
        lr=1e-3,
        device=None,
        random_state=None,
    ):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.reg = reg
        self.moment_order = moment_order
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.device = device
        self.random_state = random_state

    def fit(self, X, y, sample_domain=None, *, sample_weight=None):
        """Fit the M3SDA model using labeled sources and an unlabeled target."""
        X, y, sample_domain = check_X_y_domain(
            X,
            y,
            sample_domain,
            allow_auto_sample_domain=False,
            allow_multi_source=True,
            # Validate the single-target constraint below. Passing False here
            # currently triggers the legacy multi-target check on n_sources.
            allow_multi_target=True,
        )
        X = np.asarray(X, dtype=np.float32)
        sample_domain = np.asarray(sample_domain)
        source_mask = sample_domain >= 0
        target_mask = ~source_mask

        self.source_domains_ = np.unique(sample_domain[source_mask])
        target_domains = np.unique(sample_domain[target_mask])
        if self.source_domains_.size < 2:
            raise ValueError("M3SDA requires at least two source domains")
        if target_domains.size != 1:
            raise ValueError("M3SDA requires exactly one target domain")
        self.target_domain_ = target_domains[0]

        self.n_features_in_ = X.shape[1]
        if self.input_dim is not None and self.input_dim != self.n_features_in_:
            raise ValueError(
                f"X has {self.n_features_in_} features, but input_dim={self.input_dim}"
            )
        input_dim = self.n_features_in_ if self.input_dim is None else self.input_dim

        self.classes_, encoded_source_y = np.unique(y[source_mask], return_inverse=True)
        n_classes = self.classes_.size
        if self.num_classes is not None and self.num_classes != n_classes:
            raise ValueError(
                f"Found {n_classes} source classes, but num_classes={self.num_classes}"
            )

        if self.hidden_dim < 1:
            raise ValueError("hidden_dim must be greater than or equal to 1")
        if self.batch_size < 1:
            raise ValueError("batch_size must be greater than or equal to 1")
        if self.max_epochs < 1:
            raise ValueError("max_epochs must be greater than or equal to 1")
        if self.reg < 0:
            raise ValueError("reg must be non-negative")

        sample_weight = _check_sample_weight(sample_weight, X, dtype=np.float32)
        self.device_ = torch.device(
            self.device
            if self.device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self._set_random_state()

        self.module_ = _M3SDAModule(
            input_dim=input_dim,
            hidden_dim=self.hidden_dim,
            n_classes=n_classes,
            n_sources=self.source_domains_.size,
        ).to(self.device_)
        self.moment_loss_ = M3SDALoss(self.moment_order)
        optimizer = torch.optim.Adam(self.module_.parameters(), lr=self.lr)
        source_loaders, target_loader = self._make_loaders(
            X,
            encoded_source_y,
            sample_domain,
            source_mask,
            target_mask,
            sample_weight,
        )

        self.loss_history_ = []
        for _ in range(self.max_epochs):
            epoch_losses = self._train_epoch(source_loaders, target_loader, optimizer)
            self.loss_history_.append(float(np.mean(epoch_losses)))
        return self

    def _set_random_state(self):
        if self.random_state is not None:
            torch.manual_seed(self.random_state)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.random_state)

    def _make_loaders(
        self,
        X,
        encoded_source_y,
        sample_domain,
        source_mask,
        target_mask,
        sample_weight,
    ):
        generator = torch.Generator()
        if self.random_state is not None:
            generator.manual_seed(self.random_state)

        source_y_full = np.empty(X.shape[0], dtype=np.int64)
        source_y_full[source_mask] = encoded_source_y
        source_loaders = []
        for domain in self.source_domains_:
            mask = sample_domain == domain
            dataset = TensorDataset(
                torch.from_numpy(X[mask]),
                torch.from_numpy(source_y_full[mask]),
                torch.from_numpy(sample_weight[mask]),
            )
            source_loaders.append(
                DataLoader(
                    dataset,
                    batch_size=self.batch_size,
                    shuffle=True,
                    generator=generator,
                )
            )

        target_dataset = TensorDataset(torch.from_numpy(X[target_mask]))
        target_loader = DataLoader(
            target_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            generator=generator,
        )
        return source_loaders, target_loader

    @staticmethod
    def _next_batch(iterator, loader):
        try:
            return next(iterator), iterator
        except StopIteration:
            iterator = iter(loader)
            return next(iterator), iterator

    def _train_epoch(self, source_loaders, target_loader, optimizer):
        self.module_.train()
        source_iterators = [iter(loader) for loader in source_loaders]
        target_iterator = iter(target_loader)
        n_steps = max([len(target_loader), *(len(loader) for loader in source_loaders)])
        epoch_losses = []

        for _ in range(n_steps):
            source_batches = []
            for index, loader in enumerate(source_loaders):
                batch, source_iterators[index] = self._next_batch(
                    source_iterators[index], loader
                )
                source_batches.append(batch)
            target_batch, target_iterator = self._next_batch(
                target_iterator, target_loader
            )

            optimizer.zero_grad()
            target_X = target_batch[0].to(self.device_)
            target_features = self.module_.extract_features(target_X)
            source_features = []
            classification_loss = target_features.new_zeros(())

            for source_index, (source_X, source_y, weights) in enumerate(
                source_batches
            ):
                source_X = source_X.to(self.device_)
                source_y = source_y.to(self.device_)
                weights = weights.to(self.device_)
                features = self.module_.extract_features(source_X)
                logits = self.module_.source_logits(features, source_index)
                losses = nn.functional.cross_entropy(logits, source_y, reduction="none")
                classification_loss = classification_loss + (
                    losses * weights
                ).sum() / weights.sum().clamp_min(torch.finfo(weights.dtype).eps)
                source_features.append(features)

            classification_loss = classification_loss / len(source_batches)
            moment_loss = self.moment_loss_(source_features, target_features)
            loss = classification_loss + self.reg * moment_loss
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.detach().cpu().item())

        return epoch_losses

    def transform(self, X, sample_domain=None, *, allow_source=False):
        """Map samples through the shared feature extractor."""
        check_is_fitted(self, "module_")
        X = self._validate_prediction_data(X, sample_domain, allow_source)
        self.module_.eval()
        with torch.no_grad():
            features = self.module_.extract_features(
                torch.as_tensor(X, dtype=torch.float32, device=self.device_)
            )
        return features.cpu().numpy()

    def predict_proba(self, X, sample_domain=None, *, allow_source=False):
        """Predict by averaging probabilities from all source classifiers."""
        check_is_fitted(self, "module_")
        X = self._validate_prediction_data(X, sample_domain, allow_source)
        self.module_.eval()
        with torch.no_grad():
            features = self.module_.extract_features(
                torch.as_tensor(X, dtype=torch.float32, device=self.device_)
            )
            probabilities = self.module_.target_proba(features)
        return probabilities.cpu().numpy()

    def predict(self, X, sample_domain=None, *, allow_source=False):
        """Predict class labels for target samples."""
        indices = self.predict_proba(
            X, sample_domain=sample_domain, allow_source=allow_source
        ).argmax(axis=1)
        return self.classes_[indices]

    def score(
        self,
        X,
        y,
        sample_domain=None,
        *,
        sample_weight=None,
        allow_source=False,
    ):
        """Return mean classification accuracy."""
        predictions = self.predict(
            X, sample_domain=sample_domain, allow_source=allow_source
        )
        return accuracy_score(y, predictions, sample_weight=sample_weight)

    def _validate_prediction_data(self, X, sample_domain, allow_source):
        X, _ = check_X_domain(
            X,
            sample_domain,
            allow_source=allow_source,
            allow_multi_target=False,
        )
        X = np.asarray(X, dtype=np.float32)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but M3SDA was fitted with "
                f"{self.n_features_in_} features"
            )
        return X


# Backward-compatible name used by the initial draft of PR #341.
M3SDAAdapter = M3SDA
