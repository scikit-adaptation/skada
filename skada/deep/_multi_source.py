import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.base import BaseEstimator, ClassifierMixin
from skada import BaseAdapter

# === Feature Extractor avec domain embedding ===
class FeatureExtractor(nn.Module):
    def __init__(self, input_dim, hidden_dim, domain_embedding_dim, num_domains):
        super().__init__()
        self.domain_embedding = nn.Embedding(num_domains, domain_embedding_dim)
        self.net = nn.Sequential(
            nn.Linear(input_dim + domain_embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

    def forward(self, x, domain_ids):
        domain_embed = self.domain_embedding(domain_ids)
        x_cat = torch.cat([x, domain_embed], dim=1)
        return self.net(x_cat)

# === Classifier simple (partagé) ===
class Classifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.classifier = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.classifier(x)

# === M3SDA Adapter complet ===
class M3SDAAdapter(BaseAdapter, BaseEstimator, ClassifierMixin):
    def __init__(self, input_dim, hidden_dim=128, num_classes=2, domain_embedding_dim=8, epochs=10, batch_size=64, lr=1e-3, device=None):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.domain_embedding_dim = domain_embedding_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.is_fitted_ = False

    def fit(self, X, y=None, sample_domain=None):
        X = np.asarray(X)
        y = np.asarray(y)
        sample_domain = np.asarray(sample_domain)

        # Trouver les domaines source et cible
        all_domains = np.unique(sample_domain)
        self.source_domains_ = [d for d in all_domains if d != 'tgt']
        self.target_domain_ = 'tgt'

        self.domain_to_idx_ = {d: i for i, d in enumerate(self.source_domains_ + [self.target_domain_])}
        num_domains = len(self.domain_to_idx_)

        # Séparer par domaine
        X_sources, y_sources = [], []
        for d in self.source_domains_:
            mask = sample_domain == d
            X_sources.append(X[mask])
            y_sources.append(y[mask])

        X_target = X[sample_domain == self.target_domain_]

        # Loaders
        loaders_source = [self._to_loader(Xs, ys, self.domain_to_idx_[d]) for Xs, ys, d in zip(X_sources, y_sources, self.source_domains_)]
        loader_target = self._to_loader(X_target, domain_label=self.domain_to_idx_[self.target_domain_])

        # Init modèle
        self.feature_extractor_ = FeatureExtractor(self.input_dim, self.hidden_dim, self.domain_embedding_dim, num_domains).to(self.device)
        self.classifier_ = Classifier(self.hidden_dim, self.num_classes).to(self.device)

        self._train_model(loaders_source, loader_target)
        self.is_fitted_ = True
        return self

    def _train_model(self, loaders_source, loader_target):
        self.feature_extractor_.train()
        self.classifier_.train()
        optimizer = torch.optim.Adam(
            list(self.feature_extractor_.parameters()) + list(self.classifier_.parameters()),
            lr=self.lr
        )
        criterion = nn.CrossEntropyLoss()

        for epoch in range(self.epochs):
            for batches in zip(*loaders_source):
                optimizer.zero_grad()
                losses = []

                # Target batch
                try:
                    batch_target = next(self.target_iter)
                except:
                    self.target_iter = iter(loader_target)
                    batch_target = next(self.target_iter)

                x_t, d_t = batch_target[0].to(self.device), batch_target[1].to(self.device)
                f_t = self.feature_extractor_(x_t, d_t)

                for x_s, y_s, d_s in batches:
                    x_s, y_s, d_s = x_s.to(self.device), y_s.to(self.device), d_s.to(self.device)

                    # Forward
                    f_s = self.feature_extractor_(x_s, d_s)
                    y_pred = self.classifier_(f_s)

                    # Classification loss
                    loss_cls = criterion(y_pred, y_s)

                    # Moment matching loss
                    loss_mm = self._moment_loss(f_s, f_t)
                    losses.append(loss_cls + loss_mm)

                loss = sum(losses) / len(losses)
                loss.backward()
                optimizer.step()

    def _moment_loss(self, f_s, f_t):
        mu_s = f_s.mean(0)
        mu_t = f_t.mean(0)
        return torch.norm(mu_s - mu_t, p=2)

    def _to_loader(self, X, y=None, domain_label=None):
        X_tensor = torch.tensor(X, dtype=torch.float32)
        d_tensor = torch.full((X.shape[0],), domain_label, dtype=torch.long)
        if y is not None:
            y_tensor = torch.tensor(y, dtype=torch.long)
            dataset = TensorDataset(X_tensor, y_tensor, d_tensor)
        else:
            dataset = TensorDataset(X_tensor, d_tensor)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

    def transform(self, X, sample_domain=None):
        if not self.is_fitted_:
            raise ValueError("M3SDAAdapter must be fitted before calling transform.")

        self.feature_extractor_.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
            d_idx = self.domain_to_idx_.get(sample_domain, self.domain_to_idx_[self.target_domain_])
            d_tensor = torch.full((X.shape[0],), d_idx, dtype=torch.long).to(self.device)
            features = self.feature_extractor_(X_tensor, d_tensor)
        return features.cpu().numpy()

    def predict(self, X, sample_domain=None):
        features = self.transform(X, sample_domain=sample_domain)
        self.classifier_.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(features, dtype=torch.float32).to(self.device)
            preds = self.classifier_(X_tensor).argmax(1)
        return preds.cpu().numpy()

    def score(self, X, y, sample_domain=None):
        from sklearn.metrics import accuracy_score
        y_pred = self.predict(X, sample_domain)
        return accuracy_score(y, y_pred)