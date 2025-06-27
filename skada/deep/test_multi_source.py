import numpy as np
from _multi_source import M3SDAAdapter  
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


def generate_2d_gaussian_domains(n_samples=100, seed=42):
    np.random.seed(seed)

    def make_domain(mu0, mu1):
        X0 = np.random.normal(loc=mu0, scale=0.5, size=(n_samples, 2))
        X1 = np.random.normal(loc=mu1, scale=0.5, size=(n_samples, 2))
        X = np.vstack([X0, X1])
        y = np.array([0] * n_samples + [1] * n_samples)
        return X, y

    X_src1, y_src1 = make_domain(mu0=[-2, 0], mu1=[-2, 2])
    X_src2, y_src2 = make_domain(mu0=[0, -2], mu1=[2, -2])
    X_tgt,  y_tgt  = make_domain(mu0=[1, 1],  mu1=[3, 3])

    X_all = np.vstack([X_src1, X_src2, X_tgt])
    y_all = np.hstack([y_src1, y_src2, y_tgt])
    domains = np.array(['src1'] * len(X_src1) + ['src2'] * len(X_src2) + ['tgt'] * len(X_tgt))

    return X_all, y_all, domains, X_tgt, y_tgt, X_src1, y_src1, X_src2, y_src2


#Training


X_all, y_all, domains, X_tgt, y_tgt, X_src1, y_src1, X_src2, y_src2 = generate_2d_gaussian_domains()

adapter = M3SDAAdapter(input_dim=2, hidden_dim=32, domain_embedding_dim=4, epochs=50)
adapter.fit(X_all, y_all, sample_domain=domains)

#Display

import matplotlib.pyplot as plt
import torch

def plot_decision_boundary(model, domain_id, domain_name, color, ax, xrange=(-4, 5), yrange=(-4, 5), steps=200):
    xx, yy = np.meshgrid(np.linspace(*xrange, steps), np.linspace(*yrange, steps))
    grid = np.c_[xx.ravel(), yy.ravel()]
    with torch.no_grad():
        inputs = torch.tensor(grid, dtype=torch.float32)
        domains = torch.full((inputs.shape[0],), domain_id, dtype=torch.long)
        feats = model.feature_extractor_(inputs, domains)
        preds = model.classifier_(feats).argmax(1).numpy()
    zz = preds.reshape(xx.shape)
    ax.contourf(xx, yy, zz, levels=1, alpha=0.15, colors=[color])

def show_class_separation(adapter, X_src1, y_src1, X_src2, y_src2, X_tgt, y_tgt, domain_to_idx):
    fig, ax = plt.subplots(figsize=(8, 8))

    ax.scatter(*X_src1[y_src1==0].T, color='blue', label='src1 - class 0')
    ax.scatter(*X_src1[y_src1==1].T, color='navy', label='src1 - class 1')

    ax.scatter(*X_src2[y_src2==0].T, color='green', label='src2 - class 0')
    ax.scatter(*X_src2[y_src2==1].T, color='darkgreen', label='src2 - class 1')

    ax.scatter(*X_tgt[y_tgt==0].T, color='red', label='tgt - class 0')
    ax.scatter(*X_tgt[y_tgt==1].T, color='darkred', label='tgt - class 1')

    # Décision par domaine
    plot_decision_boundary(adapter, domain_to_idx['src1'], 'src1', 'blue', ax)
    plot_decision_boundary(adapter, domain_to_idx['src2'], 'src2', 'green', ax)
    plot_decision_boundary(adapter, domain_to_idx['tgt'],  'tgt',  'red', ax)

    ax.set_xlim(-4, 5)
    ax.set_ylim(-4, 5)
    ax.set_title("Séparation des classes + frontières de décision")
    ax.legend()
    ax.grid(True)
    plt.show()


show_class_separation(adapter,
 X_src1, y_src1, X_src2, y_src2, X_tgt, y_tgt, adapter.domain_to_idx_)

#Naive classifier



class SimpleClassifier(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=32, num_classes=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        return self.net(x)

def train_naive_classifier(X, y, epochs=50, batch_size=64, lr=1e-3):
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)
    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = SimpleClassifier()
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        for x_batch, y_batch in loader:
            optimizer.zero_grad()
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

    return model


# Utilise uniquement les sources (comme en adaptation classique)
X_sources = np.vstack([X_src1, X_src2])
y_sources = np.hstack([y_src1, y_src2])
baseline_model = train_naive_classifier(X_sources, y_sources)


def plot_baseline_decision(model, X_tgt, y_tgt):
    xx, yy = np.meshgrid(np.linspace(-4, 5, 200), np.linspace(-4, 5, 200))
    grid = np.c_[xx.ravel(), yy.ravel()]
    with torch.no_grad():
        inputs = torch.tensor(grid, dtype=torch.float32)
        preds = model(inputs).argmax(1).numpy()
    zz = preds.reshape(xx.shape)

    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, zz, levels=1, alpha=0.2, colors=["gray", "black"])
    plt.scatter(*X_tgt[y_tgt==0].T, color="red", label="Classe 0 (tgt)", alpha=0.6)
    plt.scatter(*X_tgt[y_tgt==1].T, color="darkred", label="Classe 1 (tgt)", alpha=0.6)
    plt.title("Frontière du classifieur NON adapté sur la cible")
    plt.legend()
    plt.grid(True)
    plt.show()


# Classifieur non adapté
plot_baseline_decision(baseline_model, X_tgt, y_tgt)

# Classifieur M3SDA adapté (avec séparation + plans de décision)
show_class_separation(adapter, X_src1, y_src1, X_src2, y_src2, X_tgt, y_tgt, adapter.domain_to_idx_)