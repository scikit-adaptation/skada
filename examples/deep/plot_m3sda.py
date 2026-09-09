"""
=============================================
Moment matching with multiple source domains
=============================================

This example illustrates baseline M3SDA on two labeled Gaussian source
domains and one unlabeled target domain. M3SDA uses a shared feature extractor,
one classifier per source, and aligns first- and second-order feature moments.
"""

import matplotlib.pyplot as plt
import numpy as np

from skada.deep import M3SDA


def make_domain(generator, mean_0, mean_1, n_samples=100):
    """Generate a balanced two-class Gaussian domain."""
    X_0 = generator.normal(mean_0, 0.5, (n_samples, 2))
    X_1 = generator.normal(mean_1, 0.5, (n_samples, 2))
    return np.vstack([X_0, X_1]), np.repeat([0, 1], n_samples)


generator = np.random.default_rng(42)
X_source_1, y_source_1 = make_domain(generator, [-2, 0], [-2, 2])
X_source_2, y_source_2 = make_domain(generator, [0, -2], [2, -2])
X_target, y_target = make_domain(generator, [1, 1], [3, 3])

X = np.vstack([X_source_1, X_source_2, X_target]).astype(np.float32)
y = np.concatenate([y_source_1, y_source_2, np.full(X_target.shape[0], -1)])
sample_domain = np.concatenate(
    [
        np.full(X_source_1.shape[0], 1),
        np.full(X_source_2.shape[0], 2),
        np.full(X_target.shape[0], -1),
    ]
)

model = M3SDA(
    hidden_dim=32,
    reg=0.1,
    moment_order=2,
    max_epochs=50,
    random_state=0,
)
model.fit(X, y, sample_domain=sample_domain)

grid_x, grid_y = np.meshgrid(np.linspace(-4, 5, 200), np.linspace(-4, 5, 200))
grid = np.column_stack([grid_x.ravel(), grid_y.ravel()]).astype(np.float32)
grid_prediction = model.predict(grid).reshape(grid_x.shape)

plt.figure(figsize=(7, 6))
plt.contourf(grid_x, grid_y, grid_prediction, alpha=0.2, cmap="coolwarm")
plt.scatter(
    X_target[:, 0],
    X_target[:, 1],
    c=y_target,
    cmap="coolwarm",
    edgecolor="black",
    label="Target samples",
)
plt.title("M3SDA decision boundary on the target domain")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.show()
