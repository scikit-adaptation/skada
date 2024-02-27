"""
Plot comparison of 1NN reweighting
====================================================

A comparison of 1NN reweighting and no da classifications
on a custom dataset having a covariate shift
"""

# Author: Ruben Bueno <ruben.bueno@polytechnique.edu>
#
# License: BSD 3-Clause

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np

from sklearn.inspection import DecisionBoundaryDisplay
from skada.datasets import make_shifted_datasets

from sklearn.linear_model import LogisticRegression
from skada._reweight import NearestNeighborReweightDensity
from skada import source_target_split


# Use same random seed for multiple calls to make_datasets to
# ensure same distributions
RANDOM_SEED = 1

names = [
    "Without da",
    "1NN Reweight Density",
]

classifiers = [
    LogisticRegression(),
    NearestNeighborReweightDensity(
        LogisticRegression().set_fit_request(sample_weight=True),
        laplace_smoothing=True),
]

np.random.seed(RANDOM_SEED)

# We generate a simple 2D covariate shift dataset.

X, y, sample_domain = make_shifted_datasets(
    n_samples_source=50,
    n_samples_target=50,
    noise=0.1,
    random_state=RANDOM_SEED
)

Xs, Xt, ys, yt = source_target_split(
    X, y, sample_domain=sample_domain
)

x_min, x_max = -1.5, 4.5
y_min, y_max = -1.5, 4.5

figure, axes = plt.subplots(len(classifiers) + 1, 2, figsize=(7, 21))

# just plot the dataset first
cm = plt.cm.RdBu
cm_bright = ListedColormap(["#FF0000", "#0000FF"])
ax = axes[0, 1]
ax.set_ylabel("Source data")
# Plot the source points
ax.scatter(
    Xs[:, 0],
    Xs[:, 1],
    c=ys,
    cmap=cm_bright,
    alpha=0.5,
)

ax = axes[0, 0]

ax.set_ylabel("Target data")
# Plot the target points
ax.scatter(
    Xt[:, 0],
    Xt[:, 1],
    c=ys,
    cmap=cm_bright,
    alpha=0.1,
)
ax.scatter(
    Xt[:, 0],
    Xt[:, 1],
    c=yt,
    cmap=cm_bright,
    alpha=0.5,
)
ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)
ax.set_xticks(())
ax.set_yticks(())
i = 1

# iterate over classifiers
for name, clf in zip(names, classifiers):
    print(name, clf)
    ax = axes[i, 0]
    if name == "Without da":
        clf.fit(Xs, ys)
    else:
        clf.fit(X, y, sample_domain=sample_domain)
    score = clf.score(Xt, yt)
    DecisionBoundaryDisplay.from_estimator(
        clf, Xs, cmap=cm, alpha=0.8, ax=ax, eps=0.5, response_method="predict",
    )

    if name == "1NN Reweight Density":
        size = 20*clf.named_steps[
            'nearestneighbordensityadapter'].base_estimator.get_weights(
                Xs, Xt)
    else:
        size = np.array([30]*Xs.shape[0])

    # Plot the target points
    ax.scatter(
        Xt[:, 0],
        Xt[:, 1],
        c=yt,
        cmap=cm_bright,
        alpha=0.5,
    )

    ax.set_xlim(x_min, x_max)

    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_ylabel(name)
    ax.text(
        x_max - 0.3,
        y_min + 0.3,
        ("%.2f" % score).lstrip("0"),
        size=15,
        horizontalalignment="right",
    )

    ax = axes[i, 1]

    # Plot the target points
    ax.scatter(
        Xs[:, 0],
        Xs[:, 1],
        c=ys,
        cmap=cm_bright,
        alpha=0.5,
        s=size
    )

    ax.set_xlim(x_min, x_max)

    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_ylabel("obtained weights")

    i += 1

figure.suptitle("Comparison of the weighting da methods")
plt.tight_layout()
plt.show()
