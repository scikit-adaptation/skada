"""
Plot comparison of reweighting methods
====================================================

A comparison of some reweighting methods and classifications
with no da on a dataset having a covariate shift
"""

# Author: Ruben Bueno <ruben.bueno@polytechnique.edu>
#
# License: BSD 3-Clause

# %% Imports
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np

from sklearn.inspection import DecisionBoundaryDisplay
from skada.datasets import make_shifted_datasets

from sklearn.neighbors import KernelDensity
from sklearn.linear_model import LogisticRegression
from skada._reweight import NearestNeighborReweightDensity
from skada import (
    source_target_split,
    ReweightDensity,
    GaussianReweightDensity,
    DiscriminatorReweightDensity,
    KLIEP
)

# %%
#     The reweighting methods
# ------------------------------------------
#
# The goal of reweighting methods is to estimate some weights for the
# source dataset in order to then fit a estimator on the source dataset,
# while taking those weights into account, so that the fitted estimator is
# well suitted to predicting labels from points drawn from the target distribution.

RANDOM_SEED = 42

names = [
    "Without da",
    "Reweight Density",
    "Gaussian Reweight Density",
    "Discr. Reweight Density",
    "KLIEP",
    "1NN Reweight Density",
]

classifiers = [
    LogisticRegression(),
    ReweightDensity(
        base_estimator=LogisticRegression().set_fit_request(sample_weight=True),
        weight_estimator=KernelDensity(bandwidth=0.5),
    ),
    GaussianReweightDensity(LogisticRegression().set_fit_request(sample_weight=True)),
    DiscriminatorReweightDensity(
        LogisticRegression().set_fit_request(sample_weight=True)),
    KLIEP(
        LogisticRegression().set_fit_request(
            sample_weight=True), gamma=[1, 0.1, 0.001]),
    NearestNeighborReweightDensity(
        LogisticRegression().set_fit_request(sample_weight=True),
        laplace_smoothing=True),
]

# %%
# We generate our 2D dataset with 2 classes
# ------------------------------------------
#
# We generate a simple 2D dataset with covariate shift

X, y, sample_domain = make_shifted_datasets(
    n_samples_source=50,
    n_samples_target=50,
    noise=0.1,
    random_state=RANDOM_SEED
)

Xs, Xt, ys, yt = source_target_split(
    X, y, sample_domain=sample_domain
)

x_min, x_max = -2.5, 4.5
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

ax.set_xticks(())
ax.set_yticks(())
ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)

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

# iterate over classifiers
i = 1
for name, clf in zip(names, classifiers):
    print(name, clf)
    ax = axes[i, 0]
    if name == "Without da":
        clf.fit(Xs, ys)
    else:
        clf.fit(X, y, sample_domain=sample_domain)
    score = clf.score(Xt, yt)
    DecisionBoundaryDisplay.from_estimator(
        clf, Xs, cmap=cm, alpha=0.4, ax=ax, eps=0.5, response_method="predict",
    )

    if name != "Without da":
        keys = list(clf.named_steps.keys())
        weight_estimator = clf.named_steps[
            keys[0]].base_estimator
        weight_estimator.fit(X, sample_domain=sample_domain)
        idx = sample_domain > 0
        size = 1 + 10*weight_estimator.adapt(
                X, sample_domain=sample_domain
                ).sample_weight[idx]
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
    ax.set_ylim(y_min, y_max)
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
    ax.set_ylim(y_min, y_max)
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_ylabel("obtained weights")

    i += 1

figure.suptitle("Comparison of the weighting da methods")
plt.tight_layout()
plt.show()

# %%
#     Plots of the methods
# ------------------------------------------
#
# First we plotted the dataset, and then each reweighting methods,
# On the left part we can see the prediction made by the dapipeline (mostly
# composed of a logistic regression classifier and the reweighting adapter).
# And on the right plots, we have plotted the the source dataset with the weights 
# that have been obtained by the reweighting adapter
