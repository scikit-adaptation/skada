"""
Reweighting method example on covariate shift dataset
====================================================

An example of the reweighting methods on a dataset subject
to covariate shift
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

# %%
# We generate our 2D dataset with 2 classes
# ------------------------------------------
#
# We generate a simple 2D dataset with covariate shift

RANDOM_SEED = 42

X, y, sample_domain = make_shifted_datasets(
    n_samples_source=50,
    n_samples_target=50,
    noise=0.1,
    random_state=RANDOM_SEED
)

Xs, Xt, ys, yt = source_target_split(
    X, y, sample_domain=sample_domain
)

# %%
# Plot of the dataset:
# ------------------------------------------

x_min, x_max = -2.5, 4.5
y_min, y_max = -1.5, 4.5


figsize = (8, 4)
figure, axes = plt.subplots(1, 2, figsize=figsize)

cm = plt.cm.RdBu
colormap = ListedColormap(["#FF0000", "#0000FF"])
ax = axes[1]
ax.set_title("Source data")
# Plot the source points:
ax.scatter(
    Xs[:, 0],
    Xs[:, 1],
    c=ys,
    cmap=colormap,
    alpha=0.5,
)

ax.set_xticks(()), ax.set_yticks(())
ax.set_xlim(x_min, x_max), ax.set_ylim(y_min, y_max)

ax = axes[0]

ax.set_title("Target data")
# Plot the target points:
ax.scatter(
    Xt[:, 0],
    Xt[:, 1],
    c=ys,
    cmap=colormap,
    alpha=0.1,
)
ax.scatter(
    Xt[:, 0],
    Xt[:, 1],
    c=yt,
    cmap=colormap,
    alpha=0.5,
)
figure.suptitle("Plot of the dataset")
ax.set_xticks(()), ax.set_yticks(())
ax.set_xlim(x_min, x_max), ax.set_ylim(y_min, y_max)

# We create a dict to store scores:
scores_dict = {}

# %%
#     Illustration of the problem with no domain adaptation
# ------------------------------------------
#
# When not using domain adaptatiion, the classifier won't train on
# data that is distributed as the target sample domain, it will thus
# not be performing optimaly.


def Create_section(
        clf,
        name="Without da",
        suptitle=None,
        ):
    if suptitle is None:
        suptitle = f"Illustration of the {name} method"
    figure, axes = plt.subplots(1, 2, figsize=figsize)
    ax = axes[0]
    if name == "Without da":
        clf.fit(Xs, ys)
    else:
        clf.fit(X, y, sample_domain=sample_domain)
    score = clf.score(Xt, yt)
    DecisionBoundaryDisplay.from_estimator(
        clf, Xs, cmap=ListedColormap(["w", "k"]), alpha=1, ax=ax, eps=0.5,
        response_method="predict", plot_method='contour',
    )

    if name != "Without da":
        # We get the weights
        keys = list(clf.named_steps.keys())
        weight_estimator = clf.named_steps[
            keys[0]].base_estimator
        weight_estimator.fit(X, sample_domain=sample_domain)
        idx = sample_domain > 0
        size = 1 + 8*weight_estimator.adapt(
                X, sample_domain=sample_domain
                ).sample_weight[idx]
    else:
        size = np.array([16]*Xs.shape[0])

    # Plot the target points:
    ax.scatter(
        Xt[:, 0],
        Xt[:, 1],
        c=yt,
        cmap=colormap,
        alpha=0.5,
    )

    ax.set_xticks(()), ax.set_yticks(())
    ax.set_xlim(x_min, x_max), ax.set_ylim(y_min, y_max)
    ax.set_title(f"decision_boundaries for {name}")
    ax.text(
        x_max - 0.3,
        y_min + 0.3,
        ("%.2f" % score).lstrip("0"),
        size=15,
        horizontalalignment="right",
    )
    scores_dict[name] = score

    ax = axes[1]

    # Plot the source points:
    ax.scatter(
        Xs[:, 0],
        Xs[:, 1],
        c=ys,
        cmap=colormap,
        alpha=0.5,
        s=size
    )

    ax.set_xticks(()), ax.set_yticks(())
    ax.set_xlim(x_min, x_max), ax.set_ylim(y_min, y_max)
    ax.set_title("obtained weights")
    figure.suptitle(suptitle)


Create_section(
    LogisticRegression(), "Without da",
    suptitle="Illustration of the classifier with no da")

# %%
#     Illustration of the Reweight Density method
# ------------------------------------------
#
# Here the adapter based on re-weighting samples using
# density estimation.

Create_section(
    ReweightDensity(
        base_estimator=LogisticRegression().set_fit_request(sample_weight=True),
        weight_estimator=KernelDensity(bandwidth=0.5),
    ),
    "Reweight Density")

# %%
#     Illustration of the Gaussian reweighting method
# ------------------------------------------
#
# See [1] for details:
#
# [1]  Hidetoshi Shimodaira. Improving predictive inference under
#           covariate shift by weighting the log-likelihood function.
#           In Journal of Statistical Planning and Inference, 2000.

Create_section(
    GaussianReweightDensity(LogisticRegression().set_fit_request(sample_weight=True)),
    "Gaussian Reweight Density")

# %%
#     Illustration of the Discr. reweighting method
# ------------------------------------------
#
# See [2] for details:
#
# [2] Hidetoshi Shimodaira. Improving predictive inference under
#            covariate shift by weighting the log-likelihood function.
#            In Journal of Statistical Planning and Inference, 2000.

Create_section(
    DiscriminatorReweightDensity(
        LogisticRegression().set_fit_request(sample_weight=True)),
    "Discr. Reweight Density")

# %%
#     Illustration of the KLIEP method
# ------------------------------------------
#
# The idea of KLIEP is to find an importance estimate w(x) such that
# the Kullback-Leibler (KL) divergence between the source input density
# p_source(x) to its estimate p_target(x) = w(x)p_source(x) is minimized.
#
# See [3] for details:
#
# [3] Masashi Sugiyama et. al. Direct Importance Estimation with Model Selection
#           and Its Application to Covariate Shift Adaptation.
#           In NeurIPS, 2007.

Create_section(
    KLIEP(
        LogisticRegression().set_fit_request(
            sample_weight=True), gamma=[1, 0.1, 0.001]),
    "KLIEP")

# %%
#     Illustration of the Nearest Neighbor reweighting method
# ------------------------------------------
#
# See [3] for details:
# [4] Loog, M. (2012).
#           Nearest neighbor-based importance weighting.
#           In 2012 IEEE International Workshop on Machine
#           Learning for Signal Processing, pages 1–6. IEEE

Create_section(
    NearestNeighborReweightDensity(
        LogisticRegression().set_fit_request(sample_weight=True),
        laplace_smoothing=True),
    "1NN Reweight Density")

# %%
#     Finally we can see the resulting scores:
# ------------------------------------------


def print_as_table(scores):
    keys = list(scores.keys())
    lenghts = [len(k) for k in keys]
    max_lenght = max(lenghts)
    for k in keys:
        print(f"{k}{' '*(max_lenght - len(k))} | ", end="")
        print(f"{scores[k]*100}{' '*(6-len(str(scores[k]*100)))}%")


print_as_table(scores_dict)

plt.show()
