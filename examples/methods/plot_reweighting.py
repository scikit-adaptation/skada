"""
Reweighting method example on covariate shift dataset
====================================================

An example of the reweighting methods on a dataset subject
to covariate shift
"""

# Author:   Ruben Bueno <ruben.bueno@polytechnique.edu>
#           Antoine de Mathelin
#
# License: BSD 3-Clause
# sphinx_gallery_thumbnail_number = 7

# %% Imports
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KernelDensity

from skada import (
    KLIEP,
    KMM,
    DiscriminatorReweightDensity,
    GaussianReweightDensity,
    ReweightDensity,
    source_target_split,
)
from skada._reweight import NearestNeighborReweightDensity
from skada.datasets import make_shifted_datasets
from skada.utils import extract_source_indices

# %%
#
#     The reweighting methods
# ------------------------------------------
# The goal of reweighting methods is to estimate some weights for the
# source dataset in order to then fit a estimator on the source dataset,
# while taking those weights into account, so that the fitted estimator is
# well suited to predicting labels from points drawn from the target distribution.
#
# Reweighting methods implemented and illustrated are the following:
#   * :ref:`Illustration of the Reweight Density method<Reweight Density>`
#   * :ref:`Illustration of the Gaussian reweighting method<Gaussian Reweight Density>`
#   * :ref:`Illustration of the Discr. reweighting method<Discr. Reweight Density>`
#   * :ref:`Illustration of the KLIEP method<KLIEP>`
#   * :ref:`Illustration of the Nearest Neighbor
#     reweighting method<Nearest Neighbor reweighting>`
#   * :ref:`Illustration of the Kernel Mean Matching method<Kernel Mean Matching>`
#
# For more details, look at [0].
#
# .. [0] [Sugiyama et al., 2008] Sugiyama, M., Suzuki, T., Nakajima, S., Kashima, H.,
#        von Bünau, P., and Kawanabe, M. (2008). Direct importance estimation for
#        covariate shift adaptation. Annals of the Institute of Statistical
#        Mathematics, 60(4):699–746.
#        https://www.ism.ac.jp/editsec/aism/pdf/060_4_0699.pdf


base_classifier = LogisticRegression().set_fit_request(sample_weight=True)

print(f"Will be using {base_classifier} as base classifier", end="\n\n")

# %%
# We generate our 2D dataset with 2 classes
# ------------------------------------------
#
# We generate a simple 2D dataset with covariate shift

RANDOM_SEED = 42

X, y, sample_domain = make_shifted_datasets(
    n_samples_source=20, n_samples_target=20, noise=0.1, random_state=RANDOM_SEED
)

Xs, Xt, ys, yt = source_target_split(X, y, sample_domain=sample_domain)

# %%
# Plot of the dataset:
# ------------------------------------------

x_min, x_max = -2.5, 4.5
y_min, y_max = -1.5, 4.5


figsize = (8, 4)
figure, axes = plt.subplots(1, 2, figsize=figsize)

cm = plt.cm.RdBu
colormap = ListedColormap(["#FFA056", "#6C4C7C"])
ax = axes[0]
ax.set_title("Source data")
# Plot the source points:
ax.scatter(Xs[:, 0], Xs[:, 1], c=ys, cmap=colormap, alpha=0.7, s=[25])

ax.set_xticks(()), ax.set_yticks(())
ax.set_xlim(x_min, x_max), ax.set_ylim(y_min, y_max)

ax = axes[1]

ax.set_title("Target data")
# Plot the target points:
ax.scatter(Xt[:, 0], Xt[:, 1], c=ys, cmap=colormap, alpha=0.1, s=[25])
ax.scatter(Xt[:, 0], Xt[:, 1], c=yt, cmap=colormap, alpha=0.7, s=[25])
figure.suptitle("Plot of the dataset", fontsize=16, y=1)
ax.set_xticks(()), ax.set_yticks(())
ax.set_xlim(x_min, x_max), ax.set_ylim(y_min, y_max)

# %%
#     Illustration of the problem with no domain adaptation
# ------------------------------------------
#
# When not using domain adaptation, the classifier won't train on
# data that is distributed as the target sample domain, it will thus
# not be performing optimaly.

# We create a dict to store scores:
scores_dict = {}


def create_plots(
    clf,
    weights,
    name="Without DA",
    suptitle=None,
):
    if suptitle is None:
        suptitle = f"Illustration of the {name} method"
    figure, axes = plt.subplots(1, 2, figsize=figsize)
    ax = axes[1]
    score = clf.score(Xt, yt)
    DecisionBoundaryDisplay.from_estimator(
        clf,
        Xs,
        cmap=ListedColormap(["w", "k"]),
        alpha=1,
        ax=ax,
        eps=0.5,
        response_method="predict",
        plot_method="contour",
    )

    size = 5 + 10 * weights

    # Plot the target points:
    ax.scatter(
        Xt[:, 0],
        Xt[:, 1],
        c=yt,
        cmap=colormap,
        alpha=0.7,
        s=[25],
    )

    ax.set_xticks(()), ax.set_yticks(())
    ax.set_xlim(x_min, x_max), ax.set_ylim(y_min, y_max)
    ax.set_title("Accuracy on target", fontsize=12)
    ax.text(
        x_max - 0.3,
        y_min + 0.3,
        ("%.2f" % score).lstrip("0"),
        size=15,
        horizontalalignment="right",
    )
    scores_dict[name] = score

    ax = axes[0]

    # Plot the source points:
    ax.scatter(Xs[:, 0], Xs[:, 1], c=ys, cmap=colormap, alpha=0.7, s=size)

    DecisionBoundaryDisplay.from_estimator(
        clf,
        Xs,
        cmap=ListedColormap(["w", "k"]),
        alpha=1,
        ax=ax,
        eps=0.5,
        response_method="predict",
        plot_method="contour",
    )

    ax.set_xticks(()), ax.set_yticks(())
    ax.set_xlim(x_min, x_max), ax.set_ylim(y_min, y_max)
    ax.set_title("Training with rewegihted data", fontsize=12)
    figure.suptitle(suptitle, fontsize=16, y=1)


clf = base_classifier
clf.fit(Xs, ys)
create_plots(
    base_classifier,
    name="Without DA",
    weights=np.array([2] * Xs.shape[0]),
    suptitle="Illustration of the classifier with no DA",
)

# %%
#     Illustration of the Reweight Density method
# ------------------------------------------
# .. _Reweight Density
#
# Here the adapter based on re-weighting samples using
# density estimation.

# We define our classifier, `clf` is a da pipeline
clf = ReweightDensity(
    base_estimator=base_classifier,
    weight_estimator=KernelDensity(bandwidth=0.5),
)
clf.fit(X, y, sample_domain=sample_domain)

# We get the weights:

# we first get the adapter which is estimating the weights
weight_estimator = clf[0].base_estimator
weight_estimator.fit(X, sample_domain=sample_domain)
idx = extract_source_indices(sample_domain)
weights = weight_estimator.adapt(X, sample_domain=sample_domain).sample_weight[idx]

create_plots(clf, weights=weights, name="Reweight Density")

# %%
#     Illustration of the Gaussian reweighting method
# ------------------------------------------
# .. _Gaussian Reweight Density
# See [1] for details.
#
# .. [1]  Hidetoshi Shimodaira. Improving predictive inference under
#         covariate shift by weighting the log-likelihood function.
#         In Journal of Statistical Planning and Inference, 2000.

# We define our classifier, `clf` is a da pipeline
clf = GaussianReweightDensity(base_classifier)
clf.fit(X, y, sample_domain=sample_domain)
# We get the weights
weight_estimator = clf[0].base_estimator
weight_estimator.fit(X, sample_domain=sample_domain)
idx = extract_source_indices(sample_domain)
weights = weight_estimator.adapt(X, sample_domain=sample_domain).sample_weight[idx]

create_plots(clf, weights=weights, name="Gaussian Reweight Density")

# %%
#     Illustration of the Discr. reweighting method
# ------------------------------------------
# .. _Discr. Reweight Density
#
# See [2] for details.
#
# .. [2]    Hidetoshi Shimodaira. Improving predictive inference under
#           covariate shift by weighting the log-likelihood function.
#           In Journal of Statistical Planning and Inference, 2000.

# We define our classifier, `clf` is a da pipeline
clf = DiscriminatorReweightDensity(base_classifier)
clf.fit(X, y, sample_domain=sample_domain)

# We get the weights:

# we first get the adapter which is estimating the weights
weight_estimator = clf[0].base_estimator
weight_estimator.fit(X, sample_domain=sample_domain)
idx = extract_source_indices(sample_domain)
weights = weight_estimator.adapt(X, sample_domain=sample_domain).sample_weight[idx]

create_plots(clf, weights=weights, name="Discr. Reweight Density")

# %%
#     Illustration of the KLIEP method
# ------------------------------------------
#
# The idea of KLIEP is to find an importance estimate :math:`w(x)` such that
# the Kullback-Leibler (KL) divergence between the source input density
# :math:`p_{source}(x)` to its estimate :math:`p_{target}(x) = w(x)p_{source}(x)`
# is minimized.
#
# See [3] for details.
#
# .. [3] Masashi Sugiyama et. al. Direct Importance Estimation with Model Selection
#        and Its Application to Covariate Shift Adaptation.
#        In NeurIPS, 2007.

# We define our classifier, `clf` is a da pipeline
clf = KLIEP(
    LogisticRegression().set_fit_request(sample_weight=True), gamma=[1, 0.1, 0.001]
)
clf.fit(X, y, sample_domain=sample_domain)

# We get the weights:

# we first get the adapter which is estimating the weights
weight_estimator = clf[0].base_estimator
weight_estimator.fit(X, sample_domain=sample_domain)
idx = extract_source_indices(sample_domain)
weights = weight_estimator.adapt(X, sample_domain=sample_domain).sample_weight[idx]

create_plots(clf, weights=weights, name="KLIEP")

# %%
#     Illustration of the Nearest Neighbor reweighting method
# ------------------------------------------
# .. _Nearest Neighbor reweighting
#
# This method estimate weight of a point in the source dataset by
# counting the number of points in the target set that are closer to
# it than any other points from the source dataset.
#
# See [4] for details.
#
# .. [4] Loog, M. (2012).
#        Nearest neighbor-based importance weighting.
#        In 2012 IEEE International Workshop on Machine
#        Learning for Signal Processing, pages 1–6. IEEE

# We define our classifier, `clf` is a da pipeline
clf = NearestNeighborReweightDensity(base_classifier, laplace_smoothing=True)
clf.fit(X, y, sample_domain=sample_domain)

# We get the weights:

# we first get the adapter which is estimating the weights
weight_estimator = clf[0].base_estimator
weight_estimator.fit(X, sample_domain=sample_domain)
idx = extract_source_indices(sample_domain)
weights = weight_estimator.adapt(X, sample_domain=sample_domain).sample_weight[idx]

create_plots(clf, weights=weights, name="1NN Reweight Density")

# %%
#     Illustration of the Kernel Mean Matching method
# ------------------------------------------
# .. _Kernel Mean Matching
#
# This example illustrates the use of KMM method [5] to correct covariate-shift.
#
# See [5] for details.
#
# .. [5] J. Huang, A. Gretton, K. Borgwardt, B. Schölkopf and A. J. Smola.
#        Correcting sample selection bias by unlabeled data. In NIPS, 2007.

# We define our classifier, `clf` is a da pipeline
clf = KMM(base_classifier, gamma=10.0, max_iter=1000, smooth_weights=False)
clf.fit(X, y, sample_domain=sample_domain)

# We get the weights:

# we first get the adapter which is estimating the weights
weight_estimator = clf[0].base_estimator
weight_estimator.fit(X, sample_domain=sample_domain)
idx = extract_source_indices(sample_domain)
weights = weight_estimator.adapt(X, sample_domain=sample_domain).sample_weight[idx]

create_plots(
    clf,
    weights=weights,
    name="Kernel Mean Matching",
    suptitle="Illustration of KMM without weights smoothing",
)

# We define our classifier, `clf` is a da pipeline
clf = KMM(base_classifier, gamma=10.0, max_iter=1000, smooth_weights=True)
clf.fit(X, y, sample_domain=sample_domain)

# We get the weights:

# we first get the adapter which is estimating the weights
weight_estimator = clf[0].base_estimator
weight_estimator.fit(X, sample_domain=sample_domain)
idx = extract_source_indices(sample_domain)
weights = weight_estimator.adapt(X, sample_domain=sample_domain).sample_weight[idx]

create_plots(
    clf,
    weights=weights,
    name="Kernel Mean Matching",
    suptitle="Illustration of KMM with weights smoothing",
)

# %%
#     Comparison of score between reweighting methods:
# ------------------------------------------


def print_scores_as_table(scores):
    keys = list(scores.keys())
    lengths = [len(k) for k in keys]
    max_lenght = max(lengths)
    for k in keys:
        print(f"{k}{' '*(max_lenght - len(k))} | ", end="")
        print(f"{scores[k]*100}{' '*(6-len(str(scores[k]*100)))}%")


print_scores_as_table(scores_dict)

plt.show()
