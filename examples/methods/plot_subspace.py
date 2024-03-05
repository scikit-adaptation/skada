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
from matplotlib.colors import ListedColormap
import numpy as np

from sklearn.inspection import DecisionBoundaryDisplay
from skada.datasets import make_shifted_datasets

from sklearn.neighbors import KernelDensity
from sklearn.linear_model import LogisticRegression
from skada._subspace import TJM
from skada import (
    source_target_split,
    ReweightDensity,
    GaussianReweightDensity,
    DiscriminatorReweightDensity,
    KLIEP,
    KMM,
    TransferComponentAnalysis,
    SubspaceAlignment,
)

# %%
#     The reweighting methods
# ------------------------------------------
#
# The goal of reweighting methods is to estimate some weights for the
# source dataset in order to then fit a estimator on the source dataset,
# while taking those weights into account, so that the fitted estimator is
# well suited to predicting labels from points drawn from the target distribution.
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
    n_samples_source=20,
    n_samples_target=20,
    noise=0.1,
    random_state=RANDOM_SEED,
    shift="covariate_shift",
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
colormap = ListedColormap(["#FFA056", "#6C4C7C"])
ax = axes[0]
ax.set_title("Source data")
# Plot the source points:
ax.scatter(
    Xs[:, 0],
    Xs[:, 1],
    c=ys,
    cmap=colormap,
    alpha=0.7,
    s=[15]
)

ax.set_xticks(()), ax.set_yticks(())
ax.set_xlim(x_min, x_max), ax.set_ylim(y_min, y_max)

ax = axes[1]

ax.set_title("Target data")
# Plot the target points:
ax.scatter(
    Xt[:, 0],
    Xt[:, 1],
    c=ys,
    cmap=colormap,
    alpha=0.1,
    s=[15]
)
ax.scatter(
    Xt[:, 0],
    Xt[:, 1],
    c=yt,
    cmap=colormap,
    alpha=0.7,
    s=[15]
)
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
        name="Without DA",
        suptitle=None,
        ):
    size = np.array([16]*Xs.shape[0])

    if suptitle is None:
        suptitle = f"Illustration of the {name} method"
    figure, axes = plt.subplots(1, 3, figsize=figsize)
    ax = axes[1]
    if name == "Without DA":
        clf.fit(Xs, ys)
    else:
        clf.fit(X, y, sample_domain=sample_domain)
    score = clf.score(Xt, yt)
    DecisionBoundaryDisplay.from_estimator(
        clf, Xs, cmap=colormap, alpha=0.1, ax=ax, eps=0.5,
        response_method="predict",
    )

    # Plot the target points:
    ax.scatter(
        Xt[:, 0],
        Xt[:, 1],
        c=yt,
        cmap=colormap,
        alpha=0.7,
        s=[15],
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
    ax.scatter(
        Xs[:, 0],
        Xs[:, 1],
        c=ys,
        cmap=colormap,
        alpha=0.7,
        s=size
    )

    ax.set_xticks(()), ax.set_yticks(())
    ax.set_xlim(x_min, x_max), ax.set_ylim(y_min, y_max)
    ax.set_title("Training with rewegihted data", fontsize=12)

    ax = axes[2]
    ax.scatter(
        clf.adapt(Xs),
        [0] * Xs.shape[0],
        c=ys,
        cmap=colormap,
        alpha=0.7,
        s=size
    )
    ax.set_title("Subspace")

    figure.suptitle(suptitle, fontsize=16, y=1)


create_plots(
    LogisticRegression(), "Without DA",
    suptitle="Illustration of the classifier with no DA")

# %%
#     Illustration of the Reweight Density method
# ------------------------------------------
#
# Here the adapter based on re-weighting samples using
# density estimation.
create_plots(
    TJM(base_classifier, l=0.5, k=1),
    f"tjm")

# %%
#     Illustration of the Reweight Density method
# ------------------------------------------
#
# Here the adapter based on re-weighting samples using
# density estimation.

create_plots(
    TransferComponentAnalysis(base_classifier, n_components=1),
    "tca")


# %%
#     Illustration of the Reweight Density method
# ------------------------------------------
#
# Here the adapter based on re-weighting samples using
# density estimation.

create_plots(
    SubspaceAlignment(base_classifier, n_components=1),
    "SubspaceAlignment")


# %%
#     omparisaon of score between reweighting methods:
# ------------------------------------------


def print_scores_as_table(scores):
    keys = list(scores.keys())
    lenghts = [len(k) for k in keys]
    max_lenght = max(lenghts)
    for k in keys:
        print(f"{k}{' '*(max_lenght - len(k))} | ", end="")
        print(f"{scores[k]*100}{' '*(6-len(str(scores[k]*100)))}%")


print_scores_as_table(scores_dict)

plt.show()
