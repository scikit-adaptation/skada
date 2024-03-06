"""
Subspace method example on covariate shift dataset
====================================================

An example of the subspace methods on a dataset subject
to covariate shift
"""

# Author:   Ruben Bueno <ruben.bueno@polytechnique.edu>
#
# License: BSD 3-Clause
# sphinx_gallery_thumbnail_number = 7

# %% Imports
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.linear_model import LogisticRegression

from skada import (
    SubspaceAlignment,
    TransferComponentAnalysis,
    TransferJointMatching,
    source_target_split,
)
from skada.datasets import make_shifted_datasets

# %%
#     The subspaces methods
# ------------------------------------------
#

base_classifier = LogisticRegression()

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
    shift="subspace",
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
ax.scatter(Xs[:, 0], Xs[:, 1], c=ys, cmap=colormap, alpha=0.7, s=[15])

ax.set_xticks(()), ax.set_yticks(())
ax.set_xlim(x_min, x_max), ax.set_ylim(y_min, y_max)

ax = axes[1]

ax.set_title("Target data")
# Plot the target points:
ax.scatter(Xt[:, 0], Xt[:, 1], c=ys, cmap=colormap, alpha=0.1, s=[15])
ax.scatter(Xt[:, 0], Xt[:, 1], c=yt, cmap=colormap, alpha=0.7, s=[15])
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
    size = np.array([16] * Xs.shape[0])

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
        clf,
        Xs,
        cmap=colormap,
        alpha=0.1,
        ax=ax,
        eps=0.5,
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
    ax.scatter(Xs[:, 0], Xs[:, 1], c=ys, cmap=colormap, alpha=0.7, s=size)

    ax.set_xticks(()), ax.set_yticks(())
    ax.set_xlim(x_min, x_max), ax.set_ylim(y_min, y_max)
    ax.set_title("Training with rewegihted data", fontsize=12)
    if name != "Without DA":
        ax = axes[2]
        keys = list(clf.named_steps.keys())
        subspace_estimator = clf.named_steps[keys[0]].base_estimator
        subspace_estimator.fit(X, sample_domain=sample_domain)
        ax.scatter(
            subspace_estimator.adapt(Xs),
            [0] * Xs.shape[0],
            c=ys,
            cmap=colormap,
            alpha=0.5,
            s=size,
        )
        ax.set_title("Subspace")

    figure.suptitle(suptitle, fontsize=16, y=1)


create_plots(
    base_classifier, "Without DA", suptitle="Illustration of the classifier with no DA"
)


# Subspace#     Illustration of the subspace Alignment method
# ------------------------------------------
#

create_plots(SubspaceAlignment(base_classifier, n_components=1), "Subspace Alignment")

# %%
#     Illustration of the Transfer Component Analysis method
# ------------------------------------------
#
# The TCA ...

create_plots(TransferComponentAnalysis(base_classifier, n_components=1), "tca")

# %%
#     Illustration of the TransferJointMatching method
# ------------------------------------------
#

create_plots(
    TransferJointMatching(base_classifier, regularizer=2, n_components=1, max_iter=20),
    "TransferJointMatching",
)


# %%
#     Comparisaon of score between reweighting methods:
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
