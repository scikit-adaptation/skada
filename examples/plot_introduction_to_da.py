"""
Introduction to Domain Adaptation with SKADA
====================================================

This is a basic introduction to domain adaptation (DA) using the
:mod:`skada` library. We will introduce the main concepts of DA and show how to
use SKADA to perform DA on simple datasets.
"""

# Author: Theo Gnassounou
#
# License: BSD 3-Clause
# sphinx_gallery_thumbnail_number = 1

# %% intro
# Domain Adaptation (DA)
# --------------------------
#
# Domain Adaptation (DA) is a subfield of machine learning that focuses on
# adapting models trained on a source domain to perform well on a different but
# related target domain. This is particularly useful when there is a shift in
# data distribution between the source and target domains, which can lead to
# poor performance of models trained solely on the source domain.
#
# Let's illustrate the concept of DA with a simple example. Imagine we have
# an image classification task between pears and apples. You can train a model
# to distinguish between these two classes.
#
# .. image:: ./images/image_classif3.pdf
#    :width: 400px
#    :align: center
#
# The model will learn discriminative features to separate the two classes
# like color, shape, texture, etc. Now one can give new example of pear and apple
# and the model will be able to classify them correctly.
#
# .. image:: ./images/image_classif4.pdf
#    :width: 400px
#    :align: center
#
# However, if we now want to classify images of pears and apples, that shifted
# from the initial dataset like quickdraws:
#
# .. image:: ./images/image_classif5.pdf
#    :width: 400px
#    :align: center
#
# or paintings:
#
# .. image:: ./images/image_classif6.pdf
#    :width: 400px
#    :align: center
#
# The model will likely fail to classify them correctly because the data
# distribution has changed significantly. Here, the features like color, shape,
# or texture that were useful for classification in the original dataset may no
# longer be effective in the new domains.
#
# .. image:: ./images/image_shift.pdf
#    :width: 400px
#    :align: center
#
# In domain adaptation, we suppose that we have access to different domains.
# In each domain we have the same task to solve (here classify pears and apples)
# but the data distribution is different: we have a distribution shift.
# In practice, you will suppose that you have access to **source domains** where
# you have **labeled data** and a **target domain** where you have **unlabeled data**.
# The goal of DA is to leverage the labeled data from the source domains to
# learn a model that performs well on the target domain, despite the distribution
# shift.
#
# Let's dive into SKADA to illustrate the impact of distribution shift and how DA
# can help mitigate it.

# %% imports
import matplotlib.pyplot as plt
import numpy as np
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.svm import SVC

from skada import (
    DensityReweight,
    source_target_split,
)
from skada.datasets import make_shifted_datasets
from skada.utils import extract_source_indices

color_0 = "#6382BC"
color_1 = "#CF5C4A"
color_2 = "#F7CE46"

cmap = plt.cm.colors.ListedColormap([color_0, color_1])

# %% shifts
# Type of distribution shifts
# In practice, different types of distribution shifts can occur between
# source and target domains. The most common types are:
# - **Covariate shift**: The distribution of the input features changes
#   between the source and target domains, but the conditional distribution of
#   the labels given the features remains the same.
# - **Label shift**: The distribution of the labels changes between the source
#   and target domains, but the conditional distribution of the features given
#   the labels remains the same.
# - **Conditional shift**: The conditional distribution of the labels given the
#   features or the features given the labels changes between the source and
#   target domains.
# - **Subspace assumptions**: The source and target domains share a common subspace
#   where the data distributions are similar, even if the overall distributions
#   differ.
#
# Let's illustrate these different types of shifts with simple 2D datasets using skada.

# %% plot shifts
# First let's plot different types of distribution shifts with
# :func:`make_shifted_datasets`.
# This function generates simple 2D datasets with different types of shifts.
# The data format of skada comprises:
# - :code:`X`: input data (features)
# - :code:`y`: output data (labels)
# - :code:`sample_domain`: domain of each sample
# (positive for source and negative for target)

shifts = ["covariate_shift", "target_shift", "conditional_shift", "subspace"]
shift_names = ["Covariate shift", "Target shift", "Conditional shift", "Subspace Ass."]

fig, axes = plt.subplots(1, 4, figsize=(8, 2), sharey=True, sharex=True)
axes[0].set_ylabel("Source data", fontsize=15)
axes[1].set_ylabel("Target data", fontsize=15)

for i, shift in enumerate(shifts[:-1]):
    X, y, sample_domain = make_shifted_datasets(
        20, 20, shift=shift, random_state=42, noise=0.3
    )

    X_source, X_target, y_source, y_target = source_target_split(
        X, y, sample_domain=sample_domain
    )

    if i == 0:
        xlim = (X[:, 0].min() - 0.5, X[:, 0].max() + 0.5)
        ylim = (X[:, 1].min() - 0.5, X[:, 1].max() + 0.5)
        axes[0].set_xlim(xlim)
        axes[0].set_ylim(ylim)
        c_source = np.where(y_source == 0, color_0, color_1)
        axes[0].scatter(X_source[:, 0], X_source[:, 1], c=c_source, alpha=0.5)
        axes[0].set_xticks([])
        axes[0].set_yticks([])

    c_target = np.where(y_target == 0, color_0, color_1)
    axes[i + 1].scatter(X_target[:, 0], X_target[:, 1], c=c_target, alpha=0.5)
    axes[i + 1].set_title(shift_names[i], fontsize=15)
    axes[i + 1].set_xticks([])
    axes[i + 1].set_yticks([])

plt.tight_layout()

# %% Subspace shift
# The last type of shift is the subspace assumption. Here, we suppose that one can see a
# common subspace on the diagonal.

fig, axes = plt.subplots(1, 2, figsize=(4, 2), sharey=True, sharex=True)
axes[0].set_ylabel("Source data", fontsize=15)
axes[1].set_ylabel("Target data", fontsize=15)

X, y, sample_domain = make_shifted_datasets(
    20, 20, shift="subspace", random_state=42, noise=0.3
)
X_source, X_target, y_source, y_target = source_target_split(
    X, y, sample_domain=sample_domain
)

xlim = (X[:, 0].min() - 0.5, X[:, 0].max() + 0.5)
ylim = (X[:, 1].min() - 0.5, X[:, 1].max() + 0.5)
axes[0].set_xlim(xlim)
axes[0].set_ylim(ylim)
c_source = np.where(y_source == 0, color_0, color_1)
axes[0].scatter(X_source[:, 0], X_source[:, 1], c=c_source, alpha=0.5)
axes[0].set_xticks([])
axes[0].set_yticks([])

c_target = np.where(y_target == 0, color_0, color_1)
axes[1].scatter(X_target[:, 0], X_target[:, 1], c=c_target, alpha=0.5)
axes[1].set_title(shift_names[-1], fontsize=15)
axes[1].set_xticks([])
axes[1].set_yticks([])
plt.tight_layout()

# %%
# Drop of accuracy due to distribution shift
# ------------------------------------------
# In domain adaptation settings, the target domains is **UNLABELED**. Therefore,
# we cannot directly train a model on the target domain. To illustrate the
# impact of distribution shift on model performance, we will train a simple
# classifier on the source domain and evaluate its performance on both the source
# and target domains.

fig, axes = plt.subplots(1, 4, figsize=(8, 2.3), sharey=True, sharex=True)
axes[0].set_ylabel("Source data", fontsize=15)
axes[1].set_ylabel("Target data", fontsize=15)

for i, shift in enumerate(shifts[:-1]):
    X, y, sample_domain = make_shifted_datasets(
        20, 20, shift=shift, random_state=42, noise=0.3
    )

    X_source, X_target, y_source, y_target = source_target_split(
        X, y, sample_domain=sample_domain
    )

    # Train SVC model on source data
    estimator = SVC(kernel="rbf")
    estimator.fit(X_source, y_source)

    if i == 0:
        xlim = (X[:, 0].min() - 0.5, X[:, 0].max() + 0.5)
        ylim = (X[:, 1].min() - 0.5, X[:, 1].max() + 0.5)
        axes[0].set_xlim(xlim)
        axes[0].set_ylim(ylim)
        c_source = np.where(y_source == 0, color_0, color_1)
        axes[0].scatter(X_source[:, 0], X_source[:, 1], c=c_source, alpha=0.5)
        axes[0].set_xticks([])
        axes[0].set_yticks([])
        X_for_boundary_grid = np.array(
            [
                [xlim[0], ylim[0]],
                [xlim[0], ylim[1]],
                [xlim[1], ylim[0]],
                [xlim[1], ylim[1]],
            ]
        )
        DecisionBoundaryDisplay.from_estimator(
            estimator,
            X_for_boundary_grid,
            alpha=0.3,
            eps=0,
            response_method="predict",
            cmap=cmap,
            ax=axes[0],
        )
        axes[0].set_xlabel(
            f"Acc: {estimator.score(X_source, y_source):.2f}", fontsize=13
        )

    c_target = np.where(y_target == 0, color_0, color_1)
    axes[i + 1].scatter(X_target[:, 0], X_target[:, 1], c=c_target, alpha=0.5)
    axes[i + 1].set_title(shift_names[i], fontsize=15)
    axes[i + 1].set_xticks([])
    axes[i + 1].set_yticks([])
    DecisionBoundaryDisplay.from_estimator(
        estimator,
        X_for_boundary_grid,
        alpha=0.3,
        eps=0,
        response_method="predict",
        cmap=cmap,
        ax=axes[i + 1],
    )
    axes[i + 1].set_xlabel(
        f"Acc: {estimator.score(X_target, y_target):.2f}", fontsize=13
    )

plt.tight_layout()

# %%
# As we can see the accuracy on the target domain drops compared to the
# source domain due to the distribution shift. These examples stay very simple
# that explain why some drops are not very significant.
# Same results can be observed on subspace shift:

fig, axes = plt.subplots(1, 2, figsize=(4, 2.3), sharey=True, sharex=True)
axes[0].set_ylabel("Source data", fontsize=15)
axes[1].set_ylabel("Target data", fontsize=15)
X, y, sample_domain = make_shifted_datasets(
    20, 20, shift="subspace", random_state=42, noise=0.3
)
X_source, X_target, y_source, y_target = source_target_split(
    X, y, sample_domain=sample_domain
)

estimator = SVC(kernel="rbf")
estimator.fit(X_source, y_source)

xlim = (X[:, 0].min() - 0.5, X[:, 0].max() + 0.5)
ylim = (X[:, 1].min() - 0.5, X[:, 1].max() + 0.5)
axes[0].set_xlim(xlim)
axes[0].set_ylim(ylim)
c_source = np.where(y_source == 0, color_0, color_1)
axes[0].scatter(X_source[:, 0], X_source[:, 1], c=c_source, alpha=0.5)
axes[0].set_xticks([])
axes[0].set_yticks([])
X_for_boundary_grid = np.array(
    [
        [xlim[0], ylim[0]],
        [xlim[0], ylim[1]],
        [xlim[1], ylim[0]],
        [xlim[1], ylim[1]],
    ]
)
DecisionBoundaryDisplay.from_estimator(
    estimator,
    X_for_boundary_grid,  # Use this to define the mesh for boundary plotting
    alpha=0.3,
    eps=0,  # Set eps=0 because X_for_boundary_grid already defines the full extent
    response_method="predict",
    cmap=cmap,
    ax=axes[0],
)
axes[0].set_xlabel(f"Acc: {estimator.score(X_source, y_source):.2f}", fontsize=13)
c_target = np.where(y_target == 0, color_0, color_1)
axes[1].scatter(X_target[:, 0], X_target[:, 1], c=c_target, alpha=0.5)
axes[1].set_title(shift_names[-1], fontsize=15)
axes[1].set_xticks([])
axes[1].set_yticks([])
DecisionBoundaryDisplay.from_estimator(
    estimator,
    X_for_boundary_grid,
    alpha=0.3,
    eps=0,
    response_method="predict",
    cmap=cmap,
    ax=axes[1],
)
axes[1].set_xlabel(f"Acc: {estimator.score(X_target, y_target):.2f}", fontsize=13)
plt.tight_layout()
# %%
# Reweigthing for Covariate Shift and Target Shift
# ------------------------------------------
# In the case of covariate shift and target shift, reweighting methods can be
# used to correct the distribution shift. The idea is to assign weights to the
# source samples during training such that the weighted source distribution
# matches the target distribution.
# Let's illustrate this with a simple reweighting method using SKADA.
# Let consider the method :class:`skada.DensityReweight` that uses density
# ratio estimation to compute the weights.
# When you want to train any estimator using skada the DomainAwareDataset
# that will help you to
# create the proper data format for skada.
# During training you need to mask the target labels since
# they are not available in DA settings.
# this can be done using the :meth:`skada.DomainAwareDataset.pack` method with
# the :code:`mask_target_labels` parameter set to :code:`True`.
# During evaluation on the target domain, you can unmask the target labels
# using the same method with :code:`mask_target_labels` set to :code:`False`.
dataset = make_shifted_datasets(
    20, 20, shift="covariate_shift", random_state=42, noise=0.30, return_dataset=True
)

estimator = DensityReweight(
    base_estimator=SVC(kernel="rbf").set_fit_request(sample_weight=True)
)

X, y, sample_domain = dataset.pack(
    as_sources=["s"], as_targets=["t"], mask_target_labels=True
)
estimator.fit(X, y, sample_domain=sample_domain)

X, y, sample_domain = dataset.pack(
    as_sources=[], as_targets=["t"], mask_target_labels=False
)
score_target = estimator.score(X, y)

# %% plot steps
# The DA estimator proceed in different steps that we can illustrate.
# First, the reweighting estimator computes weights for the source samples
# based on the distribution shift (here covariate shift).
# These weights are then used to train the classifier on the source data.
# Finally, the trained classifier is used to predict on the target data.
X, y, sample_domain = dataset.pack(
    as_sources=["s"], as_targets=["t"], mask_target_labels=True
)

weight_estimator = estimator[0].get_estimator()
idx = extract_source_indices(sample_domain)
weights = weight_estimator.compute_weights(X, sample_domain=sample_domain)[idx]

# step 1: Init
fig, axes = plt.subplots(1, 4, figsize=(8, 2.4), sharex=True, sharey=True)
# create cmap from color_0 and color_1
y_source_c = np.where(y[idx] == 0, color_0, color_1)
axes[0].scatter(
    X[idx, 0],
    X[idx, 1],
    c=y_source_c,
    alpha=0.5,
)
axes[0].set_xticks([])
axes[0].set_yticks([])
axes[0].set_title("Source data", fontsize=12)
xlim = axes[0].get_xlim()
ylim = axes[0].get_ylim()
# print decision boundary

axes[1].scatter(X[idx, 0], X[idx, 1], c=y_source_c, alpha=0.5, s=weights * 30)
axes[1].set_xticks([])
axes[1].set_yticks([])
axes[1].set_title("Data reweighted", fontsize=12)

axes[2].scatter(X[idx, 0], X[idx, 1], c=y_source_c, alpha=0.5, s=weights * 20)
axes[2].set_xticks([])
axes[2].set_yticks([])
axes[2].set_title("Classifier training", fontsize=12)
axes[2].set_xlim(xlim)
axes[2].set_ylim(ylim)

# print decision boundary
DecisionBoundaryDisplay.from_estimator(
    estimator,
    X,
    alpha=0.3,
    eps=0.5,
    response_method="predict",
    cmap=cmap,
    ax=axes[2],
)

X, y, sample_domain = dataset.pack(
    as_sources=["s"], as_targets=["t"], mask_target_labels=False
)
y_target_c = np.where(y[~idx] == 0, color_0, color_1)

axes[3].scatter(X[~idx, 0], X[~idx, 1], c=y_target_c, alpha=0.5)
axes[3].set_xticks([])
axes[3].set_yticks([])
axes[3].set_title("Prediction on Target", fontsize=12)
axes[3].set_xlim(xlim)
axes[3].set_ylim(ylim)

DecisionBoundaryDisplay.from_estimator(
    estimator,
    X[idx],
    alpha=0.3,
    eps=0.5,
    response_method="predict",
    cmap=cmap,
    ax=axes[3],
)

plt.tight_layout()
# %%
