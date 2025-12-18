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
# .. image:: ./image/image_classif3.pdf
#    :width: 400px
#    :align: center
#
# The model will learn discriminative features to separate the two classes
# like color, shape, texture, etc. Now one can give new example of pear and apple
# and the model will be able to classify them correctly.
#
# .. image:: ./image/image_classif4.pdf
#    :width: 400px
#    :align: center
#
# However, if we now want to classify images of pears and apples, that shifted
# from the initial dataset like quickdraws:
#
# .. image:: ./image/image_classif5.pdf
#    :width: 400px
#    :align: center
#
# or paintings:
# .. image:: ./image/image_classif6.pdf
#    :width: 400px
#    :align: center
#
# The model will likely fail to classify them correctly because the data
# distribution has changed significantly. Here, the features like color, shape,
# or texture that were useful for classification in the original dataset may no
# longer be effective in the new domains.
#
# .. image:: ./image/image_shift.pdf
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

from skada import (
    source_target_split,
)
from skada.datasets import make_shifted_datasets

color_0 = "#6382BC"
color_1 = "#CF5C4A"
color_2 = "#F7CE46"

palette = [color_0, color_1, color_2]
# %% shift

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
# Plot different types of distribution shifts with :func:`make_shifted_datasets`.
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
    X, y, sample_domain = make_shifted_datasets(20, 20, shift=shift, noise=0.1)

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

X, y, sample_domain = make_shifted_datasets(20, 20, shift="subspace", noise=0.1)
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
