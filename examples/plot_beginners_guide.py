"""
Beginners Guide
===============

This is an introduction page to SKADA: SciKit Adaptation for beginners.
SKADA is an open-source library focusing on domain adaptation methods that goes
hand in hand with scikit-learn.

In this page, we will present the key concepts and methods for starting to use SKADA
and how to use them.

* :ref:`Shifted dataset creation<Creating a shifted dataset>`:
    * :ref:`Covariate shift<Example of covariate shift>`
    * :ref:`Target shift<Example of target shift>`
    * :ref:`Conditional shift<Example of conditional shift>`
    * :ref:`Subspace shift<Example of subspace shift>`

* :ref:`Methods<Adaptation methods>`
    * :ref:`Reweighting<Source dataset reweighting>`
    * :ref:`Mapping<Source to target mapping>`
    * :ref:`Subspace<Subspace methods>`

* :ref:`Summary<Results summary>`

For better readability, only the use of SKADA is provided and the plotting code
with matplotlib is hidden (but is available in the source file of the example).
"""

# Author: Maxence Barneche
#
# License: BSD 3-Clause
# sphinx_gallery_thumbnail_number = 6

# %%
# Necessary imports

# sphinx_gallery_start_ignore
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression

# sphinx_gallery_end_ignore
from sklearn.neighbors import KernelDensity
from sklearn.svm import SVC

import skada
from skada.datasets import make_shifted_datasets
from skada.utils import extract_source_indices

# %%

# sphinx_gallery_start_ignore
fig_size = (8, 4)


def print_scores_as_table(scores):
    max_len = max(len(k) for k in scores.keys())
    for k, v in scores.items():
        print(f"{k}{' '*(max_len - len(k))} | ", end="")
        print(f"{v*100}{' '*(6-len(str(v*100)))}%")


def decision_borders_plot(X, model):
    # Create a meshgrid
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, num=100), np.linspace(y_min, y_max, num=100)
    )

    # Predict on every point of the meshgrid
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Plot decision borders
    plt.contourf(xx, yy, Z, alpha=0.4, cmap="tab10", vmax=9)


def plot_full_data(X, y, title, prediction=False, model=None, size=25):
    plt.scatter(
        X[:, 0], X[:, 1], s=size, c=y, cmap="tab10", vmax=9, label="Target", alpha=0.8
    )
    plt.xticks([])
    plt.yticks([])
    if prediction:
        decision_borders_plot(X, model)
    if title is not None:
        plt.title(title)


def source_target_comparison(
    X, y, sample_domain, title, prediction=False, model=None, size=25
):
    Xs, Xt, ys, yt = skada.source_target_split(X, y, sample_domain=sample_domain)

    plt.subplot(1, 2, 1)
    plot_full_data(Xs, ys, "Source data", prediction, model, size)

    plt.subplot(1, 2, 2)
    plot_full_data(Xt, yt, "Target data", prediction, model)

    plt.suptitle(title, fontsize=16)


# sphinx_gallery_end_ignore

# %%
# Creating a shifted dataset
# --------------------------
#
# Data shift occurs when there is a change in data distribution. It can come from
# from a change in the input dataset, the target variable or the latent relations
# between the two.
#
# This implies that a model trained on an original dataset, (called source)
# will have a decrease in performance when predicting on the shifted dataset
# (called target).
# We will see the different methods used to deal with data shift
# in the :ref:`methods<Adaptation methods>` section.
#
# DA datasets provided by SKADA are organized as follows:
#
# * :code:`X` is the input data, including the source and the target samples
# * :code:`y` is the output data to be predicted (labels on target samples are not
#   used when fitting the DA estimator)
# * :code:`sample_domain` encodes the domain of each sample (integer >=0 for
#   source and <0 for target)
#
# To create a shifted dataset, use the method :code:`make_shifted_dataset`
# from the `skada.datasets` module.

X, y, sample_domain = make_shifted_datasets(
    n_samples_source=20, n_samples_target=20, random_state=42
)

# %%
# To split the dataset between the source and the target dataset, use
# :code:`source_target_split` from the main module

Xs, Xt, ys, yt = skada.source_target_split(X, y, sample_domain=sample_domain)
# %%
# .. NOTE::
#
#   For reproducibility, we will use the seed 42 throughout the guide.
#   For simplicity, we will use a small (20) number of samples
#   in both source and target. Feel free to change these arguments to your liking once
#   you have a good understanding of the process.
#
# .. NOTE::
#
#   Though SKADA can manage multidomain datasets, we will only generate simple 2D
#   DA datasets with a single source domain and a single target domain in this guide.
#
# There are four main types of shift available with SKADA.
# To specify the type of shift, adjust the :code:`shift` argument in
# :code:`make_shifted_datasets`. By default, this argument is set to
# :code:`covariate_shift`
#
# Example of covariate shift
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Covariate shift is characterised by a change of distribution
# in one or more of the independent variables from the input data.

X, y, sample_domain = make_shifted_datasets(
    n_samples_source=20, n_samples_target=20, shift="covariate_shift", random_state=42
)

# sphinx_gallery_start_ignore
plt.figure(1, fig_size)
source_target_comparison(X, y, sample_domain, "Example covariate shift")
plt.tight_layout()
# sphinx_gallery_end_ignore

# %%
# Example of target shift
# ~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Target shift (or prior probability shift) is characterised by a change in
# target variable distribution while the source data distribution remains the same.

X, y, sample_domain = make_shifted_datasets(
    n_samples_source=20, n_samples_target=20, shift="target_shift", random_state=42
)

# sphinx_gallery_start_ignore
plt.figure(2, fig_size)
source_target_comparison(X, y, sample_domain, "Example target shift")
plt.tight_layout()
# sphinx_gallery_end_ignore

# %%
# Example of conditional shift
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Conditional shift (or concept drift) is characterised by a change in the
# relation between input and output variables.

X, y, sample_domain = make_shifted_datasets(
    n_samples_source=20, n_samples_target=20, shift="conditional_shift", random_state=42
)

# sphinx_gallery_start_ignore
plt.figure(3, fig_size)
source_target_comparison(X, y, sample_domain, "Example conditional shift")
plt.tight_layout()
# sphinx_gallery_end_ignore

# %%
# Example of subspace shift
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Subspace shift is characterised by a change in data distribution where there exists
# a subspace such that the projection of the data on that
# subspace keeps the same distribution

X, y, sample_domain = make_shifted_datasets(
    n_samples_source=20, n_samples_target=20, shift="subspace", random_state=42
)

# sphinx_gallery_start_ignore
plt.figure(4, fig_size)
source_target_comparison(X, y, sample_domain, "Example subspace shift")
plt.tight_layout()
# sphinx_gallery_end_ignore

# %%
# Adaptation methods
# ------------------
#
# As stated before, a model trained on a source dataset will have a decrease
# in performance when evaluated on a shifted target dataset. Thus, there is a need
# to train the model again to account for the shift in data distribution.
#
# In some cases, the source is the only data fully available, as the target may not
# have been classified yet. A common solution is to adapt the source data.
#
# For every shift, there is a method to adapt the source data to train the model on.
#
# Source dataset reweighting
# ~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# A common method for dealing with covariate shift is reweighting the source data.
#
# SKADA handles this with multiple reweighting methods:
#
# * Density Reweighting
# * Gaussian Reweighting
# * Discr. Reweighting
# * KLIEPReweight
# * Nearest Neighbor reweighting
# * Kernel Mean Matching
#
# Each method estimates the weight for the source dataset for an estimator
# to predict labels from the target dataset.
#
# Using a simple :code:`LogisticRegression` classifier,
# we can see that the estimator is not optimal on the target dataset.

X, y, sample_domain = make_shifted_datasets(
    n_samples_source=20, n_samples_target=20, shift="covariate_shift", random_state=42
)
Xs, Xt, ys, yt = skada.source_target_split(X, y, sample_domain=sample_domain)

base_classifier = LogisticRegression().set_fit_request(sample_weight=True)
base_classifier.fit(Xs, ys)

# compute the accuracy of the model
accuracy = base_classifier.score(Xt, yt)

# sphinx_gallery_start_ignore
plt.figure(5, fig_size)
source_target_comparison(
    X,
    y,
    sample_domain,
    "Predictions without reweighting",
    prediction=True,
    model=base_classifier,
)
plt.tight_layout()
print("Accuracy on target:", accuracy)
# sphinx_gallery_end_ignore

# %%
# The reweighting is done by transforming the source dataset using an instance of
# the :code:`Adapter` class provided by SKADA.

# We define the classifier as a da pipeline from the base classifier
clf = skada.DensityReweight(
    base_estimator=base_classifier, weight_estimator=KernelDensity(bandwidth=0.5)
)

clf.fit(X, y, sample_domain=sample_domain)

# To extract the weights, we take the weight estimator from the pipeline
weight_estimator = clf[0].get_estimator()
idx = extract_source_indices(sample_domain)
# then compute the weights corresponding to the source dataset
weights = weight_estimator.compute_weights(X, sample_domain=sample_domain)[idx]

# comput the accuracy of the newly obtained model
accuracy = clf.score(Xt, yt)

# sphinx_gallery_start_ignore
weights = 15 * weights
plt.figure(6, fig_size)
source_target_comparison(
    X,
    y,
    sample_domain,
    "Prediction on reweighted dataset",
    prediction=True,
    model=clf,
    size=weights,
)
plt.tight_layout()
print("Accuracy on target:", accuracy)
# sphinx_gallery_end_ignore


# %%
# Source to target mapping
# ~~~~~~~~~~~~~~~~~~~~~~~~
#
# The traditional way of dealing with conditional shift is via mapping.
#

X, y, sample_domain = make_shifted_datasets(
    n_samples_source=20, n_samples_target=20, shift="conditional_shift", random_state=42
)

Xs, Xt, ys, yt = skada.source_target_split(X, y, sample_domain=sample_domain)

# %%
#
#
#
# %%
# Subspace methods
# ~~~~~~~~~~~~~~~~
#
# The goal of a subspace method is to project data from its original space into a
# lower dimensional subspace. Subspace methods are particularly effective when dealing
# with subspace shift, when the source and target data have the same distribution when
# projected onto a subspace.
#
# There are multiple methods available in SKADA:
#
# * Subspace alignment
# * Transfer Component Analysis
# * Transfer Joint Matching
# * Transfer Subspace Learning
#
# Without domain adaptation, the estimator will have difficulties on the target

X, y, sample_domain = make_shifted_datasets(
    n_samples_source=20, n_samples_target=20, shift="subspace", random_state=42
)

Xs, Xt, ys, yt = skada.source_target_split(X, y, sample_domain=sample_domain)

# setting and fitting the estimator on the source data
base_classifier = SVC()
base_classifier.fit(Xs, ys)

# accuracy on target data
accuracy = base_classifier.score(Xt, yt)

# sphinx_gallery_start_ignore
plt.figure(9, fig_size)
source_target_comparison(
    X, y, sample_domain, "Prediction on dataset", True, base_classifier
)
plt.tight_layout()
print("Accuracy on target:", accuracy)
# sphinx_gallery_end_ignore

# %%
# Now, we will illustrate what happens when using the Transfer Subspace Learning method
clf = skada.TransferSubspaceLearning(base_classifier, n_components=1)
clf.fit(X, y, sample_domain=sample_domain)
accuracy = clf.score(Xt, yt)

# sphinx_gallery_start_ignore
plt.figure(10, fig_size)
source_target_comparison(X, y, sample_domain, "Prediction on dataset", True, clf)
plt.tight_layout()
print("Accuracy on target:", accuracy)
# sphinx_gallery_end_ignore

# %%
# Results summary
# ---------------

# sphinx_gallery_start_ignore
plt.figure(10, (15, 7.5))
shift_list = ["covariate_shift", "target_shift", "conditionl_shift", "subspace"]
clfs = {
    "covariate_shift": LogisticRegression().set_fit_request(sample_weight=True),
    "target_shift": LogisticRegression(),
    "conditionl_shift": SVC(),
    "subspace": SVC(),
}
shift_acc_before = {}
shift_acc_after = {}

for indx, shift in enumerate(shift_list):
    X, y, sd = make_shifted_datasets(20, 20, shift, random_state=42)
    Xs, Xt, ys, yt = skada.source_target_split(X, y, sample_domain=sd)
    shift_name = shift.capitalize().replace("_", " ")

    plt.subplot(3, 5, 2 + indx)
    plot_full_data(Xt, yt, shift_name)

    base_clf = clfs[shift]
    base_clf.fit(Xs, ys)
    shift_acc_before[shift_name] = base_clf.score(Xt, yt)
    plt.subplot(3, 5, 7 + indx)
    plot_full_data(Xt, yt, None, True, base_clf)

    if shift == "covariate_shift":
        clf = skada.DensityReweight(
            base_estimator=base_clf, weight_estimator=KernelDensity(bandwidth=0.5)
        )

    elif shift == "target_shift":
        ...

    elif shift == "conditionl_shift":
        ...

    elif shift == "subspace":
        clf = skada.TransferSubspaceLearning(base_clf, n_components=1)

    clf.fit(X, y, sample_domain=sd)
    shift_acc_after[shift_name] = clf.score(Xt, yt)
    plt.subplot(3, 5, 12 + indx)
    plot_full_data(Xt, yt, None, True, clf)

X, y, sd = make_shifted_datasets(20, 20, random_state=42)
Xs, Xt, ys, yt = skada.source_target_split(X, y, sample_domain=sd)
plt.subplot(3, 5, 1)
plot_full_data(Xs, ys, "Source data")
plt.ylabel("Data", fontsize=15)

clf = SVC()
clf.fit(Xs, ys)
plt.subplot(3, 5, 6)
plot_full_data(Xs, ys, None, True, clf)
plt.ylabel("Before DA", fontsize=15)

plt.subplot(3, 5, 11)
plot_full_data(Xs, ys, None, True, clf)
plt.ylabel("After DA", fontsize=15)

plt.suptitle("Summary plot", fontsize=20)
plt.tight_layout()

print("Accuracy scores before Domain Adaptation:")
print_scores_as_table(shift_acc_before)
print("\nAfter Domain Adaptation:")
print_scores_as_table(shift_acc_after)
# sphinx_gallery_end_ignore
