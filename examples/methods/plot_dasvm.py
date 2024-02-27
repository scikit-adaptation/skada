"""
Plot for the dasvm estimator
======================

This example illustrates the dsvm method from [21].

"""


# Author: Ruben Bueno
#
# License: BSD 3-Clause


# %% Imports
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

from skada._self_labeling import DASVMEstimator
from skada.datasets import make_dataset_from_moons_distribution
from skada import source_target_split

from sklearn.base import clone

from sklearn.svm import SVC


RANDOM_SEED = 42

# base_estimator can be any classifier equipped with `decision_function` such as:
# SVC(gamma='auto'), LogisticRegression(random_state=0), etc...
base_estimator = SVC()

target_marker = "s"
source_marker = "o"
xlim = (-1.5, 2.4)
ylim = (-1, 1.3)

figure, axis = plt.subplots(1, 2)

# %%
# We generate our dataset
# ------------------------------------------
#
# We generate a simple 2D covariate shift dataset.

X, y, sample_domain = make_dataset_from_moons_distribution(
    pos_source=[0.1, 0.2, 0.3, 0.4],
    pos_target=[0.6, 0.7, 0.8, 0.9],
    n_samples_source=10,
    n_samples_target=10,
    noise=0.1,
    random_state=RANDOM_SEED
)

Xs, Xt, ys, yt = source_target_split(
    X, y, sample_domain=sample_domain
)


# %%
#     Plots of the dataset
# ------------------------------------------
#
# As we can see, the source and target datasets have different
# distributions of the points but have the same labels for
# the same x-values.
# We are then in the case of covariate shift


axis[0].scatter(Xs[:, 0], Xs[:, 1], c=ys, marker=source_marker)
axis[0].set_xlim(xlim)
axis[0].set_ylim(ylim)
axis[0].set_title("source data points")

axis[1].scatter(Xt[:, 0], Xt[:, 1], c=yt, marker=target_marker)
axis[1].set_xlim(xlim)
axis[1].set_ylim(ylim)
axis[1].set_title("target data points")

figure.suptitle("data points", fontsize=20)


# %%
#     Usage of the DASVMEstimator
# ------------------------------------------
#
# Here we create our estimator,
# The algorithm of the dasvm consist in fitting multiple base_estimator (SVC) by:
#     - removing from the training dataset (if possible)
#     `k` points from the source dataset for which the current
#     estimator is doing well
#     - adding to the training dataset (if possible) `k`
#     points from the target dataset for which out current
#     estimator is not so sure about it's prediction (those
#     are target points in the margin band, that are close to
#     the margin)
#     - semi-labeling points that were added to the training set
#     and came from the target dataset
#     - fit a new estimator on this training set
# Here we plot the progression of the SVC classifier when training with the dasvm
# algorithm


estimator = DASVMEstimator(
    base_estimator=clone(base_estimator), k=5,
    save_estimators=True, save_indices=True).fit(
    X, y, sample_domain=sample_domain)

epsilon = 0.02
N = 3
K = len(estimator.estimators)//N
figure, axis = plt.subplots(1, N+1, figsize=(N*5, 3))
for i in list(range(0, N*K, K)) + [-1]:
    j = i//K if i != -1 else -1
    e = estimator.estimators[i]
    x_points = np.linspace(xlim[0], xlim[1], 200)
    y_points = np.linspace(ylim[0], ylim[1], 200)
    X = np.array([[x, y] for x in x_points for y in y_points])

    # plot margins
    if j == -1:
        X_ = X[np.absolute(e.decision_function(X)-1) < epsilon]
        axis[j].scatter(
            X_[:, 0], X_[:, 1], c=[1]*X_.shape[0],
            alpha=1, cmap="gray", s=[0.1]*X_.shape[0], label="margin")
        X_ = X[np.absolute(e.decision_function(X)+1) < epsilon]
        axis[j].scatter(
            X_[:, 0], X_[:, 1], c=[1]*X_.shape[0],
            alpha=1, cmap="gray", s=[0.1]*X_.shape[0])
        X_ = X[np.absolute(e.decision_function(X)) < epsilon]
        axis[j].scatter(
            X_[:, 0], X_[:, 1], c=[1]*X_.shape[0],
            alpha=1, cmap="autumn", s=[0.1]*X_.shape[0], label="decision boundary")
    else:
        X_ = X[np.absolute(e.decision_function(X)-1) < epsilon]
        axis[j].scatter(
            X_[:, 0], X_[:, 1], c=[1]*X_.shape[0],
            alpha=1, cmap="gray", s=[0.1]*X_.shape[0])
        X_ = X[np.absolute(e.decision_function(X)+1) < epsilon]
        axis[j].scatter(
            X_[:, 0], X_[:, 1], c=[1]*X_.shape[0],
            alpha=1, cmap="gray", s=[0.1]*X_.shape[0])
        X_ = X[np.absolute(e.decision_function(X)) < epsilon]
        axis[j].scatter(
            X_[:, 0], X_[:, 1], c=[1]*X_.shape[0],
            alpha=1, cmap="autumn", s=[0.1]*X_.shape[0])

    X_s = Xs[~estimator.indices_source_deleted[i]]
    X_t = Xt[estimator.indices_target_added[i]]
    X = np.concatenate((
        X_s,
        X_t))

    if sum(estimator.indices_target_added[i]) > 0:
        semi_labels = e.predict(Xt[estimator.indices_target_added[i]])
        axis[j].scatter(
            X_s[:, 0], X_s[:, 1], c=ys[~estimator.indices_source_deleted[i]],
            marker=source_marker, alpha=0.7)
        axis[j].scatter(
            X_t[:, 0], X_t[:, 1], c=semi_labels,
            marker=target_marker, alpha=0.7)
    else:
        semi_labels = np.array([])
        axis[j].scatter(
            X[:, 0], X[:, 1], c=ys[~estimator.indices_source_deleted[i]],
            alpha=0.7)
    X = Xt[~estimator.indices_target_added[i]]
    axis[j].scatter(
        X[:, 0], X[:, 1], cmap="gray",
        c=[0.5]*X.shape[0], alpha=0.5, vmax=1, vmin=0,
        marker=target_marker)

    axis[j].set_xlim(xlim)
    axis[j].set_ylim(ylim)

figure.suptitle("evolutions of predictions", fontsize=20)

margin_line = mlines.Line2D(
    [], [], color='black', marker='_', markersize=15, label='margin')
decision_boundary = mlines.Line2D(
    [], [], color='red', marker='_', markersize=15, label='decision boundary')
axis[0].legend(
    handles=[margin_line, decision_boundary], loc='lower left')
axis[-1].legend(
    handles=[margin_line, decision_boundary])

# Show the improvement of the labeling technique
figure, axis = plt.subplots(1, 2, figsize=(10, 6))
semi_labels = (
    base_estimator.fit(Xs, ys).predict(Xt),
    estimator.predict(Xt)
    )
axis[0].scatter(
    Xt[:, 0], Xt[:, 1], c=semi_labels[0],
    alpha=0.7, marker=target_marker)
axis[1].scatter(
    Xt[:, 0], Xt[:, 1], c=semi_labels[1],
    alpha=0.7, marker=target_marker)

scores = np.array([
    sum(semi_labels[0] == yt),
    sum(semi_labels[1] == yt)
    ])*100/semi_labels[0].shape[0]

axis[0].set_title(
    f"Score without method: {scores[0]}%")
axis[1].set_title(
    f"Score with dasvm: {scores[1]}%")
figure.suptitle("predictions")
plt.show()
