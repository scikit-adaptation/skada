"""
Wasserstein Barycenter Transport
================================

This example illustrates the method "Wasserstein Barycenter Transport"

"""

# Author: Eduardo Fernandes Montesuma
#
# License: BSD 3-Clause
# sphinx_gallery_thumbnail_number = 4

# %% Imports
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression

from skada import (
    WassersteinBarycenterTransportAdapter,
)
from skada.datasets import make_multi_source_da_example

np.random.seed(42)

# %%
# Generate covariate shift for multi-source DA
# --------------------------------------------
#
# We generate a simple toy example for MSDA. This
# toy example includes a set of 4 datasets (3 sources,
# 1 target), corresponding to a rotation of a base dataset
# for the angles (0.0, 10.0, 20.0, 30.0)

X, y, sample_domain = make_multi_source_da_example(
    n_datasets=4, n_samples=500, angle_min=0.0, angle_max=30, separation=10
)

# Converts labels into one-hot encoded labels
Y = np.zeros((y.size, y.max() + 1))
Y[np.arange(y.size), y] = 1

# %%
#
# Visualize the datasets
# ----------------------
#
# Here we visualize the datasets

fig, axes = plt.subplots(2, 2, figsize=(10, 10), sharex=True, sharey=True)
for k, ax in enumerate(axes.flatten()[:-1]):
    ax.scatter(
        x=X[sample_domain == k, 0],
        y=X[sample_domain == k, 1],
        c=y[sample_domain == k],
        cmap=plt.cm.coolwarm,
    )
    ax.set_title(f"Source domain {k + 1}")
axes[1, 1].scatter(
    x=X[sample_domain == -1, 0],
    y=X[sample_domain == -1, 1],
    c=y[sample_domain == -1],
    cmap=plt.cm.coolwarm,
)
axes[1, 1].set_title("Target domain")
plt.tight_layout()
plt.show()

# %%
# Fit Logistic Regression
# -----------------------
#
# Here, we fit a classifier to the target domain
clf = LogisticRegression()
clf.fit(X[sample_domain != -1], y[sample_domain != -1])
print(
    "[Source-Only] Accuracy on the target domain:"
    f" {clf.score(X[sample_domain == -1], y[sample_domain==-1])}"
)

# %%
# Fit Wasserstein Barycenter Transport
# ------------------------------------
#
# We fit the Wasserstein Barycenter Transport to the multi-source DA example.
# This algorithms models measures through empirical measures (mixtures of diracs)
# and is capable of handling non-linear shifts between domains.
wbt = WassersteinBarycenterTransportAdapter(n_samples=500, verbose=True)
wbt.fit(X, Y, sample_domain=sample_domain)
mapped_samples = wbt.transform(X, Y, w=None, sample_domain=sample_domain)

# %%
# Plots loss of barycenter algorithm
# ----------------------------------
#
# Here, we plot the loss of the empirical wasserstein barycenter
# algorithm per iteration. Note that convergence happens quite fast,
# with a few iterations.
fig, ax = plt.subplots(1, 1, figsize=(5, 5))
ax.plot(wbt.log["barycenter_computation"]["loss_hist"])
ax.set_xlabel("Iteration")
ax.set_ylabel("Barycenter loss")
plt.show()

# %%
# Fit a classifier on WBT mapped data
# -----------------------------------
#
# Here, we evaluate the performance of WBT in the target domain.
clf = LogisticRegression()
clf.fit(X=mapped_samples[-1][0], y=mapped_samples[-1][1])
print(
    "[WBT] Accuracy on the target domain:"
    f" {clf.score(X[sample_domain == -1], y[sample_domain==-1])}"
)

# %%
#
# Visualize the datasets
# ----------------------
#
# Here we visualize the target and the transported barycenter

XB, yB = mapped_samples[-1]
fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharex=True, sharey=True)
axes[0].scatter(
    x=X[sample_domain == -1, 0],
    y=X[sample_domain == -1, 1],
    c=y[sample_domain == -1],
    cmap=plt.cm.coolwarm,
)
axes[0].set_title("Target")
axes[1].scatter(
    x=XB[:, 0],
    y=XB[:, 1],
    c=yB,
    cmap=plt.cm.coolwarm,
)
axes[1].set_title("Transported Barycenter")
plt.tight_layout()
plt.show()
