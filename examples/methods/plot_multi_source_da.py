# %% Imports
import matplotlib.pyplot as plt
import numpy as np
import ot
from sklearn.linear_model import LogisticRegression

from skada import (
    LinearWassersteinBarycenterTransportAdapter,
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
# Fit Linear Wasserstein Barycenter Transport
# -------------------------------------------
#
# Next, we map the source domain data to the target domain
# through a linearized version of the Wasserstein barycenter
# transport. This algorithm assumes that the data is Gaussian,
# and models optimal transport through an affine map between
# domains.
linear_wbt = LinearWassersteinBarycenterTransportAdapter()
linear_wbt.fit(X, Y, sample_domain=sample_domain)
linear_mapped_samples = linear_wbt.transform(X, Y, w=None, sample_domain=sample_domain)

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
# Plots the results of WBT
# ------------------------
#
# Next, we compare the results for both the empirical and Gaussian
# versions of the Wasserstein barycenter transport algorithm
fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)
axes[0].scatter(
    X[sample_domain == -1, 0],
    X[sample_domain == -1, 1],
    c=y[sample_domain == -1],
    cmap=plt.cm.coolwarm,
)
axes[0].set_title("Target domain")
axes[1].scatter(
    mapped_samples[-1][0][:, 0],
    mapped_samples[-1][0][:, 1],
    c=mapped_samples[-1][1],
    cmap=plt.cm.coolwarm,
)
axes[1].set_title("WassersteinBarycenterTransport")
axes[2].scatter(
    linear_mapped_samples[-1][0][:, 0],
    linear_mapped_samples[-1][0][:, 1],
    c=linear_mapped_samples[-1][1].argmax(axis=1),
    cmap=plt.cm.coolwarm,
)
axes[2].set_title("MultiLinearMongeAlignment")
plt.show()

# %%
# Compare the distance in distribution between the measures
# ---------------------------------------------------------
#
# Next, we compare the distance in distribution between the
# obtained measures, and the target domain. As you can see,
# the empirical WBT mapping is able to better approximate
# the target domain.

a = ot.unif(len(X[sample_domain == -1, 0]))
b = ot.unif(len(mapped_samples[-1][0]))
C = ot.dist(X[sample_domain == -1], mapped_samples[-1][0], metric="sqeuclidean")
dist_w2 = ot.emd2(a, b, C)
print(f"Empirical WBT: {dist_w2}")

a = ot.unif(len(X[sample_domain == -1, 0]))
b = ot.unif(len(linear_mapped_samples[-1][0]))
C = ot.dist(X[sample_domain == -1], linear_mapped_samples[-1][0], metric="sqeuclidean")
dist_w2 = ot.emd2(a, b, C)
print(f"Linear WBT: {dist_w2}")

# %%
# Fit a classifier on WBT mapped data
# -----------------------------------
#
# Here we compare the performance of both
# algorithms in domain adaptation.
clf = LogisticRegression()
clf.fit(X=mapped_samples[-1][0], y=mapped_samples[-1][1])
print(
    "[WBT] Accuracy on the target domain:"
    f" {clf.score(X[sample_domain == -1], y[sample_domain==-1])}"
)

# %%
# Fit a classifier on WBT mapped data
#
#
clf = LogisticRegression()
clf.fit(X=linear_mapped_samples[-1][0], y=linear_mapped_samples[-1][1].argmax(axis=1))
print(
    "[LinearWBT] Accuracy on the target domain:"
    f" {clf.score(X[sample_domain == -1], y[sample_domain==-1])}"
)
