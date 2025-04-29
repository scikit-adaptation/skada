"""
Computation of feature-label joint Wasserstein Barycenters
==========================================================

This example illustrates the computation of feature-label joint Wasserstein barycenters

"""

# Author: Eduardo Fernandes Montesuma
#
# License: BSD 3-Clause

# %% Imports
import matplotlib.pyplot as plt
import numpy as np
import ot
from sklearn.datasets import make_moons

from skada._mapping import joint_wasserstein_barycenter

# %%
# Generate labeled data from multiple distributions
# -------------------------------------------------
#
# Here, we use as the base distribution the famous
# moons dataset. We then generate other 2 measure
# by translating the original dataset. This corresponds
# to applying a linear mapping :math:`T_{b}(x) = x + b` on
# each sample in the measures' support, i.e., applying
# :math:`P_{i} = T_{b,\sharp}P_{0}.`
X0, y0 = make_moons(n_samples=100, noise=0.1)
y1 = y0.copy()
y2 = y0.copy()

X1 = X0 + np.array([2, 0])[None, :]
X2 = X0 + np.array([1, np.sqrt(3)])[None, :]

Xs = [X0, X1, X2]

# Converts labels into one-hot encoded labels
Ys = []
for y in [y0, y1, y2]:
    Y = np.zeros((y.size, y.max() + 1))
    Y[np.arange(y.size), y] = 1
    Ys.append(Y)


# %%
# Gaussian Modeling
# -----------------
#
# We start our illlustration of Wasserstein barycenters
# by computing the Bures-Wasserstein barycenter. In this
# case, one assumes each measure is a Gaussian with its
# own mean vector and covariance matrix.
means = np.concatenate(
    [X0.mean(axis=0)[None, :], X1.mean(axis=0)[None, :], X2.mean(axis=0)[None, :]],
    axis=0,
)  # shape: (k, d)

covs = np.concatenate(
    [np.cov(X0.T)[None, ...], np.cov(X1.T)[None, ...], np.cov(X2.T)[None, ...]], axis=0
)  # shape: (k, d, d)


# %%
# Bures-Wasserstein Barycenter
# ----------------------------
#
# Here, we compute the Bures-Wasserstein barycenter using
# the parameters obtained in the previous section. The
# barycenter is calculated using a fixed-point algorithm.
# See, for instance (Álvarez-Esteban et al., 2016)
barycenter_mean, barycenter_cov = ot.gaussian.bures_wasserstein_barycenter(
    m=means, C=covs, eps=1e-8
)

mappings = [
    ot.gaussian.bures_wasserstein_mapping(
        ms=m, Cs=C, mt=barycenter_mean, Ct=barycenter_cov
    )
    for m, C in zip(means, covs)
]

linear_XB, linear_YB = [], []
for _X, _Y, (A, b) in zip(Xs, Ys, mappings):
    linear_XB.append(_X.dot(A) + b)
    linear_YB.append(_Y)
linear_XB = np.concatenate(linear_XB, axis=0)
linear_YB = np.concatenate(linear_YB, axis=0)

# %%
# Plots the results of Gaussian Wasserstein Barycenter
# ------------------------
#
fig, axes = plt.subplots(1, 4, figsize=(16, 4), sharex=True, sharey=True)

names = ["$X_{0}$", "$X_{1}$", "$X_{2}$", "$X_{B}$"]
for ax, _X, _Y, name in zip(
    axes,
    Xs
    + [
        linear_XB,
    ],
    Ys
    + [
        linear_YB,
    ],
    names,
):
    ax.scatter(_X[:, 0], _X[:, 1], c=_Y.argmax(axis=1), cmap=plt.cm.coolwarm)
    ax.set_title
plt.suptitle("Bures-Wasserstein Barycenter")
plt.tight_layout()
plt.show()

# %%
# Compute the Empirical Wasserstein barycenter
# --------------------------------------------
#
# Computes the Barycenter
XB, YB = joint_wasserstein_barycenter(
    Xs,
    Ys,
    mus=None,
    XB=None,
    YB=None,
    muB=None,
    measure_weights=None,
    n_samples=X0.shape[0],
    reg_e=0.0,
    verbose=True,
)

# %%
# Plots the results of Empirical Wasserstein Barycenter
# -----------------------------------------------------
#
fig, axes = plt.subplots(1, 4, figsize=(16, 4), sharex=True, sharey=True)

names = ["$X_{0}$", "$X_{1}$", "$X_{2}$", "$X_{B}$"]
for ax, _X, _Y, name in zip(
    axes,
    Xs
    + [
        XB,
    ],
    Ys
    + [
        YB,
    ],
    names,
):
    ax.scatter(_X[:, 0], _X[:, 1], c=_Y.argmax(axis=1), cmap=plt.cm.coolwarm)
    ax.set_title
plt.suptitle("Empirical Wasserstein Barycenter")
plt.tight_layout()
plt.show()

# %%
# Compares the obtained barycenters
# ---------------------------------
#
#
fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharex=True, sharey=True)

axes[0].scatter(XB[:, 0], XB[:, 1], c=YB.argmax(axis=1), cmap=plt.cm.coolwarm)
axes[0].set_title("Empirical Wasserstein Barycenter")
axes[1].scatter(
    linear_XB[:, 0], linear_XB[:, 1], c=linear_YB.argmax(axis=1), cmap=plt.cm.coolwarm
)
axes[1].set_title("Gaussian Wasserstein Barycenter")

plt.show()


# %%
# When to choose the mapping strategy
# -----------------------------------
#
# In the previous example, you saw that there
# is no much difference between the empirical
# and the Gaussian strategies for computing
# Wasserstein barycenters. This is true because
# the mapping that generates the different measures
# is an affine transformation. More generally, if
# we expect that the mappings between all the measures
# involved is affine (e.g., $T(x) = Ax + b$), then
# we can successfully use Gaussian modeling. We now
# present an example where it fails.
def non_affine_map(points, b):
    """
    Apply the non-affine map T(x, y; b) = [x^2 - y^2 + b, 2xy] to a set of points.

    Parameters
    ----------
        points (np.ndarray): An array of shape (N, 2), where each row is a point (x, y).

    Returns
    -------
        np.ndarray: An array of shape (N, 2), where each row is the transformed point.
    """
    x = points[:, 0]  # Extract x-coordinates
    y = points[:, 1]  # Extract y-coordinates

    # Apply the transformation
    x_transformed = x**2 - y**2 + b
    y_transformed = 2 * x * y

    # Stack the results into a new array of shape (N, 2)
    transformed_points = np.column_stack((x_transformed, y_transformed))
    return transformed_points


# %%
# Transforms the samples
# ----------------------
#
# Here we transform all the samples from the first
# measure
X1 = non_affine_map(X0, b=3)
y1 = y0.copy()
Xs = [X0, X1]

# Converts labels into one-hot encoded labels
Ys = []
for y in [y0, y1]:
    Y = np.zeros((y.size, y.max() + 1))
    Y[np.arange(y.size), y] = 1
    Ys.append(Y)

# %%
# Plot the measures' support
# --------------------------
#
fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharex=True, sharey=True)

axes[0].scatter(X0[:, 0], X0[:, 1], c=y0, cmap=plt.cm.coolwarm)
axes[0].set_title("Measure 0")
axes[1].scatter(X1[:, 0], X1[:, 1], c=y1, cmap=plt.cm.coolwarm)
axes[1].set_title("Measure 1")

plt.show()

# %%
# Compute the Bures-Wasserstein Barycenter
# ----------------------------------------
#
means = np.concatenate(
    [X0.mean(axis=0)[None, :], X1.mean(axis=0)[None, :]], axis=0
)  # shape: (k, d)

covs = np.concatenate(
    [np.cov(X0.T)[None, ...], np.cov(X1.T)[None, ...]], axis=0
)  # shape: (k, d, d)

mappings = [
    ot.gaussian.bures_wasserstein_mapping(
        ms=m, Cs=C, mt=barycenter_mean, Ct=barycenter_cov
    )
    for m, C in zip(means, covs)
]

linear_XB, linear_YB = [], []
for _X, _Y, (A, b) in zip(Xs, Ys, mappings):
    linear_XB.append(_X.dot(A) + b)
    linear_YB.append(_Y)
linear_XB = np.concatenate(linear_XB, axis=0)
linear_YB = np.concatenate(linear_YB, axis=0)

# %%
# Compute the Empirical Wasserstein barycenter
# --------------------------------------------
#
# Computes the Barycenter
XB, YB = joint_wasserstein_barycenter(
    Xs,
    Ys,
    mus=None,
    XB=None,
    YB=None,
    muB=None,
    measure_weights=None,
    n_samples=X0.shape[0],
    reg_e=0.0,
    verbose=True,
)

# %%
# Compares the obtained barycenters
# ---------------------------------
#
# Here, as you can see, the barycenter obtained
# with the Gaussian assumption is actually just
# a translated version of the input measures.
# The empirical barycenter is actually capable
# of capturing the non-linearity of the input
# measures.
fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharex=True, sharey=True)

axes[0].scatter(XB[:, 0], XB[:, 1], c=YB.argmax(axis=1), cmap=plt.cm.coolwarm)
axes[0].set_title("Empirical Barycenter")
axes[1].scatter(
    linear_XB[:, 0], linear_XB[:, 1], c=linear_YB.argmax(axis=1), cmap=plt.cm.coolwarm
)
axes[1].set_title("Bures-Wasserstein Barycenter")

plt.show()

# %%
# References
# ----------
# Álvarez-Esteban, Pedro C., et al. "A fixed-point approach to barycenters in
# Wasserstein space." Journal of Mathematical Analysis and Applications 441.2
# (2016): 744-762.
