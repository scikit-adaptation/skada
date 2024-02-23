"""
JDOT Regressor example
======================

This example shows how to use the JDOTRegressor [10] to learn a regression model
from source to target domain on a simple concept drift 2D exemple. We use a
simple Kernel Ridge Regression (KRR) as base estimator.

We compare the performance of the KRR on the source and target domain, and the
JDOTRegressor on the same task and illustrate the learned decision boundary and
the OT plan between samples estimated by JDOT.

[10] Courty, N., Flamary, R., Habrard, A., & Rakotomamonjy, A. (2017). Joint
 distribution optimal transportation for domain adaptation. Advances in neural
 information processing systems, 30.

"""

# Author: Remi Flamary
#
# License: BSD 3-Clause
# sphinx_gallery_thumbnail_number = 4


# %% Imports
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error
from sklearn.kernel_ridge import KernelRidge

from skada import JDOTRegressor
from skada.datasets import make_shifted_datasets
from skada import source_target_split


# %%
# Generate concept drift dataset and plot it
# ------------------------------------------
#
# We generate a simple 2D concept drift dataset.

X, y, sample_domain = make_shifted_datasets(
        n_samples_source=20,
        n_samples_target=20,
        shift="concept_drift",
        noise=0.3,
        label="regression",
        random_state=42,
    )

y = (y-y.mean())/y.std()

Xs, Xt, ys, yt = source_target_split(X, y, sample_domain=sample_domain)


plt.figure(1, (10, 5))
plt.subplot(1, 2, 1)
plt.scatter(Xs[:, 0], Xs[:, 1], c=ys, label="Source")
plt.title("Source data")
ax = plt.axis()

plt.subplot(1, 2, 2)
plt.scatter(Xt[:, 0], Xt[:, 1], c=yt, label="Target")
plt.title("Target data")
plt.axis(ax)

# %%
# Train a regressor on source data
# --------------------------------
#
# We train a simple Kernel Ridge Regression (KRR) on the source domain and
# evaluate its performance on the source and target domain. Performance is
# much lower on the target domain due to the shift. We also plot the decision
# boundary learned by the KRR.


clf = KernelRidge(kernel='rbf', alpha=0.5)
clf.fit(Xs, ys)

# Compute accuracy on source and target
ys_pred = clf.predict(Xs)
yt_pred = clf.predict(Xt)

mse_s = mean_squared_error(ys, ys_pred)
mse_t = mean_squared_error(yt, yt_pred)

print(f"MSE on source: {mse_s:.2f}")
print(f"MSE on target: {mse_t:.2f}")

XX, YY = np.meshgrid(np.linspace(ax[0], ax[1], 100), np.linspace(ax[2], ax[3], 100))
Z = clf.predict(np.c_[XX.ravel(), YY.ravel()]).reshape(XX.shape)


plt.figure(2, (10, 5))
plt.subplot(1, 2, 1)
plt.scatter(Xs[:, 0], Xs[:, 1], c=ys, label="Prediction")
plt.imshow(Z, extent=(ax[0], ax[1], ax[2], ax[3]), origin='lower', alpha=0.5)
plt.title(f"KRR Prediction on source (MSE={mse_s:.2f})")
plt.axis(ax)

plt.subplot(1, 2, 2)
plt.scatter(Xt[:, 0], Xt[:, 1], c=yt, label="Prediction")
plt.imshow(Z, extent=(ax[0], ax[1], ax[2], ax[3]), origin='lower', alpha=0.5)
plt.title(f"KRR Prediction on target (MSE={mse_t:.2f})")
plt.axis(ax)


# %%
# Train with JDOT regressor
# -------------------------
#
# We now use the JDOTRegressor to learn a regression model from source to
# target domain. We use the same KRR as base estimator. We compare the
# performance of JDOT on the source and target domain, and illustrate the
# learned decision boundary of JDOT. Performance is much better on the target
# domain than with the KRR trained on source.


jdot = JDOTRegressor(base_estimator=KernelRidge(kernel='rbf', alpha=0.5), alpha=0.01)

jdot.fit(X, y, sample_domain=sample_domain)

ys_pred = jdot.predict(Xs)
yt_pred = jdot.predict(Xt)

mse_s = mean_squared_error(ys, ys_pred)
mse_t = mean_squared_error(yt, yt_pred)

Zjdot = jdot.predict(np.c_[XX.ravel(), YY.ravel()]).reshape(XX.shape)

print(f"JDOT MSE on source: {mse_s:.2f}")
print(f"JDOT MSE on target: {mse_t:.2f}")

plt.figure(3, (10, 5))
plt.subplot(1, 2, 1)
plt.scatter(Xs[:, 0], Xs[:, 1], c=ys, label="Prediction")
plt.imshow(Zjdot, extent=(ax[0], ax[1], ax[2], ax[3]), origin='lower', alpha=0.5)
plt.title(f"JDOT Prediction on source (MSE={mse_s:.2f})")
plt.axis(ax)

plt.subplot(1, 2, 2)
plt.scatter(Xt[:, 0], Xt[:, 1], c=yt, label="Prediction")
plt.imshow(Zjdot, extent=(ax[0], ax[1], ax[2], ax[3]), origin='lower', alpha=0.5)
plt.title(f"JDOT Prediction on target (MSE={mse_t:.2f})")
plt.axis(ax)

# %%
# Illustration of the OT plan
# ---------------------------
#
# We illustrate the OT plan between samples estimated by JDOT. We plot the
# OT plan between the source and target samples. We can see that the OT plan
# is able to align the source and target samples while preserving the label.

T = jdot.sol_.plan
T = T/T.max()

plt.figure(4, (5, 5))

plt.scatter(Xs[:, 0], Xs[:, 1], c='C0', label="Source", alpha=0.7)
plt.scatter(Xt[:, 0], Xt[:, 1], c='C1', label="Target", alpha=0.7)

for i in range(Xs.shape[0]):
    for j in range(Xt.shape[0]):
        if T[i, j] > 0.01:
            plt.plot([Xs[i, 0], Xt[j, 0]], [Xs[i, 1], Xt[j, 1]], 'k', alpha=T[i, j]*0.8)
plt.legend()
plt.title("OT plan between source and target")
