"""
Plot JDOT Regressor
===================


"""
# %% Imports
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error
from sklearn.kernel_ridge import KernelRidge

from skada import JDOTRegressor
from skada.datasets import make_shifted_datasets
from skada import source_target_split


# %%
# Generate concept drift dataset
# ------------------------------

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


# %%
# Plot data
# ---------

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
# Train on source data
# --------------------


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
