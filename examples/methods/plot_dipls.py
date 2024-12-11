"""
DIPLS Regressor example
=======================

This example shows how to use the DIPLS Regressor [36] to learn a regression model
based on low-dimensional feature representations that align source and target domains
in terms of second moment differences while exhibiting a high covariance with
the response variable.

We compare the performance of ordinary Partial Least Squares (PLS) regression
on the source and target domain, the DIPLS and JDOT Regressor on the same task and
illustrate the learned (domain-invariant) features.

.. [36] Nikzad-Langerodi, R., Zellinger, W., Saminger-Platz, S., & Moser, B. A. (2020).
        Domain adaptation for regression under Beer–Lambert’s law.
        Knowledge-Based Systems, 210, 106447.

"""
# Author: Ramin Nikzad-Langerodi
#
# License: BSD 3-Clause
# sphinx_gallery_thumbnail_number = 1

# %% Imports
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_squared_error

from skada import DIPLS, JDOTRegressor
from skada._dipls import genspec

# %%
# Generate spectral dataset with concept drift and plot it
# --------------------------------------------------------
#
# We generate simple spectral-like data corresponding to two domains with a
# dataset shift.

n = 50  # Number of samples
p = 100  # Number of variables

np.random.seed(10)

# Source domain (analyte + 1 interferent)
S1 = genspec(p, 50, 15, 8, 0)  # Analyte signal
S2 = genspec(p, 70, 10, 10, 0)  # Interferent 1 signal
S = np.vstack([S1, S2])

# Analyte and interferent concentrations
Cs = 10 * np.random.rand(n, 2)
ys = Cs[:, 0]

# Spectra
Xs = Cs @ S

# Target domain (Analyte + 2 Interferents)
S1 = genspec(p, 50, 15, 8, 0)  # Analyte signal
S2 = genspec(p, 70, 10, 10, 0)  # Interferent 1 signal
S3 = genspec(p, 30, 10, 10, 0)  # Interferent 2 signal
S = np.vstack([S1, S2, S3])

# Analyte and interferent concentrations
Ct = 10 * np.random.rand(n, 3)
yt = Ct[:, 0]

# Spectra
Xt = Ct @ S

# Plot pure signals
plt.figure()

plt.subplot(211)
plt.plot(S1)
plt.plot(S2)
plt.plot(S3)
plt.legend(["Analyte", "Interferent 1", "Interferent 2"])
plt.title("Pure Signals")
plt.xlabel("X-Variables")
plt.ylabel("Signal")
plt.axvline(x=50, linestyle="-", color="k", alpha=0.5)
plt.axvline(x=70, linestyle=":", color="k", alpha=0.5)
plt.axvline(x=30, linestyle=":", color="k", alpha=0.5)

# Source domain
plt.subplot(223)
plt.plot(Xs.T, "b", alpha=0.2)
plt.title("Source Domain")
plt.xlabel("X-Variables")
plt.ylabel("Signal")
plt.axvline(x=50, linestyle="-", color="k", alpha=0.5)
plt.axvline(x=70, linestyle=":", color="k", alpha=0.5)

# Target domain
plt.subplot(224)
plt.plot(Xt.T, "r", alpha=0.2)
plt.title("Target Domain")
plt.xlabel("X-Variables")
plt.ylabel("Signal")
plt.axvline(x=50, linestyle="-", color="k", alpha=0.5)
plt.axvline(x=70, linestyle=":", color="k", alpha=0.5)
plt.axvline(x=30, linestyle=":", color="k", alpha=0.5)
plt.tight_layout()


# %%
# Train a PLS regressor on source data
# ------------------------------------
#
# We train an ordinary Partial Least Squares (PLS) Regression on the source domain and
# evaluate its performance on the source and target domain. Performance is
# much lower on the target domain due to the shift.

# Prepare data for JDOT
sample_domain_s = np.ones(Xs.shape[0])
sample_domain_t = -np.ones(Xt.shape[0]) * 2
sample_domain = np.hstack((sample_domain_s, sample_domain_t))
X = np.vstack((Xs, Xt))
y = np.vstack((ys, yt))

pls = PLSRegression(n_components=2, scale=False)
pls.fit(Xs, ys)

# Compute accuracy on source and target
ys_pred = pls.predict(Xs)
yt_pred = pls.predict(Xt)

mse_s = mean_squared_error(ys, ys_pred)
mse_t = mean_squared_error(yt, yt_pred)

print(f"MSE on source: {mse_s:.2f}")
print(f"MSE on target: {mse_t:.2f}")

# plot predictions
plt.figure(2, (10, 5))

plt.subplot(1, 2, 1)
plt.scatter(ys, ys_pred, edgecolors="k")
plt.plot([np.min(ys), np.max(ys)], [np.min(ys), np.max(ys)], "k--")
plt.title(f"PLS Prediction on source (MSE={mse_s:.2f})")
plt.xlabel("Measured")
plt.ylabel("Predicted")

plt.subplot(1, 2, 2)
plt.scatter(yt, yt_pred, edgecolors="k")
plt.plot([np.min(yt), np.max(yt)], [np.min(yt), np.max(yt)], "k--")
plt.title(f"PLS Prediction on target (MSE={mse_t:.2f})")
plt.xlabel("Measured")
plt.ylabel("Predicted")

plt.tight_layout()
# %%
# Train with DIPLS regressor
# --------------------------
#
# We now use the DIPLS Regressor to learn a regression model based on low-dimensional
# feature representations that align source and target domains in terms of second moment
# differences.

# di-PLS model with 2 components and heuristic selection of the regularization parameter
dipls = DIPLS(A=2, heuristic=True, rescale="Source")
dipls.fit(X, y.flatten(), sample_domain)
ys_pred = dipls.predict(Xs)
mse_s = mean_squared_error(ys, ys_pred)

dipls = DIPLS(A=2, heuristic=True, rescale="Target")
dipls.fit(X, y.flatten(), sample_domain)
yt_pred = dipls.predict(Xt)
mse_t = mean_squared_error(yt, yt_pred)

# plot predictions
plt.figure(2, (10, 5))

plt.subplot(1, 2, 1)
plt.scatter(ys, ys_pred, edgecolors="k")
plt.plot([np.min(ys), np.max(ys)], [np.min(ys), np.max(ys)], "k--")
plt.title(f"DIPLS Prediction on source (MSE={mse_s:.2f})")
plt.xlabel("Measured")
plt.ylabel("Predicted")

plt.subplot(1, 2, 2)
plt.scatter(yt, yt_pred, edgecolors="k")
plt.plot([np.min(yt), np.max(yt)], [np.min(yt), np.max(yt)], "k--")
plt.title(f"DIPLS Prediction on target (MSE={mse_t:.2f})")
plt.xlabel("Measured")
plt.ylabel("Predicted")

plt.tight_layout()
# %%
# Illustration of invariant-features
# ----------------------------------
#
# We illustrate the (invariant) features and regression coefficients learned by the
# DIPLS Regressor and compare them to the PLS regression coefficients. We can see
# that DIPLS has learned features that account for the additional interferent in the
# target domain, while PLS does not. We can also observe the alignment of the source
# and target domain in the low-dimensional space.

# Coefficients and projections of PLS
coef_PLS = pls.coef_
Ts_PLS = pls.x_scores_
Tt_PLS = pls.transform(Xt)

# Coefficients and projections of DIPLS
coef_DIPLS = dipls.b_
Ts_DIPLS = dipls.Ts_
Tt_DIPLS = dipls.Tt_


# Plot weights and coefficients
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(dipls.W_)
plt.title("DIPLS weights")
plt.xlabel("X-Variables")
plt.ylabel("Weights")
plt.legend(["Component 1", "Component 2"])
plt.subplot(1, 2, 2)
a = plt.plot(coef_PLS.T, "r")
b = plt.plot(coef_DIPLS, "b")
plt.legend([a[0], b[0]], ["PLS", "DIPLS"])
plt.title("Regression coefficients")
plt.xlabel("X-Variables")
plt.ylabel("Coefficients")
plt.tight_layout()

# Plot projections
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.scatter(Ts_PLS[:, 0], Ts_PLS[:, 1], edgecolors="k")
plt.scatter(Tt_PLS[:, 0], Tt_PLS[:, 1], edgecolors="k")
plt.title("PLS Projections")
plt.xlabel("Component 1")
plt.ylabel("Component 2")
plt.legend(["Source", "Target"])
plt.subplot(1, 2, 2)
plt.scatter(Ts_DIPLS[:, 0], Ts_DIPLS[:, 1], edgecolors="k")
plt.scatter(Tt_DIPLS[:, 0], Tt_DIPLS[:, 1], edgecolors="k")
plt.title("DIPLS Projections")
plt.xlabel("Component 1")
plt.ylabel("Component 2")
plt.legend(["Source", "Target"])

plt.tight_layout()

# %%
# Illustration of component-wise regularization
# ---------------------------------------------
#
# Note that with DIPLS we can also set a separate regularization parameter for each
# latent variable. In this example we only add regularization to the second component
# such that the first component captures the analyte and the second component captures
# the interferents.

dipls = DIPLS(A=2, reg_param=(0, 100), rescale="Target")
dipls.fit(X, y.flatten(), sample_domain)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(pls.x_weights_)
plt.title("PLS weights")

plt.subplot(1, 2, 2)
plt.plot(dipls.W_)
plt.title("DIPLS weights")


# %%
# Comparison of JDOT and DIPLS
# ----------------------------
#
# We now train JDOT and DIPLS Regressors using different regularization
# parameters and compare their performance on the target domain. We can see
# that for this dataset with highly correlated features, DIPLS outperforms JDOT.

# JDOT
krr = KernelRidge()
rmse_values = []
alphas = np.linspace(0, 1, 10)
for alpha in alphas:
    jdot = JDOTRegressor(base_estimator=krr, alpha=alpha)
    jdot.fit(X, y.flatten(), sample_domain)
    yt_pred = jdot.predict(Xt)
    rmse_values.append(mean_squared_error(yt_pred, yt))

# DIPLS
rmse_values_dipls = []
l_values = np.linspace(0, 1000, 10)
for reg_param in l_values:
    dipls = DIPLS(A=2, heuristic=False, reg_param=reg_param, rescale="Target")
    dipls.fit(X, y.flatten(), sample_domain)
    yt_pred = dipls.predict(Xt)
    rmse_values_dipls.append(mean_squared_error(yt_pred, yt))


# Plot RMSE values
plt.figure()
a = plt.plot(alphas, rmse_values, "o-", color="r", label="KRR-JDOT", mec="k")
plt.xlabel("Parameter alpha (JDOT)")
plt.twiny()
b = plt.plot(l_values, rmse_values_dipls, "o-", color="b", label="di-PLS", mec="k")
plt.xlabel("Parameter lambda (DIPLS)")
plt.ylabel("RMSE")
plt.legend([a[0], b[0]], ["JDOT", "DIPLS"])
plt.title("Target domain error vs. regularization parameter")
plt.tight_layout()
