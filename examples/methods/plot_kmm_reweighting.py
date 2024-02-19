"""
Kernel Mean Matching
====================
This example illustrates the use of KMM method [1] to correct covariate-shift.
.. [1] J. Huang, A. Gretton, K. Borgwardt, B. Schölkopf and A. J. Smola.
       'Correcting sample selection bias by unlabeled data.'
       In NIPS, 2007.
"""

# Author: Antoine de Mathelin
#
# License: BSD 3-Clause
# sphinx_gallery_thumbnail_number = 1

# %%
import matplotlib.pyplot as plt
from skada import KMMAdapter
from skada.datasets import make_shifted_datasets

# %%
# Generate sample bias dataset
# ----------------------------
ds = make_shifted_datasets(
        n_samples_source=12,
        n_samples_target=3,
        shift="covariate_shift",
        label="binary",
        noise=0.4,
        random_state=123,
        return_dataset=True
)
X, y, sample_domain = ds.pack_train(as_sources=['s'], as_targets=['t'])
Xs, ys = ds.get_domain("s")
Xt, yt = ds.get_domain("t")

# %%
# Illustration of Importance Weighting
# ------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

kmm = KMMAdapter(max_iter=1000, random_state=123)
kmm.fit(X, sample_domain=sample_domain)
weights = kmm.adapt(X, sample_domain=sample_domain)["sample_weight"]

src_weights = weights[sample_domain == 1]
src_weights /= src_weights.mean()
src_weights *= 30.

ax1.scatter(Xs[:, 0], Xs[:, 1], label="Source")
ax1.plot(Xt[:, 0], Xt[:, 1], "o", c="r", label="Target")
ax1.set_title("Before Reweighting", fontsize=16)
ax1.set_xlabel("X1", fontsize=14)
ax1.set_ylabel("X2", fontsize=14)
ax1.legend(loc="lower left", fontsize=14)

ax2.scatter(Xs[:, 0], Xs[:, 1], s=src_weights, label="Source")
ax2.plot(Xt[:, 0], Xt[:, 1], "o", c="r", alpha=0.8, label="Target")
ax2.set_title("After Reweighting", fontsize=16)
ax2.set_xlabel("X1", fontsize=14)
ax2.set_ylabel("X2", fontsize=14)
ax2.legend(loc="lower left", fontsize=14)

ax1.tick_params(direction="in", labelleft=False, labelbottom=False)
ax2.tick_params(direction="in", labelleft=False, labelbottom=False)
plt.show()
