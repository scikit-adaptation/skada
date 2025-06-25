# %%
import matplotlib.pyplot as plt

from skada import source_target_split
from skada._gradual_da import GradualTreeEstimator
from skada.datasets import make_shifted_datasets

# %%
"""
Gradual Domain Adaptation Using Optimal Transport
==========================================

This example illustrates the OTDA method from [1] on a simple classification task.
However, the CNN is replaced with Histogram Gradient Boosting Trees.
We add new trees at each interpolated step to "fine-tune" the model.

.. [1] Y. He, H. Wang, B. Li, H. Zhao
        Gradual Domain Adaptation: Theory and Algorithms in
        Journal of Machine Learning Research, 2024.

"""

# Author: Félix Lefebvre and Julie Alberge
#
# License: BSD 3-Clause
# sphinx_gallery_thumbnail_number = 4


# %%
# Generate conditional shift dataset
# ------------------------------
n_samples = 20
X, y, sample_domain = make_shifted_datasets(
    n_samples_source=n_samples,
    n_samples_target=n_samples + 1,
    shift="conditional_shift",
    noise=0.1,
    random_state=42,
)

# %%
X_source, X_target, y_source, y_target = source_target_split(
    X, y, sample_domain=sample_domain
)


n_tot_source = X_source.shape[0]
n_tot_target = X_target.shape[0]

plt.figure(1, figsize=(8, 3.5))
plt.subplot(121)

plt.scatter(X_source[:, 0], X_source[:, 1], c=y_source, vmax=9, cmap="tab10", alpha=0.7)
plt.title("Source domain")
lims = plt.axis()

plt.subplot(122)
plt.scatter(X_target[:, 0], X_target[:, 1], c=y_target, vmax=9, cmap="tab10", alpha=0.7)
plt.title("Target domain")
plt.axis(lims)

# %%
# Gradual Domain Adaptation
# -----------------------------------

gradual_adapter = GradualTreeEstimator(
    T=10,  # number of adaptation steps
)

gradual_adapter.fit(
    X,
    y,
    sample_domain=sample_domain,
)

# %%
# Compute accuracy on source and target
ACC_source = gradual_adapter.score(X_source, y_source)
ACC_target = gradual_adapter.score(X_target, y_target)
