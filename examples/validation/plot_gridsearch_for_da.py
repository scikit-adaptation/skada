"""
Using GridSearchCV with skada
=============================

This illustrates the use of DA scorer such as :class:`~skada.metrics.ImportanceWeightedScorer`
with `GridSearchCV <https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html>`_.
"""  # noqa
import numpy as np
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import ShuffleSplit
from sklearn.inspection import DecisionBoundaryDisplay

from skada import EntropicOTMappingAdapter
from skada.datasets import make_shifted_datasets
from skada.metrics import ImportanceWeightedScorer

import warnings
warnings.filterwarnings("ignore")

RANDOM_SEED = 0
X, y, X_target, y_target = make_shifted_datasets(
    n_samples_source=30,
    n_samples_target=20,
    shift="concept_drift",
    label="binary",
    noise=0.4,
    random_state=RANDOM_SEED,
)
estimator = EntropicOTMappingAdapter(base_estimator=SVC())

reg_e_list = [0.01, 0.05, 0.1]
cv = ShuffleSplit(n_splits=5, test_size=0.3, random_state=0)
grid_search = GridSearchCV(
    estimator,
    {"reg_e": reg_e_list},
    cv=cv,
    scoring=ImportanceWeightedScorer(X_target),
)

grid_search.fit(X, y, X_target=X_target)

best_reg_e = grid_search.cv_results_["param_reg_e"][
    np.argmax(grid_search.cv_results_["mean_test_score"])
]
print(f"Best regularization parameter: {best_reg_e}")

# %%
# Plot the results
plt.plot(
    grid_search.cv_results_["param_reg_e"],
    grid_search.cv_results_["mean_test_score"]
)
plt.xlabel("Regulariation parameter")
plt.ylabel("Importance weighted scorer")
plt.show()

# %%
DecisionBoundaryDisplay.from_estimator(
    grid_search.best_estimator_, X, alpha=0.8, eps=0.5,
)

# Plot the target points
plt.scatter(
    X_target[:, 0],
    X_target[:, 1],
    c=y_target,
    alpha=0.5,
)
plt.show()
