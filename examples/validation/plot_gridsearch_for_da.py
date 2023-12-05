"""
Using GridSearchCV with skada
=============================

This illustrates the use of DA scorer such as :class:`~skada.metrics.ImportanceWeightedScorer`
with `GridSearchCV <https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html>`_.
"""  # noqa
# %%
# Prepare dataset and the estimator

import numpy as np
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, ShuffleSplit
from sklearn.inspection import DecisionBoundaryDisplay

from skada import EntropicOTMapping
from skada.datasets import make_shifted_datasets
from skada.metrics import PredictionEntropyScorer

import warnings
warnings.filterwarnings("ignore")


dataset = make_shifted_datasets(
    n_samples_source=30,
    n_samples_target=20,
    shift="concept_drift",
    label="binary",
    noise=0.4,
    random_state=42,
    return_dataset=True
)
X, y, sample_domain = dataset.pack_for_train(as_sources=['s'], as_targets=['t'])
X_target, y_target, _ = dataset.pack_for_test(as_targets=['t'])

# %%
# Run grid search

reg_e = [0.01, 0.05, 0.1]
estimator = EntropicOTMapping(base_estimator=SVC(probability=True))
cv = ShuffleSplit(n_splits=5, test_size=0.3, random_state=0)
grid_search = GridSearchCV(
    estimator,
    {"entropicotmappingadapter__base_estimator__reg_e": reg_e},
    cv=cv,
    scoring=PredictionEntropyScorer(),
)

grid_search.fit(X, y, sample_domain=sample_domain)

best_reg_e = grid_search.best_params_['entropicotmappingadapter__base_estimator__reg_e']
print(f"Best regularization parameter: {best_reg_e}")

# %%
# Plot the results

plt.plot(
    grid_search.cv_results_["param_entropicotmappingadapter__base_estimator__reg_e"],
    grid_search.cv_results_["mean_test_score"]
)
plt.xlabel("Regulariation parameter")
plt.ylabel("Importance weighted scorer")
plt.show()

# %% Visualize the results

DecisionBoundaryDisplay.from_estimator(
    grid_search.best_estimator_,
    X_target,
    alpha=0.8,
    eps=0.5,
    response_method='predict',
)

# Plot the target points
plt.scatter(
    X_target[:, 0],
    X_target[:, 1],
    c=y_target,
    alpha=0.5,
)
plt.show()
