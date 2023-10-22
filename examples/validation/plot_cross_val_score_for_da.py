"""
Using cross_val_score with skada
================================

This illustrates the use of DA scorer such :class:`~skada.metrics.TargetAccuracyScorer`
with `cross_val_score <https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_score.html#sklearn.model_selection.cross_val_score>`_.
"""  # noqa
import numpy as np
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit

from skada import EntropicOTMappingAdapter
from skada.datasets import make_shifted_datasets
from skada.metrics import SupervisedScorer


RANDOM_SEED = 0
X, y, X_target, y_target = make_shifted_datasets(
    n_samples_source=30,
    n_samples_target=20,
    shift="concept_drift",
    label="binary",
    noise=0.4,
    random_state=RANDOM_SEED,
)

base_estimator = SVC()
estimator = EntropicOTMappingAdapter(base_estimator=base_estimator, reg_e=0.05, tol=1e-3)

cv = ShuffleSplit(n_splits=5, test_size=0.3, random_state=0)
scores = cross_val_score(
    estimator,
    X,
    y,
    cv=cv,
    fit_params={"X_target": X_target},
    scoring=SupervisedScorer(X_target, y_target),
)

scores_no_da = cross_val_score(
    base_estimator, X, y, cv=cv, scoring=SupervisedScorer(X_target, y_target)
)

# %%
# Compare scores with the dummy estimator that does not use DA

print(
    f"Cross-validation score with DA: {np.mean(scores):.2f} (+/- {np.std(scores):.2f})"
)
print(
    "Cross-validation score without DA: "
    f"{np.mean(scores_no_da):.2f} (+/- {np.std(scores_no_da):.2f})"
)

# %%

plt.figure(figsize=(6, 4))
plt.barh(
    [0, 1],
    [np.mean(scores), np.mean(scores_no_da)],
    yerr=[np.std(scores), np.std(scores_no_da)],
)
plt.yticks([0, 1], ["DA", "No DA"])
plt.xlabel("Accuracy")
plt.axvline(0.5, color="k", linestyle="--", label="Random guess")
plt.legend()
plt.show()
