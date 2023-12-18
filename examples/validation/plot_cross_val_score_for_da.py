"""
Using cross_val_score with skada
================================

This illustrates the use of DA scorer such :class:`~skada.metrics.TargetAccuracyScorer`
with `cross_val_score <https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_score.html#sklearn.model_selection.cross_val_score>`_.
"""  # noqa
# %%
# Prepare dataset and estimators

import numpy as np
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit

from skada import EntropicOTMapping, source_target_split
from skada.datasets import make_shifted_datasets
from skada.metrics import SupervisedScorer


RANDOM_SEED = 0
dataset = make_shifted_datasets(
    n_samples_source=30,
    n_samples_target=20,
    shift="concept_drift",
    label="binary",
    noise=0.4,
    random_state=RANDOM_SEED,
    return_dataset=True
)

base_estimator = SVC()
estimator = EntropicOTMapping(
    base_estimator=base_estimator,
    reg_e=0.5,
    tol=1e-3
)

X, y, sample_domain = dataset.pack_train(as_sources=['s'], as_targets=['t'])
X_source, y_source, X_target, y_target = source_target_split(X, y, sample_domain)
cv = ShuffleSplit(n_splits=5, test_size=0.3, random_state=0)

# %%
# Cross Validate using supervised labels from the target domain
#
# Supervised scoring requires target labels to be passed into the pipeline
# separately, so they are only available for the scoring.

_, target_labels, _ = dataset.pack(as_sources=['s'], as_targets=['t'], train=False)
scores_sup = cross_val_score(
    estimator,
    X,
    y,
    cv=cv,
    params={'sample_domain': sample_domain, 'target_labels': target_labels},
    scoring=SupervisedScorer(),
)

print(
    "Cross-validation score with supervised DA: "
    f"{np.mean(scores_sup):.2f} (+/- {np.std(scores_sup):.2f})"
)

# %%
# Compare scores with the simple estimator with no adaptation


def _scorer(estimator, X, y):
    return estimator.score(X_target, y_target)


scores_no_da = cross_val_score(
    base_estimator,
    X_source,
    y_source,
    cv=cv,
    scoring=_scorer,
)

print(
    "Cross-validation score without DA: "
    f"{np.mean(scores_no_da):.2f} (+/- {np.std(scores_no_da):.2f})"
)

# %%

plt.figure(figsize=(6, 4))
plt.barh(
    [0, 1],
    [np.mean(scores_sup), np.mean(scores_no_da)],
    yerr=[np.std(scores_sup), np.std(scores_no_da)],
)
plt.yticks([0, 1], ["DA", "No DA"])
plt.xlabel("Accuracy")
plt.axvline(0.5, color="k", linestyle="--", label="Random guess")
plt.legend()
plt.show()

# %%
