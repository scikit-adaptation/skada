.. _scorer:

-----------------------------------------------------
Metrics: Compute score for domain adaptation problems
-----------------------------------------------------

.. currentmodule:: skada.metrics

To evaluate an estimator or to select the best parameters for it, it is necessary to define a score.
In `sklearn <https://scikit-learn.org/>`_, several functions and objects can make use of the
scoring API like `cross_val_score <https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_score.html#sklearn.model_selection.cross_val_score>`_
or `GridSearchCV <https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html#sklearn.model_selection.GridSearchCV>`_.
To avoid overfitting, these methods split the initial data into training set and test set.
The training set is used to fit the estimator and the test set is used to compute the score.

In domain adaptation (DA) problems, source data and target data have a shift in their distributions.

Let's load a DA dataset::

    >>> from skada.datasets import make_shifted_datasets
    >>> from skada import EntropicOTmapping
    >>> from skada.metrics import TargetAccuracyScorer
    >>> RANDOM_SEED = 0
    >>> X, y, X_target, y_target = make_shifted_datasets(
    ...     n_samples_source=30,
    ...     n_samples_target=20,
    ...     shift="covariate_shift",
    ...     label="binary",
    ...     noise=0.4,
    ...     random_state=RANDOM_SEED,
    ... )

Now let's define a DA estimator to evaluate on this data::

    >>> from skada import DensityReweight
    >>> from sklearn.linear_model import LogisticRegression
    >>> base_estimator = LogisticRegression()
    >>> estimator = DensityReweight(base_estimator=base_estimator)

Having a distribution shift between the two domains means that if the validation
is done on samples from source like shown in the images below, there is high
chance that the score does not reflect the score on target because the distributions are different.

.. image:: images/source_only_scorer.png
   :width: 400px
   :height: 240px
   :alt: Source Only Scorer
   :align: center

To evaluate the estimator on the source data, one can use::

    >>> from sklearn.model_selection import cross_val_score
    >>> from sklearn.model_selection import ShuffleSplit
    >>> cv = ShuffleSplit(n_splits=5, test_size=0.3, random_state=0)
    >>> cross_val_score(
    ...     estimator,
    ...     X,
    ...     y,
    ...     cv=cv,
    ...     fit_params={'X_target': X_target},
    ...     scoring=None,
    ... )
    array([0.72222222, 0.83333333, 0.81944444])


skada offers a way to do the evaluation on the target data, while
reusing the scikit-learn methods and scoring API.

Different methods are available, to start we will use :class:`skada.metrics.SupervisedScorer`
that computes the score on the target domain::

    >>> from skada.metrics import SupervisedScorer
    >>> cv = ShuffleSplit(n_splits=5, test_size=0.3, random_state=0)
    >>> cross_val_score(
    ...     estimator,
    ...     X,
    ...     y,
    ...     cv=cv,
    ...     fit_params={'X_target': X_target},
    ...     scoring=SupervisedScorer(X_target, y_target),
    ... )
    array([0.975  , 0.95625, 0.95625])
