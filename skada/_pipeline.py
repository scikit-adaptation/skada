# Author: Theo Gnassounou <theo.gnassounou@inria.fr>
#         Remi Flamary <remi.flamary@polytechnique.edu>
#         Oleksii Kachaiev <kachayev@gmail.com>
#
# License: BSD 3-Clause

from collections import defaultdict
from typing import Callable, List, Optional, Union

from joblib import Memory
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline

from .base import BaseSelector, PerDomain, Shared

_DEFAULT_SELECTORS = {
    "shared": Shared,
    "per_domain": PerDomain,
}


# xxx(okachaiev): block 'fit_predict' as it is somewhat unexpected
def make_da_pipeline(
    *steps,
    memory: Optional[Memory] = None,
    verbose: bool = False,
    default_selector: Union[str, Callable[[BaseEstimator], BaseSelector]] = "shared",
    mask_target_labels: bool = True,
) -> Pipeline:
    """Construct a :class:`~sklearn.pipeline.Pipeline` from the given estimators.

    This is a shorthand for the :class:`sklearn.pipeline.Pipeline` constructor;
    it does not require, and does not permit, naming the estimators. Instead,
    their names will be set to the lowercase of their types automatically.

    Parameters
    ----------
    *steps : list of estimators or tuples of the form (name of step, estimator).
        List of the scikit-learn estimators that are chained together.

    memory : str or object with the joblib.Memory interface, default=None
        Used to cache the fitted transformers of the pipeline. The last step
        will never be cached, even if it is a transformer. By default, no
        caching is performed. If a string is given, it is the path to the
        caching directory. Enabling caching triggers a clone of the transformers
        before fitting. Therefore, the transformer instance given to the
        pipeline cannot be inspected directly. Use the attribute ``named_steps``
        or ``steps`` to inspect estimators within the pipeline. Caching the
        transformers is advantageous when fitting is time consuming.

    verbose : bool, default=False
        If True, the time elapsed while fitting each step will be printed as it
        is completed.

    default_selector : str or callable, default = 'shared'
        Specifies a domain selector to wrap the estimator, if it is not already
        wrapped. Refer to :class:`~skada.base.BaseSelector` for an understanding of
        selector functionalities. The available options include 'shared' and
        'per_domain'. For integrating a custom selector as the default, pass a
        callable that accepts :class:`~sklearn.base.BaseEstimator` and returns
        the estimator encapsulated within a domain selector.

    mask_target_labels : bool, default=True
        Whether to mask target labels in the pipeline.

    Returns
    -------
    p : Pipeline
        Returns a scikit-learn :class:`~sklearn.pipeline.Pipeline` object.

    Examples
    --------
    >>> from sklearn.naive_bayes import GaussianNB
    >>> from sklearn.preprocessing import StandardScaler
    >>> from skada import make_da_pipeline
    >>> make_da_pipeline(StandardScaler(), GaussianNB(priors=None))
    Pipeline(steps=[('standardscaler',
                     Shared(base_estimator=StandardScaler(), copy=True,
                            with_mean=True, with_std=True)),
                    ('gaussiannb',
                     Shared(base_estimator=GaussianNB(), priors=None,
                            var_smoothing=1e-09))])
    """
    if not steps:
        raise TypeError("Missing 1 required positional argument: 'steps'")

    names, estimators = [], []
    for step in steps:
        name, estimator = step if isinstance(step, tuple) else (None, step)
        if isinstance(estimator, Pipeline) and isinstance(estimator[0], BaseSelector):
            # this means we got DA pipeline as a step in the pipeline
            for nested_name, nested_selector in estimator.steps:
                if name is not None:
                    nested_name = f"{name}__{nested_name}"
                names.append(nested_name)
                estimators.append(nested_selector._unmark_as_final())
        else:
            names.append(name)
            estimators.append(estimator)
    wrapped_estimators = _wrap_with_selectors(
        estimators, default_selector, mask_target_labels
    )
    steps = _name_estimators(wrapped_estimators)
    named_steps = [
        (auto_name, step) if user_name is None else (user_name, step)
        for user_name, (auto_name, step) in zip(names, steps)
    ]
    named_steps[-1][1]._mark_as_final()
    return Pipeline(named_steps, memory=memory, verbose=verbose)


def _wrap_with_selector(
    estimator: BaseEstimator,
    selector: Union[str, Callable[[BaseEstimator], BaseSelector]],
    mask_target_labels: bool = True,
) -> BaseSelector:
    if (estimator is not None) and not isinstance(estimator, BaseSelector):
        if callable(selector):
            estimator = selector(estimator, mask_target_labels=mask_target_labels)
            if not isinstance(estimator, BaseSelector):
                raise ValueError(
                    "Callable `default_selector` has to return `BaseSelector` "  # noqa: E501
                    f"instance, got {type(estimator)} instead."
                )
        elif isinstance(selector, str):
            selector_cls = _DEFAULT_SELECTORS.get(selector)
            if selector_cls is None:
                raise ValueError(
                    f"Unsupported `default_selector` name: {selector}."
                    f"Use one of {_DEFAULT_SELECTORS.keys().join(', ')}"
                )
            estimator = selector_cls(estimator, mask_target_labels=mask_target_labels)
        else:
            raise ValueError(f"Unsupported `default_selector` type: {type(selector)}")
    return estimator


def _wrap_with_selectors(
    estimators: List[BaseEstimator],
    default_selector: Union[str, Callable[[BaseEstimator], BaseSelector]],
    mask_target_labels: bool = True,
) -> List[BaseEstimator]:
    wrap_list = []
    for estimator in estimators:
        if getattr(estimator, "predicts_target_labels", False):
            mask_target_labels = False

        wrap_list.append(
            _wrap_with_selector(
                estimator, default_selector, mask_target_labels=mask_target_labels
            )
        )
    return wrap_list


def _name_estimators(estimators):
    """Generate names for estimators."""
    # From scikit-learn: https://github.com/scikit-learn/scikit-learn
    # Author: Edouard Duchesnay
    #         Gael Varoquaux
    #         Virgile Fritsch
    #         Alexandre Gramfort
    #         Lars Buitinck
    # License: BSD
    names = []

    for estimator in estimators:
        # xxx(okachaiev): this logic gets progressively more
        # awkward. maybe we just need to make sure that default
        # 'Shared' selector does not get into a way of setting
        # parameters, but all others are just fine to be more
        # verbose
        if hasattr(estimator, "base_estimator"):
            name = type(estimator.base_estimator).__name__.lower()
        else:
            name = estimator.__class__.__name__.lower()
        if isinstance(estimator, PerDomain):
            name = "perdomain_" + name
        names.append(name)

    namecount = defaultdict(int)
    for est, name in zip(estimators, names):
        namecount[name] += 1

    for k, v in list(namecount.items()):
        if v == 1:
            del namecount[k]

    for i in reversed(range(len(estimators))):
        name = names[i]
        if name in namecount:
            names[i] += "-%d" % namecount[name]
            namecount[name] -= 1

    return list(zip(names, estimators))
