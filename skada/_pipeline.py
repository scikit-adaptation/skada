# Author: Theo Gnassounou <theo.gnassounou@inria.fr>
#         Remi Flamary <remi.flamary@polytechnique.edu>
#         Oleksii Kachaiev <kachayev@gmail.com>
#
# License: BSD 3-Clause

from collections import defaultdict

from typing import Callable, Optional, Union

from joblib import Memory
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline

from .base import BaseSelector, PerDomain, Shared


_DEFAULT_SELECTORS = {
    'shared': Shared,
    'per_domain': PerDomain,
}


# xxx(okachaiev): block 'fit_predict' as it is somewhat unexpected
def make_da_pipeline(
    *steps,
    memory: Optional[Memory] = None,
    verbose: bool = False,
    default_selector: Union[str, Callable[[BaseEstimator], BaseSelector]] = 'shared',
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
    # note that we generate names before wrapping estimators into the selector
    # xxx(okachaiev): unwrap from the selector when passed explicitly
    if not steps:
        raise TypeError("Missing 1 required positional argument: 'steps'")

    names = [step[0] if isinstance(step, tuple) else None for step in steps]
    estimators = [step[1] if isinstance(step, tuple) else step for step in steps]

    wrapped_estimators = _wrap_with_selectors(estimators, default_selector)
    steps = _name_estimators(wrapped_estimators)
    steps[-1][1]._mark_as_final()
    named_steps = [
        (auto_name, step) if user_name is None else (user_name, step)
        for user_name, (auto_name, step) in zip(names, steps)
    ]
    return Pipeline(named_steps, memory=memory, verbose=verbose)


def _wrap_with_selector(
    estimator: BaseEstimator,
    selector: Union[str, Callable[[BaseEstimator], BaseSelector]]
) -> BaseSelector:
    if not isinstance(estimator, BaseSelector):
        if callable(selector):
            estimator = selector(estimator)
            if not isinstance(estimator, BaseSelector):
                raise ValueError("Callable `default_selector` has to return `BaseSelector` "  # noqa: E501
                                 f"instance, got {type(estimator)} instead.")
        elif isinstance(selector, str):
            selector_cls = _DEFAULT_SELECTORS.get(selector)
            if selector_cls is None:
                raise ValueError(f"Unsupported `default_selector` name: {selector}."
                                 f"Use one of {_DEFAULT_SELECTORS.keys().join(', ')}")
            estimator = selector_cls(estimator)
        else:
            raise ValueError("Unsupported `default_selector` type: {type(selector)}")
    return estimator


def _wrap_with_selectors(
    estimators: [BaseEstimator],
    default_selector: Union[str, Callable[[BaseEstimator], BaseSelector]]
) -> [BaseEstimator]:
    return [
        (_wrap_with_selector(estimator, default_selector))
        for estimator in estimators
    ]


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
        name = type(estimator.base_estimator).__name__.lower()
        if isinstance(estimator, PerDomain):
            name = 'perdomain_' + name
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
