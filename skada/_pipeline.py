# Author: Theo Gnassounou <theo.gnassounou@inria.fr>
#         Remi Flamary <remi.flamary@polytechnique.edu>
#         Oleksii Kachaiev <kachayev@gmail.com>
#
# License: BSD 3-Clause

from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline, _name_estimators

from .base import BaseSelector, Shared


# xxx(okachaiev): block 'fit_predict' as it is somewhat unexpected
def make_da_pipeline(*steps, memory=None, verbose=False, default_selector='shared'):
    """Construct a :class:`~sklearn.pipeline.Pipeline` from the given estimators.

    This is a shorthand for the :class:`sklearn.pipeline.Pipeline` constructor;
    it does not require, and does not permit, naming the estimators. Instead,
    their names will be set to the lowercase of their types automatically.

    Parameters
    ----------
    *steps : list of Estimator objects
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
    Pipeline(steps=[('standardscaler', Shared(base_estimator=StandardScaler())),
                    ('gaussiannb', Shared(base_estimator=GaussianNB()))])
    """
    # note that we generate names before wrapping estimators into the selector
    # xxx(okachaiev): unwrap from the selector when passed explicitly
    steps = _wrap_selectors(_name_estimators(steps), default_selector=default_selector)
    return Pipeline(steps, memory=memory, verbose=verbose)


def _wrap_selectors(
    steps: [(str, BaseEstimator)],
    default_selector: str = 'shared'
) -> [(str, BaseEstimator)]:
    # xxx(okachaiev): respect 'default' configuration
    return [
        (name, estimator if isinstance(estimator, BaseSelector) else Shared(estimator))
        for (name, estimator) in steps
    ]
