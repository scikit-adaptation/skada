from abc import abstractmethod
from itertools import chain
from typing import Generator, List, Optional, Set, Tuple, Union

import numpy as np
from sklearn.base import BaseEstimator, clone
from sklearn.utils.metaestimators import available_if
from sklearn.utils.validation import check_is_fitted

# xxx(okachaiev): this should be `skada.utils.check_X_y_domain`
# rather than `skada._utils.check_X_y_domain`
from ._utils import check_X_domain, check_X_y_domain


def _estimator_has(attr):
    """Check if we can delegate a method to the underlying estimator.

    First, we check the first fitted classifier if available, otherwise we
    check the unfitted classifier.
    """
    def has_base_estimator(estimator) -> bool:
        return hasattr(estimator, "base_estimator") and hasattr(estimator.base_estimator, attr)

    # xxx(okachaiev): there should be a simple way to access selector base estimator
    def has_estimator_selector(estimator) -> bool:
        return hasattr(estimator, "estimators_") and hasattr(estimator.estimators_[0], attr)

    return lambda estimator: has_base_estimator(estimator) or has_estimator_selector(estimator)


# xxx(okachaiev): implement clone, repr so we don't have to use BaseEstimator
class BaseAdapter(BaseEstimator):

    @abstractmethod
    def adapt(self, X, y=None, sample_domain=None, *, sample_weight=None):
        """Transform samples, labels, and weights into the space in which
        the estimator is trained.
        """
        # xxx(okachaiev): I'm still not sure if it's required for adapt
        # to return new sample_domain. I can't find any example where such
        # API would be useful (rather than being annoying)
        return X, y, sample_domain, sample_weight

    @abstractmethod
    def fit(self, X, y=None, sample_domain=None, *, sample_weight=None):
        """Fit adaptation parameters"""
        return self
    
    def fit_adapt(self, X, y=None, sample_domain=None, *, sample_weight=None, **kwargs):
        self.fit(X, y=y, sample_domain=sample_domain, sample_weight=sample_weight, **kwargs)
        return self.adapt(X, y=y, sample_domain=sample_domain, sample_weight=sample_weight, **kwargs)


class BaseDomainAwareEstimator(BaseEstimator):
    """Base class for Data Adaptation estimators.

    This class forms the foundation for domain-aware estimators. Each specific
    implementation of such an estimator is expected to define the logic for two main
    functions:

    - `get_domain_estimators`: Accepts a list of domain labels and returns corresponding
    estimators with domain assignments.
    
    - `get_domain_adapters`: Takes a list of domain labels and produces adapters with
    the associated domain assignments.

    The defined adapters and estimators will be trained during the `fit()` method call.
    Subsequent predictions will utilize these, based on the domain labels provided at
    inference time. This design is versatile, accommodating a broad spectrum of use cases.
    To facilitate the development of new estimators, the library offers ready-to-use mixins
    for prevalent scenarios:

    - :class:`SingleEstimatorMixin`
    - :class:`PerDomainEstimatorMixin`
    - :class:`SourceTargetEstimatorMixin`
    - :class:`SingleAdapterMixin`
    - :class:`PerDomainAdapterMixin`
    - :class:`SourceTargetAdapterMixin`

    See the documentation for each class for details on the processing strategy.
    """

    INCLUDE_ALL_SOURCES = np.inf
    INCLUDE_ALL_TARGETS = -np.inf

    __metadata_request__fit = {'sample_domain': True}
    __metadata_request__predict = {'sample_domain': True}
    __metadata_request__predict_proba = {'sample_domain': True}
    __metadata_request__predict_log_proba = {'sample_domain': True}
    __metadata_request__decision_function = {'sample_domain': True}
    __metadata_request__score = {'sample_domain': True}

    @abstractmethod
    def get_domain_estimators(
        self,
        sample_domain: np.ndarray
    ) -> List[Tuple[BaseEstimator, Union[int, np.ndarray]]]:
        """Creates new estimators.

        Returns list of estimators each with a list of corresponding
        domain labels. In case there's a single estimator, just specify
        all labels. Note that with such API one have flexibility to
        manage which domains are eligible for 'predict'.
        """

    @abstractmethod
    def get_domain_adapters(
        self,
        sample_domain: np.ndarray
    ) -> List[Tuple[BaseAdapter, Union[int, np.ndarray]]]:
        """Creates new adapters.

        Returns list of adapter each with a list of corresponding domain
        labels. Special markers `INCLUDE_ALL_SOURCES` and `INCLUDE_ALL_TARGETS`
        could be used to indicate that corresponding adapter is universal.
        In case there's a single adapter, just specify all labels or both markers.
        Note that with such API one have flexibility to manage which domains are
        eligible for 'predict'.
        """

    def fit(self, X, y, sample_domain=None, *, sample_weight=None, **kwargs):
        """Fit the DA model on data"""
        X, y, sample_domain = check_X_y_domain(
            X,
            y,
            sample_domain,
            allow_auto_sample_domain=True,
            return_joint=True
        )
        # fit adaptation parameters
        adapters = self.get_domain_adapters(sample_domain)
        # xxx(okachaiev): this is horribly inefficient by I guess
        #                 we will be able to find better implementation
        #                 if/when we feel that API is good/appropriate
        for adapter, indices in self.select_domain_estimators(
            sample_domain,
            select_from=adapters
        ):
            adapter.fit(
                X[indices],
                y[indices],
                sample_domain=sample_domain[indices],
                sample_weight=sample_weight[indices] if sample_weight is not None else sample_weight,
            )
        self.adapters_ = adapters
        self.adapt_domains_ = self._unique_domains(adapters)
        # adapt sample, labels or weights
        # xxx(okachaiev): this is going to run pretty much they same computation
        # trying to figure out which domains go where but it's okay, we can optimize
        # performance later
        X_adapt, y_adapt, sample_domain, weight_output = self.adapt(
            X,
            y=y,
            sample_domain=sample_domain,
            sample_weight=sample_weight,
            **kwargs
        )
        estimators = self.get_domain_estimators(sample_domain)
        # xxx(okachaiev): this is horribly inefficient by I guess
        #                 we will be able to find better implementation
        #                 if/when we feel that API is good/appropriate
        for estimator, indices in self.select_domain_estimators(
            sample_domain,
            select_from=estimators
        ):
            # xxx(okachaiev): propagate weight_output
            estimator.fit(X_adapt[indices], y_adapt[indices])
        self.estimators_ = estimators
        # xxx(okachaiev): there's still a question if we should immediately
        # throw an error in case sets of domains for adaptation and estimators
        # are different
        self.fit_domains_ = self._unique_domains(estimators)
        return self

    # xxx(okachaiev): might be a util function, no need to include as a class API
    # xxx(okachaiev): the name of the function is suboptimal, as it's used both for
    #                 adapters & estimators
    def select_domain_estimators(
        self,
        sample_domain,
        select_from=None
    ) -> Generator[Tuple[BaseEstimator, np.ndarray], None, None]:
        """Yields estimators along side with indices of samples."""
        assert len(sample_domain.shape) == 1, "sample domain is a 1D array"
        if select_from is None:
            check_is_fitted(self, 'estimators_')
            select_from = self.estimators_
        for estimator, domains in select_from:
            # xxx(okachaiev): see my previous note about performance issues
            # xxx(okachaiev): it's easy to short-circuit everything when we
            # are given +/- inf as list of domains
            indices_set = []
            for domain in domains:
                if domain == self.INCLUDE_ALL_SOURCES:
                    criterion = sample_domain >= 0
                elif domain == self.INCLUDE_ALL_TARGETS:
                    criterion = sample_domain < 0
                else:
                    criterion = sample_domain == domain
                unroll_domain_idx, = np.nonzero(criterion)
                indices_set.append(unroll_domain_idx)
            indices = np.concatenate(indices_set)
            indices = np.unique(indices)
            yield estimator, indices

    def adapt(self, X, y=None, sample_domain=None, *, sample_weight=None, **kwargs):
        """Transform samples, labels, and weights into the space in which
        the estimator is trained.
        """
        check_is_fitted(self, 'adapters_')
        X_output, y_output, weight_output = None, None, None
        for adapter, indices in self.select_domain_estimators(sample_domain, select_from=self.adapters_):
            X_adapt, y_adapt, _, weight_adapt = adapter.adapt(
                X[indices],
                y=y[indices] if y is not None else None,
                sample_domain=sample_domain[indices],
                sample_weight=sample_weight[indices] if sample_weight is not None else None
            )
            if X_output is None:
                X_output = np.zeros((X.shape[0], *X_adapt.shape[1:]), dtype=X_adapt.dtype)
            X_output[indices] = X_adapt
            if y_output is None and y_adapt is not None:
                y_output = np.zeros((y.shape[0], *y_adapt.shape[1:]), dtype=y_adapt.dtype)
            if y_adapt is not None:
                y_output[indices] = y_adapt
            if weight_output is None and weight_adapt is not None:
                weight_output = np.zeros(X.shape[0], dtype=np.float32)
            if weight_adapt is not None:
                weight_output[indices] = weight_adapt
        return X_output, y_output, sample_domain, weight_output

    # xxx(okachaiev): code duplication with 'fit', for demo purpose only
    def update(self, X, y, sample_domain, *, sample_weight=None, **kwargs):
        """Domain adaptation setup when new target domain is given at a test-time."""
        check_is_fitted(self, 'estimators_')
        assert (
            self.INCLUDE_ALL_TARGETS not in self.fit_domains_ or
            self.INCLUDE_ALL_TARGETS not in self.adapt_domains_
        ), "The method does not support test time domain adaptation."
        X, y, sample_domain = check_X_y_domain(
            X,
            y,
            sample_domain,
            allow_source=False, # to make sure we only add new target domains
            return_joint=True,
            # xxx(okachaiev): we can come up with some fancy strategy here,
            # but I'm not sure it worth the effort, honestly
            allow_auto_sample_domain=False,
        )
        # fit adaptation parameters
        adapters = self.get_domain_adapters(sample_domain)
        # xxx(okachaiev): this is horribly inefficient by I guess
        #                 we will be able to find better implementation
        #                 if/when we feel that API is good/appropriate
        for adapter, indices in self.select_domain_estimators(
            sample_domain,
            select_from=adapters
        ):
            adapter.fit(
                X[indices],
                y[indices],
                sample_domain=sample_domain[indices],
                sample_weight=sample_weight[indices],
            )
        self.adapters_.extend(adapters)
        self.adapt_domains_.extend(self._unique_domains(adapters))
        # adapt sample, labels or weights
        # xxx(okachaiev): this is going to run pretty much they same computation
        # trying to figure out which domains go where but it's okay, we can optimize
        # performance later
        X_adapt, y_adapt, sample_domain, _ = self.adapt(
            X,
            y=y,
            sample_domain=sample_domain,
            sample_weight=sample_weight,
            **kwargs
        )
        estimators = self.get_domain_estimators(sample_domain)
        # xxx(okachaiev): this is horribly inefficient by I guess
        #                 we will be able to find better implementation
        #                 if/when we feel that API is good/appropriate
        for estimator, indices in self.select_domain_estimators(
            sample_domain,
            select_from=estimators
        ):
            estimator.fit(X_adapt[indices], y_adapt[indices])
        self.estimators_.extend(estimators)
        # xxx(okachaiev): there's still a question if we should immediately
        # throw an error in case sets of domains for adaptation and estimators
        # are different
        self.fit_domains_.extend(self._unique_domains(estimators))
        return self

    def _check_and_adapt(self, X, sample_domain):
        check_is_fitted(self, 'estimators_')
        X, sample_domain = check_X_domain(
            X,
            sample_domain,
            # xxx(okachaiev): right now I'm testing for domains used
            # for fitting, ideally we need to also cover the situation
            # when estimators and adapters are defined on different sets
            # of domains (likely in 'fit', a bit too late here)
            allow_domains=self.fit_domains_,
            allow_auto_sample_domain=True,
        )
        # xxx(okachaiev): this is a good case were we don't
        #                 need to adapt labels, only samples
        #                 thus y is optional in 'adapt' API
        X_adapt, _, sample_domain, _ = self.adapt(X, sample_domain=sample_domain)
        return X_adapt, sample_domain

    # xxx(okachaiev): this is, actually, not an 'int' as we are allowing for np.inf here as well
    def _unique_domains(self, estimators: List[Tuple[BaseAdapter, Union[int, np.ndarray]]]) -> Set[int]:
        """Takes a list of the estimators announced for fitting and extracts
        set of unique domain indices. This is needed for making sure that
        we don't use estimator on the domain it was not trained on (e.g. when
        running per-domain estimator or adapter strategy). For test-time DA
        methods, 'update' method is available.
        """
        return set(chain.from_iterable(
            ([domains] if isinstance(domains, int) else domains)
            for _, domains
            in estimators
        ))

    def _call_estimators_method(self, method_name: str, X, sample_domain) -> np.ndarray:
        output = None
        for estimator, indices in self.select_domain_estimators(sample_domain):
            out = getattr(estimator, method_name)(X[indices])
            if output is None:
                output = np.zeros((X.shape[0], *out.shape), dtype=out.dtype)
            output[indices] = out
        return output

    # xxx(okachaiev): i bet this should have parameters for weights as well
    def predict(self, X, sample_domain=None):
        X, sample_domain = self._check_and_adapt(X, sample_domain)
        return self._call_estimators_method('predict', X, sample_domain)

    @available_if(_estimator_has("predict_proba"))
    def predict_proba(self, X, sample_domain=None):
        X, sample_domain = self._check_and_adapt(X, sample_domain)
        return self._call_estimators_method('predict_proba', X, sample_domain)

    @available_if(_estimator_has("decision_function"))
    def decision_function(self, X, sample_domain=None):
        X, sample_domain = self._check_and_adapt(X, sample_domain)
        return self._call_estimators_method('decision_function', X, sample_domain)

    @available_if(_estimator_has("predict_log_proba"))
    def predict_log_proba(self, X, sample_domain=None):
        X, sample_domain = self._check_and_adapt(X, sample_domain)
        return self._call_estimators_method('predict_log_proba', X, sample_domain)

    @available_if(_estimator_has("score"))
    def score(self, X, y, sample_domain=None, sample_weight=None):
        X, sample_domain = self._check_and_adapt(X, sample_domain)
        scores, n_samples = [], []
        for estimator, indices in self.select_domain_estimators(sample_domain):
            scores.append(estimator.score(X[indices], y[indices]))
            n_samples.append(X[indices].shape[0])
        return np.average(scores, weights=n_samples)

    def fit_predict(self, X, y, sample_domain=None, *, sample_weight=None, **kwargs):
        """Fit and predict"""
        self.fit(X, y, sample_domain=sample_domain, sample_weight=sample_weight, **kwargs)
        return self.predict(X, sample_domain=sample_domain)

    def update_predict(self, X, y, sample_domain=None, *, sample_weight=None, **kwargs):
        """Update and predict"""
        self.update(X, y, sample_domain=sample_domain, sample_weight=sample_weight, **kwargs)
        return self.predict(X, sample_domain=sample_domain)


class BaseSelector:

    @abstractmethod
    def select(self, sample_domain: np.ndarray) -> List[Tuple[BaseEstimator, np.ndarray]]:
        """Creates new estimators.

        Returns list of estimators each with a list of corresponding
        domain labels. In case there's a single estimator, just specify
        all labels. Note that with such API one have flexibility to
        manage which domains are eligible for 'predict'.
        """

    # xxx(okachaiev): there might be a much easier way of doing this
    @abstractmethod
    def get_base_estimator(self) -> BaseEstimator:
        """Return object of the estimator suitable for property testing
        (for example, for detecting available methods of the estimator).
        """


class SingleEstimatorSelector(BaseSelector):
    """Use the same estimator (passed as `base_estimator` to the constructor)
    for all domains (including source and target).
    """

    def __init__(self, base_estimator: BaseEstimator):
        super().__init__()
        self.base_estimator = base_estimator

    def select(self, sample_domain) -> List[Tuple[BaseEstimator, np.ndarray]]:
        return [(
            clone(self.base_estimator),
            np.array([
                BaseDomainAwareEstimator.INCLUDE_ALL_SOURCES,
                BaseDomainAwareEstimator.INCLUDE_ALL_TARGETS
            ])
        )]

    def get_base_estimator(self) -> BaseEstimator:
        return self.base_estimator


class PerDomainEstimatorSelector(BaseSelector):
    """Takes a single `base_estimator` as an argument but uses them (to fit and adapt)
    separately: by creating copy per each domain.
    """

    def __init__(self, base_estimator: BaseEstimator):
        super().__init__()
        self.base_estimator = base_estimator

    def select_estimators(self, sample_domain) -> List[Tuple[BaseEstimator, np.ndarray]]:
        """Creates new estimators.

        Returns list of estimators each with a list of corresponding
        domain labels. In case there's a single estimator, just specify
        all labels. Note that with such API one have flexibility to
        manage which domains are eligible for 'predict'.
        """
        return [(clone(self.base_estimator), domain) for domain in np.unique(sample_domain)]

    def get_base_estimator(self) -> BaseEstimator:
        return self.base_estimator


class SourceTargetEstimatorSelector(BaseSelector):
    """Uses one estimator for all sources and one for all targets."""

    def __init__(self, source_estimator: BaseEstimator, target_estimator: BaseEstimator):
        super().__init__()
        self.source_estimator = source_estimator
        self.target_estimator = target_estimator

    def select(self, sample_domain) -> List[Tuple[BaseAdapter, np.ndarray]]:
        """Creates new estimators.

        Returns list of estimators each with a list of corresponding
        domain labels. In case there's a single estimator, just specify
        all labels. Note that with such API one have flexibility to
        manage which domains are eligible for 'predict'.
        """
        return [
            (clone(self.source_estimator), BaseDomainAwareEstimator.INCLUDE_ALL_SOURCES),
            (clone(self.target_estimator), BaseDomainAwareEstimator.INCLUDE_ALL_TARGETS),
        ]

    def get_base_estimator(self) -> BaseEstimator:
        return self.target_estimator


# xxx(okachaiev): get_params and set_params should propagate settings
class DomainAwareEstimator(BaseDomainAwareEstimator):
    """API to move '*Mixin'(s) into '*Selector'(s)."""

    def __init__(
        self,
        # xxx(okachaiev): I guess we can make them into a type generic
        adapter_selector: Union[BaseEstimator, BaseSelector],
        estimator_selector: Union[BaseEstimator, BaseSelector],
    ):
        super().__init__()
        if not isinstance(adapter_selector, BaseSelector):
            adapter_selector = SingleEstimatorSelector(adapter_selector)
        if not isinstance(estimator_selector, BaseSelector):
            estimator_selector = SingleEstimatorSelector(estimator_selector)
        self.adapter_selector = adapter_selector
        self.estimator_selector = estimator_selector

    @property
    def base_estimator(self) -> BaseEstimator:
        return self.estimator_selector.get_base_estimator()

    def get_domain_adapters(self, sample_domain) -> List[Tuple[BaseAdapter, np.ndarray]]:
        return self.adapter_selector.select(sample_domain)

    def get_domain_estimators(self, sample_domain) -> List[Tuple[BaseEstimator, np.ndarray]]:
        return self.estimator_selector.select(sample_domain)
