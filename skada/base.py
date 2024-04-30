# Author: Theo Gnassounou <theo.gnassounou@inria.fr>
#         Remi Flamary <remi.flamary@polytechnique.edu>
#         Oleksii Kachaiev <kachayev@gmail.com>
#
# License: BSD 3-Clause

from abc import abstractmethod
from typing import Literal, Optional, Union

import numpy as np
from sklearn.base import BaseEstimator, clone
from sklearn.exceptions import UnsetMetadataPassedError
from sklearn.utils.metadata_routing import (
    MetadataRouter,
    MethodMapping,
    _MetadataRequester,
    get_routing_for_object,
)
from sklearn.utils.metaestimators import available_if
from sklearn.utils.validation import check_is_fitted

from skada._utils import _apply_domain_masks, _remove_masked
from skada.utils import check_X_domain, check_X_y_domain, extract_source_indices


def _estimator_has(attr, base_attr_name='base_estimator'):
    """Check if we can delegate a method to the underlying estimator.

    First, we check the first fitted classifier if available, otherwise we
    check the unfitted classifier.
    """
    def has_base_estimator(estimator) -> bool:
        return hasattr(estimator, base_attr_name) and hasattr(
            getattr(estimator, base_attr_name),
            attr
        )

    # xxx(okachaiev): there should be a simple way to access selector base estimator
    def has_estimator_selector(estimator) -> bool:
        return hasattr(estimator, "estimators_") and hasattr(
            estimator.estimators_[0],
            attr
        )

    return lambda estimator: (has_base_estimator(estimator) or
                              has_estimator_selector(estimator))


class AdaptationOutput:
    """Container object for multi-key adaptation output."""

    def __init__(self, X, **params):
        self.X = X
        self.params = params

    def merge_into(self, output: Union[np.ndarray, "AdaptationOutput"]) -> "AdaptationOutput":
        if isinstance(output, self.__class__):
            for k, v in self.items():
                if k not in output.params:
                    output[k] = v
            return output
        else:
            self.X = output
            return self

    def keys(self):
        return self.params.keys()

    def values(self):
        return self.params.values()

    def items(self):
        return self.params.items()

    def get(self, key, default_value=None):
        return self.params.get(key, default_value)

    def __getattr__(self, attr_name):
        params = object.__getattribute__(self, 'params')
        if attr_name not in params:
            raise AttributeError()
        return params[attr_name]

    def __iter__(self):
        return iter(self.params)

    def __getitem__(self, k):
        return self.params[k]

    def __setitem__(self, k, v):
        self.params[k] = v
        return self


class IncompatibleMetadataError(UnsetMetadataPassedError):
    """The exception is designated to report the situation when the adapter output
    the key, like 'sample_weight', that is not explicitly consumed by the following
    estimator in the pipeline.

    The exception overrides :class:`~sklearn.exceptions.UnsetMetadataPassedError`
    when there is a reason to believe that the original exception was thrown because
    of the adapter output rather than being caused by the input to a specific function.
    """

    def __init__(self, message):
        super().__init__(message=message, unrequested_params={}, routed_params={})


class BaseAdapter(BaseEstimator):

    __metadata_request__fit = {'sample_domain': True}
    __metadata_request__transform = {'sample_domain': True, 'allow_source': True}

    @abstractmethod
    def fit_transform(self, X, y=None, *, sample_domain=None, **params):
        """Fit adapter and transforms samples, labels, and weights to be used
        to fit estimator (i.e. 'adapt' the data).
        """
        pass

    def transform(
        self,
        X,
        y=None,
        *,
        sample_domain=None,
        allow_source=False,
        **params
    ) -> np.ndarray:
        """Transforms (adapts) the data during evaluation. Default implementation
        passes through samples without changing them, as it is a default behavior
        for many adapters.
        """
        check_is_fitted(self)
        X, sample_domain = check_X_domain(
            X,
            sample_domain=sample_domain,
            allow_auto_sample_domain=True,
            allow_source=allow_source,
        )
        return X

    def fit(self, X, y=None, *, sample_domain=None, **params):
        """Fitting of the adapter is supposed to happen in `fit_transform`
        method, though for a convenience reason it might be redefined by
        by the specific implementation.
        """
        raise NotImplementedError('To fit adapter use `fit_transform` method.')


class _DAMetadataRequesterMixin(_MetadataRequester):
    """Mixin class for adding metadata related to the domain adaptation
    functionality. The mixin is primarily designed for the internal API
    and is expected to be rarely, if at all, required by end users.
    """

    __metadata_request__fit = {'sample_domain': True}
    __metadata_request__partial_fit = {'sample_domain': True}
    __metadata_request__predict = {'sample_domain': True, 'allow_source': True}
    __metadata_request__predict_proba = {'sample_domain': True, 'allow_source': True}
    __metadata_request__predict_log_proba = {
        'sample_domain': True,
        'allow_source': True
    }
    __metadata_request__score = {'sample_domain': True, 'allow_source': True}
    __metadata_request__decision_function = {
        'sample_domain': True,
        'allow_source': True
    }


class DAEstimator(BaseEstimator, _DAMetadataRequesterMixin):
    """Generic DA estimator class."""

    @abstractmethod
    def fit(self, X, y=None, sample_domain=None, *, sample_weight=None):
        """Fit adaptation parameters"""
        pass

    @abstractmethod
    def predict(self, X, sample_domain=None, *, sample_weight=None):
        """Predict using the model"""
        pass


class BaseSelector(BaseEstimator, _DAMetadataRequesterMixin):

    __metadata_request__transform = {'sample_domain': True}

    def __init__(self, base_estimator: BaseEstimator, **kwargs):
        super().__init__()
        self.base_estimator = base_estimator
        self.base_estimator.set_params(**kwargs)
        self._is_final = False
        self._is_transformer = hasattr(base_estimator, 'transform')

    def get_metadata_routing(self):
        return (
            MetadataRouter(owner=self.__class__.__name__)
            .add_self_request(self)
            .add(estimator=self.base_estimator, method_mapping=MethodMapping()
                 .add(callee='fit', caller='fit')
                 .add(callee='partial_fit', caller='partial_fit')
                 .add(callee='transform', caller='transform')
                 .add(callee='predict', caller='predict')
                 .add(callee='predict_proba', caller='predict_proba')
                 .add(callee='predict_log_proba', caller='predict_log_proba')
                 .add(callee='decision_function', caller='decision_function')
                 .add(callee='score', caller='score'))
        )

    @abstractmethod
    def get_estimator(self, *params) -> BaseEstimator:
        """Returns estimator associated with `params`.

        The set of available estimators and access to them has to be provided
        by specific implementations.
        """

    def get_params(self, deep=True):
        """Get parameters for this estimator.

        Returns the parameters of the base estimator provided in the constructor.

        Parameters
        ----------
        deep : bool, default=True
            If True, will return the parameters for this estimator and
            contained sub-objects that are estimators.

        Returns
        -------
        params : mapping of string to any
            Parameter names mapped to their values.
        """
        params = self.base_estimator.get_params(deep=deep)
        params['base_estimator'] = self.base_estimator
        return params

    def set_params(self, base_estimator=None, **kwargs):
        """Set the parameters of this estimator.

        Valid parameter keys can be listed with ``get_params()``. Note that
        you can directly set the parameters of the estimator using `base_estimator`
        attribute.

        Parameters
        ----------
        **kwargs : dict
            Parameters of of the base estimator.

        Returns
        -------
        self : object
            Selector class instance.
        """
        if base_estimator is not None:
            self.base_estimator = base_estimator
        self.base_estimator.set_params(**kwargs)
        return self

    def _mark_as_final(self):
        self._is_final = True
        return self

    def _unmark_as_final(self):
        self._is_final = False
        return self

    @abstractmethod
    def _route_to_estimator(self, method_name, X, y=None, **params) -> np.ndarray:
        """Abstract method for calling method of a base estimator based on
        the input and the routing logic associated with domain labels.
        """

    @available_if(_estimator_has('transform'))
    def transform(self, X_container, **params):
        output = self._route_to_estimator('transform', X_container, **params)
        # If the original input was an `AdaptationOutput` container,
        # we should either wrap the output back into container, or
        # merge a pair of containers if the `output` itself is an instance
        # of `AdaptationOutput`. The only case where should avoid such
        # operation is that if the selector wraps the last step in the pipeline.
        # In such a case, the `output` is supposed to be consumable from
        # the common sklearn API.
        if not self._is_final and isinstance(X_container, AdaptationOutput):
            output = X_container.merge_into(output)
        return output

    @available_if(_estimator_has('predict'))
    def predict(self, X, **params):
        return self._route_to_estimator('predict', X, **params)

    @available_if(_estimator_has('predict_proba'))
    def predict_proba(self, X, **params):
        return self._route_to_estimator('predict_proba', X, **params)

    @available_if(_estimator_has('predict_log_proba'))
    def predict_log_proba(self, X, **params):
        return self._route_to_estimator('predict_log_proba', X, **params)

    @available_if(_estimator_has('decision_function'))
    def decision_function(self, X, **params):
        return self._route_to_estimator('decision_function', X, **params)

    @available_if(_estimator_has('score'))
    def score(self, X, y, **params):
        return self._route_to_estimator('score', X, y=y, **params)

    # xxx(okachaiev): this name is terrible
    def _merge_and_route_params(self, routing_request, metadata_container, params):
        if self._is_final or not self._is_transformer:
            try:
                routed_params = routing_request._route_params(params=params)
            except UnsetMetadataPassedError as e:
                # check if every parameter given from the metadata container
                # was accepted by the downstream (base) estimator
                # xxx(okachaiev): there's still a way for this to fail,
                # if non-final estimator consumed the input that was
                # generated by one of the adapters
                for k, v in metadata_container.iter_metadata():
                    marker = routing_request.requests.get(k)
                    if v is not None and marker is None:
                        method = routing_request.method
                        raise IncompatibleMetadataError(
                            f"The adapter provided '{k}' parameter which is not explicitly set as "  # noqa
                            f"requested or not for '{routing_request.owner}.{method}'.\n"  # noqa
                            f"Make sure that metadata routing is properly setup, e.g. by calling 'set_{method}_request()'. "  # noqa
                            "See documentation at https://scikit-learn.org/stable/metadata_routing.html"  # noqa
                        ) from e
                # re-raise exception if the problem was not caused by the adapter
                raise e
        else:
            routed_params = {k: params[k] for k in routing_request._consumes(params=params)}
        return routed_params

    def _remove_masked(self, X, y, routed_params):
        """Removes masked inputs before passing them to a downstream (base) estimator,
        ensuring their compatibility with the DA pipeline, particularly for estimators
        that do not natively support DA setups, such as those from scikit-learn.

        Masked inputs are removed under the following conditions:
        - Labels `y` are provided (necessary for mask detection).
        - The estimator is not a transformer (i.e., does not define a
          'transform' function).
        - The estimator does not accept a `sample_domain` parameter
          through routing.

        In scenarios not meeting these criteria, masked input samples are retained.

        Note: This API is intended for internal use.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data
        y : array-like of shape (n_samples,)
            Labels for the data
        params : dict
            Additional parameters declared in the routing

        Returns
        -------
        X : array-like of shape (n_samples, n_features)
            Input data
        y : array-like of shape (n_samples,)
            Labels for the data
        params : dict
            Additional parameters declared in the routing
        """
        if (y is not None
                and not hasattr(self, 'transform')
                and 'sample_domain' not in routed_params):
            X, y, routed_params = _remove_masked(X, y, routed_params)
        return X, y, routed_params


class Shared(BaseSelector):

    def get_estimator(self) -> BaseEstimator:
        """Provides access to the fitted estimator."""
        check_is_fitted(self)
        return self.base_estimator_

    # xxx(okachaiev): solve the problem with parameter renaming
    def fit(self, X_container, y=None, **params):
        X, y, params = X_container.merge_out(y, **params)
        # xxx(okachaiev): for Share this should not be necessary,
        # as we expect metadata routing to select parameters for us
        # we only need to remove masked (when necessary)
        routing = get_routing_for_object(self.base_estimator)
        routed_params = self._merge_and_route_params(routing.fit, X_container, params)
        X, y, routed_params = self._remove_masked(X, y, routed_params)
        estimator = clone(self.base_estimator)
        estimator.fit(X, y, **routed_params)
        self.base_estimator_ = estimator
        self.routing_ = routing
        return self

    # xxx(okachaiev): this is not exactly true, more like
    # a) estimator has transform
    # b) estimator has fit_transform
    # c) estimator is adapter (has transform)
    @available_if(_estimator_has('transform'))
    def fit_transform(self, X_container, y=None, **params):
        # 'fit_transform' allows transformation for source domains
        # as well, that's why it calls 'adapt' directly
        # if isinstance(self.base_estimator, BaseAdapter):
        #     self.fit(X_container, y, **params)
        #     # xxx(okachaiev): adapt should take 'y' as well, as in many cases
        #     # we need to bound estimator fitting to a sub-group of the input
        #     X, y, method_params = X_container.merge_out(y, **params)
        #     routed_params = self.routing_.fit_transform._route_params(params=method_params)
        #     output = self.base_estimator_.adapt(X, **routed_params)
        if hasattr(self.base_estimator, "fit_transform"):
            X, y, method_params = X_container.merge_out(y, **params)
            routing = get_routing_for_object(self.base_estimator)
            routed_params = self._merge_and_route_params(routing.fit_transform, X_container, method_params)
            X, y, routed_params = self._remove_masked(X, y, routed_params)
            estimator = clone(self.base_estimator)
            output = estimator.fit_transform(X, y, **routed_params)
            self.base_estimator_ = estimator
            self.routing_ = routing
        else:
            self.fit(X_container, y, **params)
            X, y, method_params = X_container.merge_out(y, **params)
            transform_params = self.routing_.transform._route_params(params=method_params)
            output = self.transform(X, **transform_params)
        return X_container.merge_in(output)

    # xxx(okachaiev): fail if unknown domain is given
    def _route_to_estimator(self, method_name, X, y=None, **params):
        check_is_fitted(self)
        # xxx(okachaiev): make sure to remove this after the rework of adapters
        if isinstance(X, AdaptationOutput):
            X = X.X
        request = getattr(self.routing_, method_name)
        routed_params = {k: params[k] for k in request._consumes(params=params)}
        X, y, params = self._remove_masked(X, y, params)
        method = getattr(self.base_estimator_, method_name)
        output = method(X, **routed_params) if y is None else method(
            X, y, **routed_params
        )
        return output


class PerDomain(BaseSelector):

    def get_estimator(self, domain_label: int) -> BaseEstimator:
        """Provides access to the fitted estimator based on the domain label."""
        check_is_fitted(self)
        return self.estimators_[domain_label]

    def fit(self, X_container, y, **params):
        X, y, params = X_container.merge_out(y, **params)
        sample_domain = params['sample_domain']
        routing = get_routing_for_object(self.base_estimator)
        routed_params = self._merge_and_route_params(routing.fit, X_container, params)
        X, y, routed_params = self._remove_masked(X, y, routed_params)
        estimators = {}
        for domain_label in np.unique(sample_domain):
            idx, = np.where(sample_domain == domain_label)
            estimator = clone(self.base_estimator)
            estimator.fit(
                X[idx],
                y[idx] if y is not None else None,
                **{k: v[idx] for k, v in routed_params.items()}
            )
            estimators[domain_label] = estimator
        self.estimators_ = estimators
        self.routing_ = routing
        return self

    def _route_to_estimator(self, method_name, X, y=None, **params):
        check_is_fitted(self)
        # xxx(okachaiev): make sure to remove this after the rework of adapters
        if isinstance(X, AdaptationOutput):
            X = X.X
        request = getattr(self.routing_, method_name)
        # xxx(okachaiev): we should not use _merge_and_route_params helper
        routed_params = self._merge_and_route_params(request, None, params)
        # xxx(okachaiev): use check_*_domain to derive default domain labels
        sample_domain = params['sample_domain']
        output = None
        lst_domains = np.unique(sample_domain)
        # test if default target domain and unique target during fit and replace label
        for domain_label in lst_domains:
            # xxx(okachaiev): fail if unknown domain is given
            try:
                method = getattr(self.estimators_[domain_label], method_name)
            except KeyError:
                raise ValueError(
                    f"Domain label {domain_label} is not present in the "
                    "fitted estimators."
                )
            idx, = np.where(sample_domain == domain_label)
            X_domain = X[idx]
            y_domain = y[idx] if y is not None else None
            domain_params = {k: v[idx] for k, v in routed_params.items()}
            if y is None:
                domain_output = method(X_domain, **domain_params)
            else:
                domain_output = method(X_domain, y_domain, **domain_params)
            if output is None:
                output = np.zeros(
                    (X.shape[0], *domain_output.shape[1:]),
                    dtype=domain_output.dtype
                )
            output[idx] = domain_output
        return output


class _BaseSelectDomain(Shared):
    """Abstract class to implement selectors that are intended
    for picking specific subset of samples from the input for
    fitting the base estimator, e.g. only source, only targets,
    specific domain by its index or name, and more.

    Specific functionality is given by providing implementation
    for the `_select_indices` abstract method that receives an
    array with domain indices (i.e. `sample_domain`) and returns
    masks that should be used to filter out necessary samples
    from the input.
    """

    @abstractmethod
    def _select_indices(self, sample_domain):
        """Calculates masks for input samples.

        Parameters
        ----------
        sample_domain : array-like, shape (n_samples,)
            The domain labels.

        Returns
        -------
        filter_masks : array-like, shape (n_samples,)
            Array of boolean masks.
        """

    def fit(self, X_container, y, **params):
        X_input, params = self._unwrap_from_container(X_container, params)
        filter_masks = self._select_indices(params['sample_domain'])
        X_input, y, params = _apply_domain_masks(X_input, y, params, masks=filter_masks)
        return super().fit(X_input, y, **params)


class SelectSource(_BaseSelectDomain):
    """Selects only source domains for fitting base estimator."""

    def _select_indices(self, sample_domain):
        return extract_source_indices(sample_domain)


class SelectTarget(_BaseSelectDomain):
    """Selects only target domains for fitting base estimator."""

    def _select_indices(self, sample_domain):
        return ~extract_source_indices(sample_domain)


class SelectSourceTarget(BaseSelector):

    def __init__(self, source_estimator: BaseEstimator, target_estimator: Optional[BaseEstimator] = None):
        if target_estimator is not None \
                and hasattr(source_estimator, 'transform') \
                and not hasattr(target_estimator, 'transform'):
            raise TypeError("The provided source and target estimators must "
                            "both be transformers, or neither should be.")
        self.source_estimator = source_estimator
        self.target_estimator = target_estimator
        # xxx(okachaiev): the fact that we need to put those variables
        # here means that the choice of the base class is suboptimal
        self._is_final = False
        self._is_transformer = hasattr(source_estimator, 'transform')

    def get_metadata_routing(self):
        routing = MetadataRouter(owner=self.__class__.__name__).add_self_request(self)
        routing.add(estimator=self.source_estimator, method_mapping=MethodMapping()
            .add(callee='fit', caller='fit')
            .add(callee='partial_fit', caller='partial_fit')
            .add(callee='transform', caller='transform')
            .add(callee='predict', caller='predict')
            .add(callee='predict_proba', caller='predict_proba')
            .add(callee='predict_log_proba', caller='predict_log_proba')
            .add(callee='decision_function', caller='decision_function')
            .add(callee='score', caller='score'))
        if self.target_estimator is not None:
            routing.add(estimator=self.target_estimator, method_mapping=MethodMapping()
                .add(callee='fit', caller='fit')
                .add(callee='partial_fit', caller='partial_fit')
                .add(callee='transform', caller='transform')
                .add(callee='predict', caller='predict')
                .add(callee='predict_proba', caller='predict_proba')
                .add(callee='predict_log_proba', caller='predict_log_proba')
                .add(callee='decision_function', caller='decision_function')
                .add(callee='score', caller='score'))
        return routing

    def get_params(self, deep=True):
        """Get parameters for this estimator.

        Returns the parameters of the base estimator provided in the constructor.

        Parameters
        ----------
        deep : bool, default=True
            If True, will return the parameters for this estimator and
            contained sub-objects that are estimators.

        Returns
        -------
        params : mapping of string to any
            Parameter names mapped to their values.
        """
        return super(BaseSelector, self).get_params(deep=deep)

    def set_params(self, **kwargs):
        """Set the parameters of this estimator.

        Valid parameter keys can be listed with ``get_params()``. Note that
        you can directly set the parameters of the estimator using `base_estimator`
        attribute.

        Parameters
        ----------
        **kwargs : dict
            Parameters of of the base estimator.

        Returns
        -------
        self : object
            Selector class instance.
        """
        super(BaseSelector, self).set_params(**kwargs)
        return self

    def get_estimator(self, domain: Literal['source', 'target']) -> BaseEstimator:
        """Provides access to the fitted estimator based on the domain type."""
        assert domain in ('source', 'target')
        check_is_fitted(self)
        return self.estimators_[domain]

    def fit(self, X_container, y, **params):
        X, params = self._unwrap_from_container(X_container, params)
        # xxx(okachaiev): seems like we have this block of code rather often
        #                 maybe should be a helper
        if y is not None:
            X, y, sample_domain = check_X_y_domain(X, y, sample_domain=params.get('sample_domain'))
        else:
            X, sample_domain = check_X_domain(X, sample_domain=params.get('sample_domain'))
        params['sample_domain'] = sample_domain
        source_masks = extract_source_indices(sample_domain)
        estimators = {}
        target_estimator = self.target_estimator if self.target_estimator is not None else self.source_estimator
        for domain_type, base_estimator, domain_masks in [
            ('source', self.source_estimator, source_masks),
            ('target', target_estimator, ~source_masks)
        ]:
            if domain_masks.sum() == 0:
                # if we don't have either source or target, we should conclude that fitting
                # was not successful, otherwise prediction might be not possible
                raise ValueError(
                    "`SelectSourceTarget` requires both source and target samples for fitting. "
                    f"'{domain_type}' samples are missing in the input provided."
                )
            X_masked, y_masked, params_masked = _apply_domain_masks(X, y, params, masks=domain_masks)
            routing = get_routing_for_object(base_estimator)
            X_masked, routed_params = self._merge_and_route_params(routing.fit, X_masked, params_masked)
            X_masked, y_masked, routed_params = self._remove_masked(X_masked, y_masked, routed_params)
            estimator = clone(base_estimator)
            estimator.fit(X_masked, y_masked, **routed_params)
            estimators[domain_type] = estimator
        self.estimators_ = estimators
        return self

    def fit_transform(self, X, y=None, **params):
        """
        Fit to data, then transform it. In this case, the fitting and
        the transformation are performed on the target and source domains
        by default (allow_source=True).

        It should be used only for fitting the estimator, and not for generating
        the adaptation output. For the latter, use the `transform` method.
        """
        self.fit(X, y=y, **params)
        # assume 'fit_transform' is called to fit the estimator,
        # thus we allow for the source domain to be adapted
        return self.transform(
            X,
            y=y,
            allow_source=True,
            **params
        )

    def _route_to_estimator(self, method_name, X_container, y=None, **params):
        check_is_fitted(self)
        X, params = self._unwrap_from_container(X_container, params)
        if y is not None:
            X, y, sample_domain = check_X_y_domain(X, y, sample_domain=params.get('sample_domain'))
        else:
            X, sample_domain = check_X_domain(X, sample_domain=params.get('sample_domain'))
        source_masks = extract_source_indices(sample_domain)
        params['sample_domain'] = sample_domain
        output_X, output_params = None, {}
        for domain_estimator, domain_masks in [
            (self.estimators_['source'], source_masks),
            (self.estimators_['target'], ~source_masks),
        ]:
            if domain_masks.sum() == 0:
                # if domain type is not present, just skip
                continue
            X_domain, y_domain, params_domain = _apply_domain_masks(X, y, params, masks=domain_masks)
            request = getattr(get_routing_for_object(domain_estimator), method_name)
            X_domain, routed_params = self._merge_and_route_params(request, X_domain, params_domain)
            method = getattr(domain_estimator, method_name)
            if y_domain is None or method_name == 'transform':
                domain_output = method(X_domain, **routed_params)
            else:
                domain_output = method(X_domain, y_domain, **routed_params)
            # xxx(okachaiev): I would imagine a proper merging procedure should go to the utils
            domain_output_X = domain_output.X if isinstance(domain_output, AdaptationOutput) else domain_output
            if output_X is None:
                output_X = np.zeros((X.shape[0], *domain_output_X.shape[1:]), dtype=domain_output_X.dtype)
            output_X[domain_masks] = domain_output_X
            if isinstance(domain_output, AdaptationOutput):
                for param_name in domain_output:
                    param_value = domain_output[param_name]
                    if param_name not in output_params:
                        output_params[param_name] = np.zeros(
                            (X.shape[0], *param_value.shape[1:]),
                            dtype=param_value.dtype
                        )
                    output_params[param_name][domain_masks] = param_value
        if not output_params:
            # no additional parameters were generated by downstream estimators
            return output_X
        elif self._is_final:
            # the selector is a final step, no need to track params
            return output_X
        else:
            return AdaptationOutput(output_X, **output_params)

    @available_if(_estimator_has('transform', base_attr_name='source_estimator'))
    def transform(self, X_container, **params):
        # xxx(okachaiev): code duplication
        output = self._route_to_estimator('transform', X_container, **params)
        if not self._is_final and isinstance(X_container, AdaptationOutput):
            output = X_container.merge_into(output)
        return output

    # xxx(okachaiev): i guess this should return 'True' only if both
    # estimators have the same method. though practical advantage is,
    # surely, questionable
    @available_if(_estimator_has('predict_proba', base_attr_name='source_estimator'))
    def predict_proba(self, X, **params):
        return self._route_to_estimator('predict_proba', X, **params)

    @available_if(_estimator_has('predict_log_proba', base_attr_name='source_estimator'))
    def predict_log_proba(self, X, **params):
        return self._route_to_estimator('predict_log_proba', X, **params)

    @available_if(_estimator_has('decision_function', base_attr_name='source_estimator'))
    def decision_function(self, X, **params):
        return self._route_to_estimator('decision_function', X, **params)

    @available_if(_estimator_has('score', base_attr_name='source_estimator'))
    def score(self, X, y, **params):
        return self._route_to_estimator('score', X, y=y, **params)
