# Author: Theo Gnassounou <theo.gnassounou@inria.fr>
#         Remi Flamary <remi.flamary@polytechnique.edu>
#         Oleksii Kachaiev <kachayev@gmail.com>
#
# License: BSD 3-Clause

from abc import abstractmethod
from typing import Union

import numpy as np
from sklearn.base import BaseEstimator, clone
from sklearn.utils import Bunch
from sklearn.utils.metadata_routing import get_routing_for_object
from sklearn.utils.metaestimators import available_if
from sklearn.utils.validation import check_is_fitted

from skada.utils import check_X_domain
from skada._utils import _DEFAULT_MASKED_TARGET_CLASSIFICATION_LABEL, _find_y_type


def _estimator_has(attr):
    """Check if we can delegate a method to the underlying estimator.

    First, we check the first fitted classifier if available, otherwise we
    check the unfitted classifier.
    """
    def has_base_estimator(estimator) -> bool:
        return hasattr(estimator, "base_estimator") and hasattr(
            estimator.base_estimator,
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


class AdaptationOutput(Bunch):
    pass


class BaseAdapter(BaseEstimator):

    __metadata_request__fit = {'sample_domain': True}
    __metadata_request__transform = {'sample_domain': True, 'allow_source': True}

    @abstractmethod
    def adapt(
        self,
        X,
        y=None,
        sample_domain=None,
        **params
    ) -> Union[np.ndarray, AdaptationOutput]:
        """Transform samples, labels, and weights into the space in which
        the estimator is trained.
        """

    @abstractmethod
    def fit(self, X, y=None, sample_domain=None, *, sample_weight=None):
        """Fit adaptation parameters"""

    def fit_transform(self, X, y=None, sample_domain=None, **params):
        """
        Fit to data, then transform it.
        In this case, the fitting and the transformation are performed on
        the target and source domains by default (allow_source=True).

        It should be used only for fitting the estimator, and not for
        generating the adaptation output.
        For the latter, use the `transform` method.
        """
        self.fit(X, y=y, sample_domain=sample_domain, **params)
        # assume 'fit_transform' is called to fit the estimator,
        # thus we allow for the source domain to be adapted
        return self.transform(
            X,
            y=y,
            sample_domain=sample_domain,
            allow_source=True,
            **params
        )

    def transform(
        self,
        X,
        y=None,
        sample_domain=None,
        allow_source=False,
        **params
    ) -> Union[np.ndarray, AdaptationOutput]:
        check_is_fitted(self)
        X, sample_domain = check_X_domain(
            X,
            sample_domain=sample_domain,
            allow_auto_sample_domain=True,
            allow_source=allow_source,
        )
        return self.adapt(
            X,
            y=y,
            sample_domain=sample_domain,
            **params
        )


class BaseSelector(BaseEstimator):

    def __init__(self, base_estimator: BaseEstimator, **kwargs):
        super().__init__()
        self.base_estimator = base_estimator
        self.base_estimator.set_params(**kwargs)
        self._is_final = False

    # xxx(okachaiev): should this be a metadata routing object instead of request?
    def get_metadata_routing(self):
        request = get_routing_for_object(self.base_estimator)
        request.fit.add_request(param='sample_domain', alias=True)
        request.transform.add_request(param='sample_domain', alias=True)
        request.predict.add_request(param='sample_domain', alias=True)
        if hasattr(self.base_estimator, 'predict_proba'):
            request.predict_proba.add_request(param='sample_domain', alias=True)
        if hasattr(self.base_estimator, 'predict_log_proba'):
            request.predict_log_proba.add_request(param='sample_domain', alias=True)
        if hasattr(self.base_estimator, 'decision_function'):
            request.decision_function.add_request(param='sample_domain', alias=True)
        if hasattr(self.base_estimator, 'score'):
            request.score.add_request(param='sample_domain', alias=True)
        return request

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

    @abstractmethod
    def _route_to_estimator(self, method_name, X, y=None, **params) -> np.ndarray:
        """Abstract method for calling method of a base estimator based on
        the input and the routing logic associated with domain labels.
        """

    @available_if(_estimator_has('transform'))
    def transform(self, X, **params):
        return self._route_to_estimator('transform', X, **params)

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

    def _mark_as_final(self) -> 'BaseSelector':
        """Internal API for keeping track of which estimator is final
        in the Pipeline.
        """
        self._is_final = True
        return self

    def _remove_masked(self, X, y, routed_params):
        """Internal API for removing masked samples before passing them
        to the final estimator. Only applicable for the final estimator
        within the Pipeline.
        Exception: if the final estimator has a transform method, we don't
        need to do anything.
        """
        # If the estimator is not final, we don't need to do anything
        # If the estimator has a transform method, we don't need to do anything
        if not self._is_final or hasattr(self, 'transform'):
            return X, y, routed_params

        # in case the estimator is marked as final in the pipeline,
        # the selector is responsible for removing masked labels
        # from the targets
        y_type = _find_y_type(y)
        if y_type == 'classification':
            unmasked_idx = (y != _DEFAULT_MASKED_TARGET_CLASSIFICATION_LABEL)
        elif y_type == 'continuous':
            unmasked_idx = np.isfinite(y)

        X = X[unmasked_idx]
        y = y[unmasked_idx]
        routed_params = {
            # this is somewhat crude way to test is `v` is indexable
            k: v[unmasked_idx] if (
                hasattr(v, "__len__") and len(v) > len(unmasked_idx)
                ) else v
            for k, v
            in routed_params.items()
        }
        return X, y, routed_params


class Shared(BaseSelector):

    def get_estimator(self) -> BaseEstimator:
        """Provides access to the fitted estimator."""
        check_is_fitted(self)
        return self.base_estimator_

    def fit(self, X, y, **params):
        # xxx(okachaiev): this code is awkward, and it's duplicated everywhere
        routing = get_routing_for_object(self.base_estimator)
        routed_params = routing.fit._route_params(params=params)
        # xxx(okachaiev): code duplication
        if isinstance(X, AdaptationOutput):
            for k, v in X.items():
                if k != 'X' and k in routed_params:
                    routed_params[k] = v
            X = X['X']
        X, y, routed_params = self._remove_masked(X, y, routed_params)
        estimator = clone(self.base_estimator)
        estimator.fit(X, y, **routed_params)
        self.base_estimator_ = estimator
        self.routing_ = get_routing_for_object(estimator)
        return self

    # xxx(okachaiev): check if underlying estimator supports 'fit_transform'
    def fit_transform(self, X, y=None, **params):
        self.fit(X, y, **params)
        routed_params = self.routing_.fit_transform._route_params(params=params)
        # 'fit_transform' allows transformation for source domains
        # as well, that's why it calls 'adapt' directly
        if isinstance(self.base_estimator_, BaseAdapter):
            # xxx(okachaiev): adapt should take 'y' as well, as in many cases
            # we need to bound estimator fitting to a sub-group of the input
            output = self.base_estimator_.adapt(X, **routed_params)
        else:
            output = self.base_estimator_.transform(X, **routed_params)
        return output

    # xxx(okachaiev): fail if unknown domain is given
    def _route_to_estimator(self, method_name, X, y=None, **params):
        check_is_fitted(self)
        routed_params = getattr(self.routing_, method_name)._route_params(params=params)
        if isinstance(X, AdaptationOutput):
            for k, v in X.items():
                if k != 'X' and k in routed_params:
                    routed_params[k] = v
            X = X['X']
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

    def fit(self, X, y, **params):
        # xxx(okachaiev): use check_*_domain to derive default domain labels
        sample_domain = params['sample_domain']
        # xxx(okachaiev): this code is awkward, and it's duplicated everywhere
        routing = get_routing_for_object(self.base_estimator)
        routed_params = routing.fit._route_params(params=params)
        # xxx(okachaiev): code duplication
        if isinstance(X, AdaptationOutput):
            for k, v in X.items():
                if k != 'X' and k in routed_params:
                    routed_params[k] = v
            X = X['X']
        X, y, routed_params = self._remove_masked(X, y, routed_params)
        estimators = {}
        # xxx(okachaiev): maybe return_index?
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
        routed_params = getattr(self.routing_, method_name)._route_params(params=params)
        # xxx(okachaiev): again, code duplication
        if isinstance(X, AdaptationOutput):
            for k, v in X.items():
                if k != 'X' and k in routed_params:
                    routed_params[k] = v
            X = X['X']
        # xxx(okachaiev): use check_*_domain to derive default domain labels
        sample_domain = params['sample_domain']
        output = None
        # xxx(okachaiev): maybe return_index?
        for domain_label in np.unique(sample_domain):
            # xxx(okachaiev): fail if unknown domain is given
            method = getattr(self.estimators_[domain_label], method_name)
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
