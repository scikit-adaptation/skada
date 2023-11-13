from abc import abstractmethod
from typing import List, Tuple, Union

import numpy as np
from sklearn.base import BaseEstimator, clone
from sklearn.utils import Bunch
from sklearn.utils.metadata_routing import get_routing_for_object
from sklearn.utils.metaestimators import available_if
from sklearn.utils.validation import check_is_fitted


# xxx(okachaiev): this should be `skada.utils.check_X_y_domain`
# rather than `skada._utils.check_X_y_domain`
from ._utils import check_X_domain


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
        pass

    @abstractmethod
    def fit(self, X, y=None, sample_domain=None, *, sample_weight=None):
        """Fit adaptation parameters"""
        pass

    def fit_transform(self, X, y=None, sample_domain=None, **params):
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

    # xxx(okachaiev): this is wrong, it should take routing information from
    #                 for downstream estimators rather than declaring on its own
    __metadata_request__fit = {'sample_domain': True}
    __metadata_request__transform = {'sample_domain': True}
    __metadata_request__predict = {'sample_domain': True}
    __metadata_request__predict_proba = {'sample_domain': True}
    __metadata_request__predict_log_proba = {'sample_domain': True}
    __metadata_request__decision_function = {'sample_domain': True}
    __metadata_request__score = {'sample_domain': True}

    @abstractmethod
    def select(
        self,
        sample_domain: np.ndarray
    ) -> List[Tuple[BaseEstimator, np.ndarray]]:
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


# xxx(okachaiev): the default flow for this selector would look
# like the following:
# * fit: adapter takes source & target, transforms source
# * fit: estimator takes transformed source
# * predict: adapter takes target and transforms it, when necessary
# * predict: estimator works with whatever it got from the adapter
#
# a few notes:
# 1) for per-domain that would look very differently
# 2) semi-supervised learning would require us to transform
#    both source and target for fitting
# 3) it still feels valuable to have ability to use the
#    estimator for source data (specifically in the case of
#    learning latent space in the adaptation phase). allow_source
#    flog seems somewhat fragile from that perspective
class Shared(BaseSelector):

    def __init__(self, base_estimator: BaseEstimator):
        super().__init__()
        self.base_estimator = base_estimator

    # xxx(okachaiev): should this be a metadata routing object instead of request?
    def get_metadata_routing(self):
        request = get_routing_for_object(self.base_estimator)
        request.fit.add_request(param='sample_domain', alias=True)
        request.transform.add_request(param='sample_domain', alias=True)
        request.predict.add_request(param='sample_domain', alias=True)
        if hasattr(self.base_estimator, 'predict_proba'):
            request.predict_proba.add_request(param='sample_domain', alias=True)
        if hasattr(self.base_estimator, 'score'):
            request.score.add_request(param='sample_domain', alias=True)
        return request

    # xxx(okachaiev): check if X is `AdapterOutput` class to update routing params
    def fit(self, X, y, **params):
        if 'sample_domain' in params:
            domains = set(np.unique(params['sample_domain']))
        else:
            domains = set([1, -2])  # default source and target labels
        # xxx(okachaiev): this code is awkward, and it's duplicated everywhere
        routing = get_routing_for_object(self.base_estimator)
        routed_params = routing.fit._route_params(params=params)
        # xxx(okachaiev): this should be done in each method
        if isinstance(X, AdaptationOutput):
            for k, v in X.items():
                if k != 'X' and k in routed_params:
                    routed_params[k] = v
            X = X['X']
        estimator = clone(self.base_estimator)
        estimator.fit(X, y, **routed_params)
        self.base_estimator_ = estimator
        self.domains_ = domains
        self.routing_ = get_routing_for_object(self.base_estimator)
        return self

    # xxx(okachaiev): fail if sources are given
    # xxx(okachaiev): fail if unknown domain is given
    # xxx(okachaiev): only defined when underlying estimator supports transform
    def transform(self, X, **params):
        check_is_fitted(self)
        routed_params = self.routing_.transform._route_params(params=params)
        output = self.base_estimator_.transform(X, **routed_params)
        return output

    # xxx(okachaiev): check if underlying estimator supports 'fit_transform'
    def fit_transform(self, X, y=None, **params):
        self.fit(X, y, **params)
        routed_params = self.routing_.fit_transform._route_params(params=params)
        # 'fit_transform' allows transformation for source domains
        # as well, that's why it calls 'adapt' directly
        if isinstance(self.base_estimator_, BaseAdapter):
            output = self.base_estimator_.adapt(X, **routed_params)
        else:
            output = self.base_estimator_.transform(X, **routed_params)
        return output

    def predict(self, X, **params):
        check_is_fitted(self)
        routed_params = self.routing_.predict._route_params(params=params)
        # xxx(okachaiev): this should be done in each method
        if isinstance(X, AdaptationOutput):
            for k, v in X.items():
                if k != 'X' and k in routed_params:
                    routed_params[k] = v
            X = X['X']
        output = self.base_estimator_.predict(X, **routed_params)
        return output

    # xxx(okachaiev): code duplication
    @available_if(_estimator_has("predict_proba"))
    def predict_proba(self, X, **params):
        check_is_fitted(self)
        routed_params = self.routing_.predict_proba._route_params(params=params)
        # xxx(okachaiev): this should be done in each method
        if isinstance(X, AdaptationOutput):
            for k, v in X.items():
                if k != 'X' and k in routed_params:
                    routed_params[k] = v
            X = X['X']
        output = self.base_estimator_.predict_proba(X, **routed_params)
        return output

    # xxx(okachaiev): code duplication
    @available_if(_estimator_has("score"))
    def score(self, X, y, **params):
        check_is_fitted(self)
        routed_params = self.routing_.score._route_params(params=params)
        # xxx(okachaiev): this should be done in each method
        if isinstance(X, AdaptationOutput):
            for k, v in X.items():
                if k != 'X' and k in routed_params:
                    routed_params[k] = v
                elif k == 'y':
                    y = X['y']
            X = X['X']
        output = self.base_estimator_.score(X, y, **routed_params)
        return output
