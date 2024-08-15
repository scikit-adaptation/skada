# Author: Theo Gnassounou <theo.gnassounou@inria.fr>
#         Remi Flamary <remi.flamary@polytechnique.edu>
#         Oleksii Kachaiev <kachayev@gmail.com>
#         Yanis Lalou <yanis.lalou@polytechnique.edu>
#
# License: BSD 3-Clause

import warnings
from abc import abstractmethod
from copy import deepcopy

import numpy as np
from sklearn.base import BaseEstimator, clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, check_scoring
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import LabelEncoder, Normalizer
from sklearn.utils import check_random_state
from sklearn.utils.extmath import softmax
from sklearn.utils.metadata_routing import _MetadataRequester, get_routing_for_object

from ._utils import (
    _DEFAULT_MASKED_TARGET_CLASSIFICATION_LABEL,
    _DEFAULT_MASKED_TARGET_REGRESSION_LABEL,
    Y_Type,
    _check_y_masking,
    _find_y_type,
)
from .utils import check_X_y_domain, extract_source_indices, source_target_split


# xxx(okachaiev): maybe it would be easier to reuse _BaseScorer?
# xxx(okachaiev): add proper __repr__/__str__
# xxx(okachaiev): support clone()
class _BaseDomainAwareScorer(_MetadataRequester):
    __metadata_request__score = {"sample_domain": True}

    @abstractmethod
    def _score(self, estimator, X, y, sample_domain=None, **params):
        pass

    def __call__(self, estimator, X, y=None, sample_domain=None, **params):
        return self._score(estimator, X, y, sample_domain=sample_domain, **params)


class SupervisedScorer(_BaseDomainAwareScorer):
    """Compute score on supervised dataset.

    Parameters
    ----------
    scoring : str or callable, default=None
        A string (see model evaluation documentation) or
        a scorer callable object / function with signature
        ``scorer(estimator, X, y)``.
        If None, the provided estimator object's `score` method is used.
    greater_is_better : bool, default=True
        Whether `scorer` is a score function (default), meaning high is
        good, or a loss function, meaning low is good. In the latter case, the
        scorer object will sign-flip the outcome of the `scorer`.
    """

    __metadata_request__score = {"target_labels": True}

    def __init__(self, scoring=None, greater_is_better=True):
        super().__init__()
        self.scoring = scoring
        self._sign = 1 if greater_is_better else -1

    def _score(
        self, estimator, X, y=None, sample_domain=None, target_labels=None, **params
    ):
        scorer = check_scoring(estimator, self.scoring)

        X, y, sample_domain = check_X_y_domain(X, y, sample_domain)
        source_idx = extract_source_indices(sample_domain)

        return self._sign * scorer(
            estimator,
            X[~source_idx],
            target_labels[~source_idx],
            sample_domain=sample_domain[~source_idx],
            **params,
        )


class ImportanceWeightedScorer(_BaseDomainAwareScorer):
    """Score based on source data using sample weight.

    See [17]_ for details.

    Parameters
    ----------
    weight_estimator : estimator object, optional
        The estimator to use to estimate the densities of source and target
        observations. If None, a KernelDensity estimator is with
        default parameters used.
    scoring : str or callable, default=None
        A string (see model evaluation documentation) or
        a scorer callable object / function with signature
        ``scorer(estimator, X, y)``.
        If None, the provided estimator object's `score` method is used.
    greater_is_better : bool, default=True
        Whether `scorer` is a score function (default), meaning high is
        good, or a loss function, meaning low is good. In the latter case, the
        scorer object will sign-flip the outcome of the `scorer`.


    Attributes
    ----------
    weight_estimator_source_ : object
        The estimator object fitted on the source data.
    weight_estimator_target_ : object
        The estimator object fitted on the target data.

    References
    ----------
    .. [17] Masashi Sugiyama et al. Covariate Shift Adaptation
            by Importance Weighted Cross Validation.
            Journal of Machine Learning Research, 2007.
    """

    def __init__(
        self,
        weight_estimator=None,
        scoring=None,
        greater_is_better=True,
    ):
        super().__init__()
        self.weight_estimator = weight_estimator
        self.scoring = scoring
        self._sign = 1 if greater_is_better else -1

    def _fit(self, X_source, X_target):
        """Fit adaptation parameters.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The source data.
        X_target : array-like, shape (n_samples, n_features)
            The target data.

        Returns
        -------
        self : object
            Returns self.
        """
        weight_estimator = self.weight_estimator
        if weight_estimator is None:
            weight_estimator = KernelDensity()
        self.weight_estimator_source_ = clone(weight_estimator)
        self.weight_estimator_target_ = clone(weight_estimator)
        self.weight_estimator_source_.fit(X_source)
        self.weight_estimator_target_.fit(X_target)
        return self

    def _score(self, estimator, X, y, sample_domain=None, **params):
        scorer = check_scoring(estimator, self.scoring)
        if "sample_weight" not in get_routing_for_object(estimator).consumes(
            "score", ["sample_weight"]
        ):
            raise ValueError(
                "The estimator passed should accept 'sample_weight' parameter. "
                f"The estimator {estimator!r} does not."
            )

        X, y, sample_domain = check_X_y_domain(X, y, sample_domain)
        X_source, X_target, y_source, _ = source_target_split(
            X, y, sample_domain=sample_domain
        )
        self._fit(X_source, X_target)
        ws = self.weight_estimator_source_.score_samples(X_source)
        wt = self.weight_estimator_target_.score_samples(X_source)
        weights = np.exp(wt - ws)

        if weights.sum() != 0:
            weights /= weights.sum()
        else:
            warnings.warn("All weights are zero. Using uniform weights.")
            weights = np.ones_like(weights) / len(weights)

        return self._sign * scorer(
            estimator,
            X_source,
            y_source,
            sample_domain=sample_domain[sample_domain >= 0],
            sample_weight=weights,
            allow_source=True,
            **params,
        )


class PredictionEntropyScorer(_BaseDomainAwareScorer):
    """Score based on the entropy of predictions on unsupervised dataset.

    See [18]_ for details.

    Parameters
    ----------
    greater_is_better : bool, default=False
        Whether `scorer` is a score function (default), meaning high is
        good, or a loss function, meaning low is good. In the latter case, the
        scorer object will sign-flip the outcome of the `scorer`.
    reduction: str, default='mean'
        Specifies the reduction to apply to the entropy values.
        Must be one of ['none', 'mean', 'sum'].
        If 'none', the entropy values for each sample are returned ([1]_ method).
        If 'mean', the mean of the entropy values is returned.
        If 'sum', the sum of the entropy values is returned.

    Returns
    -------
    entropy : float or ndarray of floats
        If `reduction` is 'none', then ndarray of shape (n_samples,).
        Otherwise float.

    References
    ----------
    .. [18] Pietro Morerio et al. Minimal-Entropy correlation alignment
            for unsupervised deep domain adaptation.
            ICLR, 2018.
    """

    def __init__(self, greater_is_better=False, reduction="mean"):
        super().__init__()
        self._sign = 1 if greater_is_better else -1
        self.reduction = reduction

        if self.reduction not in ["none", "mean", "sum"]:
            raise ValueError(
                f"Unknown reduction '{self.reduction}'. "
                "Valid options are: 'none', 'mean', 'sum'."
            )

    def _score(self, estimator, X, y, sample_domain=None, **params):
        if not hasattr(estimator, "predict_proba"):
            raise AttributeError(
                "The estimator passed should have a 'predict_proba' method. "
                f"The estimator {estimator!r} does not."
            )
        X, y, sample_domain = check_X_y_domain(X, y, sample_domain)
        source_idx = extract_source_indices(sample_domain)
        proba = estimator.predict_proba(
            X[~source_idx], sample_domain=sample_domain[~source_idx], **params
        )
        if hasattr(estimator, "predict_log_proba"):
            log_proba = estimator.predict_log_proba(
                X[~source_idx], sample_domain=sample_domain[~source_idx], **params
            )
        else:
            log_proba = np.log(proba + 1e-7)
        entropy_per_sample = -proba * log_proba
        if self.reduction == "none":
            return self._sign * entropy_per_sample
        elif self.reduction == "sum":
            return self._sign * np.sum(entropy_per_sample)
        elif self.reduction == "mean":
            return self._sign * np.mean(entropy_per_sample)
        else:
            raise ValueError(
                f"Unknown reduction '{self.reduction}'. "
                "Valid options are: 'none', 'mean', 'sum'."
            )


class SoftNeighborhoodDensity(_BaseDomainAwareScorer):
    """Score based on the entropy of similarity between unsupervised dataset.

    See [19]_ for details.

    Parameters
    ----------
    T :  float
        Temperature in the Eq. 2 in [1]_.
        Default is set to 0.05, the value proposed in the paper.
    greater_is_better : bool, default=True
        Whether `scorer` is a score function (default), meaning high is
        good, or a loss function, meaning low is good. In the latter case, the
        scorer object will sign-flip the outcome of the `scorer`.

    References
    ----------
    .. [19] Kuniaki Saito et al. Tune it the Right Way: Unsupervised Validation
            of Domain Adaptation via Soft Neighborhood Density.
            International Conference on Computer Vision, 2021.
    """

    def __init__(self, T=0.05, greater_is_better=True):
        super().__init__()
        self.T = T
        self._sign = 1 if greater_is_better else -1

    def _score(self, estimator, X, y, sample_domain=None, **params):
        if not hasattr(estimator, "predict_proba"):
            raise AttributeError(
                "The estimator passed should have a 'predict_proba' method. "
                f"The estimator {estimator!r} does not."
            )

        X, y, sample_domain = check_X_y_domain(X, y, sample_domain)
        source_idx = extract_source_indices(sample_domain)
        proba = estimator.predict_proba(
            X[~source_idx], sample_domain=sample_domain[~source_idx], **params
        )
        proba = Normalizer(norm="l2").fit_transform(proba)

        similarity_matrix = proba @ proba.T / self.T
        np.fill_diagonal(similarity_matrix, -np.diag(similarity_matrix))
        similarity_matrix = softmax(similarity_matrix)

        entropy = np.sum(-similarity_matrix * np.log(similarity_matrix), axis=1)
        return self._sign * np.mean(entropy)


class DeepEmbeddedValidation(_BaseDomainAwareScorer):
    """Loss based on source data using features representation to weight samples.

    See [20]_ for details.

    Parameters
    ----------
    domain_classifier : sklearn classifier, optional
        Classifier used to predict the domains. If None, a
        LogisticRegression is used.
    loss_func : callable
        Loss function with signature
        `loss_func(y, y_pred, **kwargs)`.
        The loss function need not to be reduced.
    random_state : int, RandomState instance or None, default=None
        Determines random number generation for train_test_split. Pass an int
        for reproducible output across multiple function calls.
    greater_is_better : bool, default=False
        Whether `scorer` is a score function (default), meaning high is
        good, or a loss function, meaning low is good. In the latter case, the
        scorer object will sign-flip the outcome of the `scorer`.

    References
    ----------
    .. [20] Kaichao You et al. Towards Accurate Model Selection
            in Deep Unsupervised Domain Adaptation.
            ICML, 2019.
    """

    def __init__(
        self,
        domain_classifier=None,
        loss_func=None,
        random_state=None,
        greater_is_better=False,
    ):
        super().__init__()
        self.domain_classifier = domain_classifier
        self._loss_func = (
            loss_func if loss_func is not None else self._no_reduc_log_loss
        )
        self.random_state = random_state
        self._sign = 1 if greater_is_better else -1

    def _no_reduc_log_loss(self, y, y_pred):
        return np.array(
            [self.cross_entropy_loss(y[i], y_pred[i]) for i in range(len(y))]
        )

    def _fit_adapt(self, features, features_target):
        domain_classifier = self.domain_classifier
        if domain_classifier is None:
            domain_classifier = LogisticRegression()
        self.domain_classifier_ = clone(domain_classifier)
        y_domain = np.concatenate((len(features) * [1], len(features_target) * [0]))
        self.domain_classifier_.fit(
            np.concatenate((features, features_target)), y_domain
        )
        return self

    def _score(self, estimator, X, y, sample_domain=None, **kwargs):
        if not hasattr(estimator, "predict_proba"):
            raise AttributeError(
                "The estimator passed should have a 'predict_proba' method. "
                f"The estimator {estimator!r} does not."
            )

        has_transform_method = False

        if not isinstance(estimator, BaseEstimator):
            # The estimator is a deep model
            transformer = estimator.predict_features
            has_transform_method = True
        else:
            # We need to find the last layer of the pipeline with a transform method
            pipeline_steps = list(enumerate(estimator.named_steps.items()))

            for index_transformer, (_, step_process) in reversed(pipeline_steps):
                if hasattr(step_process, "transform"):
                    transformer = estimator[: index_transformer + 1].transform
                    has_transform_method = True
                    break  # Stop after the first occurrence if there are multiple

        def identity(x):
            return x

        if not has_transform_method:
            # We use the input data as features
            transformer = identity

        X, y, sample_domain = check_X_y_domain(X, y, sample_domain)
        source_idx = extract_source_indices(sample_domain)
        rng = check_random_state(self.random_state)
        X_train, X_val, _, y_val, _, sample_domain_val = train_test_split(
            X[source_idx],
            y[source_idx],
            sample_domain[source_idx],
            test_size=0.33,
            random_state=rng,
        )

        features_train = transformer(X_train)
        features_val = transformer(X_val)
        features_target = transformer(X[~source_idx])

        # 2 cases:
        # - features_train is a numpy array --> Do nothing
        # - features_train is a torch.Tensor --> call detach().numpy()
        if not isinstance(features_train, np.ndarray):
            # The transformer comes from a deep model
            # and returns a torch.Tensor
            features_train = features_train.detach().numpy()
            features_val = features_val.detach().numpy()
            features_target = features_target.detach().numpy()

        self._fit_adapt(features_train, features_target)
        N_train, N_target = len(features_train), len(features_target)
        domain_pred = self.domain_classifier_.predict_proba(features_val)
        weights = (N_train / N_target) * domain_pred[:, :1] / domain_pred[:, 1:]
        if not isinstance(estimator, BaseEstimator):
            # Deep estimators dont accept allow_source parameter
            y_pred = estimator.predict_proba(X_val, sample_domain_val)
        else:
            y_pred = estimator.predict_proba(
                X_val, sample_domain=sample_domain_val, allow_source=True
            )

        error = self._loss_func(y_val, y_pred)
        assert weights.shape[0] == error.shape[0]

        weighted_error = weights * error
        weights_m = np.concatenate((weighted_error, weights), axis=1)
        cov = np.cov(weights_m, rowvar=False)[0, 1]
        var_w = np.var(weights, ddof=1)
        if var_w == 0:
            # If var_w == 0, we set eta to 0
            eta = 0
        else:
            eta = -cov / var_w
        return self._sign * (np.mean(weighted_error) + eta * np.mean(weights) - eta)

    def cross_entropy_loss(self, y_true, y_pred, epsilon=1e-15):
        """
        Compute cross-entropy loss for a single sample between the true label
        and the predicted probability estimates.

        This loss allows for a changing number of classes over the validation process.

        Parameters
        ----------
        - y_true: int
            True label (integer label).
        - y_pred: array-like
            Predicted probabilities for each class.
        - epsilon: float, optional (default=1e-15)
            A small constant to avoid numerical instability.

        Returns
        -------
        - float
            Cross-entropy loss for the single sample.
        """
        num_classes = y_pred.shape[0]

        # Convert integer labels to one-hot encoding
        y_true_one_hot = np.eye(num_classes)[y_true]

        # Clip predicted probabilities to avoid log(0) or log(1)
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

        cross_entropy = -np.sum(y_true_one_hot * np.log(y_pred))
        return cross_entropy


class CircularValidation(_BaseDomainAwareScorer):
    """Score based on a circular validation strategy.

    This scorer retrain the estimator, with the exact same parameters,
    on the predicted target domain samples. Then, the retrained estimator
    is used to predict the source domain labels. The score is then
    computed using the source_scorer between the true source labels and
    the predicted source labels.

    See [11]_ for details.

    Parameters
    ----------
    source_scorer : callable, default = `sklearn.metrics.balanced_accuracy_score`
        Scorer used on the source domain samples.
        It should be a callable of the form `source_scorer(y, y_pred)`.
    greater_is_better : bool, default=False
        Whether `scorer` is a score function (default), meaning high is
        good, or a loss function, meaning low is good. In the latter case, the
        scorer object will sign-flip the outcome of the `scorer`.

    References
    ----------
    .. [11] Bruzzone, L., & Marconcini, M. 'Domain adaptation problems: A DASVM
            classification technique and a circular validation strategy.'
            IEEE transactions on pattern analysis and machine intelligence, (2009).
    """

    def __init__(
        self,
        source_scorer=balanced_accuracy_score,
        greater_is_better=True,
    ):
        super().__init__()
        if not callable(source_scorer):
            raise ValueError(
                "The source scorer should be a callable. "
                f"The scorer {source_scorer} is not."
            )

        self.source_scorer = source_scorer
        self._sign = 1 if greater_is_better else -1

    def _score(self, estimator, X, y, sample_domain=None):
        """
        Compute the score based on a circular validation strategy.

        Parameters
        ----------
        estimator : object
            A trained estimator.
        X : array-like or sparse matrix
            The input samples.
        y : array-like
            The true labels.
        sample_domain : array-like, default=None
            Domain labels for each sample.

        Returns
        -------
        float
            The computed score.
        """
        X, y, sample_domain = check_X_y_domain(X, y, sample_domain)

        try:
            _check_y_masking(y)
        except ValueError:
            raise ValueError(
                "The labels should be masked before calling the scorer. "
                "CircularValidation doesn't support semi-supervised DA"
            )

        source_idx = extract_source_indices(sample_domain)

        # TODO: Check if skorch works with deepcopy/clone
        try:
            backward_estimator = deepcopy(estimator)
        except (TypeError, AttributeError):
            backward_estimator = clone(estimator)

        y_pred_target = estimator.predict(
            X[~source_idx], sample_domain=sample_domain[~source_idx]
        )

        if len(np.unique(y_pred_target)) == 1:
            # Otherwise, we can get ValueError exceptions
            # when fitting the backward estimator
            # (happened with SVC)
            warnings.warn("The predicted target domain labels" "are all the same. ")

            # Here we assume that the backward_estimator trained on
            # the target domain will predict the same label for all
            # the source domain samples
            score = self.source_scorer(
                y[source_idx], np.ones_like(y[source_idx]) * y_pred_target[0]
            )
        else:
            y_type = _find_y_type(y[source_idx])

            if y_type == Y_Type.DISCRETE:
                # We need to re-encode the target labels
                # since some estimator like XGBoost
                # only supports labels in [0, num_classes-1]
                # and y_pred_target may not be in this range
                le = LabelEncoder()
                le.fit(y_pred_target)
                y_pred_target = le.transform(y_pred_target)

            backward_sample_domain = -sample_domain

            backward_y = np.zeros_like(y)
            backward_y[~source_idx] = y_pred_target

            if y_type == Y_Type.DISCRETE:
                backward_y[source_idx] = _DEFAULT_MASKED_TARGET_CLASSIFICATION_LABEL
            else:
                backward_y[source_idx] = _DEFAULT_MASKED_TARGET_REGRESSION_LABEL

            backward_estimator.fit(X, backward_y, sample_domain=backward_sample_domain)
            y_pred_source = backward_estimator.predict(
                X[source_idx], sample_domain=backward_sample_domain[source_idx]
            )

            if y_type == Y_Type.DISCRETE:
                # We go back to the original labels
                y_pred_source = le.inverse_transform(y_pred_source)

            # We can now compute the score
            score = self.source_scorer(y[source_idx], y_pred_source)

        return self._sign * score


class MixValScorer(_BaseDomainAwareScorer):
    """
    MixVal scorer for unsupervised domain adaptation.

    This scorer uses mixup to create mixed samples from the target domain,
    and evaluates the model's consistency on these mixed samples.

    See [32]_ for details.

    Parameters
    ----------
    alpha : float, default=0.55
        Mixing parameter for mixup.
    ice_type : {'both', 'intra', 'inter'}, default='both'
        Type of ICE score to compute:
        - 'both': Compute both intra-cluster and inter-cluster ICE scores (average).
        - 'intra': Compute only intra-cluster ICE score.
        - 'inter': Compute only inter-cluster ICE score.
    scoring : str or callable, default=None
        A string (see model evaluation documentation) or
        a scorer callable object / function with signature
        ``scorer(estimator, X, y)``.
        If None, the provided estimator object's `score` method is used.
    greater_is_better : bool, default=True
        Whether higher scores are better.
    random_state : int, RandomState instance or None, default=None
        Controls the randomness of the mixing process.

    Attributes
    ----------
    alpha : float
        Mixing parameter.
    random_state : RandomState
        Random number generator.
    _sign : int
        1 if greater_is_better is True, -1 otherwise.
    ice_type : str
        Type of ICE score to compute.

    References
    ----------
    .. [32] Dapeng Hu et al. Mixed Samples as Probes for Unsupervised Model
            Selection in Domain Adaptation.
            NeurIPS, 2023.
    """

    def __init__(
        self,
        alpha=0.55,
        ice_type="both",
        scoring=None,
        greater_is_better=True,
        random_state=None,
    ):
        super().__init__()
        self.alpha = alpha
        self.ice_type = ice_type
        self.scoring = scoring
        self._sign = 1 if greater_is_better else -1
        self.random_state = random_state

        if self.ice_type not in ["both", "intra", "inter"]:
            raise ValueError("ice_type must be 'both', 'intra', or 'inter'")

    def _score(self, estimator, X, y=None, sample_domain=None, **params):
        """
        Compute the Interpolation Consistency Evaluation (ICE) score.

        Parameters
        ----------
        estimator : object
            The fitted estimator to evaluate.
        X : array-like of shape (n_samples, n_features)
            The input samples.
        y : Ignored
            Not used, present for API consistency by convention.
        sample_domain : array-like, default=None
            Domain labels for each sample.

        Returns
        -------
        score : float
            The ICE score.
        """
        scorer = check_scoring(estimator, self.scoring)

        X, _, sample_domain = check_X_y_domain(X, y, sample_domain)
        source_idx = extract_source_indices(sample_domain)
        X_target = X[~source_idx]

        # Check from y values if it is a classification problem
        y_type = _find_y_type(y)
        if y_type != Y_Type.DISCRETE:
            raise ValueError("MixVal scorer only supports classification problems.")

        rng = check_random_state(self.random_state)
        rand_idx = rng.permutation(X_target.shape[0])

        # Get predictions for target samples
        labels_a = estimator.predict(X_target, sample_domain=sample_domain[~source_idx])
        labels_b = labels_a[rand_idx]

        # Intra-cluster and inter-cluster mixup
        same_idx = (labels_a == labels_b).nonzero()[0]
        diff_idx = (labels_a != labels_b).nonzero()[0]

        # Mixup with X_target and hard pseudo labels
        mix_inputs = self.alpha * X_target + (1 - self.alpha) * X_target[rand_idx]
        if self.alpha > 0.5:
            mix_labels = labels_a
        else:
            mix_labels = labels_b

        # Calculate ICE scores based on ice_type
        if self.ice_type in ["both", "intra"]:
            ice_same = scorer(
                estimator,
                mix_inputs[same_idx],
                mix_labels[same_idx],
                sample_domain=sample_domain[~source_idx][same_idx],
            )

        if self.ice_type in ["both", "inter"]:
            ice_diff = scorer(
                estimator,
                mix_inputs[diff_idx],
                mix_labels[diff_idx],
                sample_domain=sample_domain[~source_idx][diff_idx],
            )

        if self.ice_type == "both":
            ice_score = (ice_same + ice_diff) / 2
        elif self.ice_type == "intra":
            ice_score = ice_same
        else:  # self.ice_type == 'inter'
            ice_score = ice_diff

        return self._sign * ice_score
