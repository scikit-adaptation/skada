# Author: Theo Gnassounou <theo.gnassounou@inria.fr>
#         Oleksii Kachaiev <kachayev@gmail.com>
#         Ambroise Odonnat <ambroiseodonnattechnologie@gmail.com>
# License: BSD 3-Clause
import pytest

torch = pytest.importorskip("torch")

import numpy as np
from torch.nn import BCELoss, CrossEntropyLoss

from skada.datasets import make_shifted_datasets
from skada.deep import CDAN, DANN, MDD
from skada.deep.modules import DomainClassifier, ToyModule2D


@pytest.mark.parametrize(
    "domain_classifier, domain_criterion, num_features",
    [
        (DomainClassifier(num_features=10), BCELoss(), None),
        (DomainClassifier(num_features=10), None, None),
        (None, None, 10),
        (None, BCELoss(), 10),
    ],
)
def test_dann(domain_classifier, domain_criterion, num_features):
    module = ToyModule2D()
    module.eval()

    n_samples = 20
    dataset = make_shifted_datasets(
        n_samples_source=n_samples,
        n_samples_target=n_samples,
        shift="concept_drift",
        noise=0.1,
        random_state=42,
        return_dataset=True,
    )
    method = DANN(
        ToyModule2D(),
        reg=1,
        domain_classifier=domain_classifier,
        num_features=num_features,
        domain_criterion=domain_criterion,
        layer_name="dropout",
        batch_size=10,
        max_epochs=10,
        train_split=None,
    )

    X, y, sample_domain = dataset.pack_train(as_sources=["s"], as_targets=["t"])
    method.fit(X.astype(np.float32), y, sample_domain)

    X_test, y_test, sample_domain_test = dataset.pack_test(as_targets=["t"])

    y_pred = method.predict(X_test.astype(np.float32), sample_domain_test)

    assert y_pred.shape[0] == X_test.shape[0]

    history = method.history_

    assert history[0]["train_loss"] > history[-1]["train_loss"]


@pytest.mark.parametrize(
    "domain_classifier, domain_criterion, num_feature, max_feature, n_classes",
    [
        (DomainClassifier(num_features=20), BCELoss(), None, 4096, 2),
        (DomainClassifier(num_features=20), None, None, 4096, 2),
        (None, None, 10, 4096, 2),
        (None, BCELoss(), 10, 4096, 2),
        (None, BCELoss(), 10, 10, 2),
    ],
)
def test_cdan(domain_classifier, domain_criterion, num_feature, max_feature, n_classes):
    module = ToyModule2D()
    module.eval()

    n_samples = 20
    dataset = make_shifted_datasets(
        n_samples_source=n_samples,
        n_samples_target=n_samples,
        shift="concept_drift",
        noise=0.1,
        random_state=42,
        return_dataset=True,
    )

    method = CDAN(
        ToyModule2D(),
        reg=1,
        domain_classifier=domain_classifier,
        num_features=num_feature,
        n_classes=n_classes,
        domain_criterion=domain_criterion,
        max_features=max_feature,
        layer_name="dropout",
        batch_size=10,
        max_epochs=10,
        train_split=None,
    )

    X, y, sample_domain = dataset.pack_train(as_sources=["s"], as_targets=["t"])
    method.fit(X.astype(np.float32), y, sample_domain)

    X_test, y_test, sample_domain_test = dataset.pack_test(as_targets=["t"])

    y_pred = method.predict(X_test.astype(np.float32), sample_domain_test)

    assert y_pred.shape[0] == X_test.shape[0]

    history = method.history_

    assert history[0]["train_loss"] > history[-1]["train_loss"]


@pytest.mark.parametrize(
    "domain_classifier, criterion, num_features",
    [
        (DomainClassifier(num_features=10), CrossEntropyLoss(), None),
        (DomainClassifier(num_features=10), None, None),
        (None, None, 10),
    ],
)
def test_mdd(domain_classifier, criterion, num_features):
    module = ToyModule2D()
    module.eval()

    n_samples = 20
    dataset = make_shifted_datasets(
        n_samples_source=n_samples,
        n_samples_target=n_samples,
        shift="concept_drift",
        noise=0.1,
        random_state=42,
        return_dataset=True,
    )

    method = MDD(
        ToyModule2D(),
        reg=1,
        gamma=4.0,
        domain_classifier=domain_classifier,
        num_features=num_features,
        criterion=criterion,
        layer_name="dropout",
        batch_size=10,
        max_epochs=10,
        train_split=None,
    )

    X, y, sample_domain = dataset.pack_train(as_sources=["s"], as_targets=["t"])
    method.fit(X.astype(np.float32), y, sample_domain)

    X_test, y_test, sample_domain_test = dataset.pack_test(as_targets=["t"])

    y_pred = method.predict(X_test.astype(np.float32), sample_domain_test)

    assert y_pred.shape[0] == X_test.shape[0]

    history = method.history_

    assert history[0]["train_loss"] > history[-1]["train_loss"]


def test_missing_num_features():
    with pytest.raises(ValueError):
        DANN(
            ToyModule2D(),
            reg=1,
            domain_classifier=None,
            num_features=None,
            domain_criterion=BCELoss(),
            layer_name="dropout",
            batch_size=10,
            max_epochs=10,
            train_split=None,
        )

    with pytest.raises(ValueError):
        CDAN(
            ToyModule2D(),
            reg=1,
            domain_classifier=None,
            num_features=None,
            domain_criterion=BCELoss(),
            layer_name="dropout",
            batch_size=10,
            max_epochs=10,
            train_split=None,
        )


def test_missing_n_classes():
    with pytest.raises(ValueError):
        CDAN(
            ToyModule2D(),
            reg=1,
            domain_classifier=None,
            num_features=10,
            n_classes=None,
            domain_criterion=BCELoss(),
            layer_name="dropout",
            batch_size=10,
            max_epochs=10,
            train_split=None,
        )


def test_return_features():
    num_features = 10
    n_classes = 2
    module = ToyModule2D(num_features=num_features, n_classes=n_classes)
    module.eval()

    n_samples = 20
    dataset = make_shifted_datasets(
        n_samples_source=n_samples,
        n_samples_target=n_samples,
        shift="concept_drift",
        noise=0.1,
        random_state=42,
        return_dataset=True,
        label="multiclass",
    )

    method = CDAN(
        ToyModule2D(),
        reg=1,
        domain_classifier=None,
        num_features=num_features,
        n_classes=n_classes,
        domain_criterion=BCELoss(),
        layer_name="dropout",
        batch_size=10,
        max_epochs=10,
        train_split=None,
    )

    X_test, _, _ = dataset.pack_test(as_targets=["t"])
    X_test = X_test.astype(np.float32)

    # without dict
    method.initialize()
    features = method.predict_features(torch.tensor(X_test))
    assert features.shape[1] == num_features
    assert features.shape[0] == X_test.shape[0]
