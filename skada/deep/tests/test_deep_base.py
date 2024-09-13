# Author: Theo Gnassounou <theo.gnassounou@inria.fr>
#         Oleksii Kachaiev <kachayev@gmail.com>
#         Yanis Lalou <yanis.lalou@polytechnique.edu>
#         Antoine Collas <contact@antoinecollas.fr>
#
# License: BSD 3-Clause
import pytest

torch = pytest.importorskip("torch")

import numpy as np
from skorch.dataset import Dataset

from skada.datasets import make_shifted_datasets
from skada.deep.base import (
    BaseDALoss,
    DomainAwareCriterion,
    DomainAwareModule,
    DomainAwareNet,
    DomainBalancedDataLoader,
    DomainBalancedSampler,
)
from skada.deep.losses import TestLoss
from skada.deep.modules import ToyModule2D


def test_domainawaremodule_features_differ_between_domains():
    num_features = 10
    module = ToyModule2D(num_features=num_features)
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

    # Prepare data
    X, y, sample_domain = dataset.pack_train(as_sources=["s"], as_targets=["t"])
    X = X.astype(np.float32)
    sample_domain = np.array(sample_domain)

    # Convert to torch tensors
    X_tensor = torch.tensor(X)
    sample_domain_tensor = torch.tensor(sample_domain)

    # Create an instance of DomainAwareModule
    domain_module = DomainAwareModule(module, layer_name="dropout")

    # Run forward pass
    with torch.no_grad():
        output = domain_module(
            X_tensor,
            sample_domain=sample_domain_tensor,
            is_fit=True,
            return_features=True,
        )

    # Unpack output
    y_pred, domain_pred, features, sample_domain_output = output

    # Separate features for source and target domains
    source_mask = sample_domain_tensor >= 0
    target_mask = sample_domain_tensor < 0
    features_s = features[source_mask]
    features_t = features[target_mask]

    # Ensure we have features from both domains
    assert features_s.size(0) > 0, "No source domain features extracted."
    assert features_t.size(0) > 0, "No target domain features extracted."

    # Compute mean features for source and target
    mean_features_s = features_s.mean(dim=0)
    mean_features_t = features_t.mean(dim=0)

    # Check that the mean features are different
    difference = torch.abs(mean_features_s - mean_features_t)
    max_difference = difference.max().item()

    assert max_difference > 0.1, "Features of source and target domains are too similar."


def test_domainawaretraining():
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
    method = DomainAwareNet(
        DomainAwareModule(module, "dropout"),
        iterator_train=DomainBalancedDataLoader,
        criterion=DomainAwareCriterion(torch.nn.CrossEntropyLoss(), TestLoss()),
        batch_size=10,
        max_epochs=2,
        train_split=None,
    )

    X, y, sample_domain = dataset.pack_train(as_sources=["s"], as_targets=["t"])
    X_test, y_test, sample_domain_test = dataset.pack_test(as_targets=["t"])
    X = X.astype(np.float32)
    X_test = X_test.astype(np.float32)

    # without dict
    method.fit(X, y, sample_domain=sample_domain)

    y_pred = method.predict(X_test, sample_domain_test)
    _ = method.predict_proba(X, sample_domain)
    method.score(X_test, y_test, sample_domain_test)

    assert y_pred.shape[0] == X_test.shape[0]

    # with dict
    X_dict = {"X": X, "sample_domain": sample_domain}
    method.fit(
        X_dict,
        y,
    )

    X_dict_test = {"X": X_test, "sample_domain": sample_domain_test}

    y_pred = method.predict(X_dict_test)
    _ = method.predict_proba(X_dict)
    method.score(X_dict_test, y_test)

    assert y_pred.shape[0] == X_test.shape[0]

    # numpy input
    method.fit(X, y, sample_domain)
    y_pred = method.predict(X_test, sample_domain_test)
    _ = method.predict_proba(X, sample_domain)
    method.score(X_test, y_test, sample_domain_test)

    assert y_pred.shape[0] == X_test.shape[0]

    # tensor input
    method.fit(torch.tensor(X), torch.tensor(y), torch.tensor(sample_domain))
    y_pred = method.predict(torch.tensor(X_test), torch.tensor(sample_domain_test))
    _ = method.predict_proba(torch.tensor(X), torch.tensor(sample_domain))
    method.score(
        torch.tensor(X_test), torch.tensor(y_test), torch.tensor(sample_domain_test)
    )

    assert y_pred.shape[0] == X_test.shape[0]

    # dataset input
    X_dict = {"X": X, "sample_domain": sample_domain}

    torch_dataset = Dataset(X_dict, y)
    method.fit(torch_dataset, y=None)

    # Test dataset without dict
    torch_dataset = Dataset(X, y)

    with pytest.raises(ValueError):
        method.fit(torch_dataset, y=None)

    with pytest.raises(ValueError):
        method.predict(
            torch_dataset,
        )
        method.predict_proba(
            torch_dataset,
        )

    # Test keys name in the dict
    X, y, sample_domain = dataset.pack_train(as_sources=["s"], as_targets=["t"])
    X_dict = {"bad_name": X.astype(np.float32), "sample_domain": sample_domain}
    with pytest.raises(ValueError):
        method.fit(
            X_dict,
            y,
        )

    with pytest.raises(ValueError):
        method.predict(
            X_dict,
        )
        method.predict_proba(
            X_dict,
        )

    with pytest.raises(ValueError):
        method.score(
            X_dict,
            y,
        )

    torch_dataset = Dataset(X_dict, y)

    with pytest.raises(ValueError):
        method.fit(torch_dataset, y=None)

    with pytest.raises(ValueError):
        method.predict(
            torch_dataset,
        )
        method.predict_proba(
            torch_dataset,
        )

    X_dict = {"X": X.astype(np.float32), "bad_name": sample_domain}
    with pytest.raises(ValueError):
        method.fit(
            X_dict,
            y,
        )

    with pytest.raises(ValueError):
        method.predict(
            X_dict,
        )
        method.predict_proba(
            X_dict,
        )

    with pytest.raises(ValueError):
        method.score(
            X_dict,
            y,
        )

    torch_dataset = Dataset(X_dict, y)

    with pytest.raises(ValueError):
        method.fit(torch_dataset, y=None)

    with pytest.raises(ValueError):
        method.predict(
            torch_dataset,
        )
        method.predict_proba(
            torch_dataset,
        )


def test_return_features():
    num_features = 10
    module = ToyModule2D(num_features=num_features)
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

    method = DomainAwareNet(
        DomainAwareModule(module, "dropout"),
        iterator_train=DomainBalancedDataLoader,
        criterion=DomainAwareCriterion(torch.nn.CrossEntropyLoss(), BaseDALoss()),
        batch_size=10,
        max_epochs=2,
        train_split=None,
    )

    X_test, _, _ = dataset.pack_test(as_targets=["t"])
    X_test = X_test.astype(np.float32)

    # without dict
    features = method.predict_features(torch.tensor(X_test))
    assert features.shape[1] == num_features
    assert features.shape[0] == X_test.shape[0]

    # Test the feature_infer method
    _, features = method.feature_infer(torch.tensor(X_test))
    assert features.shape == (X_test.shape[0], num_features)

    # Test the feature_infer method with dictionary input
    X_test_dict = {"X": X_test, "sample_domain": np.zeros(len(X_test))}
    _, features = method.feature_infer(X_test_dict)
    assert features.shape == (X_test.shape[0], num_features)


def test_domain_balanced_sampler():
    n_samples = 20
    dataset = make_shifted_datasets(
        n_samples_source=n_samples,
        n_samples_target=n_samples,
        shift="concept_drift",
        noise=0.1,
        random_state=42,
        return_dataset=True,
    )
    X, y, sample_domain = dataset.pack_train(as_sources=["s"], as_targets=["t"])
    X_dict = {"X": X.astype(np.float32), "sample_domain": sample_domain}

    dataset = Dataset(X_dict, y)

    sampler = DomainBalancedSampler(dataset, 10)
    assert len(sampler) == 2 * np.sum(sample_domain > 0)


def test_domain_balanced_dataloader():
    n_samples = 20
    dataset = make_shifted_datasets(
        n_samples_source=n_samples,
        n_samples_target=n_samples,
        shift="concept_drift",
        noise=0.1,
        random_state=42,
        return_dataset=True,
    )
    X, y, sample_domain = dataset.pack_train(as_sources=["s"], as_targets=["t"])
    X_dict = {"X": X.astype(np.float32), "sample_domain": sample_domain}

    dataset = Dataset(X_dict, y)

    dataloader = DomainBalancedDataLoader(dataset, batch_size=10)

    for batch in dataloader:
        X, y = batch
        sample_domain = X["sample_domain"]
        assert len(sample_domain > 0) == len(sample_domain < 0)

    # with more source than target
    dataset = make_shifted_datasets(
        n_samples_source=2 * n_samples,
        n_samples_target=n_samples,
        shift="concept_drift",
        noise=0.1,
        random_state=42,
        return_dataset=True,
    )
    X, y, sample_domain = dataset.pack_train(as_sources=["s"], as_targets=["t"])
    X_dict = {"X": X.astype(np.float32), "sample_domain": sample_domain}

    dataset = Dataset(X_dict, y)

    dataloader = DomainBalancedDataLoader(dataset, batch_size=10)

    for batch in dataloader:
        X, y = batch
        sample_domain = X["sample_domain"]
        assert len(sample_domain > 0) == len(sample_domain < 0)

    # with more target than source
    dataset = make_shifted_datasets(
        n_samples_source=n_samples,
        n_samples_target=2 * n_samples,
        shift="concept_drift",
        noise=0.1,
        random_state=42,
        return_dataset=True,
    )
    X, y, sample_domain = dataset.pack_train(as_sources=["s"], as_targets=["t"])
    X_dict = {"X": X.astype(np.float32), "sample_domain": sample_domain}

    dataset = Dataset(X_dict, y)

    dataloader = DomainBalancedDataLoader(dataset, batch_size=10)

    for batch in dataloader:
        X, y = batch
        sample_domain = X["sample_domain"]
        assert len(sample_domain > 0) == len(sample_domain < 0)


def test_sample_weight():
    n_samples = 10
    num_features = 5
    module = ToyModule2D(num_features=num_features)
    module.eval()

    # Create a simple dataset with a known class imbalance
    dataset = make_shifted_datasets(
        n_samples_source=n_samples,
        n_samples_target=n_samples,
        shift="concept_drift",
        noise=0.1,
        random_state=42,
        return_dataset=True,
    )

    # Initialize the domain aware network
    method = DomainAwareNet(
        DomainAwareModule(module, "dropout"),
        iterator_train=DomainBalancedDataLoader,
        criterion=DomainAwareCriterion(
            torch.nn.CrossEntropyLoss(), TestLoss(), reduction="none"
        ),
        batch_size=5,
        max_epochs=1,
        train_split=None,
    )

    # Prepare the training data
    X, y, sample_domain = dataset.pack_train(as_sources=["s"], as_targets=["t"])
    X = X.astype(np.float32)
    sample_weight = np.ones_like(y, dtype=np.float32)

    # Prepare the test data
    X_test, y_test, sample_domain_test = dataset.pack_test(as_targets=["t"])
    X_test = X_test.astype(np.float32)
    sample_weight_test = np.ones_like(y_test, dtype=np.float32)

    # Fit the model with sample weights and numpy inputs
    method.fit(X, y, sample_domain=sample_domain, sample_weight=sample_weight)
    assert method.history[-1]["train_loss"] > 0.1  # loss should be non-zero
    method.score(X_test, y_test, sample_domain_test, sample_weight=sample_weight_test)

    # Check that the loss is 0 when the sample weights are 0
    sample_weight = np.zeros_like(y, dtype=np.float32)
    method.fit(X, y, sample_domain=sample_domain, sample_weight=sample_weight)
    assert method.history[-1]["train_loss"] == 0

    # tensor input
    method.fit(
        torch.tensor(X),
        torch.tensor(y),
        sample_domain=torch.tensor(sample_domain),
        sample_weight=torch.tensor(sample_weight),
    )
    method.score(
        torch.tensor(X_test),
        torch.tensor(y_test),
        sample_domain=torch.tensor(sample_domain_test),
        sample_weight=torch.tensor(sample_weight_test),
    )

    # dataset input
    X_dict = {"X": X, "sample_domain": sample_domain, "sample_weight": sample_weight}

    torch_dataset = Dataset(X_dict, y)
    method.fit(torch_dataset, y=None)


def test_sample_weight_error_with_reduction_none():
    n_samples = 10
    num_features = 5
    module = ToyModule2D(num_features=num_features)
    module.eval()

    # Create a simple dataset
    dataset = make_shifted_datasets(
        n_samples_source=n_samples,
        n_samples_target=n_samples,
        shift="concept_drift",
        noise=0.1,
        random_state=42,
        return_dataset=True,
    )

    # Initialize the domain aware network with reduction set to 'mean'
    method = DomainAwareNet(
        DomainAwareModule(module, "dropout"),
        iterator_train=DomainBalancedDataLoader,
        criterion=DomainAwareCriterion(
            torch.nn.CrossEntropyLoss(), TestLoss(), reduction="mean"
        ),
        batch_size=5,
        max_epochs=1,
        train_split=None,
    )

    # Prepare the training data
    X, y, sample_domain = dataset.pack_train(as_sources=["s"], as_targets=["t"])
    X = X.astype(np.float32)
    sample_weight = np.ones_like(y, dtype=np.float32)

    # Expect an error when fitting with reduction='none' and sample weights provided
    with pytest.raises(ValueError):
        method.fit(X, y, sample_domain=sample_domain, sample_weight=sample_weight)
