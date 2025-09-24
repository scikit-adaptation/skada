# Author: Theo Gnassounou <theo.gnassounou@inria.fr>
#         Oleksii Kachaiev <kachayev@gmail.com>
#         Yanis Lalou <yanis.lalou@polytechnique.edu>
#         Antoine Collas <contact@antoinecollas.fr>
#
# License: BSD 3-Clause
import pytest

torch = pytest.importorskip("torch")

import numpy as np
from pandas import DataFrame
from skorch.dataset import Dataset

from skada.datasets import make_shifted_datasets
from skada.deep.base import (
    BaseDALoss,
    DeepDADataset,
    DomainAwareCriterion,
    DomainAwareModule,
    DomainAwareNet,
    DomainBalancedDataLoader,
    DomainBalancedSampler,
    DomainOnlyDataLoader,
    DomainOnlySampler,
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
        shift="conditional_shift",
        noise=0.1,
        random_state=42,
        return_dataset=True,
    )

    # Prepare data
    X, y, sample_domain = dataset.pack(
        as_sources=["s"], as_targets=["t"], mask_target_labels=True
    )
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
    y_pred, domain_pred, features, sample_domain_output, sample_idx = output

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

    assert (
        max_difference > 0.1
    ), "Features of source and target domains are too similar."


def test_domainawaretraining():
    module = ToyModule2D()
    module.eval()

    n_samples = 20
    dataset = make_shifted_datasets(
        n_samples_source=n_samples,
        n_samples_target=n_samples,
        shift="conditional_shift",
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

    X, y, sample_domain = dataset.pack(
        as_sources=["s"], as_targets=["t"], mask_target_labels=True
    )
    X_test, y_test, sample_domain_test = dataset.pack(
        as_sources=[], as_targets=["t"], mask_target_labels=False
    )
    X = X.astype(np.float32)
    X_test = X_test.astype(np.float32)

    # without dict
    method.fit(X, y, sample_domain=sample_domain)

    y_pred = method.predict(X_test, sample_domain_test)
    _ = method.predict_proba(X, sample_domain, allow_source=True)
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
    _ = method.predict_proba(X_dict, allow_source=True)
    method.score(X_dict_test, y_test)

    assert y_pred.shape[0] == X_test.shape[0]

    # numpy input
    method.fit(X, y, sample_domain)
    y_pred = method.predict(X_test, sample_domain_test)
    _ = method.predict_proba(X, sample_domain, allow_source=True)
    method.score(X_test, y_test, sample_domain_test)

    assert y_pred.shape[0] == X_test.shape[0]

    # tensor input
    method.fit(torch.tensor(X), torch.tensor(y), torch.tensor(sample_domain))
    y_pred = method.predict(torch.tensor(X_test), torch.tensor(sample_domain_test))
    _ = method.predict_proba(
        torch.tensor(X), torch.tensor(sample_domain), allow_source=True
    )
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
    X, y, sample_domain = dataset.pack(
        as_sources=["s"], as_targets=["t"], mask_target_labels=True
    )
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
        shift="conditional_shift",
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

    X_test, _, _ = dataset.pack(
        as_sources=[], as_targets=["t"], mask_target_labels=False
    )
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


@pytest.mark.parametrize(
    "max_samples",
    [
        "max",
        "source",
        "target",
        "min",
    ],
)
def test_domain_balanced_sampler(max_samples):
    n_samples = 20
    dataset = make_shifted_datasets(
        n_samples_source=n_samples,
        n_samples_target=n_samples,
        shift="conditional_shift",
        noise=0.1,
        random_state=42,
        return_dataset=True,
    )
    X, y, sample_domain = dataset.pack(
        as_sources=["s"], as_targets=["t"], mask_target_labels=True
    )
    X_dict = {"X": X.astype(np.float32), "sample_domain": sample_domain}

    n_samples_source = np.sum(sample_domain > 0)
    n_samples_target = np.sum(sample_domain < 0)

    dataset = Dataset(X_dict, y)

    sampler = DomainBalancedSampler(dataset, 10, max_samples=max_samples)
    if max_samples == "max":
        assert len(sampler) == 2 * max(n_samples_source, n_samples_target)
    elif max_samples == "source":
        assert len(sampler) == 2 * n_samples_source
    elif max_samples == "target":
        assert len(sampler) == 2 * n_samples_target
    elif max_samples == "min":
        assert len(sampler) == 2 * min(n_samples_source, n_samples_target)


def test_domain_balanced_dataloader():
    n_samples = 20
    dataset = make_shifted_datasets(
        n_samples_source=n_samples,
        n_samples_target=n_samples,
        shift="conditional_shift",
        noise=0.1,
        random_state=42,
        return_dataset=True,
    )
    X, y, sample_domain = dataset.pack(
        as_sources=["s"], as_targets=["t"], mask_target_labels=True
    )
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
        shift="conditional_shift",
        noise=0.1,
        random_state=42,
        return_dataset=True,
    )
    X, y, sample_domain = dataset.pack(
        as_sources=["s"], as_targets=["t"], mask_target_labels=True
    )
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
        shift="conditional_shift",
        noise=0.1,
        random_state=42,
        return_dataset=True,
    )
    X, y, sample_domain = dataset.pack(
        as_sources=["s"], as_targets=["t"], mask_target_labels=True
    )
    X_dict = {"X": X.astype(np.float32), "sample_domain": sample_domain}

    dataset = Dataset(X_dict, y)

    dataloader = DomainBalancedDataLoader(dataset, batch_size=10)

    for batch in dataloader:
        X, y = batch
        sample_domain = X["sample_domain"]
        assert len(sample_domain > 0) == len(sample_domain < 0)


@pytest.mark.parametrize(
    "domain_used",
    [
        "source",
        "target",
    ],
)
def test_domain_only_sampler(domain_used):
    n_samples = 20
    dataset = make_shifted_datasets(
        n_samples_source=n_samples,
        n_samples_target=n_samples,
        shift="conditional_shift",
        noise=0.1,
        random_state=42,
        return_dataset=True,
    )
    X, y, sample_domain = dataset.pack(
        as_sources=["s"], as_targets=["t"], mask_target_labels=True
    )
    X_dict = {"X": X.astype(np.float32), "sample_domain": sample_domain}

    dataset = Dataset(X_dict, y)

    sampler = DomainOnlySampler(dataset, 10, domain_used=domain_used)
    assert (
        len(sampler) == np.sum(sample_domain > 0)
        if domain_used == "source"
        else np.sum(sample_domain < 0)
    )


@pytest.mark.parametrize(
    "domain_used",
    [
        "source",
        "target",
    ],
)
def test_domain_only_dataloader(domain_used):
    n_samples = 20
    dataset = make_shifted_datasets(
        n_samples_source=n_samples,
        n_samples_target=n_samples,
        shift="conditional_shift",
        noise=0.1,
        random_state=42,
        return_dataset=True,
    )
    X, y, sample_domain = dataset.pack(
        as_sources=["s"], as_targets=["t"], mask_target_labels=True
    )
    X_dict = {"X": X.astype(np.float32), "sample_domain": sample_domain}

    dataset = Dataset(X_dict, y)

    dataloader = DomainOnlyDataLoader(dataset, batch_size=10, domain_used=domain_used)

    for batch in dataloader:
        X, y = batch
        sample_domain = X["sample_domain"]
        assert (
            (sample_domain > 0).all()
            if domain_used == "source"
            else (sample_domain < 0).all()
        )


def test_sample_weight():
    n_samples = 10
    num_features = 5
    module = ToyModule2D(num_features=num_features)
    module.eval()

    # Create a simple dataset with a known class imbalance
    dataset = make_shifted_datasets(
        n_samples_source=n_samples,
        n_samples_target=n_samples,
        shift="conditional_shift",
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
    X, y, sample_domain = dataset.pack(
        as_sources=["s"], as_targets=["t"], mask_target_labels=True
    )
    X = X.astype(np.float32)
    sample_weight = np.ones_like(y, dtype=np.float32)

    # Prepare the test data
    X_test, y_test, sample_domain_test = dataset.pack(
        as_sources=[], as_targets=["t"], mask_target_labels=False
    )
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
        shift="conditional_shift",
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
    X, y, sample_domain = dataset.pack(
        as_sources=["s"], as_targets=["t"], mask_target_labels=True
    )
    X = X.astype(np.float32)
    sample_weight = np.ones_like(y, dtype=np.float32)

    # Expect an error when fitting with reduction='none' and sample weights provided
    with pytest.raises(ValueError):
        method.fit(X, y, sample_domain=sample_domain, sample_weight=sample_weight)


@pytest.mark.parametrize(
    "base_criterion",
    [
        torch.nn.CrossEntropyLoss(),
    ],
)
def test_predict_proba(da_dataset, base_criterion):
    X_train, y_train, sample_domain_train = da_dataset.pack(
        as_sources=["s"], as_targets=["t"], mask_target_labels=True
    )
    X_train = X_train.astype(np.float32)
    n_classes = len(np.unique(y_train))

    module = ToyModule2D(n_classes=n_classes)

    # Initialize the domain aware network
    method = DomainAwareNet(
        DomainAwareModule(module, "dropout"),
        iterator_train=DomainBalancedDataLoader,
        criterion=DomainAwareCriterion(base_criterion, TestLoss(), reduction="mean"),
        batch_size=5,
        max_epochs=1,
        train_split=None,
    )

    # Fit the model
    method.fit(X_train, y_train, sample_domain=sample_domain_train)

    # Predict probabilities
    X_test, y_test, sample_domain_test = da_dataset.pack(
        as_sources=[], as_targets=["t"], mask_target_labels=False
    )
    X_test = X_test.astype(np.float32)
    y_proba = method.predict_proba(X_test, sample_domain=sample_domain_test)

    assert y_proba.shape == (len(y_test), n_classes)
    assert np.allclose(y_proba.sum(axis=1), 1)
    assert np.all(y_proba >= 0)


def test_allow_source():
    module = ToyModule2D()
    module.eval()

    n_samples = 20
    dataset = make_shifted_datasets(
        n_samples_source=n_samples,
        n_samples_target=n_samples,
        shift="conditional_shift",
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

    X, y, sample_domain = dataset.pack(
        as_sources=["s"], as_targets=["t"], mask_target_labels=True
    )
    X_test, y_test, sample_domain_test = dataset.pack(
        as_sources=[], as_targets=["t"], mask_target_labels=False
    )
    X = X.astype(np.float32)
    X_test = X_test.astype(np.float32)

    # without dict
    method.fit(X, y, sample_domain=sample_domain)
    method.predict_proba(X, sample_domain, allow_source=False)


@pytest.mark.parametrize(
    "label_type",
    [
        "binary",
        "regression",
    ],
)
def test_deep_domain_aware_dataset(label_type):
    def are_DDAD_equal(a, b):
        """Helper function to compare two DDADs."""
        correct = True
        if not isinstance(a, DeepDADataset) or not isinstance(b, DeepDADataset):
            print("One of the datasets is not a DeepDADataset.")
            correct = False
        elif len(a) != len(b):
            print(f"Length differs: {len(a)} != {len(b)}")
            correct = False
        else:
            for attr in a.__dict__:
                if not hasattr(b, attr):
                    correct = False
                try:
                    if a.__dict__[attr] != b.__dict__[attr]:
                        print(
                            f"Attribute {attr} differs between datasets: "
                            f"{a.__dict__[attr]} != {b.__dict__[attr]}"
                        )
                except RuntimeError:
                    try:
                        if not np.array_equal(
                            a.__dict__[attr], b.__dict__[attr], equal_nan=True
                        ):
                            print(
                                f"Attribute {attr} differs between datasets: "
                                f"{a.__dict__[attr]} != {b.__dict__[attr]}"
                            )
                            correct = False
                    except RuntimeError:
                        raise RuntimeError(
                            f"Attribute {attr} cannot be compared: "
                            f"{a.__dict__[attr]} vs {b.__dict__[attr]}"
                        )

        return correct

    _RANDOM_STATE_ = 42
    # data creation
    dataset = make_shifted_datasets(
        20,
        20,
        random_state=_RANDOM_STATE_,
        label=label_type,
        return_dataset=True,
    )
    raw_data = dataset.pack(as_sources=["s"], as_targets=["t"], mask_target_labels=True)
    X, y, sd = raw_data
    raw_data_dict = {"X": X, "y": y, "sample_domain": sd}
    # though these are not technically weights, they will act as such for the tests
    weights = np.ones_like(y, dtype=np.float32)
    weighted_raw_data_dict = {
        "X": X,
        "y": y,
        "sample_domain": sd,
        "sample_weight": weights,
    }
    df = DataFrame(
        {
            "X": list(X),
            "y": y,
            "sample_domain": sd,
            "sample_weight": weights,
        }
    )

    # Dataset creation
    empty = DeepDADataset()
    dataset = DeepDADataset(*raw_data)
    weighted_dataset = DeepDADataset(weighted_raw_data_dict)
    assert empty.is_empty()
    assert (
        DeepDADataset([]).is_empty()
        == DeepDADataset({}).is_empty()
        == DeepDADataset(None).is_empty()
        == empty.is_empty()
    )
    assert are_DDAD_equal(DeepDADataset(raw_data_dict), dataset)
    assert are_DDAD_equal(DeepDADataset(dataset), dataset)
    assert are_DDAD_equal(dataset.add_weights(weights), weighted_dataset)
    assert are_DDAD_equal(dataset.remove_weights(), dataset)
    assert are_DDAD_equal(DeepDADataset({"X": X}, y, sd, weights), weighted_dataset)
    assert are_DDAD_equal(DeepDADataset(df), weighted_dataset)

    with pytest.raises(ValueError):
        DeepDADataset({"bad_name": X})

    with pytest.raises(TypeError):
        DeepDADataset("incorrect_type")

    # Dataset manipulation
    assert are_DDAD_equal(dataset.merge(empty), dataset)
    assert are_DDAD_equal(empty.merge(dataset), dataset)
    assert len(dataset.merge(dataset)) == len(dataset) + len(dataset)

    # representation
    d1 = weighted_dataset.as_dict(sample_indices=False)
    assert d1.keys() == weighted_raw_data_dict.keys()
    for key in d1.keys():
        assert np.array_equal(d1[key], weighted_raw_data_dict[key], equal_nan=True)
    for v1, v2 in zip(weighted_dataset.as_arrays(), (X, y, sd, weights)):
        assert np.array_equal(v1, v2, equal_nan=True)

    # Data selection
    source_data = (X[sd >= 0], y[sd >= 0], sd[sd >= 0], weights[sd >= 0])
    target_data = (X[sd < 0], y[sd < 0], sd[sd < 0], weights[sd < 0])
    domain_id = 1
    domain_data = (
        X[sd == domain_id],
        y[sd == domain_id],
        sd[sd == domain_id],
        weights[sd == domain_id],
    )
    selec_y = y == 2
    selec_X = X[:, 0] >= 0
    selec_w = weights == 1
    assert are_DDAD_equal(weighted_dataset.select_source(), DeepDADataset(*source_data))
    assert are_DDAD_equal(weighted_dataset.select_target(), DeepDADataset(*target_data))
    assert are_DDAD_equal(
        weighted_dataset.select_domain(domain_id), DeepDADataset(*domain_data)
    )
    assert are_DDAD_equal(
        weighted_dataset.select(lambda sd: sd >= 0, on="sample_domain"),
        weighted_dataset.select_source(),
    )
    assert are_DDAD_equal(
        dataset.select(lambda y: y == 2, on="y"),
        DeepDADataset(X[selec_y], y[selec_y], sd[selec_y]),
    )
    assert are_DDAD_equal(
        weighted_dataset.select(lambda x: x[:, 0] >= 0, on="X"),
        DeepDADataset(X[selec_X], y[selec_X], sd[selec_X], weights[selec_X]),
    )
    assert are_DDAD_equal(
        weighted_dataset.select(lambda w: w == 1, on="sample_weight"),
        DeepDADataset(X[selec_w], y[selec_w], sd[selec_w], weights[selec_w]),
    )
