import numpy as np
import pytest
from skada.datasets import DomainAwareDataset, make_shifted_blobs, make_shifted_datasets


# xxx(okachaiev): old API has to be gone when re-writing is done
@pytest.fixture(scope="session")
def tmp_da_dataset():
    centers = np.array(
        [
            [0, 0],
            [1, 1],
        ]
    )
    _, n_features = centers.shape

    X, y, sample_domain = make_shifted_blobs(
        n_samples=100,
        centers=centers,
        n_features=n_features,
        shift=0.13,
        random_state=42,
        cluster_std=0.05,
        return_X_y=True,
    )

    return (
        X[sample_domain > 0], y[sample_domain > 0],
        X[sample_domain < 0], y[sample_domain < 0],
    )


@pytest.fixture(scope='session')
def da_reg_dataset():
    X, y, sample_domain = make_shifted_datasets(
        n_samples_source=20,
        n_samples_target=21,
        shift="concept_drift",
        noise=0.3,
        label="regression",
        random_state=42,
    )
    return X, y, sample_domain


@pytest.fixture(scope='session')
def da_dataset() -> DomainAwareDataset:
    centers = np.array([[0, 0], [1, 1]])
    _, n_features = centers.shape
    dataset = make_shifted_blobs(
        n_samples=100,
        centers=centers,
        n_features=n_features,
        shift=0.13,
        random_state=42,
        cluster_std=0.05,
        return_dataset=True,
    )
    centers = np.array([[2, 0], [-1, 2]])
    _, n_features = centers.shape
    dataset2 = make_shifted_blobs(
        n_samples=100,
        centers=centers,
        n_features=n_features,
        shift=0.13,
        random_state=42,
        cluster_std=0.05,
        return_dataset=True,
    )
    return dataset.merge(dataset2, names_mapping={'s': 's2', 't': 't2'})


@pytest.fixture(scope="session")
def tmp_folder(tmpdir_factory):
    folder = tmpdir_factory.mktemp("skada_datasets")
    return str(folder)
