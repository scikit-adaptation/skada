# Author: Theo Gnassounou <theo.gnassounou@inria.fr>
#         Remi Flamary <remi.flamary@polytechnique.edu>
#         Oleksii Kachaiev <kachayev@gmail.com>
#
# License: BSD 3-Clause

import numpy as np
import pytest

from skada.datasets import (
    Office31CategoriesPreset,
    Office31Domain,
    fetch_office31_decaf,
    fetch_office31_decaf_all,
    fetch_office31_surf,
    fetch_office31_surf_all,
)

_CALTECH256 = Office31CategoriesPreset.CALTECH256
_ALL = Office31CategoriesPreset.ALL


# mark all the test with the marker dataset
pytestmark = pytest.mark.dataset


@pytest.mark.parametrize(
    "domain, X_shape, y_shape, categories, n_categories",
    [
        # load by name
        ("amazon", (2817, 1000), (2817,), None, 31),
        ("webcam", (795, 1000), (795,), None, 31),
        ("dslr", (498, 1000), (498,), None, 31),
        # load by enum
        (Office31Domain.AMAZON, (2817, 1000), (2817,), None, 31),
        (Office31Domain.WEBCAM, (795, 1000), (795,), None, 31),
        (Office31Domain.DSLR, (498, 1000), (498,), None, 31),
        # caltech categories
        (Office31Domain.AMAZON, (958, 1000), (958,), _CALTECH256, 10),
        (Office31Domain.WEBCAM, (295, 1000), (295,), _CALTECH256, 10),
        (Office31Domain.DSLR, (157, 1000), (157,), _CALTECH256, 10),
        # other categories
        (Office31Domain.AMAZON, (2817, 1000), (2817,), _ALL, 31),
        (Office31Domain.AMAZON, (82, 1000), (82,), ["bike"], 1),
    ],
)
def test_decaf_fetcher(tmp_folder, domain, X_shape, y_shape, categories, n_categories):
    X, y = fetch_office31_decaf(
        domain,
        data_home=tmp_folder,
        categories=categories,
        return_X_y=True,
    )
    assert X.shape == X_shape
    assert y.shape == y_shape
    assert np.unique(y).shape[0] == n_categories


@pytest.mark.parametrize(
    "domain, X_shape, y_shape, categories, n_categories",
    [
        # load by name
        ("amazon", (2813, 800), (2813,), None, 31),
        ("webcam", (795, 800), (795,), None, 31),
        ("dslr", (498, 800), (498,), None, 31),
        # load by enum
        (Office31Domain.AMAZON, (2813, 800), (2813,), None, 31),
        (Office31Domain.WEBCAM, (795, 800), (795,), None, 31),
        (Office31Domain.DSLR, (498, 800), (498,), None, 31),
        # caltech categories
        (Office31Domain.AMAZON, (958, 800), (958,), _CALTECH256, 10),
        (Office31Domain.WEBCAM, (295, 800), (295,), _CALTECH256, 10),
        (Office31Domain.DSLR, (157, 800), (157,), _CALTECH256, 10),
        # other categories
        (Office31Domain.AMAZON, (2813, 800), (2813,), _ALL, 31),
        (Office31Domain.AMAZON, (82, 800), (82,), ["bike"], 1),
    ],
)
def test_surf_fetcher(tmp_folder, domain, X_shape, y_shape, categories, n_categories):
    X, y = fetch_office31_surf(
        domain,
        data_home=tmp_folder,
        categories=categories,
        return_X_y=True,
    )
    assert X.shape == X_shape
    assert y.shape == y_shape
    assert np.unique(y).shape[0] == n_categories


# xxx(okachaiev): i guess it would be much better to keep detailed test cases
# for DomainAwareDataset separately, better with a randomly generated data (faster)
@pytest.mark.parametrize(
    "load_all, load_domain",
    [
        (fetch_office31_surf_all, fetch_office31_surf),
        (fetch_office31_decaf_all, fetch_office31_decaf),
    ],
)
def test_surf_all_fetcher(tmp_folder, load_all, load_domain):
    dataset = load_all(data_home=tmp_folder)
    X, y = dataset.get_domain("amazon")
    X_amazon, y_amazon = load_domain("amazon", data_home=tmp_folder, return_X_y=True)
    assert np.array_equal(X, X_amazon), "single domain samples"
    assert np.array_equal(y, y_amazon), "single domain labels"
    X, y, sample_domain = dataset.pack(
        as_sources=["amazon"], as_targets=["webcam"], mask_target_labels=False
    )
    X_webcam, y_webcam = load_domain("webcam", data_home=tmp_folder, return_X_y=True)
    assert np.array_equal(X[sample_domain > 0], X_amazon), "correct sources"
    assert np.array_equal(X[sample_domain < 0], X_webcam), "correct targets"
    assert y.shape[0] == y_amazon.shape[0] + y_webcam.shape[0], "correct selection size"
    _, _, sample_domain_rev = dataset.pack(
        as_sources=["webcam"], as_targets=["amazon"], mask_target_labels=False
    )
    uniq_domain = np.unique(sample_domain)
    rev_uniq_domain = -1 * np.unique(sample_domain_rev)
    assert set(uniq_domain) == set(rev_uniq_domain), "same domain labels"


def test_categories_mapping(tmp_folder):
    categories = ["bike", "mouse"]
    data = fetch_office31_surf(
        Office31Domain.AMAZON,
        data_home=tmp_folder,
        categories=categories,
    )
    assert data.target_names == categories


def test_unknown_domain_failure(tmp_folder):
    with pytest.raises(ValueError):
        fetch_office31_surf("unknown-domain", data_home=tmp_folder)


def test_unknown_category_warning(tmp_folder):
    with pytest.warns():
        fetch_office31_surf(
            Office31Domain.AMAZON,
            data_home=tmp_folder,
            categories=["bike", "mug", "this-wont-be-found"],
        )
