# Author: Yanis Lalou <yanis.lalou@polytechnique.edu>
#
# License: BSD 3-Clause

import numpy as np
import pytest

from skada.datasets import (
    OfficeHomeDomain,
    fetch_office_home,
    fetch_office_home_all,
)

# mark all the test with the marker dataset
pytestmark = pytest.mark.dataset


@pytest.mark.parametrize(
    "domain, X_shape, y_shape",
    [
        # load by name
        ("art", (2427, 2048), (2427,)),
        ("clipart", (4365, 2048), (4365,)),
        ("product", (4439, 2048), (4439,)),
        ("realworld", (4357, 2048), (4357,)),
        # load by enum
        (OfficeHomeDomain.ART, (2427, 2048), (2427,)),
        (OfficeHomeDomain.CLIPART, (4365, 2048), (4365,)),
        (OfficeHomeDomain.PRODUCT, (4439, 2048), (4439,)),
        (OfficeHomeDomain.REALWORLD, (4357, 2048), (4357,)),
    ],
)
def test_office_home_fetcher(tmp_folder, domain, X_shape, y_shape):
    X, y = fetch_office_home(
        domain,
        data_home=tmp_folder,
        return_X_y=True,
    )
    assert X.shape == X_shape
    assert y.shape == y_shape


def test_art_review_all_fetcher(tmp_folder):
    dataset = fetch_office_home_all(data_home=tmp_folder)
    X, y = dataset.get_domain("art")
    X_art, y_art = fetch_office_home("art", data_home=tmp_folder, return_X_y=True)
    assert np.array_equal(X, X_art), "single domain samples"
    assert np.array_equal(y, y_art), "single domain labels"
    X, y, sample_domain = dataset.pack(
        as_sources=["art"], as_targets=["product"], mask_target_labels=False
    )
    X_product, y_product = fetch_office_home(
        "product", data_home=tmp_folder, return_X_y=True
    )
    assert np.array_equal(X[sample_domain > 0], X_art), "correct sources"
    assert np.array_equal(X[sample_domain < 0], X_product), "correct targets"
    assert y.shape[0] == y_art.shape[0] + y_product.shape[0], "correct selection size"
    _, _, sample_domain_rev = dataset.pack(
        as_sources=["product"], as_targets=["art"], mask_target_labels=False
    )
    uniq_domain = np.unique(sample_domain)
    rev_uniq_domain = -1 * np.unique(sample_domain_rev)
    assert set(uniq_domain) == set(rev_uniq_domain), "same domain labels"


def test_unknown_domain_failure(tmp_folder):
    with pytest.raises(ValueError):
        fetch_office_home("unknown-domain", data_home=tmp_folder)
