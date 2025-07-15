# Author: Yanis Lalou <yanis.lalou@polytechnique.edu>
#
# License: BSD 3-Clause

import numpy as np
import pytest

from skada.datasets import (
    AmazonReviewDomain,
    fetch_amazon_review,
    fetch_amazon_review_all,
)

# mark all the test with the marker dataset
pytestmark = pytest.mark.dataset


@pytest.mark.parametrize(
    "domain, X_shape, y_shape",
    [
        # load by name
        ("books", (2000, 400), (2000,)),
        ("dvd", (1999, 400), (1999,)),
        ("kitchen", (1999, 400), (1999,)),
        ("elec", (1998, 400), (1998,)),
        # load by enum
        (AmazonReviewDomain.BOOKS, (2000, 400), (2000,)),
        (AmazonReviewDomain.DVD, (1999, 400), (1999,)),
        (AmazonReviewDomain.KITCHEN, (1999, 400), (1999,)),
        (AmazonReviewDomain.ELEC, (1998, 400), (1998,)),
    ],
)
def test_amazon_review_fetcher(tmp_folder, domain, X_shape, y_shape):
    X, y = fetch_amazon_review(
        domain,
        data_home=tmp_folder,
        return_X_y=True,
    )
    assert X.shape == X_shape
    assert y.shape == y_shape


def test_books_review_all_fetcher(tmp_folder):
    dataset = fetch_amazon_review_all(data_home=tmp_folder)
    X, y = dataset.get_domain("books")
    X_books, y_books = fetch_amazon_review(
        "books", data_home=tmp_folder, return_X_y=True
    )
    assert np.array_equal(X, X_books), "single domain samples"
    assert np.array_equal(y, y_books), "single domain labels"
    X, y, sample_domain = dataset.pack(
        as_sources=["books"], as_targets=["dvd"], mask_target_labels=False
    )
    X_dvd, y_dvd = fetch_amazon_review("dvd", data_home=tmp_folder, return_X_y=True)
    assert np.array_equal(X[sample_domain > 0], X_books), "correct sources"
    assert np.array_equal(X[sample_domain < 0], X_dvd), "correct targets"
    assert y.shape[0] == y_books.shape[0] + y_dvd.shape[0], "correct selection size"
    _, _, sample_domain_rev = dataset.pack(
        as_sources=["dvd"], as_targets=["books"], mask_target_labels=False
    )
    uniq_domain = np.unique(sample_domain)
    rev_uniq_domain = -1 * np.unique(sample_domain_rev)
    assert set(uniq_domain) == set(rev_uniq_domain), "same domain labels"


def test_unknown_domain_failure(tmp_folder):
    with pytest.raises(ValueError):
        fetch_amazon_review("unknown-domain", data_home=tmp_folder)
