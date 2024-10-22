# Author: Yanis Lalou <yanis.lalou@polytechnique.edu>
#
# License: BSD 3-Clause


from skada import make_da_pipeline
from skada.transformers import SubsampleTransformer


def test_SubsampleTransformer(da_dataset):
    X, y, sample_domain = da_dataset.pack_train(as_sources=["s"], as_targets=["t"])

    n_subsample = 10

    transformer = SubsampleTransformer(n_subsample=n_subsample)

    X_subsampled, y_subsampled, dic = transformer.fit_transform(
        X, y, sample_domain=sample_domain
    )

    assert X_subsampled.shape[0] == n_subsample
    assert y_subsampled.shape[0] == n_subsample
    assert "sample_domain" in dic

    X_target, y_target, sample_domain = da_dataset.pack_test(as_targets=["t"])

    X_target_subsampled, y_target_subsampled, dic = transformer.transform(
        X_target, y_target, sample_domain=sample_domain
    )

    assert X_target_subsampled.shape[0] == X_target.shape[0]
    assert y_target_subsampled.shape[0] == X_target.shape[0]

    # now with a pipeline
    transformer = SubsampleTransformer(n_subsample=n_subsample)
    pipeline = make_da_pipeline(transformer)

    X_subsampled, y_subsampled, dic = pipeline.fit_transform(
        X, y, sample_domain=sample_domain
    )

    assert X_subsampled.shape[0] == n_subsample
    assert y_subsampled.shape[0] == n_subsample
    assert "sample_domain" in dic
