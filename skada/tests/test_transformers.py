# Author: Yanis Lalou <yanis.lalou@polytechnique.edu>
#
# License: BSD 3-Clause


from sklearn.preprocessing import StandardScaler

from skada import CORAL, make_da_pipeline
from skada.transformers import SubsampleTransformer


def test_SubsampleTransformer(da_dataset):
    X, y, sample_domain = da_dataset.pack_train(as_sources=["s"], as_targets=["t"])

    n_subsample = 10

    transformer = SubsampleTransformer(n_subsample=n_subsample)

    X_subsampled, y_subsampled, params = transformer.fit_transform(
        X, y, sample_domain=sample_domain
    )

    assert X_subsampled.shape[0] == n_subsample
    assert y_subsampled.shape[0] == n_subsample
    assert "sample_domain" in params

    X_target, y_target, sample_domain_target = da_dataset.pack_test(as_targets=["t"])

    X_target_subsampled = transformer.transform(
        X_target, y_target, sample_domain=sample_domain_target
    )

    assert X_target_subsampled.shape[0] == X_target.shape[0]

    # within a pipeline

    transformer = SubsampleTransformer(n_subsample=n_subsample)
    pipeline = make_da_pipeline(StandardScaler(), transformer)

    temp = pipeline.fit_transform(X, y, sample_domain=sample_domain)

    assert temp is not None

    # now with a pipeline with end task
    transformer = SubsampleTransformer(n_subsample=n_subsample)
    pipeline = make_da_pipeline(StandardScaler(), transformer, CORAL())

    pipeline.fit(X, y, sample_domain=sample_domain)

    ypred = pipeline.predict(X_target, sample_domain=sample_domain_target)
    assert ypred.shape[0] == X_target.shape[0]
    assert ypred.shape[0] == X_target.shape[0]
