import numpy as np

from sklearn.datasets import make_blobs


def _generate_unif_circle(n_samples, rng):
    angle = rng.rand(n_samples, 1) * 2 * np.pi
    r = np.sqrt(rng.rand(n_samples, 1))

    x = np.concatenate((r * np.cos(angle), r * np.sin(angle)), 1)
    return x


def _generate_data_2d_classif(n_samples, rng, label='binary'):
    """Generate 2d classification data.

    Parameters
    ----------
    n_samples : int
        It is the total number of points among one clusters.
        At the end the number of point are 8*n_samples
    rng : random generator
        Generator for dataset creation
    label : tuple, default='binary'
        If 'binary, return binary class
        If 'multiclass', return multiclass
    """
    n2 = n_samples
    n1 = n2 * 4
    # make data of class 1
    Sigma1 = np.array([[2, -0.5], [-0.5, 2]])
    mu1 = np.array([2, 2])
    x1 = _generate_unif_circle(n1, rng).dot(Sigma1) + mu1[None, :]

    # make data of the first cluster of class 2
    Sigma2 = np.array([[0.15, 0], [0, 0.3]])
    mu2 = np.array([-1.5, 3])

    x21 = rng.randn(n2, 2).dot(Sigma2) + mu2[None, :]

    # make data of the second cluster of class 2
    Sigma2 = np.array([[0.2, -0.1], [-0.1, 0.2]])
    mu2 = np.array([-0.5, 1])

    x22 = rng.randn(n2, 2).dot(Sigma2) + mu2[None, :]

    # make data of the third cluster of class 2
    Sigma2 = np.array([[0.17, -0.05], [-0.05, 0.17]])
    mu2 = np.array([1, -0.4])

    x23 = rng.randn(n2, 2).dot(Sigma2) + mu2[None, :]

    # make data of the fourth cluster of class 2
    Sigma2 = np.array([[0.3, -0.0], [-0.0, 0.15]])
    mu2 = np.array([3, -1])

    x24 = rng.randn(n2, 2).dot(Sigma2) + mu2[None, :]

    # concatenate data
    x = np.concatenate((x1, x21, x22, x23, x24), 0)

    # make labels
    if label == 'binary':
        y = np.concatenate((np.zeros(n1), np.ones(4*n2)), 0)
    elif label == 'multiclass':
        y = np.zeros(n1)
        for i in range(4):
            y = np.concatenate((y, (i+1)*np.ones(n2)), 0)
    return x, y.astype(int)


def make_shifted_blobs(
    n_samples_source=100,
    n_samples_target=100,
    n_features=2,
    shift=0.10,
    centers=None,
    cluster_std=1.0,
    shuffle=True,
    random_state=None,
):
    """Generate source and shift target isotropic Gaussian blobs .

    Parameters
    ----------
    n_samples_source : int or array-like, default=100
        If int, it is the total number of points equally divided among
        source clusters.
        If array-like, each element of the sequence indicates
        the number of samples per source cluster.
    n_samples_target : int or array-like, default=100
        If int, it is the total number of points equally divided among
        target clusters.
        If array-like, each element of the sequence indicates
        the number of samples per target cluster.
    n_features : int, default=2
        The number of features for each sample.
    shift : float or array like, default=0.10
        If float, it is the value of the translation for every target feature.
        If array_like, each element of the sequence indicates the value of
        the translation for each target features.
    centers : int or ndarray of shape (n_centers, n_features), default=None
        The number of centers to generate, or the fixed center locations.
        If n_samples is an int and centers is None, 3 centers are generated.
        If n_samples is array-like, centers must be
        either None or an array of length equal to the length of n_samples.
    cluster_std : float or array-like of float, default=1.0
        The standard deviation of the clusters.
    shuffle : bool, default=True
        Shuffle the samples.
    random_state : int, RandomState instance or None, default=None
        Determines random number generation for dataset creation. Pass an int
        for reproducible output across multiple function calls.

    Returns
    -------
    X_source : ndarray of shape (n_samples, n_features)
        The generated source samples.
    y_source : ndarray of shape (n_samples,)
        The integer labels for cluster membership of each source sample.
    X_target : ndarray of shape (n_samples, n_features)
        The generated target samples.
    y_target : ndarray of shape (n_samples,)
        The integer labels for cluster membership of each target sample.
    """

    X_source, y_source = make_blobs(
        n_samples=n_samples_source,
        centers=centers,
        n_features=n_features,
        random_state=random_state,
        cluster_std=cluster_std,
    )

    X_target, y_target = make_blobs(
        n_samples=n_samples_target,
        centers=centers,
        n_features=n_features,
        random_state=random_state,
        cluster_std=cluster_std,
    )
    X_target += shift

    return X_source, y_source, X_target, y_target


def make_shifted_datasets(
    n_samples_source=100,
    n_samples_target=100,
    shift="covariate_shift",
    noise=None,
    ratio=0.9,
    mean=1,
    sigma=0.7,
    gamma=2,
    center=[[0, 2]],
    random_state=None,
):
    """Generate source and shift target.

    Parameters
    ----------
    n_samples_source : int, default=100
        It is the total number of points among one
        source clusters. At the end 8*n_samples points.
    n_samples_target : int, default=100
        It is the total number of points among one
        target clusters. At the end 8*n_samples points.
    shift : tuple, default='cs'
        Choose the nature of the shift.
        If 'covariate_shift', use covariate shift.
        If 'target_shift', use target shift.
        If 'concept_drift', use concept drift.
        If 'sample_bias', use sample-selection bias.
        Detailed descriptionof each shift in [1].
    noise : float, default=None
        Standard deviation of Gaussian noise added to the data.
    ratio : float, default=0.9
        Ratio of the number of data in class 1 selected
        in the target shift and the sample_selection bias
    mean : float, default=1
        value of the translation in the concept drift.
    sigma : float, default=0.7
        multiplicative value of the concept drift.
    gamma :  float, default=2

    center : ndarray of shape (1, 2), default=[[0, 2]]

    random_state : int, RandomState instance or None, default=None
        Determines random number generation for dataset creation. Pass an int
        for reproducible output across multiple function calls.

    Returns
    -------
    X_source : ndarray of shape (n_samples, n_features)
        The generated source samples.
    y_source : ndarray of shape (n_samples,)
        The integer labels for cluster membership of each source sample.
    X_target : ndarray of shape (n_samples, n_features)
        The generated target samples.
    y_target : ndarray of shape (n_samples,)
        The integer labels for cluster membership of each target sample.

    References
    ----------
    .. [1]  Moreno-Torres, J. G., Raeder, T., Alaiz-Rodriguez,
            R., Chawla, N. V., and Herrera, F. (2012).
            A unifying view on dataset shift in classification.
            Pattern recognition, 45(1):521-530.
    """

    rng = np.random.RandomState(random_state)
    X_source, y_source = _generate_data_2d_classif(n_samples_source, rng)

    if shift == "covariate_shift":
        n_samples_target_temp = n_samples_target * 100
        X_target, y_target = _generate_data_2d_classif(n_samples_target_temp, rng)
        if noise is not None:
            X_target += rng.normal(scale=noise, size=X_target.shape)

        w = np.exp(-gamma * np.sum((X_target - np.array(center)) ** 2, 1))
        w /= w.sum()

        isel = rng.choice(len(w), size=(n_samples_target,), replace=False, p=w)

        X_target = X_target[isel]
        y_target = y_target[isel]

    elif shift == "target_shift":
        n_samples_target_temp = n_samples_target * 3
        X_target, y_target = _generate_data_2d_classif(n_samples_target_temp, rng)
        if noise is not None:
            X_target += rng.normal(scale=noise, size=X_target.shape)

        isel1 = rng.choice(
            n_samples_target_temp // 2,
            size=(int(n_samples_target * ratio),),
            replace=False
        )
        isel2 = (
            rng.choice(
                n_samples_target_temp // 2,
                size=(int(n_samples_target * (1 - ratio)),),
                replace=False
            )
        ) + n_samples_target_temp // 2
        isel = np.concatenate((isel1, isel2))

        X_target = X_target[isel]
        y_target = y_target[isel]

    elif shift == "concept_drift":
        X_target, y_target = _generate_data_2d_classif(n_samples_target, rng)
        if noise is not None:
            X_target += rng.normal(scale=noise, size=X_target.shape)

        X_target = X_target * sigma + mean

    elif shift == "sample_bias":
        n_samples_target_temp = n_samples_target * 100
        X_target, y_target = _generate_data_2d_classif(n_samples_target_temp, rng)
        if noise is not None:
            X_target += rng.normal(scale=noise, size=X_target.shape)

        w = np.exp(-gamma * np.sum((X_target - np.array(center)) ** 2, 1))

        w[y_target == 1] *= 1 - ratio
        w[y_target == 0] *= ratio

        w /= w.sum()

        isel = rng.choice(len(w), size=(n_samples_target,), replace=False, p=w)

        X_target = X_target[isel]
        y_target = y_target[isel]

    else:
        raise NotImplementedError("unknown shift {}".format(shift))

    return X_source, y_source, X_target, y_target


def make_out_of_distribution_dataset(
    n_samples_source=100,
    n_samples_target=100,
    noise=None,
    pos_sources=[0.1],
    pos_targets=[0.2],
    random_state=None
):
    """Make out-of-distribution dataset.
    A simple toy dataset to create an out-of-distribution dataset
    for classification.
    Parameters
    ----------
    n_samples_source : int, default=100
        It is the total number of points equally divided among
        source clusters.
    n_samples_target : int, default=100
        It is the total number of points equally divided among
        target clusters.
    noise : float, default=None
        Standard deviation of Gaussian noise added to the data.
    pos_sources : array-like of float, default=[0.1]
        Each element of the sequence indicates the position
        of the center of each source cluster.
    pos_sources : array-like of float, default=[0.1]
        Each element of the sequence indicates the position
        of the center of each target cluster.
    random_state : int, RandomState instance or None, default=None
        Determines random number generation for dataset creation. Pass an int
        for reproducible output across multiple function calls.

    Returns
    -------
    X_source : ndarray of shape (n_samples, n_features)
        The generated source samples.
    y_source : ndarray of shape (n_samples,)
        The integer labels for cluster membership of each source sample.
    X_target : ndarray of shape (n_samples, n_features)
        The generated target samples.
    y_target : ndarray of shape (n_samples,)
        The integer labels for cluster membership of each target sample.
    """

    rng = np.random.RandomState(random_state)

    n_sources = len(pos_sources)
    n_targets = len(pos_targets)

    n_samples_circ = 100
    outer_distr_circ_x = np.cos(np.linspace(0, np.pi, n_samples_circ))
    outer_distr_circ_y = np.sin(np.linspace(0, np.pi, n_samples_circ))
    inner_distr_circ_x = 1 - np.cos(np.linspace(0, np.pi, n_samples_circ))
    inner_distr_circ_y = 1 - np.sin(np.linspace(0, np.pi, n_samples_circ)) - 0.5

    X_source = np.zeros((n_sources, 2*n_samples_source, 2))
    y_source = np.zeros((n_sources, 2*n_samples_source))
    cov = [[0.01, 0], [0, 0.01]]
    for i, pos_source in enumerate(pos_sources):
        index = int(pos_source * 100)
        center1 = np.array([outer_distr_circ_x[index], outer_distr_circ_y[index]])
        center2 = np.array([inner_distr_circ_x[index], inner_distr_circ_y[index]])

        X_source[i] = np.concatenate(
            [rng.multivariate_normal(center1, cov, size=n_samples_source),
             rng.multivariate_normal(center2, cov, size=n_samples_source)]
        )
        y_source[i] = np.concatenate(
            [np.zeros(n_samples_source),
             np.ones(n_samples_source)]
        )

    X_target = np.zeros((n_targets, 2*n_samples_target, 2))
    y_target = np.zeros((n_sources, 2*n_samples_target))
    for i, pos_target in enumerate(pos_targets):
        index = int(pos_target * 100)
        center1 = np.array([outer_distr_circ_x[index], outer_distr_circ_y[index]])
        center2 = np.array([inner_distr_circ_x[index], inner_distr_circ_y[index]])

        X_target[i] = np.concatenate(
            [rng.multivariate_normal(center1, cov, size=n_samples_target),
             rng.multivariate_normal(center2, cov, size=n_samples_target)]
        )
        y_target[i] = np.concatenate(
            [np.zeros(n_samples_target),
             np.ones(n_samples_target)]
        )

    X_source = np.concatenate(X_source)
    y_source = np.concatenate(y_source)
    X_target = np.concatenate(X_target)
    y_target = np.concatenate(y_target)

    return X_source, y_source, X_target, y_target
