# Author: Theo Gnassounou <theo.gnassounou@inria.fr>
#         Remi Flamary <remi.flamary@polytechnique.edu>
#         Oleksii Kachaiev <kachayev@gmail.com>
#         Bueno Ruben <ruben.bueno@polytechnique.edu>
#
# License: BSD 3-Clause

import numbers

import numpy as np
from scipy import signal
from scipy.fftpack import irfft, rfft
from scipy.stats import multivariate_normal
from sklearn.datasets import make_blobs

from ._base import DomainAwareDataset


def _generate_unif_circle(n_samples, rng):
    angle = rng.rand(n_samples, 1) * 2 * np.pi
    r = np.sqrt(rng.rand(n_samples, 1))
    x = np.concatenate((r * np.cos(angle), r * np.sin(angle)), 1)
    return x


def _generate_data_2d_classif(
    n_samples,
    rng,
    mu_regression=None,
    sigma_regression=None,
    regression_scaling_constant=27,
    label="binary",
):
    """Generate 2d classification data.

    Parameters
    ----------
    n_samples : int
        It is the total number of points among one clusters.
        At the end the number of point are 8*n_samples
    rng : random generator
        Generator for dataset creation
    mu_regression : np.array, default=np.array([0, 0])
        Will only be used if label=='regression'
        When it's value is None, it will be changed to be the default one
    sigma_regression : np.array, default=np.array([[1, 0], [0, 1]])
        Will only be used if label=='regression'
        When it's value is None, it will be changed to be the default one
    regression_scaling_constant: float, default=27
        Constant by which we multiply the y-value when label=='regression'
    label : tuple, default='binary'
        If 'binary, return binary class.
        If 'multiclass', return multiclass.
        If 'regression', return regression's y-values
    """
    if mu_regression is None:
        mu_regression = np.array([0, 0])
    if sigma_regression is None:
        sigma_regression = np.array([[1, 0], [0, 1]])

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
    if label == "binary":
        y = np.concatenate((np.zeros(n1), np.ones(4 * n2)), 0)
        y = y.astype(int)
    elif label == "multiclass":
        y = np.zeros(n1)
        for i in range(4):
            y = np.concatenate((y, (i + 1) * np.ones(n2)), 0)
            y = y.astype(int)
    elif label == "regression":
        # create label y with gaussian distribution
        normal_rv = multivariate_normal(mu_regression, sigma_regression)
        y = normal_rv.pdf(x) * regression_scaling_constant
    else:
        raise ValueError(
            f"Invalid label value: {label}. The label should either be "
            "'binary', 'multiclass' or 'regression'"
        )
    return x, y


def _generate_data_2d_classif_subspace(
    n_samples,
    rng,
    mu_regression=None,
    sigma_regression=None,
    regression_scaling_constant=27,
    label="binary",
):
    """Generate 2d classification data.

    Parameters
    ----------
    n_samples : int
        It is the total number of points among one clusters.
        At the end the number of point are 8*n_samples
    rng : random generator
        Generator for dataset creation
    mu_regression : float, default=0.
        Will only be used if label=='regression'
    sigma_regression : float, default=1.
        Will only be used if label=='regression'
        When it's value is None, it will be changed to be the default one
    regression_scaling_constant: float, default=27
        Constant by which we multiply the y-value when label=='regression'
        When it's value is None, it will be changed to be the default one
    label : str, default='binary'
        If 'binary, return binary class.
        If 'multiclass', return multiclass.
        If 'regression', return regression's y-values
    """
    if mu_regression is None:
        mu_regression = 0
    if sigma_regression is None:
        sigma_regression = 1

    n2 = n_samples
    n1 = n2 * 2
    # make data of class 1
    Sigma1 = np.array([[0.5, 0], [0, 0.5]])
    mu1 = np.array([-1, 1])
    x1 = rng.randn(n1, 2).dot(Sigma1) + mu1[None, :]

    # make data of the first cluster of class 2
    Sigma2 = np.array([[0.1, 0], [0, 0.1]])
    mu2 = np.array([1.5, 0.5])

    x21 = rng.randn(n2, 2).dot(Sigma2) + mu2[None, :]

    # make data of the second cluster of class 2
    Sigma2 = np.array([[0.2, 0], [0, 0.2]])
    mu2 = np.array([-0.5, -1.5])

    x22 = rng.randn(n2, 2).dot(Sigma2) + mu2[None, :]

    # concatenate data
    x = np.concatenate((x1, x21, x22), 0)

    # make labels
    if label == "binary":
        y = np.concatenate((np.zeros(n1), np.ones(2 * n2)), 0)
        y = y.astype(int)
    elif label == "multiclass":
        k = 4
        y = np.zeros(n1 + n1 % k)
        for i in range(k):
            y = np.concatenate((y, (i + 1) * np.ones(n1 // k)), 0)
            y = y.astype(int)
    elif label == "regression":
        # create label y with gaussian distribution
        normal_rv = multivariate_normal(mu_regression, sigma_regression)
        # We project on the line: y=x
        X_projected = (x[:, 0] + x[:, 1]) / 2
        y = normal_rv.pdf(X_projected) * regression_scaling_constant
    else:
        raise ValueError(
            f"Invalid label value: {label}. The label should either be "
            "'binary', 'multiclass' or 'regression'"
        )
    return x, y


def _generate_data_from_moons(n_samples, index, rng):
    """Generate two gaussian clusters with centers draw from two moons.

    Parameters
    ----------
    n_samples : int
        It is the total number of points among one cluster.
    index : float,
        Give the position of the centers in the moons
    rng : random generator
        Generator for dataset creation
    """
    n_samples_circ = 100
    outer_circ_x = np.cos(np.linspace(0, np.pi, n_samples_circ))
    outer_circ_y = np.sin(np.linspace(0, np.pi, n_samples_circ))
    inner_circ_x = 1 - np.cos(np.linspace(0, np.pi, n_samples_circ))
    inner_circ_y = 1 - np.sin(np.linspace(0, np.pi, n_samples_circ)) - 0.5

    index = int(index * n_samples_circ)
    cov = [[0.01, 0], [0, 0.01]]
    center1 = np.array([outer_circ_x[index], outer_circ_y[index]])
    center2 = np.array([inner_circ_x[index], inner_circ_y[index]])

    X = np.concatenate(
        [
            rng.multivariate_normal(center1, cov, size=n_samples),
            rng.multivariate_normal(center2, cov, size=n_samples),
        ]
    )
    y = np.concatenate([np.zeros(n_samples), np.ones(n_samples)])

    return X, y


def _generate_signal_with_peak_frequency(
    n_samples, n_channels, input_size, frequencies, band_size, sigma_freq, fs, rng
):
    X = []
    y = []
    n_classes, n_frequencies = frequencies.shape
    for n_class in range(n_classes):
        X_new = np.zeros((n_samples, n_channels, input_size))
        for n_frequency in range(n_frequencies):
            channel_weights = rng.uniform(0.5, 1, size=(n_channels))
            X_random = rng.normal(0, 1, size=(n_samples, n_channels, input_size))
            for i in range(n_samples):
                frequency = (
                    rng.normal(
                        frequencies[n_class, n_frequency], sigma_freq * band_size
                    )
                    + 1e-5
                )
                if frequency < 0:
                    frequency = -frequency
                sos = signal.butter(
                    10,
                    [frequency, frequency + band_size],
                    "bandpass",
                    fs=fs,
                    output="sos",
                )

                X_filtered = signal.sosfilt(sos, X_random[i])

                for j in range(n_channels):
                    X_fft = rfft(X_filtered[:, j])
                    X_filtered[:, j] = irfft(X_fft * channel_weights[j])
                X_new[i] += X_filtered
        X.append(X_new)
        y.append([n_class for _ in range(n_samples)])
    X = np.concatenate(X)
    y = np.concatenate(y)
    return X, y


def make_shifted_blobs(
    n_samples=100,
    n_features=2,
    shift=0.10,
    noise=None,
    centers=None,
    cluster_std=1.0,
    random_state=None,
    return_X_y=True,
    return_dataset=False,
):
    """Generate source and shift target isotropic Gaussian blobs .

    Parameters
    ----------
    n_samples : int, default=100
        It is the total number of points equally divided among clusters.
    n_features : int, default=2
        The number of features for each sample.
    shift : float or array like, default=0.10
        If float, it is the value of the translation for every target feature.
        If array_like, each element of the sequence indicates the value of
        the translation for each target features.
    noise : float or array_like, default=None
        If float, standard deviation of Gaussian noise added to the data.
        If array-like, each element of the sequence indicate standard
        deviation of Gaussian noise added to the source and target data.
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
    return_X_y : boolean, optional (default=True)
        Returns source and target dataset as a pair of (X, y) tuples (for
        the source and the target respectively). Otherwise returns tuple of
        (X, y, sample_domain) where `sample_domain` is a categorical label
        for the domain where sample is taken.
    return_dataset : boolean, optional (default=False)
        When set to `True`, the function returns
        :class:`~skada.datasets.DomainAwareDataset` object.

    Returns
    -------
    (X, y, sample_domain) : tuple if `return_X_y=True`
        Tuple of (data, target, sample_domain), see the description below.

    data : :class:`~sklearn.utils.Bunch`
        Dictionary-like object, with the following attributes.

        X: ndarray
            Samples from all sources and all targets given.
        y : ndarray
            Labels from all sources and all targets.
        sample_domain : ndarray
            The integer label for domain the sample was taken from.
            By convention, source domains have non-negative labels,
            and target domain label is always < 0.
        domain_names : dict
            The names of domains and associated domain labels.

    dataset : :class:`~skada.datasets.DomainAwareDataset`
        Dataset object.
    """
    rng = np.random.RandomState(random_state)

    X_source, y_source = make_blobs(
        n_samples=n_samples,
        centers=centers,
        n_features=n_features,
        random_state=random_state,
        cluster_std=cluster_std,
    )

    X_target = X_source + shift
    y_target = y_source

    if isinstance(noise, numbers.Real):
        X_source += rng.normal(scale=noise, size=X_source.shape)
        X_target += rng.normal(scale=noise, size=X_target.shape)
    elif noise is not None:
        X_source += rng.normal(scale=noise[0], size=X_source.shape)
        X_target += rng.normal(scale=noise[1], size=X_target.shape)

    dataset = DomainAwareDataset(
        domains=[
            (X_source, y_source, "s"),
            (X_target, y_target, "t"),
        ]
    )
    if return_dataset:
        return dataset
    else:
        return dataset.pack(
            as_sources=["s"],
            as_targets=["t"],
            return_X_y=return_X_y,
            mask_target_labels=False,
        )


def make_shifted_datasets(
    n_samples_source=100,
    n_samples_target=100,
    shift="covariate_shift",
    noise=None,
    label="binary",
    ratio=0.9,
    mean=1,
    sigma=0.7,
    gamma=2,
    mu_regression=None,
    sigma_regression=None,
    regression_scaling_constant=27,
    center=((0, 2)),
    center_cov_shift=((0, 2)),
    standardize=False,
    random_state=None,
    return_X_y=True,
    return_dataset=False,
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
    shift : tuple, default='covariate_shift'
        Choose the nature of the shift.
        If 'covariate_shift', use covariate shift.
        If 'target_shift', use target shift.
        If 'conditional_shift', use conditional shift.
        If 'subspace', a subspace where the classes are separable
        independently of the domains exists.
        See detailed description of each shift in [1]_.
    noise : float or array_like, default=None
        If float, standard deviation of Gaussian noise added to the data.
        If array-like, each element of the sequence indicate standard
        deviation of Gaussian noise added to the source and target data.
    label : str, default='binary'
        If 'binary, generates binary class labels.
        If 'multiclass', generates multiclass labels.
        If 'regression', generates regression's y-values.
    ratio : float, default=0.9
        Ratio of the number of data in class 1 selected
        in the target shift and the sample_selection bias
    mean : float, default=1
        value of the translation in the conditional shift.
    sigma : float, default=0.7
        multiplicative value of the conditional shift.
    mu_regression : np.array|float, default=None
        Will only be used if label=='regression'
        should be 2x1 matrix when shift != 'subspace'
        should be a scalar when shift == 'subspace'
    sigma_regression : np.array|float, default=None
        Will only be used if label=='regression'
        should be a 2x2 matrix when shift != 'subspace'
        should be a scalar when shift == 'subspace'
    regression_scaling_constant: float, default=27
        Constant by which we multiply the y-value when label=='regression'
    gamma :  float, default=2
        Parameter of the RBF kernel.
    center : array-like of shape (1, 2), default=((0, 2))
        Center of the distribution.
    center_cov_shift : array-like of shape (1, 2), default=((0, 2))
        Center of the covariate-shift.
    standardize : bool, default=False
        If True, the data is standardized.
        The standard score of a sample x is calculated as:
        z = (x - u) / s
        where u is the mean and s is the standard deviation of the data.
    random_state : int, RandomState instance or None, default=None
        Determines random number generation for dataset creation. Pass an int
        for reproducible output across multiple function calls.
    return_X_y : boolean, optional (default=True)
        Returns source and target dataset as a pair of (X, y) tuples (for
        the source and the target respectively). Otherwise returns tuple of
        (X, y, sample_domain) where `sample_domain` is a categorical label
        for the domain where sample is taken.
    return_dataset : boolean, optional (default=False)
        When set to `True`, the function returns
        :class:`~skada.datasets.DomainAwareDataset` object.

    Returns
    -------
    (X, y, sample_domain) : tuple if `return_X_y=True`
        Tuple of (data, target, sample_domain), see the description below.

    data : :class:`~sklearn.utils.Bunch`
        Dictionary-like object, with the following attributes.

        X: ndarray
            Samples from all sources and all targets given.
        y : ndarray
            Labels from all sources and all targets.
        sample_domain : ndarray
            The integer label for domain the sample was taken from.
            By convention, source domains have non-negative labels,
            and target domain label is always < 0.
        domain_names : dict
            The names of domains and associated domain labels.

    dataset : :class:`~skada.datasets.DomainAwareDataset`
        Dataset object.

    References
    ----------
    .. [1]  Moreno-Torres, J. G., Raeder, T., Alaiz-Rodriguez,
            R., Chawla, N. V., and Herrera, F. (2012).
            A unifying view on dataset shift in classification.
            Pattern recognition, 45(1):521-530.
    """
    rng = np.random.RandomState(random_state)
    X_source, y_source = _generate_data_2d_classif(
        n_samples_source,
        rng,
        mu_regression,
        sigma_regression,
        regression_scaling_constant,
        label,
    )

    if shift == "covariate_shift":
        n_samples_target_temp = n_samples_target * 100
        X_target, y_target = _generate_data_2d_classif(
            n_samples_target_temp,
            rng,
            mu_regression,
            sigma_regression,
            regression_scaling_constant,
            label,
        )

        w = np.exp(-gamma * np.sum((X_target - np.array(center_cov_shift)) ** 2, 1))
        w /= w.sum()

        isel = rng.choice(len(w), size=(8 * n_samples_target,), replace=False, p=w)

        X_target = X_target[isel]
        y_target = y_target[isel]

    elif shift == "target_shift":
        n_samples_target_temp = n_samples_target * 3
        X_target, y_target = _generate_data_2d_classif(
            n_samples_target_temp,
            rng,
            mu_regression,
            sigma_regression,
            regression_scaling_constant,
            label,
        )

        n_samples1 = int(8 * n_samples_target * ratio)
        n_samples2 = 8 * n_samples_target - n_samples1
        isel1 = rng.choice(
            8 * n_samples_target_temp // 2, size=(n_samples1,), replace=False
        )
        isel2 = (
            rng.choice(
                8 * n_samples_target_temp // 2, size=(n_samples2,), replace=False
            )
        ) + 8 * n_samples_target_temp // 2
        isel = np.concatenate((isel1, isel2))

        X_target = X_target[isel]
        y_target = y_target[isel]

    elif shift == "conditional_shift":
        X_target, y_target = _generate_data_2d_classif(
            n_samples_target,
            rng,
            mu_regression,
            sigma_regression,
            regression_scaling_constant,
            label,
        )
        X_target = X_target * sigma + mean

    elif shift == "subspace":
        X_source, y_source = _generate_data_2d_classif_subspace(
            n_samples_source,
            rng,
            mu_regression,
            sigma_regression,
            regression_scaling_constant,
            label,
        )
        X_target, y_target = _generate_data_2d_classif_subspace(
            n_samples_target,
            rng,
            mu_regression,
            sigma_regression,
            regression_scaling_constant,
            label,
        )
        X_target *= -1

    else:
        raise ValueError(
            f"Invalid shift value: {shift}. The shift should either be "
            "'covariate_shift', 'target_shift', 'conditional_shift' "
            "or 'subspace'"
        )

    if isinstance(noise, numbers.Real):
        X_source += rng.normal(scale=noise, size=X_source.shape)
        X_target += rng.normal(scale=noise, size=X_target.shape)
    elif noise is not None:
        X_source += rng.normal(scale=noise[0], size=X_source.shape)
        X_target += rng.normal(scale=noise[1], size=X_target.shape)

    if standardize:
        X = np.vstack((X_source, X_target))
        mean = X.mean(axis=0)
        std = X.std(axis=0)
        X_source = (X_source - mean) / std
        X_target = (X_target - mean) / std

    dataset = DomainAwareDataset(
        domains=[
            (X_source, y_source, "s"),
            (X_target, y_target, "t"),
        ]
    )
    if return_dataset:
        return dataset
    else:
        return dataset.pack(
            as_sources=["s"],
            as_targets=["t"],
            return_X_y=return_X_y,
            mask_target_labels=False,
        )


def make_dataset_from_moons_distribution(
    n_samples_source=10,
    n_samples_target=10,
    noise=None,
    pos_source=0.1,
    pos_target=0.2,
    random_state=None,
    return_X_y=True,
    return_dataset=False,
):
    """Make dataset from moons.

    Parameters
    ----------
    n_samples_source : int, default=100
        It is the total number of points among one
        source cluster.
    n_samples_target : int, default=100
        It is the total number of points among one
        target cluster.
    noise : float or array_like, default=None
        If float, standard deviation of Gaussian noise added to the data.
        If array-like, each element of the sequence indicate standard
        deviation of Gaussian noise added to the source and target data.
    pos_source : float or array-like, default=0.1
        If float, indicate the center of the source cluster.
        If array-like, each element of the sequence indicates the position
        of the center of each source cluster.
    pos_target : float or array-like, default=0.2
        If float, indicate the center of the source cluster.
        If array-like, each element of the sequence indicates the position
        of the center of each target cluster.
    random_state : int, RandomState instance or None, default=None
        Determines random number generation for dataset creation. Pass an int
        for reproducible output across multiple function calls.
    return_X_y : boolean, optional (default=True)
        Returns source and target dataset as a pair of (X, y) tuples (for
        the source and the target respectively). Otherwise returns tuple of
        (X, y, sample_domain) where `sample_domain` is a categorical label
        for the domain where sample is taken.
    return_dataset : boolean, optional (default=False)
        When set to `True`, the function returns
        :class:`~skada.datasets.DomainAwareDataset` object.

    Returns
    -------
    (X, y, sample_domain) : tuple if `return_X_y=True`
        Tuple of (data, target, sample_domain), see the description below.

    data : :class:`~sklearn.utils.Bunch`
        Dictionary-like object, with the following attributes.

        X: ndarray
            Samples from all sources and all targets given.
        y : ndarray
            Labels from all sources and all targets.
        sample_domain : ndarray
            The integer label for domain the sample was taken from.
            By convention, source domains have non-negative labels,
            and target domain label is always < 0.
        domain_names : dict
            The names of domains and associated domain labels.

    dataset : :class:`~skada.datasets.DomainAwareDataset`
        Dataset object.
    """
    rng = np.random.RandomState(random_state)

    dataset = DomainAwareDataset(domains=[])
    sources = []
    targets = []

    if isinstance(pos_source, numbers.Real):
        X_source, y_source = _generate_data_from_moons(
            n_samples_source, pos_source, rng
        )
        if isinstance(noise, numbers.Real):
            X_source += rng.normal(scale=noise, size=X_source.shape)
        elif noise is not None:
            X_source += rng.normal(scale=noise[0], size=X_source.shape)
        dataset.add_domain(X_source, y_source, "s")
        sources.append("s")
    else:
        for i, pos in enumerate(pos_source):
            X, y = _generate_data_from_moons(n_samples_source, pos, rng)
            if isinstance(noise, numbers.Real):
                X += rng.normal(scale=noise, size=X.shape)
            elif noise is not None:
                X += rng.normal(scale=noise[0], size=X.shape)
            dataset.add_domain(X, y, f"s{i}")
            sources.append(f"s{i}")

    if isinstance(pos_target, numbers.Real):
        X_target, y_target = _generate_data_from_moons(
            n_samples_target, pos_target, rng
        )
        if isinstance(noise, numbers.Real):
            X_target += rng.normal(scale=noise, size=X_target.shape)
        elif noise is not None:
            X_target += rng.normal(scale=noise[1], size=X_target.shape)
        dataset.add_domain(X_target, y_target, "t")
        targets.append("t")
    else:
        for i, pos in enumerate(pos_target):
            X, y = _generate_data_from_moons(n_samples_target, pos, rng)
            if isinstance(noise, numbers.Real):
                X += rng.normal(scale=noise, size=X.shape)
            elif noise is not None:
                X += rng.normal(scale=noise[1], size=X.shape)
            dataset.add_domain(X, y, f"t{i}")
            targets.append(f"t{i}")

    if return_dataset:
        return dataset
    else:
        return dataset.pack(
            as_sources=sources,
            as_targets=targets,
            return_X_y=return_X_y,
            mask_target_labels=False,
        )


def make_variable_frequency_dataset(
    n_samples_source=10,
    n_samples_target=10,
    n_channels=1,
    n_frequencies=1,
    n_classes=3,
    delta_f=1,
    band_size=1,
    sigma_freq=0.25,
    sigma_ch=1,
    noise=None,
    random_state=None,
    return_X_y=True,
    return_dataset=False,
):
    """Make dataset with different peak frequency.

    Parameters
    ----------
    n_samples_source : int, default=100
        It is the total number of points among one
        source cluster.
    n_samples_target : int, default=100
        It is the total number of points among one
        target cluster.
    n_channels : int, default=1
        Number of channels in the signal.
    n_frequency_source : int, default=1
        Number of channels which generate frequency peak and propagate
        to other channels.
    n_classes : int, default=3
        Number of classes in the signals. One class correspond to a
        specific frequency band.
    delta_f : float, default=1
        Band frequency shift of the target data.
    band_size :  float, default=1
        Size of the frequency band.
    sigma_ch : float, default=1
        Std for the gaussian on the channels.
    noise : float or array_like, default=None
        If float, standard deviation of Gaussian noise added to the data.
        If array-like, each element of the sequence indicate standard
        deviation of Gaussian noise added to the source and target data.
    random_state : int, RandomState instance or None, default=None
        Determines random number generation for dataset creation. Pass an int
        for reproducible output across multiple function calls.
    return_X_y : boolean, optional (default=True)
        Returns source and target dataset as a pair of (X, y) tuples (for
        the source and the target respectively). Otherwise returns tuple of
        (X, y, sample_domain) where `sample_domain` is a categorical label
        for the domain where sample is taken.
    return_dataset : boolean, optional (default=False)
        When set to `True`, the function returns
        :class:`~skada.datasets.DomainAwareDataset` object.

    Returns
    -------
    (X, y, sample_domain) : tuple if `return_X_y=True`
        Tuple of (data, target, sample_domain), see the description below.

    data : :class:`~sklearn.utils.Bunch`
        Dictionary-like object, with the following attributes.

        X: ndarray
            Samples from all sources and all targets given.
        y : ndarray
            Labels from all sources and all targets.
        sample_domain : ndarray
            The integer label for domain the sample was taken from.
            By convention, source domains have non-negative labels,
            and target domain label is always < 0.
        domain_names : dict
            The names of domains and associated domain labels.

    dataset : :class:`~skada.datasets.DomainAwareDataset`
        Dataset object.
    """
    rng = np.random.RandomState(random_state)
    input_size = 3000
    fs = 100
    highest_frequency = 15
    frequencies = rng.choice(
        highest_frequency, size=(n_classes, n_frequencies), replace=False
    )

    X_source, y_source = _generate_signal_with_peak_frequency(
        n_samples_source,
        n_channels,
        input_size,
        frequencies,
        band_size,
        sigma_freq,
        fs,
        rng,
    )

    X_target, y_target = _generate_signal_with_peak_frequency(
        n_samples_target,
        n_channels,
        input_size,
        frequencies + delta_f,
        band_size,
        sigma_freq,
        fs,
        rng,
    )

    if isinstance(noise, numbers.Real):
        X_source += rng.normal(scale=noise, size=X_source.shape)
        X_target += rng.normal(scale=noise, size=X_target.shape)
    elif noise is not None:
        X_source += rng.normal(scale=noise[0], size=X_source.shape)
        X_target += rng.normal(scale=noise[1], size=X_target.shape)

    dataset = DomainAwareDataset(
        domains=[
            (X_source, y_source, "s"),
            (X_target, y_target, "t"),
        ]
    )
    if return_dataset:
        return dataset
    else:
        return dataset.pack(
            as_sources=["s"],
            as_targets=["t"],
            return_X_y=return_X_y,
            mask_target_labels=False,
        )
