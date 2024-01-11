import numbers

import numpy as np

from scipy import signal
from scipy.fftpack import rfft, irfft
from scipy.stats import multivariate_normal

from sklearn.datasets import make_blobs

from skada.datasets._base import DomainAwareDataset


from functools import reduce
import os
from typing import Dict, Iterable, List, Mapping, Optional, Tuple, Union

import numpy as np
from sklearn.utils import Bunch






DEFAULT_HOME_FOLDER_KEY = "SKADA_DATA_FOLDER"
_DEFAULT_HOME_FOLDER = "~/skada_datasets"

# xxx(okachaiev): if we use -1 as a detector for targets,
# we should not allow non-labeled dataset or... we need
# to come up with a way to pack them properly
DomainDataType = Union[
    # (name, X, y)
    Tuple[str, np.ndarray, np.ndarray],
    # (X, y)
    Tuple[np.ndarray, np.ndarray],
    # (X,)
    Tuple[np.ndarray, ],
]

PackedDatasetType = Union[
    Bunch,
    Tuple[np.ndarray, np.ndarray, np.ndarray]
]







#U distribution, on a circlar shape (not rectangular)
def _generate_unif_circle(n_samples, rng):
    angle = rng.rand(n_samples, 1) * 2 * np.pi
    r = np.sqrt(rng.rand(n_samples, 1))

    x = np.concatenate((r * np.cos(angle), r * np.sin(angle)), 1)
    return x


def _generate_data_2d_supervised(n_samples, rng, label='binary'):
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
        y = np.concatenate((np.zeros(n1), np.ones(4 * n2)), 0)
        y = y.astype(int)
    elif label == 'multiclass':
        y = np.zeros(n1)
        for i in range(4):
            y = np.concatenate((y, (i + 1) * np.ones(n2)), 0)
            y = y.astype(int)
    elif label == "regression":
        # create label y with gaussian distribution
        sigma = np.array([[3, -0.75], [-0.75, 2]])
        mu = [0.5, 0.5]
        normal_rv = multivariate_normal(mu, sigma)
        y = normal_rv.pdf(x)
    return x, y


def _generate_data_2d_supervised_subspace(n_samples, rng, label='binary'):
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
    n1 = n2 * 2
    # make data of class 1
    Sigma1 = np.array([[0.5, 0], [0, 0.5]])
    mu1 = np.array([-2, 2])
    x1 = rng.randn(n1, 2).dot(Sigma1) + mu1[None, :]

    # make data of the first cluster of class 2
    Sigma2 = np.array([[0.1, 0], [0, 0.1]])
    mu2 = np.array([2.5, 0])

    x21 = rng.randn(n2, 2).dot(Sigma2) + mu2[None, :]

    # make data of the second cluster of class 2
    Sigma2 = np.array([[0.2, 0], [0, 0.2]])
    mu2 = np.array([0, -2.5])

    x22 = rng.randn(n2, 2).dot(Sigma2) + mu2[None, :]

    # concatenate data
    x = np.concatenate((x1, x21, x22), 0)

    # make labels
    if label == 'binary':
        y = np.concatenate((np.zeros(n1), np.ones(2 * n2)), 0).astype(int)
    elif label == 'multiclass':
        y = np.zeros(n1)
        for i in range(2):
            y = np.concatenate((y, (i + 1) * np.ones(n2)), 0).astype(int)
    elif label == 'regression':
        # create label y with gaussian distribution
        normal_rv = multivariate_normal(mu1, Sigma1)
        y = normal_rv.pdf(x)
    return x, y
def make_shifted_datasets(
    n_samples_source=100,
    n_samples_target=100,
    shift="covariate_shift",
    noise=None,
    label='binary',
    ratio=0.9,
    mean=1,
    sigma=0.7,
    gamma=2,
    center=((0, 2)),
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
    shift : tuple, default='covariate_shift'
        Choose the nature of the shift.
        If 'covariate_shift', use covariate shift.
        If 'target_shift', use target shift.
        If 'concept_drift', use concept drift.
        If 'subspace', a subspace where the classes are separable
        independently of the domains exists.
        See detailed description of each shift in [1]_.
    noise : float or array_like, default=None
        If float, standard deviation of Gaussian noise added to the data.
        If array-like, each element of the sequence indicate standard
        deviation of Gaussian noise added to the source and target data.
    ratio : float, default=0.9
        Ratio of the number of data in class 1 selected
        in the target shift and the sample_selection bias
    mean : float, default=1
        value of the translation in the concept drift.
    sigma : float, default=0.7
        multiplicative value of the concept drift.
    gamma :  float, default=2
        Parameter of the RBF kernel.
    center : array-like of shape (1, 2), default=((0, 2))
        Center of the distribution.
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
    X_source, y_source = _generate_data_2d_supervised(n_samples_source, rng, label)

    if shift == "covariate_shift":
        n_samples_target_temp = n_samples_target * 100
        X_target, y_target = _generate_data_2d_supervised(
            n_samples_target_temp, rng, label
        )

        w = np.exp(-gamma * np.sum((X_target - np.array(center)) ** 2, 1))
        w /= w.sum()

        isel = rng.choice(len(w), size=(8 * n_samples_target,), replace=False, p=w)

        X_target = X_target[isel]
        y_target = y_target[isel]

    elif shift == "target_shift":
        n_samples_target_temp = n_samples_target * 3
        X_target, y_target = _generate_data_2d_supervised(
            n_samples_target_temp, rng, label
        )

        n_samples1 = int(8 * n_samples_target * ratio)
        n_samples2 = 8 * n_samples_target - n_samples1
        isel1 = rng.choice(
            8 * n_samples_target_temp // 2,
            size=(n_samples1,),
            replace=False
        )
        isel2 = (
            rng.choice(
                8 * n_samples_target_temp // 2,
                size=(n_samples2,),
                replace=False
            )
        ) + 8 * n_samples_target_temp // 2
        isel = np.concatenate((isel1, isel2))

        X_target = X_target[isel]
        y_target = y_target[isel]

    elif shift == "concept_drift":
        X_target, y_target = _generate_data_2d_supervised(n_samples_target, rng, label)
        X_target = X_target * sigma + mean

    elif shift == "subspace":
        X_source, y_source = _generate_data_2d_supervised_subspace(
            n_samples_source, rng, label
        )
        X_target, y_target = _generate_data_2d_supervised_subspace(
            n_samples_target, rng, label
        )
        X_target *= -1
        if label == "multiclass":
            y_target[2*n_samples_target:3*n_samples_target] += 1
            y_target[3*n_samples_target:] -= 1

    else:
        raise NotImplementedError("unknown shift {}".format(shift))

    if isinstance(noise, numbers.Real):
        X_source += rng.normal(scale=noise, size=X_source.shape)
        X_target += rng.normal(scale=noise, size=X_target.shape)
    elif noise is not None:
        X_source += rng.normal(scale=noise[0], size=X_source.shape)
        X_target += rng.normal(scale=noise[1], size=X_target.shape)

    return X_source, y_source, X_target, y_target




#We create source and target datasets for regression (the y values are continuous)
X_source, y_source, X_target, y_target = make_shifted_datasets(label="regression")
#AS we can see the y-values are floats
print(y_source[0:5])
print(type(y_source[0]))
print(y_target[0:5])
print(type(y_target[0]))
#We then create a dataset with those previously generated datasets
dataset = DomainAwareDataset(domains=[
        (X_source, y_source, 's'),
        (X_target, y_target, 't'),
    ])
x, y, d = dataset.pack(as_sources= ["s"], as_targets = ["t"], return_X_y = True)
# As we can see we have floats in the sample_domain array
print(d[0:5])
print(type(d[0]))
























# Here we are fixing the class, follow the changes thanks to the: #-
# In fact the problem came from np.ones_like in the pack function: when you give an array containing floats (such as the y-s here), the np.ones_like generated array will also contains ones that are floats (1. and not 1)
# We then just need to cast the np.ones_like arrays to change the type of the elements from float to int.


class DomainAwareDataset2:

    def __init__(
        self,
        # xxx(okachaiev): not sure if dictionary is a good format :thinking:
        domains: Union[List[DomainDataType], Dict[str, DomainDataType], None] = None
    ):
        self.domains_ = []
        self.domain_names_ = {}
        # xxx(okachaiev): there should be a simpler way for adding those
        if domains is not None:
            for d in domains:
                if len(d) == 2:
                    X, y = d
                    domain_name = None
                elif len(d) == 3:
                    X, y, domain_name = d
                self.add_domain(X, y=y, domain_name=domain_name)

    def add_domain(
        self,
        X,
        y=None,
        domain_name: Optional[str] = None
    ) -> 'DomainAwareDataset':
        if domain_name is not None:
            # check the name is unique
            # xxx(okachaiev): ValueError would be more appropriate
            assert domain_name not in self.domain_names_
        else:
            domain_name = f"_{len(self.domain_names_)+1}"
        domain_id = len(self.domains_)+1
        self.domains_.append((X, y) if y is not None else (X,))
        self.domain_names_[domain_name] = domain_id
        return self

    def merge(
        self,
        dataset: 'DomainAwareDataset',
        names_mapping: Optional[Mapping] = None
    ) -> 'DomainAwareDataset':
        for domain_name in dataset.domain_names_:
            # xxx(okachaiev): this needs to be more flexible
            # as it should be possible to pass only X with y=None
            # i guess best way of doing so is to change 'add_domain' API
            X, y = dataset.get_domain(domain_name)
            if names_mapping is not None:
                domain_name = names_mapping.get(domain_name, domain_name)
            self.add_domain(X, y, domain_name)
        return self

    def get_domain(self, domain_name: str) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        domain_id = self.domain_names_[domain_name]
        return self.domains_[domain_id-1]

    def select_domain(
        self,
        sample_domain: np.ndarray,
        domains: Union[str, Iterable[str]]
    ) -> np.ndarray:
        return select_domain(self.domain_names_, sample_domain, domains)

    # xxx(okachaiev): i guess, if we are using names to pack domains into array,
    # we should not autogenerate them... otherwise it might be not obvious at all
    def pack(
        self,
        as_sources: List[str] = None,
        as_targets: List[str] = None,
        return_X_y: bool = True,
        train: bool = False,
        mask: Union[None, int, float] = None,
    ) -> PackedDatasetType:
        """Aggregates datasets from all domains into a unified domain-aware
        representation, ensuring compatibility with domain adaptation (DA)
        estimators.

        Parameters
        ----------
        as_sources : list
            List of domain names to be used as sources.
        as_targets : list
            List of domain names to be used as targets.
        return_X_y : bool, default=True
            When set to True, returns a tuple (X, y, sample_domain). Otherwise
            returns :class:`~sklearn.utils.Bunch` object with the structure
            described below.
        train: bool, default=False
            When set to True, masks labels for target domains with -1
            (or a `mask` given), so they are not available at train time.
        mask: int | float (optional), default=None
            Value to mask labels at training time.

        Returns
        -------
        data : :class:`~sklearn.utils.Bunch`
            Dictionary-like object, with the following attributes.

            data: ndarray
                Samples from all sources and all targets given.
            target : ndarray
                Target labels from all sources and all targets.
            sample_domain : ndarray
                The integer label for domain the sample was taken from.
                By convention, source domains have non-negative labels,
                and target domain label is always < 0.
            domain_names : dict
                The names of domains and associated domain labels.

        (X, y, sample_domain) : tuple if `return_X_y=True`
            Tuple of (data, target, sample_domain), see the description above.
        """
        Xs, ys, sample_domains = [], [], []
        domain_labels = {}
        if as_sources is None:
            as_sources = []
        if as_targets is None:
            as_targets = []
        for domain_name in as_sources:
            domain_id = self.domain_names_[domain_name]
            source = self.get_domain(domain_name)
            if len(source) == 1:
                X, = source
                y = -np.ones(X.shape[0], dtype=np.int32)
            elif len(source) == 2:
                X, y = source
            else:
                raise ValueError("Invalid definition for domain data")
            # xxx(okachaiev): this is horribly inefficient, re-write when API is fixed
            Xs.append(X)
            ys.append(y)
            #- here is the first change: -= *domain_id) ### += .astype(int)*domain_id)
            sample_domains.append(np.ones_like(y).astype(int)*domain_id)
            domain_labels[domain_name] = domain_id
        # xxx(okachaiev): code duplication, re-write when API is fixed
        dtype = None
        for domain_name in as_targets:
            domain_id = self.domain_names_[domain_name]
            target = self.get_domain(domain_name)
            if len(target) == 1:
                X, = target
                # xxx(okachaiev): for what it's worth, we should likely to
                # move the decision about dtype to the very end of the list
                y = -np.ones(X.shape[0], dtype=np.int32)
            elif len(target) == 2:
                X, y = target
            else:
                raise ValueError("Invalid definition for domain data")
            if train:
                if mask is not None:
                    y = np.array([mask] * X.shape[0], dtype=dtype)
                elif y.dtype in (np.int32, np.int64):
                    y = -np.ones(X.shape[0], dtype=y.dtype)
                    # make sure that the mask is reused on the next iteration
                    mask, dtype = -1, y.dtype
                elif y.dtype in (np.float32, np.float64):
                    y = np.array([np.nan] * X.shape[0], dtype=y.dtype)
                    # make sure that the  mask is reused on the next iteration
                    mask, dtype = np.nan, y.dtype
            # xxx(okachaiev): this is horribly inefficient, rewrite when API is fixed
            Xs.append(X)
            ys.append(y)
            #- here is the second change: -= ) ### += .astype(int) )
            sample_domains.append(-1 * domain_id * np.ones_like(y).astype(int) )
            domain_labels[domain_name] = -1 * domain_id

        # xxx(okachaiev): so far this only works if source and target has the same size
        data = np.concatenate(Xs)
        target = np.concatenate(ys)
        sample_domain = np.concatenate(sample_domains)
        return (data, target, sample_domain) if return_X_y else Bunch(
            data=data,
            target=target,
            sample_domain=sample_domain,
            domain_names=domain_labels,
        )

    def pack_train(
        self,
        as_sources: List[str],
        as_targets: List[str],
        return_X_y: bool = True,
        mask: Union[None, int, float] = None,
    ) -> PackedDatasetType:
        """Same as `pack`.

        Masks labels for target domains with -1 so they are not available
        at training time.
        """
        return self.pack(
            as_sources=as_sources,
            as_targets=as_targets,
            return_X_y=return_X_y,
            train=True,
            mask=mask,
        )

    def pack_test(
        self,
        as_targets: List[str],
        return_X_y: bool = True,
    ) -> PackedDatasetType:
        return self.pack(
            as_sources=[],
            as_targets=as_targets,
            return_X_y=return_X_y,
            train=False,
        )

    def pack_lodo(self, return_X_y: bool = True) -> PackedDatasetType:
        """Packages all domains in a format compatible with the Leave-One-Domain-Out
        cross-validator (refer to :class:`~skada.model_selection.LeaveOneDomainOut` for
        more details). To enable the splitter's dynamic assignment of source and target
        domains, data from each domain is included in the output twice — once as a
        source and once as a target.

        Exercise caution when using this output for purposes other than its intended
        use, as this could lead to incorrect results and data leakage.

        Parameters
        ----------
        return_X_y : bool, default=True
            When set to True, returns a tuple (X, y, sample_domain). Otherwise
            returns :class:`~sklearn.utils.Bunch` object with the structure
            described below.

        Returns
        -------
        data : :class:`~sklearn.utils.Bunch`
            Dictionary-like object, with the following attributes.

            data: np.ndarray
                Samples from all sources and all targets given.
            target : np.ndarray
                Target labels from all sources and all targets.
            sample_domain : np.ndarray
                The integer label for domain the sample was taken from.
                By convention, source domains have non-negative labels,
                and target domain label is always < 0.
            domain_names : dict
                The names of domains and associated domain labels.

        (X, y, sample_domain) : tuple if `return_X_y=True`
            Tuple of (data, target, sample_domain), see the description above.
        """
        return self.pack(
            as_sources=list(self.domain_names_.keys()),
            as_targets=list(self.domain_names_.keys()),
            return_X_y=return_X_y,
            train=True,
        )

    def __str__(self) -> str:
        return f"DomainAwareDataset(domains={list(self.domain_names_.keys())})"

    def __repr__(self) -> str:
        return self.__str__()


# Then we can retry the same experience:
X_source, y_source, X_target, y_target = make_shifted_datasets(label="regression")
# Again the y-values are floats
print(y_source[0:5])
print(type(y_source[0]))
print(y_target[0:5])
print(type(y_target[0]))
#The new dataset is using the new class
dataset = DomainAwareDataset2(domains=[
        (X_source, y_source, 's'),
        (X_target, y_target, 't'),
    ])
X, y, d = dataset.pack(as_sources= ["s"], as_targets = ["t"], return_X_y = True)
# However we cna notice here we have int values
print(d[0:5])
print(type(d[0]))
print(X_source, X_source.shape)
