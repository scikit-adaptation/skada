from functools import reduce
import os
from typing import Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
from sklearn.utils import Bunch


_DEFAULT_HOME_FOLDER_KEY = "SKADA_DATA_FOLDER"
_DEFAULT_HOME_FOLDER = "~/skada_datasets"

DomainDataType = Union[
    # (name, X, y)
    Tuple[str, np.ndarray, np.ndarray],
    # (X, y)
    Tuple[np.ndarray, np.ndarray],
    # (X,)
    Tuple[np.ndarray,],
]


def get_data_home(data_home: Optional[Union[str, os.PathLike]] = None) -> str:
    """Return the path of the `skada` data folder.

    This folder is used by some large dataset loaders to avoid downloading the
    data several times.

    By default the data directory is set to a folder named 'skada_datasets' in the
    user home folder.

    Alternatively, it can be set by the 'SKADA_DATA_FOLDER' environment
    variable or programmatically by giving an explicit folder path. The '~'
    symbol is expanded to the user home folder.

    If the folder does not already exist, it is automatically created.

    Parameters
    ----------
    data_home : str or path-like, default=None
        The path to `skada` data folder. If `None`, the default path
        is `~/skada_datasets`.

    Returns
    -------
    data_home: str
        The path to `skada` data folder.
    """
    if data_home is None:
        data_home = os.environ.get(_DEFAULT_HOME_FOLDER_KEY, _DEFAULT_HOME_FOLDER)
    data_home = os.path.expanduser(data_home)
    os.makedirs(data_home, exist_ok=True)
    return data_home


def _make_da_dataset(
    sources: List[DomainDataType],
    targets: List[DomainDataType],
    return_X_y: bool = True,
    train: bool = False,
) -> Union[Bunch, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Takes all domain datasets and packs them into a single
    domain-aware representation compatible with DA estimators.

    Parameters
    ----------
    sources : list
        Data samples and labels associated with the source domain.
        Each item of the list should be tuple of (X, y). Optionally,
        the name for the source could be provided (source_name, X, y).
        Otherwise it will be generated automatically.
    targets : list
        Data samples and labels (optional) associated with the
        target domain. Each item of the list should be tuple of (X, y).
        Optionally, the name for the target could be provided:
        (target_name, X, y). Otherwise it will be generated automatically.
    return_X_y : bool, default=True
        When set to True, returns a tuple (X, y, sample_domain). Otherwise
        returns :class:`~sklearn.utils.Bunch` object with the structure
        described below.
    train: bool, default=False
        When set to True, masks labels for target domains with -1, so
        they are not available at train time.

    Returns
    -------
    data : :class:`~sklearn.utils.Bunch`
        Dictionary-like object, with the following attributes.

        data: ndarray
            Samples from all sources and all targets given.
        target : ndarray
            Target labels from all sources and all targets.
        sample_domain : ndarray)
            The integer label for domain the sample was taken from.
            By convention, source domains have non-negative labels,
            and target domain label is always < 0.
        domain_names : dict
            The names of domains and associated domain labels.

    (X, y, sample_domain) : tuple if `return_X_y=True`
        Tuple of (data, target, sample_domain), see the description above.
    """
    source_names, target_names = [], []
    Xs, ys, sample_domains = [], [], []
    for source in sources:
        if len(source) == 1:
            X, = source
            y = -np.ones_like(X)
            source_name = None
        elif len(source) == 2:
            X, y = source
            source_name = None
        elif len(source) == 3:
            source_name, X, y = source
        else:
            raise ValueError("Invalid definition for domain data")
        if source_name is None:
            source_name = f"source_{len(source_names)+1}"
        assert X.shape[0] == y.shape[0]
        assert source_name not in source_names
        source_names.append(source_name)
        # xxx(okachaiev): this is horribly inefficient, re-write when API is fixed
        Xs.append(X)
        ys.append(y)
        sample_domains.append(np.ones_like(y)*(len(source_names)-1))
    # xxx(okachaiev): code duplication, re-write when API is fixed
    for target in targets:
        if len(target) == 1:
            X, = target
            y = -np.ones_like(X)
            target_name = None
        elif len(target) == 2:
            X, y = target
            target_name = None
        elif len(target) == 3:
            target_name, X, y = target
        else:
            raise ValueError("Invalid definition for domain data")
        if train:
            # always mask target labels for training dataset
            y = -np.ones_like(X)
        if target_name is None:
            target_name = f"target_{len(target_names)+1}"
        assert X.shape[0] == y.shape[0]
        assert target_name not in target_names
        target_names.append(target_name)
        # xxx(okachaiev): this is horribly inefficient, rewrite when API is fixed
        Xs.append(X)
        ys.append(y)
        sample_domains.append(-len(target_names) * np.ones_like(y))

    domain_labels = {}
    for idx, source in enumerate(source_names):
        domain_labels[source] = 1+idx
    for idx, target in enumerate(target_names):
        domain_labels[target] = -1-idx

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


class DomainAwareDataset:

    def __init__(
        self,
        # xxx(okachaiev): not sure if dictionary is a good format :thinking:
        domains: Union[List[DomainDataType], Dict[str, DomainDataType], None] = None
    ):
        self.domains_ = domains or []
        # xxx(okachaiev): fill this in if domains are given
        self.domain_names_ = {}

    def add_domain(self, X, y=None, domain_name: Optional[str] = None) -> 'DomainAwareDataset':
        if domain_name is not None:
            # check the name is unique
            # xxx(okachaiev): ValueError would be more appropriate
            assert domain_name not in self.domain_names_
        else:
            domain_name = f"_{len(self.domain_names_)+1}"
        domain_id = len(self.domains_)+1
        self.domains_.append((X, y))
        self.domain_names_[domain_name] = domain_id
        return self

    def get_domain(self, domain_name: str) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        domain_id = self.domain_names_[domain_name]
        return self.domains_[domain_id]

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
        as_sources: List[str],
        as_targets: List[str],
        return_X_y: bool = True,
        train: bool = False,
    ) -> Union[Bunch, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Takes all domain datasets and packs them into a single
        domain-aware representation compatible with DA estimators.

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
            When set to True, masks labels for target domains with -1, so
            they are not available at train time.

        Returns
        -------
        data : :class:`~sklearn.utils.Bunch`
            Dictionary-like object, with the following attributes.

            data: ndarray
                Samples from all sources and all targets given.
            target : ndarray
                Target labels from all sources and all targets.
            sample_domain : ndarray)
                The integer label for domain the sample was taken from.
                By convention, source domains have non-negative labels,
                and target domain label is always < 0.
            domain_names : dict
                The names of domains and associated domain labels.

        (X, y, sample_domain) : tuple if `return_X_y=True`
            Tuple of (data, target, sample_domain), see the description above.
        """
        # xxx(okachaiev): validate there's no intersections
        return _make_da_dataset(
            sources=[(domain, *self.get_domain(domain)) for domain in as_sources],
            targets=[(domain, *self.get_domain(domain)) for domain in as_targets],
            return_X_y=return_X_y,
            train=train,
        )

    def pack_for_train(
        self,
        as_sources: List[str],
        as_targets: List[str],
        return_X_y: bool = True,
    ):
        """Same as pack.
        
        Masks labels for target domains with -1 so they are not available
        at training time.
        """
        return _make_da_dataset(
            sources=[(domain, *self.get_domain(domain)) for domain in as_sources],
            targets=[(domain, *self.get_domain(domain)) for domain in as_targets],
            return_X_y=return_X_y,
            train=True,
        )

    def pack_for_test(
        self,
        as_sources: List[str],
        as_targets: List[str],
        return_X_y: bool = True,
    ):
        return _make_da_dataset(
            sources=[(domain, *self.get_domain(domain)) for domain in as_sources],
            targets=[(domain, *self.get_domain(domain)) for domain in as_targets],
            return_X_y=return_X_y,
            train=False,
        )


# xxx(okachaiev): putting `domain_names` first argument so it's compatible with `partial`
def select_domain(
    domain_names: Dict[str, int],
    sample_domain: np.ndarray,
    domains: Union[str, Iterable[str]]
) -> np.ndarray:
    if isinstance(domains, str):
        domains = [domains]
    # xxx(okachaiev: this version is not the most efficient)
    return reduce(np.logical_or, (sample_domain == domain_names[domain] for domain in domains))
