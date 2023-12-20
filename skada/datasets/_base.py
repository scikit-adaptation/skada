# Author: Oleksii Kachaiev <kachayev@gmail.com>
#
# License: BSD 3-Clause

from functools import reduce
import os
from typing import Dict, Iterable, List, Mapping, Optional, Tuple, Union

import numpy as np
from sklearn.utils import Bunch


_DEFAULT_HOME_FOLDER_KEY = "SKADA_DATA_FOLDER"
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


def get_data_home(data_home: Union[str, os.PathLike, None]) -> str:
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


class DomainAwareDataset:

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
            sample_domains.append(np.ones_like(y)*domain_id)
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
            sample_domains.append(-1 * domain_id * np.ones_like(y))
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


# xxx(okachaiev): putting `domain_names` first argument
# so it's compatible with `partial`
def select_domain(
    domain_names: Dict[str, int],
    sample_domain: np.ndarray,
    domains: Union[str, Iterable[str]]
) -> np.ndarray:
    if isinstance(domains, str):
        domains = [domains]
    # xxx(okachaiev): this version is not the most efficient
    return reduce(
        np.logical_or,
        (sample_domain == domain_names[domain] for domain in domains)
    )
