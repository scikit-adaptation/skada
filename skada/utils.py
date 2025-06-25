# Author: Yanis Lalou <yanis.lalou@polytechnique.edu>
#         Antoine Collas <contact@antoinecollas.fr>
#
# License: BSD 3-Clause

import warnings
from itertools import chain
from typing import Optional, Sequence, Set

import numpy as np
from scipy.optimize import LinearConstraint, minimize
from sklearn.utils import check_array, check_consistent_length

from skada._utils import (
    _DEFAULT_MASKED_TARGET_CLASSIFICATION_LABEL,
    _DEFAULT_MASKED_TARGET_REGRESSION_LABEL,
    _DEFAULT_SOURCE_DOMAIN_LABEL,
    _DEFAULT_TARGET_DOMAIN_LABEL,
    _DEFAULT_TARGET_DOMAIN_ONLY_LABEL,
    _check_y_masking,
    Y_Type,
    _find_y_type
)


def check_X_y_domain(
    X,
    y,
    sample_domain=None,
    allow_source: bool = True,
    allow_multi_source: bool = True,
    allow_target: bool = True,
    allow_multi_target: bool = True,
    allow_common_domain_idx: bool = True,
    allow_auto_sample_domain: bool = True,
    allow_nd: bool = False,
    allow_label_masks: bool = True,
):
    """
    Input validation for domain adaptation (DA) estimator.
    If we work in single-source and single target mode, return source and target
    separately to avoid additional scan for 'sample_domain' array.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Input features
    y : array-like of shape (n_samples,)
        Target variable
    sample_domain : array-like, scalar, or None, optional (default=None)
        Array specifying the domain labels for each sample. A scalar value
        can be provided to assign all samples to the same domain.
    allow_source : bool, optional (default=True)
        Allow the presence of source domains.
    allow_multi_source : bool, optional (default=True)
        Allow multiple source domains.
    allow_target : bool, optional (default=True)
        Allow the presence of target domains.
    allow_multi_target : bool, optional (default=True)
        Allow multiple target domains.
    allow_common_domain_idx : bool, optional (default=True)
        Allow the same domain index to be used for source and target domains, e.g 1 for a source domain and -1 for a target domain.
    allow_auto_sample_domain : bool, optional (default=True)
        Allow automatic generation of sample_domain if not provided.
    allow_nd : bool, optional (default=False)
        Allow X and y to be N-dimensional arrays.
    allow_label_masks : bool, optional (default=True)
        Allow NaNs in y.

    Returns
    -------
    X : array
        Input features
    y : array
        Target variable
    sample_domain : array
        Array specifying the domain labels for each sample.
    """
    X = check_array(X, input_name='X', allow_nd=allow_nd)
    y = check_array(y, ensure_all_finite=not allow_label_masks, ensure_2d=False, input_name='y')
    check_consistent_length(X, y)

    if sample_domain is None and not allow_auto_sample_domain:
        raise ValueError("Either 'sample_domain' or 'allow_auto_sample_domain' "
                         "should be set")
    elif sample_domain is None and allow_auto_sample_domain:
        y_type = _check_y_masking(y)
        sample_domain = _DEFAULT_SOURCE_DOMAIN_LABEL*np.ones_like(y)
        # labels masked with -1 (for classification) are recognized as targets,
        # labels masked with nan (for regression) are recognized as targets,
        # the rest is treated as a source
        if y_type == Y_Type.DISCRETE:
            mask = (y == _DEFAULT_MASKED_TARGET_CLASSIFICATION_LABEL)
        else:
            mask = (np.isnan(y))
        sample_domain[mask] = _DEFAULT_TARGET_DOMAIN_LABEL

    if np.isscalar(sample_domain):
        sample_domain = sample_domain*np.ones_like(y)

    source_idx = extract_source_indices(sample_domain)

    # xxx(okachaiev): this needs to be re-written to accommodate for a
    # a new domain labeling convention without "intersections"
    n_sources = np.unique(sample_domain[source_idx]).shape[0]
    n_targets = np.unique(sample_domain[~source_idx]).shape[0]

    if not allow_source and n_sources > 0:
        raise ValueError(f"Number of sources provided is {n_sources} "
                         "and 'allow_source' is set to False")
    if not allow_target and n_targets > 0:
        raise ValueError(f"Number of targets provided is {n_targets} "
                         "and 'allow_target' is set to False")
    if not allow_multi_source and n_sources > 1:
        raise ValueError(f"Number of sources provided is {n_sources} "
                         "and 'allow_multi_source' is set to False")
    if not allow_multi_target and n_sources > 1:
        raise ValueError(f"Number of targets provided is {n_targets} "
                         "and 'allow_multi_target' is set to False")

    # Check for unique domain idx
    if not allow_common_domain_idx:
        unique_domain_idx = np.unique(sample_domain)
        unique_domain_idx_abs = np.abs(unique_domain_idx)
        if len(unique_domain_idx) != len(np.unique(unique_domain_idx_abs)):
            raise ValueError("Domain labels should be unique: the same domain "
                             "index should not be used both for source and target")

    return X, y, sample_domain


# xxx(okachaiev): code duplication, just for testing
def check_X_domain(
    X,
    sample_domain,
    *,
    allow_domains: Optional[Set[int]] = None,
    allow_source: bool = True,
    allow_multi_source: bool = True,
    allow_target: bool = True,
    allow_multi_target: bool = True,
    allow_common_domain_idx: bool = True,
    allow_auto_sample_domain: bool = True,
    allow_nd: bool = False,
):
    """
    Input validation for domain adaptation (DA) estimator.
    If we work in single-source and single target mode, return source and target
    separately to avoid additional scan for 'sample_domain' array.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Input features.
    sample_domain : array-like, scalar, or None, optional (default=None)
        Array specifying the domain labels for each sample. A scalar value
        can be provided to assign all samples to the same domain.
    allow_domains : set of int, optional (default=None)
        Set of allowed domain labels. If provided, only these domain labels are allowed.
    allow_source : bool, optional (default=True)
        Allow the presence of source domains.
    allow_multi_source : bool, optional (default=True)
        Allow multiple source domains.
    allow_target : bool, optional (default=True)
        Allow the presence of target domains.
    allow_multi_target : bool, optional (default=True)
        Allow multiple target domains.
    allow_common_domain_idx : bool, optional (default=True)
        Allow the same domain index to be used for source and target domains, e.g 1 for a source domain and -1 for a target domain.
    allow_auto_sample_domain : bool, optional (default=True)
        Allow automatic generation of sample_domain if not provided.
    allow_nd : bool, optional (default=False)
        Allow X and y to be N-dimensional arrays.

    Returns
    -------
    X : array
        Input features.
    sample_domain : array
        Combined domain labels for source and target domains.
    """
    X = check_array(X, input_name='X', allow_nd=allow_nd)

    if sample_domain is None and not allow_auto_sample_domain:
        raise ValueError("Either 'sample_domain' or 'allow_auto_sample_domain' "
                         "should be set")
    elif sample_domain is None and allow_auto_sample_domain:
        # default target domain when sample_domain is not given
        # The idea is that with no labels we always assume
        # target domain (_DEFAULT_TARGET_DOMAIN_ONLY_LABEL)
        sample_domain = (
            _DEFAULT_TARGET_DOMAIN_ONLY_LABEL * np.ones(X.shape[0], dtype=np.int32)
        )

    if np.isscalar(sample_domain):
        sample_domain = sample_domain * np.ones(X.shape[0], dtype=np.int32)

    source_idx = extract_source_indices(sample_domain)
    check_consistent_length(X, sample_domain)

    if allow_domains is not None:
        for domain in np.unique(sample_domain):
            # xxx(okachaiev): re-definition of the wildcards
            wildcard = np.inf if domain >= 0 else -np.inf
            if domain not in allow_domains and wildcard not in allow_domains:
                raise ValueError(f"Unknown domain label '{domain}' given")

    n_sources = np.unique(sample_domain[source_idx]).shape[0]
    n_targets = np.unique(sample_domain[~source_idx]).shape[0]

    if not allow_source and n_sources > 0:
        raise ValueError(f"Number of sources provided is {n_sources} "
                         "and 'allow_source' is set to False")
    if not allow_target and n_targets > 0:
        raise ValueError(f"Number of targets provided is {n_targets} "
                         "and 'allow_target' is set to False")
    if not allow_multi_source and n_sources > 1:
        raise ValueError(f"Number of sources provided is {n_sources} "
                         "and 'allow_multi_source' is set to False")
    if not allow_multi_target and n_sources > 1:
        raise ValueError(f"Number of targets provided is {n_targets} "
                         "and 'allow_multi_target' is set to False")
    
    # Check for unique domain idx
    if not allow_common_domain_idx:
        unique_domain_idx = np.unique(sample_domain)
        unique_domain_idx_abs = np.abs(unique_domain_idx)
        if len(unique_domain_idx) != len(np.unique(unique_domain_idx_abs)):
            raise ValueError("Domain labels should be unique: the same domain "
                             "index should not be used both for source and target")

    return X, sample_domain


def extract_source_indices(sample_domain):
    """Extract the indices of the source samples.

    Parameters
    ----------
    sample_domain : array-like of shape (n_samples,)
        Array specifying the domain labels for each sample.

    Returns
    -------
    source_idx : array
        Boolean array indicating source indices.
    """
    sample_domain = check_array(
        sample_domain,
        dtype=np.int32,
        ensure_2d=False,
        input_name='sample_domain'
    )

    source_idx = (sample_domain >= 0)
    return source_idx


def extract_domains_indices(sample_domain, split_source_target=False):
    """Extract the indices of the specific
    domain samples.

    Parameters
    ----------
    sample_domain : array-like of shape (n_samples,)
        Array specifying the domain labels for each sample.
    split_source_target : bool, optional (default=False)
        Whether to split the source and target domains.

    Returns
    -------
    domains_idx : dict
        A dictionary where keys are unique domain labels
        and values are indexes of the samples belonging to
        the corresponding domain.
    If split_source_target is True, then two dictionaries are returned:
        - source_idx: keys >= 0
        - target_idx: keys < 0
    """
    sample_domain = check_array(
        sample_domain,
        dtype=np.int32,
        ensure_2d=False,
        input_name='sample_domain'
    )

    domain_idx = {}

    unique_domains = np.unique(sample_domain)
    for domain in unique_domains:
        source_idx = (sample_domain == domain)
        domain_idx[domain] = np.flatnonzero(source_idx)

    if split_source_target:
        source_domain_idx = {k: v for k, v in domain_idx.items() if k >= 0}
        target_domain_idx = {k: v for k, v in domain_idx.items() if k < 0}
        return source_domain_idx, target_domain_idx
    else:
        return domain_idx


def source_target_split(
    *arrays,
    sample_domain
):
    r"""Split data into source and target domains

    Parameters
    ----------
    *arrays : sequence of array-like of identical shape (n_samples, n_features)
        Input features and target variable(s), and or sample_weights to be
        split. All arrays should have the same length except if None is given
        then a couple of None variables are returned to allow for optional
        sample_weight.
    sample_domain : array-like of shape (n_samples,)
        Array specifying the domain labels for each sample.

    Returns
    -------
    splits : list, length=2 * len(arrays)
        List containing source-target split of inputs.
    """
    if len(arrays) == 0:
        raise ValueError("At least one array required as input")

    check_consistent_length(arrays)

    source_idx = extract_source_indices(sample_domain)

    return list(chain.from_iterable(
        (a[source_idx], a[~source_idx]) if a is not None else (None, None)
        for a in arrays
    ))

def per_domain_split(
    *arrays,
    sample_domain):
    r"""Split data into multiple source and target domains

    Parameters
    ----------
    *arrays : sequence of array-like of identical shape (n_samples, n_features)
        Input features and target variable(s), and or sample_weights to be
        split. All arrays should have the same length except if None is given
        then a couple of None variables are returned to allow for optional
        sample_weight.
    sample_domain : array-like of shape (n_samples,)
        Array specifying the domain labels for each sample.
    split_source_target : bool, optional (default=False)
        Whether to split the source and target domains.

    Returns
    -------
    source_dict : dict
        dict of source domains (contain tuple of subset of arrays).
    target_dict : dict
        dict of target domains (contain tuple of subset of arrays).
    """
    if len(arrays) == 0:
        raise ValueError("At least one array required as input")

    check_consistent_length(arrays)

    domain_idx = extract_domains_indices(sample_domain, False)

    source_dict = {}
    target_dict = {}

    for domain, idx in domain_idx.items():
        if domain >= 0:
            source_dict[domain] = list(chain.from_iterable(
                (a[idx],) if a is not None else (None,)
                for a in arrays
            ))
        else:
            target_dict[domain] = list(chain.from_iterable(
                (a[idx],) if a is not None else (None,)
                for a in arrays
            ))

    return source_dict, target_dict


def source_target_merge(
    *arrays,
    sample_domain: Optional[np.ndarray] = None
) -> Sequence[np.ndarray]:
    """Merge source and target domain data based on sample domain labels.

    Parameters
    ----------
    *arrays : sequence of array-like
        (, n_features). The number of arrays must be even
        since we consider pairs of source-target arrays each time.
        Each pair should have at least one non-empty array.
        In each pair the first array is considered as the source and
        the second as the target. If one of the array is None or empty,
        it's value will be inferred from the other array and the sample_domain
        (depending on the type of the arrays, they'll have a value of
        {_DEFAULT_MASKED_TARGET_CLASSIFICATION_LABEL} or
        {_DEFAULT_MASKED_TARGET_REGRESSION_LABEL}).
    sample_domain : array-like of shape (n_samples,)
        Array specifying the domain labels for each sample. If None or empty
        the domain labels will be inferred from the the *arrays, being a
        default source and target domain (depending on the type of the arrays,
        they'll have a value of {_DEFAULT_SOURCE_DOMAIN_LABEL}
        or {_DEFAULT_TARGET_DOMAIN_LABEL}).

    Returns
    -------
    merges : list, length=len(arrays)/2
        List containing merged data based on the sample domain labels.
    sample_domain : array-like of shape (n_samples,)
        Array specifying the domain labels for each sample.

    Examples
    --------
    >>> X_source = np.array([[1, 2], [3, 4], [5, 6]])
    >>> X_target = np.array([[7, 8], [9, 10]])
    >>> X, sample_domain = source_target_merge(X_source, X_target)
    >>> X
    array([[ 1,  2],
           [ 3,  4],
           [ 5,  6],
           [ 7,  8],
           [ 9, 10]])
    >>> sample_domain
    array([ 1.,  1.,  1., -2., -2.])

    >>> X_source = np.array([[1, 2], [3, 4], [5, 6]])
    >>> X_target = np.array([[7, 8], [9, 10]])
    >>> y_source = np.array([0, 1, 1])
    >>> y_target = None
    >>> X, y, _ = source_target_merge(X_source, X_target, y_source, y_target)
    >>> X
    array([[ 1,  2],
           [ 3,  4],
           [ 5,  6],
           [ 7,  8],
           [ 9, 10]])
    >>> y
    array([ 0,  1,  1, {_DEFAULT_MASKED_TARGET_CLASSIFICATION_LABEL}, {_DEFAULT_MASKED_TARGET_CLASSIFICATION_LABEL}])
    """

    arrays = list(arrays)   # Convert to list to be able to modify it

    # ################ Check *arrays #################
    if len(arrays) < 2:
        raise ValueError("At least two array required as input")

    if len(arrays) % 2 != 0:
        raise ValueError("Even number of arrays required as input")

    for i in range(0, len(arrays), 2):
        if arrays[i] is None or arrays[i].shape[0] == 0:
            arrays[i] = np.array([])

        if arrays[i+1] is None or arrays[i+1].shape[0] == 0:
            arrays[i+1] = np.array([])

        # Check no pair is empty
        if (np.size(arrays[i]) == 0 and np.size(arrays[i+1]) == 0):
            raise ValueError("Only one array can be None or empty in each pair")

        # Check consistent dim of arrays
        if (np.size(arrays[i]) != 0 and np.size(arrays[i+1]) != 0):
            if arrays[i].shape[1:] != arrays[i+1].shape[1:]:
                raise ValueError(
                    "Inconsistent number of features in source-target arrays"
                )

    # ################ Check sample_domain #################
    # If sample_domain is None, we need to infer it from the target array
    if sample_domain is None or sample_domain.shape[0] == 0:
        # We need to infer the domain from the target array
        warnings.warn(
            "sample_domain is None or empty, it will be inferred from the arrays"
        )

        # By assuming that the first array is the source and the second the target
        source_assumed_index = 0
        target_assumed_index = 1
        sample_domain = np.concatenate((
            _DEFAULT_SOURCE_DOMAIN_LABEL*np.ones(
                arrays[source_assumed_index].shape[0]
            ),
            _DEFAULT_TARGET_DOMAIN_LABEL*np.ones(
                arrays[target_assumed_index].shape[0]
            )
        ))

    # To test afterward if the number of samples in source-target arrays
    # and the number inferred in the sample_domain are consistent
    source_indices = extract_source_indices(sample_domain)
    source_samples = np.count_nonzero(source_indices)
    target_samples = np.count_nonzero(~source_indices)

    # ################ Merge arrays #################
    merges = []  # List of merged arrays

    for i in range(0, len(arrays), 2):

        # If one of the array is empty, we need to infer its values
        index_is_empty = None  # Index of the array that was None
        if np.size(arrays[i]) == 0:
            index_is_empty = i

        if np.size(arrays[i+1]) == 0:
            index_is_empty = i+1

        if index_is_empty is not None:
            # We need to infer the value of the empty array in the pair
            # warnings.warn(
            #    "One of the arrays in a pair is empty, it will be inferred"
            # )

            pair_index = i+1 if index_is_empty == i else i

            # Infer the value of the empty array
            # only if the shape of the new array is not (0, ...)
            shape_to_complete = (sample_domain.shape[0] - arrays[pair_index].shape[0],) + arrays[pair_index].shape[1:]
            if shape_to_complete[0] > 0:
                y_type = _find_y_type(arrays[pair_index])
                if y_type == Y_Type.DISCRETE:
                    default_masked_label = _DEFAULT_MASKED_TARGET_CLASSIFICATION_LABEL
                elif y_type == Y_Type.CONTINUOUS:
                    default_masked_label = _DEFAULT_MASKED_TARGET_REGRESSION_LABEL

                arrays[index_is_empty] = (
                    default_masked_label *
                    np.ones(shape_to_complete)
                )

        # Check consistent number of samples in source-target arrays
        # and the number inferred in the sample_domain
        if (sample_domain is not None) and (sample_domain.shape[0] != 0):
            if (
                (arrays[i].shape[0] != source_samples) or
                (arrays[i+1].shape[0] != target_samples)
            ):
                raise ValueError(
                    "Inconsistent number of samples in source-target arrays "
                    "and the number inferred in the sample_domain"
                )

        merges.append(
            _merge_arrays(arrays[i], arrays[i+1], sample_domain=sample_domain)
        )

    return (*merges, sample_domain)


# Update the docstring to replace placeholders with actual values
source_target_merge.__doc__ = source_target_merge.__doc__.format(
    _DEFAULT_MASKED_TARGET_CLASSIFICATION_LABEL=_DEFAULT_MASKED_TARGET_CLASSIFICATION_LABEL,
    _DEFAULT_MASKED_TARGET_REGRESSION_LABEL=_DEFAULT_MASKED_TARGET_REGRESSION_LABEL,
    _DEFAULT_SOURCE_DOMAIN_LABEL=_DEFAULT_SOURCE_DOMAIN_LABEL,
    _DEFAULT_TARGET_DOMAIN_LABEL=_DEFAULT_TARGET_DOMAIN_LABEL
)


def _merge_arrays(
    array_source,
    array_target,
    sample_domain
):
    """Merge source and target domain data based on sample domain labels.

    Parameters
    ----------
    array_source : sequence of array-like of identical number of samples
        (n_samples_source, n_features).
    array_target : sequence of array-like of identical number of samples
        (n_samples_target, n_features).
    sample_domain : array-like of shape (n_samples_source + n_samples_target,)
        Array specifying the domain labels for each sample.

    Returns
    -------
    merge : sequence of array-like of identical number of samples
        (n_samples_source + n_samples_target, n_features)
    """
    if array_source.shape[0] > 0 and array_target.shape[0] > 0:
        output = np.zeros_like(
            np.concatenate((array_source, array_target)), dtype=array_source.dtype
        )
        output[sample_domain >= 0] = array_source
        output[sample_domain < 0] = array_target

    elif array_source.shape[0] > 0:
        output = np.zeros_like(
            array_source, dtype=array_source.dtype
        )
        output[sample_domain >= 0] = array_source

    else:
        output = np.zeros_like(
            array_target, dtype=array_target.dtype
        )
        output[sample_domain < 0] = array_target

    return output


def qp_solve(Q, c=None, A=None, b=None, Aeq=None, beq=None,
             lb=None, ub=None, x0=None, tol=1e-6, max_iter=1000,
             verbose=False, log=False, solver="scipy"):
    r""" Solves a standard quadratic program

    Solve the following optimization problem:

    .. math::
        \min_x \quad  \frac{1}{2}x^TQx + x^Tc


        lb <= x <= ub

        Ax <= b

        A_{eq} x = b_{eq}

    Return val as None if optimization failed.

    All constraint parameters are optional, they will be ignored
    if set as None.

    Note that the Frank-wolfe solver can only be used to solve
    optimization problem defined as follows:

    .. math::
        \min_x \quad  \frac{1}{2}x^TQx + x^Tc

        A_{eq} x = b_{eq}

        x >= 0

    Or,

    .. math::
        \min_x \quad  \frac{1}{2}x^TQx + x^Tc

        A x <= b

        x >= 0

    With Aeq and beq of respective dimension (1,d) and (1,),
    and A and b of respective dimension (1,d) and (1,) or
    (2,d) and (2,), with A[0] = -A[1] and b[0] >= -b[1].

    Parameters
    ----------
    Q : (d,d) ndarray, float64, optional
        Quadratic cost matrix matrix
    c : (d,) ndarray, float64, optional
        Linear cost vector
    A : (n,d) ndarray, float64, optional
        Linear inequality constraint matrix.
    b : (n,) ndarray, float64, optional
        Linear inequality constraint vector.
    Aeq : (n,d) ndarray, float64, optional
        Linear equality constraint matrix .
    beq : (n,) ndarray, float64, optional
        Linear equality constraint vector.   .
    lb : (d) ndarray, float64, optional
        Lower bound constraint, -np.inf if not provided.
    ub : (d) ndarray, float64, optional
        Upper bound constraint, np.inf if not provided.
    x0 : (d,) ndarray, float64, optional
        Initialization. Ones by default.
    tol : float, optional
        Tolerance for termination.
    max_iter : int, optional
        Maximum number of iterations to perform.
    verbose : boolean, optional
        Print optimization information.
    log : boolean, optional
        Return a dictionary with optim information in addition to x and val
    solver : str, optional default='scipy'
        Available solvers : 'scipy', 'frank-wolfe'

    Returns
    -------
    x: (d,) ndarray
        Optimal solution x
    val: float
        optimal value of the objective (None if optimization error)
    log: dict
        Optional log output
    """
    solver_list = ["scipy", "frank-wolfe"]

    if solver == "scipy":
        return _qp_solve_scipy(Q, c, A, b, Aeq, beq, lb, ub,
                               x0, tol, max_iter, verbose, log)
    elif solver == "frank-wolfe":
        return _qp_solve_frank_wolfe(Q, c, A, b, Aeq, beq,
                                     x0, max_iter)
    else:
        raise ValueError("`solver` argument should be included in %s,"
                         " got '%s'" % (solver_list, str(solver)))


def _qp_solve_scipy(Q, c=None, A=None, b=None, Aeq=None, beq=None,
                    lb=None, ub=None, x0=None, tol=1e-6, max_iter=1000,
                    verbose=False, log=False):
    r""" Solves a standard quadratic program

    Solve the following optimization problem:

    .. math::
        \min_x \quad  \frac{1}{2}x^TQx + x^Tc


        lb <= x <= ub

        Ax <= b

        A_{eq} x = b_{eq}

    All constraint parameters are optional, they will be ignored
    if set as None.

    Parameters
    ----------
    Q : (d,d) ndarray, float64, optional
        Quadratic cost matrix matrix
    c : (d,) ndarray, float64, optional
        Linear cost vector
    A : (n,d) ndarray, float64, optional
        Linear inequality constraint matrix.
    b : (n,) ndarray, float64, optional
        Linear inequality constraint vector.
    Aeq : (n,d) ndarray, float64, optional
        Linear equality constraint matrix .
    beq : (n,) ndarray, float64, optional
        Linear equality constraint vector.   .
    lb : (d) ndarray, float64, optional
        Lower bound constraint, -np.inf if not provided.
    ub : (d) ndarray, float64, optional
        Upper bound constraint, np.inf if not provided.
    x0 : (d,) ndarray, float64, optional
        Initialization. Ones by default.
    tol : float, optional
        Tolerance for termination.
    max_iter : int, optional
        Maximum number of iterations to perform.
    verbose : boolean, optional
        Print optimization information.
    log : boolean, optional
        Return a dictionary with optim information in addition to x and val

    Returns
    -------
    x: (d,) ndarray
        Optimal solution x
    val: float
        optimal value of the objective (None if optimization error)
    log: dict
        Optional log output
    """
    # Constraints
    constraints = []
    if A is not None:
        constraints.append(LinearConstraint(A, ub=b))
    if Aeq is not None:
        constraints.append(LinearConstraint(Aeq, lb=beq, ub=beq))

    # Objective function
    if c is None:
        def func(x):
            return (1/2) * x @ (Q @ x)

        def jac(x):
            return Q @ x
    else:
        def func(x):
            return (1/2) * x @ (Q @ x) + x @ c

        def jac(x):
            return Q @ x + c

    if x0 is None:
        x0 = np.ones(Q.shape[0])

    # Bounds
    if lb is None and ub is None:
        bounds = None
    else:
        if lb is None:
            bounds = [(-np.inf, b) for b in ub]
        elif ub is None:
            bounds = [(b, np.inf) for b in lb]
        else:
            bounds = [(b1, b2) for b1, b2 in zip(lb, ub)]

    # Optimization
    results = minimize(func,
                       x0=x0,
                       method="SLSQP",
                       jac=jac,
                       bounds=bounds,
                       constraints=constraints,
                       tol=tol,
                       options={"maxiter": max_iter,
                                "disp": verbose})

    if not results.success:
        warnings.warn(results.message)

    outputs = (results['x'], results['fun'])

    if log:
        outputs += (results,)

    return outputs


def _qp_solve_frank_wolfe(Q, c=None, A=None, b=None,
                          Aeq=None, beq=None, x0=None,
                          max_iter=1000):
    r""" Solves a quadratic program with Frank-Wolfe algorithm

    Solve the following optimization problem:

    .. math::
        \min_x \quad  \frac{1}{2}x^TQx + x^Tc

        A_{eq} x = b_{eq}

        x >= 0

    Or,

    .. math::
        \min_x \quad  \frac{1}{2}x^TQx + x^Tc

        A x <= b

        x >= 0

    With Aeq and beq of respective dimension (1,d) and (1,),
    and A and b of respective dimension (1,d) and (1,) or
    (2,d) and (2,), with A[0] = -A[1] and b[0] >= -b[1].

    Parameters
    ----------
    Q : (d,d) ndarray, float64, optional
        Quadratic cost matrix matrix
    c : (d,) ndarray, float64, optional
        Linear cost vector
    A : (n,d) ndarray, float64, optional
        Linear inequality constraint matrix.
    b : (n,) ndarray, float64, optional
        Linear inequality constraint vector.
    Aeq : (n,d) ndarray, float64, optional
        Linear equality constraint matrix .
    beq : (n,) ndarray, float64, optional
        Linear equality constraint vector.
    x0 : (d,) ndarray, float64, optional
        Initialization. Ones by default.
    max_iter : int, optional
        Maximum number of iterations to perform.

    Returns
    -------
    x: (d,) ndarray
        Optimal solution x
    val: float
        optimal value of the objective (None if optimization error)
    """
    if Aeq is not None and Aeq.shape[0] > 1:
        raise ValueError("`Aeq.shape[0]` must be equal to 1"
                         " when using the 'frank-wolfe' solver,"
                         " got '%s'" % str(Aeq.shape[0]))
    if A is not None:
        if A.shape[0] > 2:
            raise ValueError("`A.shape[2]` must be lower than 2"
                             " when using the 'frank-wolfe' solver,"
                             " got '%s'" % str(A.shape[0]))
        if A.shape[0] == 2:
            if not np.allclose(A[0], -A[1]):
                raise ValueError("`A[0]` must be equal to `-A[1]`"
                                 " when using the 'frank-wolfe' solver"
                                 " with A.shape[0]=2")
            if b[0] < -b[1]:
                raise ValueError("`b[0]` must be greater or equal to"
                                 " `-b[1]` when using the 'frank-wolfe'"
                                 " solver with A.shape[0]=2")
    if c is None:
        def func(x):
            return (1/2) * x @ (Q @ x)

        def jac(x):
            return Q @ x
    else:
        def func(x):
            return (1/2) * x @ (Q @ x) + x @ c

        def jac(x):
            return Q @ x + c

    if Aeq is not None:
        x = frank_wolfe(jac, Aeq[0], beq[0], beq[0],
                        x0, max_iter)
    elif A is not None:
        if A.shape[0] > 1:
            x = frank_wolfe(jac, A[0], -b[1], b[0],
                            x0, max_iter)
        else:
            x = frank_wolfe(jac, A[0], b[0], b[0],
                            x0, max_iter)
    else:
        raise ValueError("`A` or `Aeq` must be given when"
                         " using the 'frank-wolfe' solver")
    return x, func(x)


def frank_wolfe(jac, c, clb=1., cub=1., x0=None, max_iter=1000):
    r"""Frank-Wolfe algorithm for convex programming

    Solve the following convex optimization problem:

    .. math::
        \min_x \quad  f(x)

        clb <= \\langle x, c \\rangle <= cub

        x >= 0

    Parameters
    ----------
    jac : callable
        The jacobian of f, return a vector of shape (d,).
    c : (d,) ndarray, float64
        Linear constraint vector.
    clb : float64, optional, default=1.
        Lower bound of the linear constraint.
    cub : float64, optional, default=1.
        Upper bound of the linear constraint.
    max_iter : int, optional, default=1000
        Maximum number of iterations to perform.

    Returns
    -------
    x: (d,) ndarray
        Optimal solution x
    """
    inv_c = 1. / c

    if x0 is None:
        x0 = inv_c * ((cub - clb) / 2) / c.shape[0]

    x = x0

    for k in range(1, max_iter+1):
        grad = jac(x)
        product = grad * inv_c
        index = np.argmin(product)
        vect = np.zeros(c.shape[0])
        if product[index] >= 0:
            vect[index] = inv_c[index] * clb
        else:
            vect[index] = inv_c[index] * cub
        lr = 2. / (k + 1.)
        x = (1 - lr) * x + lr * vect
    return x


def torch_minimize(loss, x0, tol=1e-6, max_iter=1000, verbose=False):
    r""" Solves unconstrained optimization problem using pytorch

    Solve the following optimization problem:

    .. math::
        \min_x \quad loss(x)

    Parameters
    ----------
    loss : callable
        Objective function to be minimized.
    x0 : list of ndarrays or torch.Tensor
        Initialization.
    tol : float, optional
        Tolerance on the gradient for termination.
    max_iter : int, optional
        Maximum number of iterations to perform.
    verbose : bool, optional
        If True, print the final gradient norm.

    Returns
    -------
    x: ndarray or list of ndarrays
        Optimal solution x
    val: float
        final value of the objective
    """
    try:
        import torch
    except ImportError:
        raise ImportError("torch_minimize requires pytorch to be installed")

    if type(x0) not in (list, tuple):
        x0 = [x0]
    x0 = [torch.tensor(x, requires_grad=True, dtype=torch.float64) for x in x0]

    optimizer = torch.optim.LBFGS(
        x0,
        max_iter=max_iter,
        tolerance_grad=tol,
        line_search_fn="strong_wolfe"
    )

    def closure():
        optimizer.zero_grad()
        loss_value = loss(*x0)
        loss_value.backward()
        return loss_value

    optimizer.step(closure)

    grad_norm = torch.cat([x.grad.flatten() for x in x0]).abs().max()

    if verbose:
        print(f"Final gradient norm: {grad_norm:.2e}")

    if grad_norm > tol:
        warnings.warn(
            "Optimization did not converge. "
            f"Final gradient maximum value: {grad_norm:.2e} > {tol:.2e}"
        )

    solution = [x.detach().numpy() for x in x0]
    if len(solution) == 1:
        solution = solution[0]
    loss_val = loss(*x0).item()

    return solution, loss_val
