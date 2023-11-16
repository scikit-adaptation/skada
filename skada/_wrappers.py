from ._mapping import (
    OTMappingAdapter,
    EntropicOTMappingAdapter,
    ClassRegularizerOTMappingAdapter,
    LinearOTMappingAdapter,
    CORALAdapter,
)
from ._pipeline import make_da_pipeline
from sklearn.svm import SVC


def OTMapping(
    base_estimator=SVC(kernel="rbf"),
    metric="sqeuclidean",
    norm=None,
    max_iter=100000
):
    """Returns a the OT mapping method with adapter and estimator.

    Parameters
    ----------
    base_estimator : object, optional (default=SVC(kernel="rbf"))
        The base estimator to fit on the target dataset.
    metric : string, optional (default="sqeuclidean")
        The ground metric for the Wasserstein problem
    norm : string, optional (default=None)
        If given, normalize the ground metric to avoid numerical errors that
        can occur with large metric values.
    max_iter : int, optional (default=100_000)
        The maximum number of iterations before stopping OT algorithm if it
        has not converged.
    """
    ot_mapping = make_da_pipeline(
        OTMappingAdapter(metric=metric, norm=norm, max_iter=max_iter),
        base_estimator,
    )
    return ot_mapping


def EntropicOTMapping(
    base_estimator=SVC(kernel="rbf"),
    metric="sqeuclidean",
    norm=None,
    max_iter=1000,
    reg_e=1,
    tol=1e-8,
):
    """Returns a the entropic OT mapping method with adapter and estimator.

    Parameters
    ----------
    base_estimator : object, optional (default=SVC(kernel="rbf"))
        The base estimator to fit on the target dataset.
    reg_e : float, default=1
        Entropic regularization parameter.
    metric : string, optional (default="sqeuclidean")
        The ground metric for the Wasserstein problem.
    norm : string, optional (default=None)
        If given, normalize the ground metric to avoid numerical errors that
        can occur with large metric values.
    max_iter : int, float, optional (default=1000)
        The minimum number of iteration before stopping the optimization
        of the Sinkhorn algorithm if it has not converged
    tol : float, optional (default=10e-9)
        The precision required to stop the optimization of the Sinkhorn
        algorithm.
    """
    ot_mapping = make_da_pipeline(
        EntropicOTMappingAdapter(
            metric=metric,
            norm=norm,
            max_iter=max_iter,
            reg_e=reg_e,
            tol=tol
        ),
        base_estimator,
    )
    return ot_mapping


def ClassRegularizerOTMapping(
    base_estimator=SVC(kernel="rbf"),
    metric="sqeuclidean",
    norm="lpl1",
    max_iter=10,
    max_inner_iter=200,
    reg_e=1,
    reg_cl=0.1,
    tol=1e-8,
):
    """Returns a the class regularized OT mapping method with adapter and estimator.

    Parameters
    ----------
    base_estimator : object, optional (default=SVC(kernel="rbf"))
        The base estimator to fit on the target dataset.
    reg_e : float, default=1
        Entropic regularization parameter.
    reg_cl : float, default=0.1
        Class regularization parameter.
    norm : string, default="lpl1"
        Norm use for the regularizer of the class labels.
        If "lpl1", use the lp l1 norm.
        If "l1l2", use the l1 l2 norm.
    metric : string, optional (default="sqeuclidean")
        The ground metric for the Wasserstein problem
    max_iter : int, float, optional (default=10)
        The minimum number of iteration before stopping the optimization
        algorithm if it has not converged
    max_inner_iter : int, float, optional (default=200)
        The number of iteration in the inner loop
    tol : float, optional (default=10e-9)
        Stop threshold on error (inner sinkhorn solver) (>0)
    """
    ot_mapping = make_da_pipeline(
        ClassRegularizerOTMappingAdapter(
            metric=metric,
            norm=norm,
            max_iter=max_iter,
            max_inner_iter=max_inner_iter,
            reg_e=reg_e,
            reg_cl=reg_cl,
            tol=tol
        ),
        base_estimator,
    )
    return ot_mapping


def LinearOTMapping(
    base_estimator=SVC(kernel="rbf"),
    reg=1,
    bias=True,
):
    """Returns a the linear OT mapping method with adapter and estimator.

    Parameters
    ----------
    base_estimator : object, optional (default=SVC(kernel="rbf"))
        The base estimator to fit on the target dataset.
    reg : float, (default=1e-08)
        regularization added to the diagonals of covariances.
    bias: boolean, optional (default=True)
        estimate bias.
    """
    ot_mapping = make_da_pipeline(
        LinearOTMappingAdapter(
            reg=reg,
            bias=bias,
        ),
        base_estimator,
    )
    return ot_mapping


def CORAL(
    base_estimator=SVC(kernel="rbf"),
    reg="auto",
):
    """Returns a the CORAL method with adapter and estimator.

    Parameters
    ----------
    base_estimator : object, optional (default=SVC(kernel="rbf"))
        The base estimator to fit on the target dataset.
    reg : 'auto' or float, default="auto"
        The regularization parameter of the covariance estimator.
        Possible values:

          - None: no shrinkage).
          - 'auto': automatic shrinkage using the Ledoit-Wolf lemma.
          - float between 0 and 1: fixed shrinkage parameter.
    """
    ot_mapping = make_da_pipeline(
        CORALAdapter(reg=reg),
        base_estimator,
    )
    return ot_mapping
