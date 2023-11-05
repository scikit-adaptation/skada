from .base import BaseAdapter, DomainAwareEstimator
from ._mapping import (
    ClassRegularizerOTMappingAdapter,
    CORALAdapter,
    EntropicOTMappingAdapter,
    LinearOTMappingAdapter,
    OTMappingAdapter,
)
from ._reweight import (
    DiscriminatorReweightDensityAdapter,
    GaussianReweightDensityAdapter,
    KLIEPAdapter,
    ReweightDensityAdapter,
)
from ._subspace import (
    SubspaceAlignmentAdapter,
    TransferComponentAnalysisAdapter,
)
from ._pipeline import make_da_pipeline


__all__ = [
    "BaseAdapter",
    "DomainAwareEstimator",

    "ClassRegularizerOTMappingAdapter",
    "CORALAdapter",
    "EntropicOTMappingAdapter",
    "LinearOTMappingAdapter",
    "OTMappingAdapter",

    "DiscriminatorReweightDensityAdapter",
    "GaussianReweightDensityAdapter",
    "KLIEPAdapter",
    "ReweightDensityAdapter",

    "SubspaceAlignmentAdapter",
    "TransferComponentAnalysisAdapter",

    "make_da_pipeline",
]
