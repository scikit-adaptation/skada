from .base import BaseAdapter, DomainAwareEstimator
from ._mapping import (
    ClassRegularizerOTMappingAdapter,
    CORALAdapter,
    EntropicOTMappingAdapter,
    LinearOTMappingAdapter,
    OTMappingAdapter,
)
from ._reweight import (
    DiscriminatorReweightDensity,
    DiscriminatorReweightDensityAdapter,
    GaussianReweightDensity,
    GaussianReweightDensityAdapter,
    KLIEP,
    KLIEPAdapter,
    ReweightDensity,
    ReweightDensityAdapter,
)
from ._subspace import (
    SubspaceAlignment,
    SubspaceAlignmentAdapter,
    TransferComponentAnalysis,
    TransferComponentAnalysisAdapter,
)
# from ._pipeline import DAPipeline


__all__ = [
    "BaseAdapter",
    "DomainAwareEstimator",

    "ClassRegularizerOTMappingAdapter",
    "CORALAdapter",
    "EntropicOTMappingAdapter",
    "LinearOTMappingAdapter",
    "OTMappingAdapter",

    "DiscriminatorReweightDensity",
    "DiscriminatorReweightDensityAdapter",
    "GaussianReweightDensity",
    "GaussianReweightDensityAdapter",
    "KLIEP",
    "KLIEPAdapter",
    "ReweightDensity",
    "ReweightDensityAdapter",

    "SubspaceAlignment",
    "SubspaceAlignmentAdapter",
    "TransferComponentAnalysis",
    "TransferComponentAnalysisAdapter",

    # "DAPipeline",
]
