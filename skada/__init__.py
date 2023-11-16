import sklearn

from . import model_selection
from . import metrics
from .base import BaseAdapter
from ._mapping import (
    ClassRegularizerOTMappingAdapter,
    ClassRegularizerOTMapping,
    CORALAdapter,
    CORAL,
    EntropicOTMappingAdapter,
    EntropicOTMapping,
    LinearOTMappingAdapter,
    LinearOTMapping,
    OTMappingAdapter,
    OTMapping,
)
from ._reweight import (
    DiscriminatorReweightDensityAdapter,
    DiscriminatorReweightDensity,
    GaussianReweightDensityAdapter,
    GaussianReweightDensity,
    KLIEPAdapter,
    KLIEP,
    ReweightDensityAdapter,
    ReweightDensity,
)
from ._subspace import (
    SubspaceAlignmentAdapter,
    SubspaceAlignment,
    TransferComponentAnalysisAdapter,
    TransferComponentAnalysis,
)
from ._pipeline import make_da_pipeline

# make sure that the usage of the library is not possible
# without metadata routing being enabled in the configuration
sklearn.set_config(enable_metadata_routing=True)

__all__ = [
    "metrics",
    "model_selection",

    "BaseAdapter",

    "ClassRegularizerOTMappingAdapter",
    "ClassRegularizerOTMapping",
    "CORALAdapter",
    "CORAL",
    "EntropicOTMappingAdapter",
    "EntropicOTMapping",
    "LinearOTMappingAdapter",
    "LinearOTMapping",
    "OTMappingAdapter",
    "OTMapping",

    "DiscriminatorReweightDensityAdapter",
    "DiscriminatorReweightDensity",
    "GaussianReweightDensityAdapter",
    "GaussianReweightDensity",
    "KLIEPAdapter",
    "KLIEP",
    "ReweightDensityAdapter",
    "ReweightDensity",

    "SubspaceAlignmentAdapter",
    "SubspaceAlignment",
    "TransferComponentAnalysisAdapter",
    "TransferComponentAnalysis",

    "make_da_pipeline",
]
