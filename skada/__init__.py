# Author: Theo Gnassounou <theo.gnassounou@inria.fr>
#         Remi Flamary <remi.flamary@polytechnique.edu>
#         Oleksii Kachaiev <kachayev@gmail.com>
#
# License: BSD 3-Clause

import sklearn

from . import metrics, model_selection
from ._mapping import (
    CORAL,
    ClassRegularizerOTMapping,
    ClassRegularizerOTMappingAdapter,
    CORALAdapter,
    EntropicOTMapping,
    EntropicOTMappingAdapter,
    LinearOTMapping,
    LinearOTMappingAdapter,
    MMDLSConSMapping,
    MMDLSConSMappingAdapter,
    OTMapping,
    OTMappingAdapter,
)
from ._ot import (
    JDOTClassifier,
    JDOTRegressor,
    solve_jdot_classification,
    solve_jdot_regression,
)
from ._pipeline import make_da_pipeline
from ._reweight import (
    KLIEP,
    KMM,
    DiscriminatorReweightDensity,
    DiscriminatorReweightDensityAdapter,
    GaussianReweightDensity,
    GaussianReweightDensityAdapter,
    KLIEPAdapter,
    KMMAdapter,
    MMDTarSReweight,
    MMDTarSReweightAdapter,
    ReweightDensity,
    ReweightDensityAdapter,
)
from ._self_labeling import DASVMClassifier
from ._subspace import (
    SubspaceAlignment,
    SubspaceAlignmentAdapter,
    TransferComponentAnalysis,
    TransferComponentAnalysisAdapter,
)
from .base import BaseAdapter, PerDomain, SelectSourceTarget, Shared
from .utils import source_target_split
from .version import __version__  # noqa: F401

# make sure that the usage of the library is not possible
# without metadata routing being enabled in the configuration
sklearn.set_config(enable_metadata_routing=True)

__all__ = [
    "metrics",
    "model_selection",
    "BaseAdapter",
    "PerDomain",
    "Shared",
    "SelectSourceTarget",
    "ClassRegularizerOTMappingAdapter",
    "ClassRegularizerOTMapping",
    "CORALAdapter",
    "CORAL",
    "EntropicOTMappingAdapter",
    "EntropicOTMapping",
    "LinearOTMappingAdapter",
    "LinearOTMapping",
    "MMDLSConSMappingAdapter",
    "MMDLSConSMapping",
    "OTMappingAdapter",
    "OTMapping",
    "DiscriminatorReweightDensityAdapter",
    "DiscriminatorReweightDensity",
    "GaussianReweightDensityAdapter",
    "GaussianReweightDensity",
    "KLIEPAdapter",
    "KLIEP",
    "KMMAdapter",
    "KMM",
    "ReweightDensityAdapter",
    "ReweightDensity",
    "MMDTarSReweightAdapter",
    "MMDTarSReweight",
    "SubspaceAlignmentAdapter",
    "SubspaceAlignment",
    "TransferComponentAnalysisAdapter",
    "TransferComponentAnalysis",
    "DASVMClassifier",
    "solve_jdot_regression",
    "JDOTRegressor",
    "solve_jdot_classification",
    "JDOTClassifier",
    "make_da_pipeline",
    "source_target_split",
]
