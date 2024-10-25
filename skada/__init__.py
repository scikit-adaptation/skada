# Author: Theo Gnassounou <theo.gnassounou@inria.fr>
#         Remi Flamary <remi.flamary@polytechnique.edu>
#         Oleksii Kachaiev <kachayev@gmail.com>
#
# License: BSD 3-Clause

import sklearn

from .version import __version__  # noqa: F401
from . import model_selection
from . import metrics
from .base import BaseAdapter, PerDomain, Shared, SelectSource, SelectTarget, SelectSourceTarget
from ._mapping import (
    ClassRegularizerOTMappingAdapter,
    ClassRegularizerOTMapping,
    CORALAdapter,
    CORAL,
    EntropicOTMappingAdapter,
    EntropicOTMapping,
    LinearOTMappingAdapter,
    LinearOTMapping,
    MMDLSConSMappingAdapter,
    MMDLSConSMapping,
    OTMappingAdapter,
    OTMapping,
    MultiLinearMongeAlignmentAdapter,
    MultiLinearMongeAlignment
)
from ._reweight import (
    DiscriminatorReweightAdapter,
    DiscriminatorReweight,
    GaussianReweightAdapter,
    GaussianReweight,
    KLIEPReweightAdapter,
    KLIEPReweight,
    KMMReweightAdapter,
    KMMReweight,
    DensityReweightAdapter,
    DensityReweight,
    NearestNeighborReweightAdapter,
    NearestNeighborReweight,
    MMDTarSReweightAdapter,
    MMDTarSReweight
)
from ._subspace import (
    SubspaceAlignmentAdapter,
    SubspaceAlignment,
    TransferComponentAnalysisAdapter,
    TransferComponentAnalysis,
    TransferJointMatching,
    TransferJointMatchingAdapter,
    TransferSubspaceLearning,
    TransferSubspaceLearningAdapter,
)
from ._ot import (
    solve_jdot_regression,
    JDOTRegressor,
    solve_jdot_classification,
    JDOTClassifier,
    OTLabelPropAdapter,
    OTLabelProp,
    JCPOTLabelPropAdapter,
    JCPOTLabelProp)
from .transformers import Subsampler, DomainSubsampler, StratifiedDomainSubsampler
from ._self_labeling import DASVMClassifier
from ._pipeline import make_da_pipeline
from .utils import source_target_split, per_domain_split


# make sure that the usage of the library is not possible
# without metadata routing being enabled in the configuration
sklearn.set_config(enable_metadata_routing=True)

__all__ = [
    "metrics",
    "model_selection",

    "BaseAdapter",
    "PerDomain",
    "Shared",
    "SelectSource",
    "SelectTarget",
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
    "MultiLinearMongeAlignmentAdapter",
    "MultiLinearMongeAlignment",
    "OTLabelPropAdapter",
    "OTLabelProp",
    "JCPOTLabelPropAdapter",
    "JCPOTLabelProp",

    "DiscriminatorReweightAdapter",
    "DiscriminatorReweight",
    "GaussianReweightAdapter",
    "GaussianReweight",
    "KLIEPReweightAdapter",
    "KLIEPReweight",
    "KMMReweightAdapter",
    "KMMReweight",
    "DensityReweightAdapter",
    "DensityReweight",
    "NearestNeighborReweightAdapter",
    "NearestNeighborReweight",
    "MMDTarSReweightAdapter",
    "MMDTarSReweight",

    "SubspaceAlignmentAdapter",
    "SubspaceAlignment",
    "TransferComponentAnalysisAdapter",
    "TransferComponentAnalysis",
    "TransferJointMatchingAdapter",
    "TransferJointMatching",
    "TransferSubspaceLearningAdapter",
    "TransferSubspaceLearning",

    "DASVMClassifier",
    "solve_jdot_regression",
    "JDOTRegressor",
    "solve_jdot_classification",
    "JDOTClassifier",
    "OTLabelPropAdapter",

    "SubsampleTransformer",
    "DomainStratifiedSubsampleTransformer",

    "make_da_pipeline",

    "source_target_split",
    "per_domain_split",
]
