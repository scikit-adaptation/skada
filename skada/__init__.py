# Author: Theo Gnassounou <theo.gnassounou@inria.fr>
#         Remi Flamary <remi.flamary@polytechnique.edu>
#         Oleksii Kachaiev <kachayev@gmail.com>
#
# License: BSD 3-Clause

import sklearn

from .version import __version__  # noqa: F401
from . import model_selection
from . import metrics
from . import utils
from .base import BaseAdapter, PerDomain, Shared
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
from ._ot import solve_jdot_regression, JDOTRegressor
from ._pipeline import make_da_pipeline
from .utils import source_target_split


# make sure that the usage of the library is not possible
# without metadata routing being enabled in the configuration
sklearn.set_config(enable_metadata_routing=True)

__all__ = [
    "metrics",
    "model_selection",
    "utils",

    "BaseAdapter",
    "PerDomain",
    "Shared",

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

    "solve_jdot_regression",
    "JDOTRegressor",

    "make_da_pipeline",

    "source_target_split",
]
