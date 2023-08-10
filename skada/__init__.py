# -*- coding: utf-8 -*-
"""
Domain Adaptation estimators

"""
# Author: Theo Gnassounou <theo.gnassounou@inria.fr>
#         Remi Flamary <remi.flamary@polytechnique.edu>
#         Alexandre Gramfort <alexandre.gramfort@inria.fr>
#
# License: BSD 3-Clause
from ._mapping import (
    ClassRegularizerOTMapping,
    CORAL,
    EntropicOTMapping,
    LinearOTMapping,
    OTMapping,
)
from ._reweight import (
    DiscriminatorReweightDensity, 
    GaussianReweightDensity,
    KLIEP,
    ReweightDensity,
)
from ._subspace import SubspaceAlignment, TransferComponentAnalysis
from ._pipeline import DAPipeline

__all__ = [
    "ReweightDensity",
    "GaussianReweightDensity",
    "DiscriminatorReweightDensity",
    "KLIEP",
    "SubspaceAlignment",
    "TransferComponentAnalysis",
    "OTMapping",
    "EntropicOTMapping",
    "ClassRegularizerOTMapping",
    "LinearOTMapping",
    "CORAL",
    "DAPipeline",
]
