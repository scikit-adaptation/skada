# -*- coding: utf-8 -*-
"""
Domain Adaptation estimators

"""
# Author: Theo Gnassounou <theo.gnassounou@inria.fr>
#         Remi Flamary <remi.flamary@polytechnique.edu>
#         Alexandre Gramfort <alexandre.gramfort@inria.fr>
#
# License: BSD 3-Clause
from ._reweight import (
    ReweightDensity, GaussianReweightDensity, DiscriminatorReweightDensity, KLIEP
)
from ._subspace import SubspaceAlignment, TransferComponentAnalysis

from ._mapping import (
    OTmapping, EntropicOTmapping, ClassRegularizerOTmapping, LinearOTmapping
)

from ._mapping import CORAL

__all__ = [
    "ReweightDensity",
    "GaussianReweightDensity",
    "DiscriminatorReweightDensity",
    "KLIEP",
    "SubspaceAlignment",
    "TransferComponentAnalysis",
    "OTmapping",
    "EntropicOTmapping",
    "ClassRegularizerOTmapping",
    "LinearOTmapping",
    "CORAL"
]
