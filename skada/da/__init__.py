# -*- coding: utf-8 -*-
"""
Domain Adaptation estimators

"""

# Author: Remi Flamary <remi.flamary@polytechnique.edu>
#         Alexandre Gramfort <firstname.lastname@inria.fr>
#
# License: BSD 3-Clause

from ._reweight import (
    ReweightDensity, GaussianReweightDensity, ClassifierReweightDensity
)
from ._subspace import SubspaceAlignment, TransferComponentAnalysis

from ._mapping import (
    OTmapping, EntropicOTmapping, ClassRegularizerOTmapping, LinearOTmapping
)

__all__ = [
    "ReweightDensity",
    "GaussianReweightDensity",
    "ClassifierReweightDensity",
    "SubspaceAlignment",
    "TransferComponentAnalysis",
    "OTmapping",
    "EntropicOTmapping",
    "ClassRegularizerOTmapping",
    "LinearOTmapping",
]
