# -*- coding: utf-8 -*-
"""
Domain Adaptation estimators

"""

# Author: Remi Flamary <remi.flamary@polytechnique.edu>
#         Alexandre Gramfort <firstname.lastname@inria.fr>
#
# License: MIT License

from ._reweight import (
    ReweightDensity, GaussianReweightDensity, ClassifierReweightDensity
)
from ._subspace import SubspaceAlignment, TransferComponentAnalysis

__all__ = [
    "ReweightDensity",
    "GaussianReweightDensity",
    "ClassifierReweightDensity",
    "SubspaceAlignment",
    "TransferComponentAnalysis"
]
