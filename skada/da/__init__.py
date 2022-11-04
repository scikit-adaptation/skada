# -*- coding: utf-8 -*-
"""
Domain Adaptation estimators

"""

# Author: Remi Flamary <remi.flamary@polytechnique.edu>
#         Alexandre Gramfort <firstname.lastname@inria.fr>
#
# License: MIT License

from . import reweight

from .reweight import (
    ReweightDensity, GaussianReweightDensity, ClassifierReweightDensity
)
from .subspace import SubspaceAlignment, TCA

from .mapping import (
    EMDTransport, SinkhornTransport, SinkhornLpl1Transport, SinkhornL1l2Transport
)

__all__ = [
    "reweight",
    "ReweightDensity",
    "GaussianReweightDensity",
    "ClassifierReweightDensity",
    "SubspaceAlignment",
    "TCA",
    "EMDTransport",
    "SinkhornTransport",
    "SinkhornLpl1Transport",
    "SinkhornL1l2Transport"
]
