# Author: Theo Gnassounou <theo.gnassounou@inria.fr>
#         Remi Flamary <remi.flamary@polytechnique.edu>
#
# License: BSD 3-Clause

"""
Some methods for deep domain adaptation.
"""
try:
    import torch  # noqa: F401
    import skorch  # noqa: F401
except ImportError as e:
    raise ImportError(
        "torch and skorch are required for importing deep's method."
    ) from e

from ._divergence import DeepCoral, DeepCoralLoss, DANLoss, DAN
from ._optimal_transport import DeepJDOT, DeepJDOTLoss
from ._adversarial import DANN, CDAN, DANNLoss, CDANLoss

from . import losses
from . import modules

__all__ = [
    'losses',
    'modules',
    'DeepCoralLoss',
    'DeepCoral',
    'DANLoss',
    'DAN',
    'DeepJDOTLoss',
    'DeepJDOT',
    'DANNLoss',
    'DANN',
    'CDANLoss',
    'CDAN',
]
