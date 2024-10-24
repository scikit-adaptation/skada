# Author: Theo Gnassounou <theo.gnassounou@inria.fr>
#         Remi Flamary <remi.flamary@polytechnique.edu>
#         Yanis Lalou <yanis.lalou@polytechnique.edu>
#
# License: BSD 3-Clause

"""
Some methods for deep domain adaptation.
"""
try:
    import torch  # noqa: F401
    import skorch  # noqa: F401
except (ImportError, ModuleNotFoundError) as e:
    raise ImportError(
        "torch and skorch are required for importing skada.deep.* modules."
    ) from e

from ._divergence import DeepCoral, DeepCoralLoss, DANLoss, DAN, CAN, CANLoss
from ._optimal_transport import DeepJDOT, DeepJDOTLoss
from ._adversarial import DANN, CDAN, DANNLoss, CDANLoss
from ._baseline import SourceOnly, TargetOnly

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
    'CANLoss',
    'CAN',
    'SourceOnly',
    'TargetOnly',
]
