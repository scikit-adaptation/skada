# Author: Theo Gnassounou <theo.gnassounou@inria.fr>
#         Remi Flamary <remi.flamary@polytechnique.edu>
#         Yanis Lalou <yanis.lalou@polytechnique.edu>
#         Ambroise Odonnat <ambroiseodonnattechnologie@gmail.com>
#
# License: BSD 3-Clause

"""Some methods for deep domain adaptation."""

try:
    import skorch  # noqa: F401
    import torch  # noqa: F401
except (ImportError, ModuleNotFoundError) as e:
    raise ImportError(
        "torch and skorch are required for importing skada.deep.* modules."
    ) from e

from . import losses, modules
from ._adversarial import (
    CDAN,
    DANN,
    MDD,
    CDANLoss,
    DANNLoss,
    MDDLoss,
    ModifiedCrossEntropyLoss,
)
from ._baseline import SourceOnly, TargetOnly
from ._class_confusion import MCC, MCCLoss
from ._divergence import CAN, DAN, CANLoss, DANLoss, DeepCoral, DeepCoralLoss
from ._graph_alignment import SPA, SPALoss
from ._multi_source import M3SDA, M3SDAAdapter, M3SDALoss
from ._optimal_transport import DeepJDOT, DeepJDOTLoss

__all__ = [
    "losses",
    "modules",
    "DeepCoralLoss",
    "DeepCoral",
    "DANLoss",
    "DAN",
    "DeepJDOTLoss",
    "DeepJDOT",
    "DANNLoss",
    "DANN",
    "CDANLoss",
    "CDAN",
    "MCCLoss",
    "MCC",
    "MDDLoss",
    "MDD",
    "ModifiedCrossEntropyLoss",
    "CANLoss",
    "CAN",
    "SPALoss",
    "SPA",
    "SourceOnly",
    "TargetOnly",
    "M3SDA",
    "M3SDAAdapter",
    "M3SDALoss",
]
