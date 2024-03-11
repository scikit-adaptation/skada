# Author: Theo Gnassounou <theo.gnassounou@inria.fr>
#         Remi Flamary <remi.flamary@polytechnique.edu>
#         Oleksii Kachaiev <kachayev@gmail.com>
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
from ._adversarial import CDAN, DANN, CDANLoss, DANNLoss
from ._divergence import DAN, DANLoss, DeepCoral, DeepCoralLoss
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
]
