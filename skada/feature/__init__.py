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
    print("ERROR : torch and skorch are required for importing feature's method.")
    raise e

from ._losses import dan_loss, deepcoral_loss, deepjdot_loss
from ._divergence import DeepCORAL, DAN
from ._optimal_transport import DeepJDOT
from ._adversarial import DANN, CDAN

__all__ = [
    'dan_loss',
    'deepcoral_loss',
    'deepjdot_loss',
    'DeepCORAL',
    'DeepJDOT',
    'DANN',
    'CDAN',
    'DAN',
]
