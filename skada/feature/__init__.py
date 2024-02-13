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
from ._divergence import DeepCoral, DeepCoralLoss
from ._optimal_transport import DeepJDOT
from ._adversarial import DANN, CDAN
from ._modules import ToyModule, ToyCNN, DomainClassifier

__all__ = [
    'dan_loss',
    'deepcoral_loss',
    'deepjdot_loss',
    'DeepCoralLoss',
    'DeepCoral',
    'DeepJDOT',
    'DANN',
    'CDAN',
    'ToyModule',
    'ToyCNN',
    'DomainClassifier',
]
