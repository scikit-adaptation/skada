"""
Some methods for deep domain adaptation.
"""
try:
    import torch  # noqa: F401
    import skorch  # noqa: F401

except ImportError as e:
    print("ERROR : torch and skorch are required for importing feature's method.")
    raise e

from ._divergence import DeepCORAL, DAN
from ._optimal_transport import DeepJDOT
from ._adversarial import DANN

__all__ = ['DeepCORAL', 'DeepJDOT', 'DANN', 'DAN']
