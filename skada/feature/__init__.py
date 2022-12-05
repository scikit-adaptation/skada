"""
Some methods for deep domain adaptation.
"""
try:
    import torch
    import skorch
except ImportError as e:
    print("ERROR: torch and skorch are required for importing feature's method.")
    raise(e)

from ._deepcoral import DeepCORAL
from ._deepjdot import DeepJDOT
from ._dann import DANN
from ._dan import DAN

__all__ = ['DeepCORAL', 'DeepJDOT', 'DANN', "DAN"]
