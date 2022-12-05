"""
Some methods for deep domain adaptation.
"""
from ._deepcoral import DeepCORAL
from ._deepjdot import DeepJDOT
from ._dann import DANN
from ._dan import DAN

__all__ = ['DeepCORAL', 'DeepJDOT', 'DANN', "DAN"]
