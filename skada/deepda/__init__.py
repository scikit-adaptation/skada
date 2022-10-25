"""
Some methods for deep domain adaptation.
"""
from .base import BaseDANetwork
from .deepcoral import DeepCORAL
from .deepjdot import DeepJDOT

__all__ = ['BaseDANetwork', 'DeepCORAL', 'DeepJDOT']
