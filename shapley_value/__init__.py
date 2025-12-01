"""
Shapley Value Calculator

A comprehensive Python package for calculating Shapley values in cooperative game theory.
Provides fair allocation solutions for coalition games with multiple calculation methods
and performance optimizations.
"""

__version__ = '0.0.6'
__author__ = 'Bowen Song'
__license__ = 'MIT'

from .calculator import ShapleyValue
from .framework import ShapleyValueCalculator
from .combinations import ShapleyCombinations

__all__ = [
    'ShapleyValue',
    'ShapleyValueCalculator', 
    'ShapleyCombinations',
]