"""
ML Pipeline V2 - Features Production-Ready

Version améliorée du pipeline ML avec uniquement des features disponibles en production.
"""

__version__ = '2.0.0'
__author__ = 'Moteur de Scoring Team'

from .preprocessor_v2 import ProductionPreprocessorV2
from .model_comparison_v2 import ModelComparisonV2

__all__ = ['ProductionPreprocessorV2', 'ModelComparisonV2']
