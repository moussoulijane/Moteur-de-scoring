"""Tests for preprocessor."""
import pytest
import pandas as pd
import numpy as np
import sys
sys.path.append('..')

from src.preprocessing import ProductionPreprocessor


def test_preprocessor_initialization():
    """Test preprocessor initialization."""
    preprocessor = ProductionPreprocessor(min_samples_stats=30)
    assert preprocessor.min_samples_stats == 30
    assert not preprocessor._is_fitted


def test_preprocessor_fit_transform():
    """Test fit and transform."""
    # Create dummy data
    df = pd.DataFrame({
        'Montant demandé': [1000, 2000, 3000, 4000, 5000],
        'Délai estimé': [10, 20, 30, 40, 50],
        'Famille Produit': ['A', 'B', 'A', 'B', 'A'],
        'Catégorie': ['C1', 'C2', 'C1', 'C2', 'C1'],
        'Sous-catégorie': ['SC1', 'SC2', 'SC1', 'SC2', 'SC1'],
        'Segment': ['S1', 'S2', 'S1', 'S2', 'S1'],
        'Marché': ['M1', 'M2', 'M1', 'M2', 'M1'],
        'anciennete_annees': [1, 2, 3, 4, 5],
        'PNB analytique (vision commerciale) cumulé': [10000, 20000, 30000, 40000, 50000],
        'Fondee': [1, 0, 1, 0, 1]
    })

    preprocessor = ProductionPreprocessor(min_samples_stats=1)
    X = preprocessor.fit_transform(df)

    assert preprocessor._is_fitted
    assert X.shape[0] == len(df)
    assert X.shape[1] > 0


def test_clean_numeric_column():
    """Test numeric column cleaning."""
    preprocessor = ProductionPreprocessor()

    # Test various formats
    series = pd.Series(['1000', '2,000', '3.000,50', '500 mad', 'abc'])
    cleaned = preprocessor._clean_numeric_column(series)

    assert cleaned[0] == 1000
    assert cleaned[1] == 2000
    assert cleaned[2] == 3000.50
    assert cleaned[3] == 500
    assert cleaned[4] == 0
