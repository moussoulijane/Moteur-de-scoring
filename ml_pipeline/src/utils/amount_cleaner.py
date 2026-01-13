"""
Utilitaire pour nettoyer les colonnes de montants
Gère les formats: "500,00 mad", "1 234,56 DH", "789.50", etc.
"""
import pandas as pd
import numpy as np
import re


def clean_amount_column(series):
    """
    Nettoie une colonne de montants pour la convertir en float

    Formats supportés:
    - "500,00 mad" -> 500.00
    - "1 234,56 DH" -> 1234.56
    - "789.50" -> 789.50
    - "1.234,56" -> 1234.56
    - "500 mad" -> 500.00
    - "N/A", "", None -> NaN

    Args:
        series: pd.Series contenant les montants

    Returns:
        pd.Series de floats nettoyés
    """
    if series is None or len(series) == 0:
        return series

    # Si déjà numérique, retourner directement
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(float)

    def clean_value(val):
        """Nettoie une valeur individuelle"""
        if pd.isna(val) or val == '' or val is None:
            return np.nan

        # Convertir en string
        val_str = str(val).strip()

        # Remplacer les valeurs vides
        if val_str.lower() in ['', 'n/a', 'na', 'null', 'none', '-']:
            return np.nan

        # Enlever les devises (mad, dh, eur, $, €, etc.)
        val_str = re.sub(r'\s*(mad|dh|eur|usd|euro|euros|dirhams?)\s*', '', val_str, flags=re.IGNORECASE)

        # Enlever les symboles de devises
        val_str = val_str.replace('€', '').replace('$', '').replace('£', '')

        # Enlever les espaces (1 234,56 -> 1234,56)
        val_str = val_str.replace(' ', '')

        # Détecter le format: virgule comme séparateur décimal ou milliers?
        # Si format "1.234,56" (européen) -> convertir en "1234.56"
        # Si format "1,234.56" (US) -> convertir en "1234.56"
        # Si format "1234,56" (européen sans milliers) -> convertir en "1234.56"

        # Compter les points et virgules
        n_dots = val_str.count('.')
        n_commas = val_str.count(',')

        if n_dots > 0 and n_commas > 0:
            # Les deux présents -> déterminer lequel est le séparateur décimal
            last_dot_pos = val_str.rfind('.')
            last_comma_pos = val_str.rfind(',')

            if last_comma_pos > last_dot_pos:
                # Format européen: 1.234,56
                val_str = val_str.replace('.', '').replace(',', '.')
            else:
                # Format US: 1,234.56
                val_str = val_str.replace(',', '')
        elif n_commas > 0:
            # Seulement des virgules
            # Si virgule suivie de 2 chiffres à la fin -> séparateur décimal
            # Ex: "1234,56" -> décimal, "1,234" -> milliers
            if re.search(r',\d{2}$', val_str):
                # Format européen: virgule = décimal
                val_str = val_str.replace(',', '.')
            else:
                # Virgule = milliers
                val_str = val_str.replace(',', '')
        # Si seulement des points, on garde tel quel (format US ou pas de milliers)

        # Nettoyer les caractères restants
        val_str = re.sub(r'[^\d\.\-]', '', val_str)

        # Convertir en float
        try:
            return float(val_str)
        except (ValueError, TypeError):
            return np.nan

    # Appliquer le nettoyage
    return series.apply(clean_value)


def clean_amount_columns(df, amount_columns=None):
    """
    Nettoie automatiquement les colonnes de montants dans un DataFrame

    Args:
        df: DataFrame
        amount_columns: Liste des colonnes à nettoyer
                       Si None, détecte automatiquement les colonnes contenant 'montant'

    Returns:
        DataFrame avec montants nettoyés
    """
    df = df.copy()

    # Détecter automatiquement les colonnes si non spécifié
    if amount_columns is None:
        amount_columns = [
            col for col in df.columns
            if 'montant' in col.lower() or 'pnb' in col.lower()
        ]

    # Nettoyer chaque colonne
    for col in amount_columns:
        if col in df.columns:
            print(f"  🧹 Nettoyage colonne: {col}")
            original_type = df[col].dtype
            df[col] = clean_amount_column(df[col])

            # Statistiques de nettoyage
            n_null_before = df[col].isna().sum()
            if original_type == 'object':
                print(f"     ✅ Converti de {original_type} -> float64")
            if n_null_before > 0:
                print(f"     ⚠️  {n_null_before} valeurs manquantes détectées")

    return df


# Fonction de test
def test_clean_amount():
    """Test de la fonction de nettoyage"""
    test_cases = [
        ("500,00 mad", 500.00),
        ("1 234,56 DH", 1234.56),
        ("789.50", 789.50),
        ("1.234,56", 1234.56),
        ("1,234.56", 1234.56),
        ("500 mad", 500.00),
        ("2.500,00 MAD", 2500.00),
        ("N/A", np.nan),
        ("", np.nan),
        (None, np.nan),
        ("123456", 123456.00),
        ("1234,50", 1234.50),
    ]

    print("\n🧪 Test de nettoyage des montants:")
    print("-" * 60)

    for input_val, expected in test_cases:
        result = clean_amount_column(pd.Series([input_val]))[0]
        status = "✅" if (pd.isna(result) and pd.isna(expected)) or result == expected else "❌"
        print(f"{status} '{input_val}' -> {result} (attendu: {expected})")


if __name__ == "__main__":
    test_clean_amount()
