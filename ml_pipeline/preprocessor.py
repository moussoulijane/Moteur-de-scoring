"""
PREPROCESSOR PRODUCTION
Classe de preprocessing réutilisable pour l'entraînement et l'inférence
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler


class ProductionPreprocessor:
    """Preprocessing production avec gestion stricte de l'ordre des colonnes"""

    def __init__(self):
        self.scaler = RobustScaler()
        self.family_medians = {}
        self.categorical_encodings = {}
        self.feature_names_fitted = None

    def fit(self, df):
        """Fit sur données 2024"""
        print("\n🔧 Configuration du preprocessing...")

        X = df.copy()

        # Nettoyer et convertir les colonnes numériques AVANT tout traitement
        print("🔧 Conversion des colonnes numériques...")
        numeric_columns = ['Montant demandé', 'PNB analytique (vision commerciale) cumulé', 'anciennete_annees']
        for col in numeric_columns:
            if col in X.columns:
                X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)
                X[col] = X[col].replace([np.inf, -np.inf], 0).clip(lower=0)

        # Convertir catégorielles en string AVANT les calculs
        print("🔢 Conversion catégorielles en string...")
        categorical_cols = ['Marché', 'Segment', 'Famille Produit', 'Catégorie', 'Sous-catégorie']
        for col in categorical_cols:
            if col in X.columns:
                X[col] = X[col].astype(str).fillna('UNKNOWN')

        # Calculer médianes par famille (APRÈS conversion en string)
        print("📊 Calcul médianes par famille (base 2024)...")
        self.family_medians = X.groupby('Famille Produit')['Montant demandé'].median().to_dict()
        print(f"   ✅ {len(self.family_medians)} familles")

        # Encoder catégorielles (fréquences)
        print("🔢 Encodage fréquences catégorielles...")
        for col in categorical_cols:
            if col in X.columns:
                self.categorical_encodings[col] = X[col].value_counts().to_dict()
                X[f'{col}_freq'] = X[col].map(self.categorical_encodings[col]).fillna(0)

        # Features
        X = self._create_features(X, fit_mode=True)

        # Colonnes numériques
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [c for c in numeric_cols if c != 'Fondee']

        # CRITICAL: Sauvegarder l'ordre des colonnes (trié alphabétiquement)
        self.feature_names_fitted = sorted(numeric_cols)
        print(f"📋 Features finales: {len(self.feature_names_fitted)}")

        # Fit scaler avec colonnes dans le bon ordre
        X_ordered = X[self.feature_names_fitted]
        self.scaler.fit(X_ordered)

        print(f"✅ Preprocessing configuré: {len(self.feature_names_fitted)} features")

        return self

    def transform(self, df):
        """Transform sur données 2024 ou 2025"""
        X = df.copy()

        # Nettoyer et convertir les colonnes numériques AVANT tout traitement
        numeric_columns = ['Montant demandé', 'PNB analytique (vision commerciale) cumulé', 'anciennete_annees']
        for col in numeric_columns:
            if col in X.columns:
                X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)
                X[col] = X[col].replace([np.inf, -np.inf], 0).clip(lower=0)

        # Convertir catégorielles en string AVANT les calculs
        categorical_cols = ['Marché', 'Segment', 'Famille Produit', 'Catégorie', 'Sous-catégorie']
        for col in categorical_cols:
            if col in X.columns:
                X[col] = X[col].astype(str).fillna('UNKNOWN')

        # Encoder fréquences avec encodages de 2024
        for col in categorical_cols:
            if col in X.columns and col in self.categorical_encodings:
                X[f'{col}_freq'] = X[col].map(self.categorical_encodings[col]).fillna(0)

        # Features
        X = self._create_features(X, fit_mode=False)

        # Colonnes numériques
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [c for c in numeric_cols if c != 'Fondee']

        # CRITICAL: Utiliser EXACTEMENT les mêmes colonnes dans le même ordre
        # Gérer les colonnes manquantes (ajouter avec valeur 0)
        for col in self.feature_names_fitted:
            if col not in X.columns:
                X[col] = 0

        # Garder seulement les colonnes utilisées lors du fit, dans le bon ordre
        X = X[self.feature_names_fitted]

        # Scaler
        X[self.feature_names_fitted] = self.scaler.transform(X[self.feature_names_fitted])

        return X

    def _create_features(self, X, fit_mode=True):
        """Création des features engineered"""
        df = X.copy()

        # 1. Ratio couverture PNB
        if 'PNB analytique (vision commerciale) cumulé' in df.columns and 'Montant demandé' in df.columns:
            df['ratio_pnb_montant'] = (
                df['PNB analytique (vision commerciale) cumulé'] /
                (df['Montant demandé'] + 1)
            )

        # 2. Écart à la médiane de la famille (calculé sur 2024, appliqué partout)
        if 'Famille Produit' in df.columns and 'Montant demandé' in df.columns:
            df['ecart_mediane_famille'] = df.apply(
                lambda row: (
                    row['Montant demandé'] -
                    self.family_medians.get(row['Famille Produit'], row['Montant demandé'])
                ) / (self.family_medians.get(row['Famille Produit'], 1) + 1),
                axis=1
            )

        # 3. Log transformations
        if 'Montant demandé' in df.columns:
            df['log_montant'] = np.log1p(np.abs(df['Montant demandé']))

        if 'PNB analytique (vision commerciale) cumulé' in df.columns:
            df['log_pnb'] = np.log1p(np.abs(df['PNB analytique (vision commerciale) cumulé']))

        if 'anciennete_annees' in df.columns:
            df['log_anciennete'] = np.log1p(np.abs(df['anciennete_annees']))

        # 4. Features d'interaction
        if 'Montant demandé' in df.columns and 'anciennete_annees' in df.columns:
            df['montant_x_anciennete'] = df['Montant demandé'] * df['anciennete_annees']

        if 'PNB analytique (vision commerciale) cumulé' in df.columns and 'anciennete_annees' in df.columns:
            df['pnb_x_anciennete'] = df['PNB analytique (vision commerciale) cumulé'] * df['anciennete_annees']

        # Sélectionner colonnes numériques
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        # Garder colonnes nécessaires
        keep_cols = [col for col in numeric_cols
                     if col != 'Fondee']

        df_result = df[keep_cols]

        # CRITICAL: Nettoyer les inf et NaN
        df_result = self._clean_numeric_data(df_result)

        return df_result

    def _clean_numeric_data(self, df):
        """Nettoie les NaN et inf dans les colonnes numériques"""
        df_clean = df.copy()

        for col in df_clean.columns:
            if df_clean[col].dtype in [np.float64, np.float32, np.int64, np.int32]:
                # Remplacer inf et -inf par NaN
                df_clean[col] = df_clean[col].replace([np.inf, -np.inf], np.nan)

                # Remplacer NaN par la médiane ou 0
                median_val = df_clean[col].median()
                if pd.isna(median_val):
                    median_val = 0.0
                df_clean[col] = df_clean[col].fillna(median_val)

        return df_clean

    def fit_transform(self, df):
        """Fit puis transform"""
        self.fit(df)
        return self.transform(df)
