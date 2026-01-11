"""
Preprocessing robuste pour les réclamations bancaires
Inclut feature engineering, encodage, et gestion des outliers
"""
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Création de features avancés"""

    def __init__(self):
        self.family_stats = {}
        self.segment_stats = {}
        self.global_stats = {}

    def fit(self, X, y=None):
        """Calcule les statistiques pour feature engineering"""
        df = X.copy()

        # Statistiques globales
        self.global_stats['montant_median'] = df['Montant_demande'].median()
        self.global_stats['pnb_median'] = df['PNB_cumule'].median()
        self.global_stats['anciennete_mean'] = df['Anciennete_annees'].mean()

        # Statistiques par famille produit
        self.family_stats = df.groupby('Famille_Produit').agg({
            'Montant_demande': ['mean', 'median', 'std'],
            'PNB_cumule': ['mean', 'median'],
            'Anciennete_annees': 'mean'
        }).to_dict()

        # Statistiques par segment
        self.segment_stats = df.groupby('Segment').agg({
            'Montant_demande': ['mean', 'median'],
            'PNB_cumule': ['mean', 'median'],
            'Nb_reclamations_precedentes': 'mean'
        }).to_dict()

        return self

    def transform(self, X):
        """Crée les nouvelles features"""
        df = X.copy()

        # === FEATURES DE RATIOS ===
        # Ratio PNB/Montant (importance du client vs montant demandé)
        df['ratio_pnb_montant'] = df['PNB_cumule'] / (df['Montant_demande'] + 1)

        # Ratio Montant/Médiane famille (montant relativement élevé pour cette famille?)
        df['ratio_montant_famille'] = df.apply(
            lambda row: row['Montant_demande'] / (
                self.family_stats.get(('Montant_demande', 'median'), {}).get(row['Famille_Produit'], 1) + 1
            ),
            axis=1
        )

        # Ratio PNB/Médiane globale
        df['ratio_pnb_median'] = df['PNB_cumule'] / (self.global_stats['pnb_median'] + 1)

        # === FEATURES TEMPORELLES ===
        if 'Date_de_Qualification' in df.columns:
            df['Date_de_Qualification'] = pd.to_datetime(df['Date_de_Qualification'])
            df['mois'] = df['Date_de_Qualification'].dt.month
            df['trimestre'] = df['Date_de_Qualification'].dt.quarter
            df['jour_semaine'] = df['Date_de_Qualification'].dt.dayofweek
            df['est_weekend'] = (df['jour_semaine'] >= 5).astype(int)
            df['debut_mois'] = (df['Date_de_Qualification'].dt.day <= 7).astype(int)
            df['fin_mois'] = (df['Date_de_Qualification'].dt.day >= 24).astype(int)

        # === FEATURES D'AGRÉGATION CLIENT ===
        # Ratio produits/ancienneté
        df['ratio_produits_anciennete'] = df['Nb_produits'] / (df['Anciennete_annees'] + 1)

        # Taux de réclamations par année
        df['taux_reclamations_annuel'] = df['Nb_reclamations_precedentes'] / (df['Anciennete_annees'] + 0.5)

        # === FEATURES DE CATÉGORISATION ===
        # Client à forte valeur
        df['is_high_value'] = (df['PNB_cumule'] > self.global_stats['pnb_median'] * 2).astype(int)

        # Réclamant fréquent
        df['is_frequent_claimer'] = (df['Nb_reclamations_precedentes'] >= 2).astype(int)

        # Montant élevé
        df['is_high_amount'] = (df['Montant_demande'] > df['Montant_demande'].quantile(0.75)).astype(int)

        # Client senior
        df['is_senior'] = (df['Age_client'] >= 65).astype(int)

        # Client jeune
        df['is_young'] = (df['Age_client'] <= 30).astype(int)

        # Ancienneté forte
        df['is_long_customer'] = (df['Anciennete_annees'] > 5).astype(int)

        # === FEATURES D'INTERACTION ===
        # Montant × Ancienneté
        df['montant_x_anciennete'] = df['Montant_demande'] * df['Anciennete_annees']

        # PNB × Segment (via mapping)
        segment_mapping = {'Grand Public': 1, 'Particuliers': 2, 'Premium': 3}
        df['segment_encoded'] = df['Segment'].map(segment_mapping)
        df['pnb_x_segment'] = df['PNB_cumule'] * df['segment_encoded']

        # Délai × Nombre produits
        df['delai_x_produits'] = df['Delai_traitement_jours'] * df['Nb_produits']

        # === FEATURES LOGARITHMIQUES (pour normaliser distributions) ===
        df['log_montant'] = np.log1p(df['Montant_demande'])
        df['log_pnb'] = np.log1p(df['PNB_cumule'])
        df['log_anciennete'] = np.log1p(df['Anciennete_annees'])

        return df


class TargetEncoder(BaseEstimator, TransformerMixin):
    """
    Target Encoding pour variables catégorielles
    Utilise smoothing pour éviter l'overfitting
    """

    def __init__(self, columns, smoothing=10.0):
        self.columns = columns
        self.smoothing = smoothing
        self.encodings = {}
        self.global_mean = 0.0

    def fit(self, X, y):
        """Calcule les moyennes par catégorie avec smoothing"""
        df = X.copy()
        df['__target__'] = y

        self.global_mean = y.mean()

        for col in self.columns:
            if col not in df.columns:
                continue

            # Calcul moyennes par catégorie
            stats = df.groupby(col)['__target__'].agg(['mean', 'count'])

            # Smoothing: (count * mean + smoothing * global_mean) / (count + smoothing)
            stats['smoothed'] = (
                (stats['count'] * stats['mean'] + self.smoothing * self.global_mean) /
                (stats['count'] + self.smoothing)
            )

            self.encodings[col] = stats['smoothed'].to_dict()

        return self

    def transform(self, X):
        """Applique l'encodage"""
        df = X.copy()

        for col in self.columns:
            if col not in df.columns or col not in self.encodings:
                continue

            # Encoder avec fallback sur global_mean pour nouvelles catégories
            df[f'{col}_encoded'] = df[col].map(self.encodings[col]).fillna(self.global_mean)

        return df


class OutlierHandler(BaseEstimator, TransformerMixin):
    """Détection et traitement des outliers avec IQR"""

    def __init__(self, columns, method='clip', factor=3.0):
        self.columns = columns
        self.method = method  # 'clip' ou 'remove'
        self.factor = factor
        self.bounds = {}

    def fit(self, X, y=None):
        """Calcule les bornes pour chaque colonne"""
        df = X.copy()

        for col in self.columns:
            if col not in df.columns:
                continue

            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1

            lower_bound = Q1 - self.factor * IQR
            upper_bound = Q3 + self.factor * IQR

            self.bounds[col] = {'lower': lower_bound, 'upper': upper_bound}

        return self

    def transform(self, X):
        """Traite les outliers"""
        df = X.copy()

        if self.method == 'clip':
            for col in self.columns:
                if col not in df.columns or col not in self.bounds:
                    continue

                df[col] = df[col].clip(
                    lower=self.bounds[col]['lower'],
                    upper=self.bounds[col]['upper']
                )

        return df


class RobustPreprocessor:
    """
    Pipeline complet de preprocessing
    """

    def __init__(self, target_col='Fondee'):
        self.target_col = target_col
        self.feature_engineer = FeatureEngineer()
        self.target_encoder = None
        self.outlier_handler = None
        self.scaler = RobustScaler()  # Plus robuste aux outliers que StandardScaler
        self.categorical_cols = []
        self.numerical_cols = []
        self.engineered_features = []

    def fit(self, df):
        """
        Fit tous les transformers
        """
        X = df.drop(columns=[self.target_col] if self.target_col in df.columns else [])
        y = df[self.target_col] if self.target_col in df.columns else None

        # 1. Feature Engineering
        print("⚙️  Feature Engineering...")
        X = self.feature_engineer.fit_transform(X, y)

        # Identifier colonnes catégorielles et numériques
        self.categorical_cols = [
            'Famille_Produit', 'Categorie', 'Motif_Reclamation',
            'Banque_Privee', 'Segment', 'Canal_Reclamation'
        ]

        self.numerical_cols = [
            col for col in X.select_dtypes(include=[np.number]).columns
            if col not in ['No_Demande', 'ID_Client']
        ]

        # 2. Target Encoding pour catégorielles
        print("🔢 Target Encoding...")
        self.target_encoder = TargetEncoder(
            columns=self.categorical_cols,
            smoothing=10.0
        )
        if y is not None:
            X = self.target_encoder.fit_transform(X, y)

        # 3. Traitement des outliers
        print("📊 Traitement des outliers...")
        outlier_cols = ['Montant_demande', 'PNB_cumule', 'Delai_traitement_jours']
        self.outlier_handler = OutlierHandler(
            columns=outlier_cols,
            method='clip',
            factor=3.0
        )
        X = self.outlier_handler.fit_transform(X)

        # 4. Standardisation des features numériques
        print("📏 Standardisation...")
        # Ne standardiser QUE les colonnes numériques qui existent
        cols_to_scale = [col for col in self.numerical_cols if col in X.columns]
        self.scaler.fit(X[cols_to_scale])

        print(f"✅ Preprocessing configuré: {len(cols_to_scale)} features numériques")
        return self

    def transform(self, df):
        """
        Applique tous les transformers
        """
        X = df.drop(columns=[self.target_col] if self.target_col in df.columns else [])

        # Conserver colonnes ID
        id_cols = ['No_Demande', 'ID_Client']
        ids = X[id_cols].copy() if all(col in X.columns for col in id_cols) else None

        # 1. Feature Engineering
        X = self.feature_engineer.transform(X)

        # 2. Target Encoding
        if self.target_encoder:
            X = self.target_encoder.transform(X)

        # 3. Outliers
        if self.outlier_handler:
            X = self.outlier_handler.transform(X)

        # 4. Standardisation
        cols_to_scale = [col for col in self.numerical_cols if col in X.columns]
        X_scaled = self.scaler.transform(X[cols_to_scale])
        X[cols_to_scale] = X_scaled

        # Supprimer colonnes non numériques et IDs pour modélisation
        cols_to_drop = id_cols + self.categorical_cols + ['Date_de_Qualification']
        cols_to_drop = [col for col in cols_to_drop if col in X.columns]
        X = X.drop(columns=cols_to_drop, errors='ignore')

        return X

    def fit_transform(self, df):
        """Fit et transform"""
        self.fit(df)
        return self.transform(df)
