"""
Preprocessing robuste pour les réclamations bancaires
ADAPTÉ AUX VRAIES COLONNES DE PRODUCTION
"""
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, RobustScaler
import warnings
warnings.filterwarnings('ignore')


class RealColumnFeatureEngineer(BaseEstimator, TransformerMixin):
    """Création de features avancés avec les vraies colonnes"""

    def __init__(self):
        self.family_stats = {}
        self.segment_stats = {}
        self.global_stats = {}

    def fit(self, X, y=None):
        """Calcule les statistiques pour feature engineering"""
        df = X.copy()

        # Statistiques globales
        self.global_stats['montant_median'] = df['Montant demandé'].median()
        self.global_stats['pnb_median'] = df['PNB analytique (vision commerciale) cumulé'].median()
        self.global_stats['anciennete_mean'] = df['anciennete_annees'].mean()

        # Statistiques par famille produit
        if 'Famille Produit' in df.columns:
            self.family_stats = df.groupby('Famille Produit').agg({
                'Montant demandé': ['mean', 'median', 'std'],
                'PNB analytique (vision commerciale) cumulé': ['mean', 'median'],
                'anciennete_annees': 'mean'
            }).to_dict()

        # Statistiques par segment
        if 'Segment' in df.columns:
            self.segment_stats = df.groupby('Segment').agg({
                'Montant demandé': ['mean', 'median'],
                'PNB analytique (vision commerciale) cumulé': ['mean', 'median']
            }).to_dict()

        return self

    def transform(self, X):
        """Crée les nouvelles features"""
        df = X.copy()

        # === FEATURES DE RATIOS ===
        # Ratio PNB/Montant
        df['ratio_pnb_montant'] = df['PNB analytique (vision commerciale) cumulé'] / (df['Montant demandé'] + 1)

        # Ratio Montant/Médiane famille
        if 'Famille Produit' in df.columns:
            df['ratio_montant_famille'] = df.apply(
                lambda row: row['Montant demandé'] / (
                    self.family_stats.get(('Montant demandé', 'median'), {}).get(row['Famille Produit'], 1) + 1
                ),
                axis=1
            )
        else:
            df['ratio_montant_famille'] = 1.0

        # Ratio PNB/Médiane globale
        df['ratio_pnb_median'] = df['PNB analytique (vision commerciale) cumulé'] / (self.global_stats['pnb_median'] + 1)

        # === FEATURES TEMPORELLES ===
        if 'Date de Qualification' in df.columns:
            df['Date de Qualification'] = pd.to_datetime(df['Date de Qualification'], errors='coerce')
            df['mois'] = df['Date de Qualification'].dt.month
            df['trimestre'] = df['Date de Qualification'].dt.quarter
            df['jour_semaine'] = df['Date de Qualification'].dt.dayofweek
            df['est_weekend'] = (df['jour_semaine'] >= 5).astype(int)
            df['debut_mois'] = (df['Date de Qualification'].dt.day <= 7).astype(int)
            df['fin_mois'] = (df['Date de Qualification'].dt.day >= 24).astype(int)

        # === FEATURES D'AGRÉGATION CLIENT ===
        # Taux montant demandé / montant réponse
        if 'Montant de réponse' in df.columns:
            df['taux_acceptation_montant'] = df['Montant de réponse'] / (df['Montant demandé'] + 1)

        # Ratio ancienneté
        df['ratio_anciennete_median'] = df['anciennete_annees'] / (self.global_stats['anciennete_mean'] + 1)

        # === FEATURES DE CATÉGORISATION ===
        # Client à forte valeur
        df['is_high_value'] = (df['PNB analytique (vision commerciale) cumulé'] > self.global_stats['pnb_median'] * 2).astype(int)

        # Montant élevé
        df['is_high_amount'] = (df['Montant demandé'] > df['Montant demandé'].quantile(0.75)).astype(int)

        # Ancienneté forte
        df['is_long_customer'] = (df['anciennete_annees'] > 5).astype(int)

        # Client banque privée
        if 'Banque Privé' in df.columns:
            df['is_banque_privee'] = (df['Banque Privé'] == 'OUI').astype(int)

        # Réclamation financière
        if 'Financière ou non' in df.columns:
            df['is_financiere'] = (df['Financière ou non'] == 'OUI').astype(int)

        # Canal digital
        if 'Canal de Réception' in df.columns:
            canaux_digitaux = ['Email', 'Application mobile', 'Réseaux sociaux']
            df['is_canal_digital'] = df['Canal de Réception'].isin(canaux_digitaux).astype(int)

        # Priorité haute
        if 'Priorité Client' in df.columns:
            df['is_priorite_haute'] = (df['Priorité Client'] == 'Haute').astype(int)

        # === FEATURES D'INTERACTION ===
        # Montant × Ancienneté
        df['montant_x_anciennete'] = df['Montant demandé'] * df['anciennete_annees']

        # PNB × Segment (via mapping)
        if 'Segment' in df.columns:
            segment_mapping = {'Grand Public': 1, 'Particuliers': 2, 'Premium': 3, 'VVIP': 4}
            df['segment_numeric'] = df['Segment'].map(segment_mapping).fillna(1)
            df['pnb_x_segment'] = df['PNB analytique (vision commerciale) cumulé'] * df['segment_numeric']

        # Délai × montant
        if 'Délai Estimé (j)' in df.columns:
            df['delai_x_montant'] = df['Délai Estimé (j)'] * df['Montant demandé']

        # === FEATURES LOGARITHMIQUES ===
        df['log_montant'] = np.log1p(df['Montant demandé'])
        df['log_pnb'] = np.log1p(df['PNB analytique (vision commerciale) cumulé'])
        df['log_anciennete'] = np.log1p(df['anciennete_annees'])

        return df


class RealColumnTargetEncoder(BaseEstimator, TransformerMixin):
    """Target Encoding pour variables catégorielles avec les vraies colonnes"""

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

            # Smoothing
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

            df[f'{col}_encoded'] = df[col].map(self.encodings[col]).fillna(self.global_mean)

        return df


class OutlierHandler(BaseEstimator, TransformerMixin):
    """Détection et traitement des outliers avec IQR"""

    def __init__(self, columns, method='clip', factor=3.0):
        self.columns = columns
        self.method = method
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


class RealColumnRobustPreprocessor:
    """Pipeline complet de preprocessing avec les vraies colonnes"""

    def __init__(self, target_col='Fondee'):
        self.target_col = target_col
        self.feature_engineer = RealColumnFeatureEngineer()
        self.target_encoder = None
        self.outlier_handler = None
        self.scaler = RobustScaler()
        self.categorical_cols = []
        self.numerical_cols = []

    def fit(self, df):
        """Fit tous les transformers"""
        X = df.drop(columns=[self.target_col] if self.target_col in df.columns else [])
        y = df[self.target_col] if self.target_col in df.columns else None

        # 1. Feature Engineering
        print("⚙️  Feature Engineering...")
        X = self.feature_engineer.fit_transform(X, y)

        # Identifier colonnes catégorielles et numériques
        self.categorical_cols = [
            'Famille Produit', 'Catégorie', 'Sous-catégorie',
            'Segment', 'Canal de Réception', 'Région',
            'Réseau', 'Groupe', 'PP/PM', 'Marché'
        ]
        # Filtrer seulement celles qui existent
        self.categorical_cols = [c for c in self.categorical_cols if c in X.columns]

        self.numerical_cols = [
            col for col in X.select_dtypes(include=[np.number]).columns
            if col not in ['No Demande', 'N compte', 'idtfcl', 'numero_compte']
        ]

        # 2. Target Encoding pour catégorielles
        print("🔢 Target Encoding...")
        self.target_encoder = RealColumnTargetEncoder(
            columns=self.categorical_cols,
            smoothing=10.0
        )
        if y is not None:
            X = self.target_encoder.fit_transform(X, y)

        # 3. Traitement des outliers
        print("📊 Traitement des outliers...")
        outlier_cols = [
            'Montant demandé',
            'PNB analytique (vision commerciale) cumulé',
            'Délai Estimé (j)' if 'Délai Estimé (j)' in X.columns else None
        ]
        outlier_cols = [c for c in outlier_cols if c is not None and c in X.columns]

        self.outlier_handler = OutlierHandler(
            columns=outlier_cols,
            method='clip',
            factor=3.0
        )
        X = self.outlier_handler.fit_transform(X)

        # 4. Standardisation des features numériques
        print("📏 Standardisation...")
        cols_to_scale = [col for col in self.numerical_cols if col in X.columns]
        self.scaler.fit(X[cols_to_scale])

        print(f"✅ Preprocessing configuré: {len(cols_to_scale)} features numériques")
        return self

    def transform(self, df):
        """Applique tous les transformers"""
        X = df.drop(columns=[self.target_col] if self.target_col in df.columns else [])

        # Conserver colonnes ID
        id_cols = ['No Demande', 'N compte', 'idtfcl', 'numero_compte']
        id_cols = [c for c in id_cols if c in X.columns]

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
        cols_to_drop = id_cols + self.categorical_cols + [
            'Date de Qualification', 'Ouvert', 'dt_debrel',
            'Nom', 'Statut', 'Type Demande',
            'Code Agence / CA Principal', 'Libellé Agence / CA Principal',
            'Code Entité Source', 'Libellé Entité Source',
            'Motif d\'irrecevabilité', 'Recevable',
            'Entité Resp', 'Entité Resp.',
            'Priorité Client', 'Banque Privé', 'Financière ou non',
            'Wafacash', 'Demandeur',
            'Source', 'BAS',  # 2024 only
            'Code GAB', 'Code anomalie GAB', 'Motif dérogation', 'Acteur dérogation',  # 2025 only
            'Motif de rejet réponse UT', 'Date Rejet réponse UT',
            'Motif de rejet UT', 'Date Rejet UT'
        ]
        cols_to_drop = [col for col in cols_to_drop if col in X.columns]
        X = X.drop(columns=cols_to_drop, errors='ignore')

        return X

    def fit_transform(self, df):
        """Fit et transform"""
        self.fit(df)
        return self.transform(df)
