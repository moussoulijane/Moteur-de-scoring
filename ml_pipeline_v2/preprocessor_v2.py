"""
PREPROCESSOR PRODUCTION V2
Features uniquement disponibles en temps réel
Calcul de statistiques sur 2024 pour utilisation en production
"""
import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import RobustScaler


class ProductionPreprocessorV2:
    """
    Preprocessing production avec features disponibles en temps réel uniquement

    Colonnes utilisées (disponibles en production):
    - Montant demandé
    - Délai estimé
    - Famille Produit
    - Catégorie
    - Sous-catégorie
    - Segment
    - Marché
    - anciennete_annees
    - PNB analytique (vision commerciale) cumulé

    Features calculées:
    - Taux de fondée par famille/catégorie/sous-catégorie (sur 2024)
    - Écart à la médiane par famille (montant et PNB)
    - Ratios et interactions (montant, délai, ancienneté, PNB)
    - Log transformations (montant, délai, ancienneté, PNB)
    """

    def __init__(self, min_samples_stats=30):
        """
        Args:
            min_samples_stats: Nombre minimum d'échantillons pour calculer des stats fiables
        """
        self.scaler = RobustScaler()
        self.min_samples_stats = min_samples_stats

        # Statistiques calculées sur 2024
        self.family_stats = {}
        self.category_stats = {}
        self.subcategory_stats = {}
        self.segment_stats = {}
        self.family_medians = {}  # Médiane montant par famille
        self.family_pnb_medians = {}  # Médiane PNB par famille
        self.categorical_encodings = {}

        self.feature_names_fitted = None

    def _clean_numeric_column(self, series):
        """
        Nettoyer une colonne numérique qui peut contenir du texte

        Exemples:
        - "500 mad" -> 500.0
        - "1 000 DH" -> 1000.0
        - "1,500.50" -> 1500.50
        - "1.500,50" -> 1500.50
        - "abc" -> 0.0
        """
        if series.dtype in ['float64', 'float32', 'int64', 'int32']:
            # Déjà numérique
            return pd.to_numeric(series, errors='coerce').fillna(0)

        # Convertir en string pour traitement
        cleaned_series = series.astype(str)

        def clean_value(val):
            if pd.isna(val) or val in ['', 'nan', 'None', 'NaN']:
                return 0.0

            # Convertir en string minuscule
            val_str = str(val).lower().strip()

            # Retirer les mots comme 'mad', 'dh', 'dirham', etc.
            val_str = re.sub(r'\b(mad|dh|dirham|dirhams|€|euro|euros)\b', '', val_str, flags=re.IGNORECASE)

            # Retirer tous les caractères sauf chiffres, points, virgules, espaces, et signes
            val_str = re.sub(r'[^\d\s\.,\-\+]', '', val_str)

            # Retirer les espaces (utilisés comme séparateurs de milliers)
            val_str = val_str.replace(' ', '')

            # Gérer le format européen (1.500,50) vs anglais (1,500.50)
            # Si une virgule est suivie de 2 chiffres à la fin, c'est probablement un séparateur décimal
            if re.search(r',\d{2}$', val_str):
                # Format européen: 1.500,50
                val_str = val_str.replace('.', '').replace(',', '.')
            else:
                # Format anglais: 1,500.50 ou pas de décimales
                val_str = val_str.replace(',', '')

            # Convertir en float
            try:
                return float(val_str) if val_str else 0.0
            except (ValueError, TypeError):
                return 0.0

        # Appliquer le nettoyage
        cleaned = cleaned_series.apply(clean_value)

        # S'assurer que c'est bien numérique
        return pd.to_numeric(cleaned, errors='coerce').fillna(0)

    def fit(self, df):
        """
        Fit sur données 2024 - calcule toutes les statistiques

        IMPORTANT: df doit contenir la colonne 'Fondee' pour calculer les taux
        """
        print("\n🔧 Configuration du preprocessing V2...")

        X = df.copy()

        # Vérifier que Fondee existe
        if 'Fondee' not in X.columns:
            raise ValueError("La colonne 'Fondee' est nécessaire pour l'entraînement")

        # 1. Nettoyer et convertir les colonnes numériques (avec traitement texte)
        print("🔧 Conversion des colonnes numériques (texte -> float)...")
        numeric_columns = ['Montant demandé', 'Délai estimé', 'anciennete_annees',
                          'PNB analytique (vision commerciale) cumulé']
        for col in numeric_columns:
            if col in X.columns:
                X[col] = self._clean_numeric_column(X[col])
                X[col] = X[col].replace([np.inf, -np.inf], 0).clip(lower=0)

        # 2. Convertir catégorielles en string
        print("🔢 Conversion catégorielles en string...")
        categorical_cols = ['Marché', 'Segment', 'Famille Produit', 'Catégorie', 'Sous-catégorie']
        for col in categorical_cols:
            if col in X.columns:
                X[col] = X[col].astype(str).fillna('UNKNOWN')

        # 3. Calculer les TAUX DE FONDÉE par catégorie (statistiquement robustes)
        print("📊 Calcul des taux de fondée (2024 - statistiquement renforcés)...")

        # Taux par Famille Produit
        if 'Famille Produit' in X.columns:
            family_grouped = X.groupby('Famille Produit').agg({
                'Fondee': ['mean', 'count']
            })
            family_grouped.columns = ['taux_fondee', 'count']

            # Ne garder que si count >= min_samples_stats
            family_grouped_filtered = family_grouped[family_grouped['count'] >= self.min_samples_stats]

            self.family_stats = {
                'taux': family_grouped_filtered['taux_fondee'].to_dict(),
                'count': family_grouped['count'].to_dict(),
                'taux_global': X['Fondee'].mean()  # Fallback pour nouvelles familles
            }

            print(f"   ✅ Famille Produit: {len(self.family_stats['taux'])} familles avec stats robustes (≥{self.min_samples_stats} cas)")
            print(f"      Taux global de fondée: {self.family_stats['taux_global']:.2%}")

        # Taux par Catégorie
        if 'Catégorie' in X.columns:
            cat_grouped = X.groupby('Catégorie').agg({
                'Fondee': ['mean', 'count']
            })
            cat_grouped.columns = ['taux_fondee', 'count']
            cat_grouped_filtered = cat_grouped[cat_grouped['count'] >= self.min_samples_stats]

            self.category_stats = {
                'taux': cat_grouped_filtered['taux_fondee'].to_dict(),
                'count': cat_grouped['count'].to_dict(),
                'taux_global': X['Fondee'].mean()
            }

            print(f"   ✅ Catégorie: {len(self.category_stats['taux'])} catégories avec stats robustes")

        # Taux par Sous-catégorie
        if 'Sous-catégorie' in X.columns:
            subcat_grouped = X.groupby('Sous-catégorie').agg({
                'Fondee': ['mean', 'count']
            })
            subcat_grouped.columns = ['taux_fondee', 'count']
            subcat_grouped_filtered = subcat_grouped[subcat_grouped['count'] >= self.min_samples_stats]

            self.subcategory_stats = {
                'taux': subcat_grouped_filtered['taux_fondee'].to_dict(),
                'count': subcat_grouped['count'].to_dict(),
                'taux_global': X['Fondee'].mean()
            }

            print(f"   ✅ Sous-catégorie: {len(self.subcategory_stats['taux'])} sous-catégories avec stats robustes")

        # Taux par Segment
        if 'Segment' in X.columns:
            seg_grouped = X.groupby('Segment').agg({
                'Fondee': ['mean', 'count']
            })
            seg_grouped.columns = ['taux_fondee', 'count']
            seg_grouped_filtered = seg_grouped[seg_grouped['count'] >= self.min_samples_stats]

            self.segment_stats = {
                'taux': seg_grouped_filtered['taux_fondee'].to_dict(),
                'count': seg_grouped['count'].to_dict(),
                'taux_global': X['Fondee'].mean()
            }

            print(f"   ✅ Segment: {len(self.segment_stats['taux'])} segments avec stats robustes")

        # 4. Calculer médianes par famille (pour écart)
        print("📊 Calcul médianes par famille (base 2024)...")
        if 'Famille Produit' in X.columns and 'Montant demandé' in X.columns:
            self.family_medians = X.groupby('Famille Produit')['Montant demandé'].median().to_dict()
            print(f"   ✅ Montant: {len(self.family_medians)} familles")

        if 'Famille Produit' in X.columns and 'PNB analytique (vision commerciale) cumulé' in X.columns:
            pnb_data = X[X['PNB analytique (vision commerciale) cumulé'] > 0]
            if len(pnb_data) > 0:
                self.family_pnb_medians = pnb_data.groupby('Famille Produit')['PNB analytique (vision commerciale) cumulé'].median().to_dict()
                print(f"   ✅ PNB: {len(self.family_pnb_medians)} familles")

        # 5. Encoder fréquences catégorielles
        print("🔢 Encodage fréquences catégorielles...")
        for col in categorical_cols:
            if col in X.columns:
                self.categorical_encodings[col] = X[col].value_counts().to_dict()
                X[f'{col}_freq'] = X[col].map(self.categorical_encodings[col]).fillna(0)

        # 6. Créer features engineered
        X = self._create_features(X, fit_mode=True)

        # 7. Colonnes numériques finales
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [c for c in numeric_cols if c != 'Fondee']

        # Ordonner alphabétiquement
        self.feature_names_fitted = sorted(numeric_cols)
        print(f"📋 Features finales: {len(self.feature_names_fitted)}")

        # 8. Fit scaler
        X_ordered = X[self.feature_names_fitted]
        self.scaler.fit(X_ordered)

        print(f"✅ Preprocessing V2 configuré: {len(self.feature_names_fitted)} features")

        return self

    def transform(self, df):
        """
        Transform sur données 2025 ou nouvelles données
        Utilise les statistiques calculées sur 2024
        """
        X = df.copy()

        # 1. Nettoyer et convertir numériques (avec traitement texte)
        numeric_columns = ['Montant demandé', 'Délai estimé', 'anciennete_annees',
                          'PNB analytique (vision commerciale) cumulé']
        for col in numeric_columns:
            if col in X.columns:
                X[col] = self._clean_numeric_column(X[col])
                X[col] = X[col].replace([np.inf, -np.inf], 0).clip(lower=0)

        # 2. Convertir catégorielles en string
        categorical_cols = ['Marché', 'Segment', 'Famille Produit', 'Catégorie', 'Sous-catégorie']
        for col in categorical_cols:
            if col in X.columns:
                X[col] = X[col].astype(str).fillna('UNKNOWN')

        # 3. Encoder fréquences avec encodages de 2024
        for col in categorical_cols:
            if col in X.columns and col in self.categorical_encodings:
                X[f'{col}_freq'] = X[col].map(self.categorical_encodings[col]).fillna(0)

        # 4. Créer features engineered
        X = self._create_features(X, fit_mode=False)

        # 5. Colonnes numériques
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [c for c in numeric_cols if c != 'Fondee']

        # 6. Gérer colonnes manquantes
        for col in self.feature_names_fitted:
            if col not in X.columns:
                X[col] = 0

        # 7. Garder seulement les colonnes du fit, dans le bon ordre
        X = X[self.feature_names_fitted]

        # 8. Scaler
        X[self.feature_names_fitted] = self.scaler.transform(X[self.feature_names_fitted])

        return X

    def _create_features(self, X, fit_mode=True):
        """Création des features engineered"""
        df = X.copy()

        # 1. TAUX DE FONDÉE (calculés sur 2024, appliqués partout)
        if 'Famille Produit' in df.columns and self.family_stats:
            df['taux_fondee_famille'] = df['Famille Produit'].map(self.family_stats['taux'])
            df['taux_fondee_famille'] = df['taux_fondee_famille'].fillna(self.family_stats['taux_global'])
            df['count_famille'] = df['Famille Produit'].map(self.family_stats['count']).fillna(0)

        if 'Catégorie' in df.columns and self.category_stats:
            df['taux_fondee_categorie'] = df['Catégorie'].map(self.category_stats['taux'])
            df['taux_fondee_categorie'] = df['taux_fondee_categorie'].fillna(self.category_stats['taux_global'])

        if 'Sous-catégorie' in df.columns and self.subcategory_stats:
            df['taux_fondee_souscategorie'] = df['Sous-catégorie'].map(self.subcategory_stats['taux'])
            df['taux_fondee_souscategorie'] = df['taux_fondee_souscategorie'].fillna(self.subcategory_stats['taux_global'])

        if 'Segment' in df.columns and self.segment_stats:
            df['taux_fondee_segment'] = df['Segment'].map(self.segment_stats['taux'])
            df['taux_fondee_segment'] = df['taux_fondee_segment'].fillna(self.segment_stats['taux_global'])

        # 2. Écart à la médiane de la famille
        if 'Famille Produit' in df.columns and 'Montant demandé' in df.columns and self.family_medians:
            df['ecart_mediane_famille'] = df.apply(
                lambda row: (
                    row['Montant demandé'] -
                    self.family_medians.get(row['Famille Produit'], row['Montant demandé'])
                ) / (self.family_medians.get(row['Famille Produit'], 1) + 1),
                axis=1
            )

        # 3. Ratio montant / délai
        if 'Montant demandé' in df.columns and 'Délai estimé' in df.columns:
            df['ratio_montant_delai'] = df['Montant demandé'] / (df['Délai estimé'] + 1)

        # 4. Log transformations
        if 'Montant demandé' in df.columns:
            df['log_montant'] = np.log1p(np.abs(df['Montant demandé']))

        if 'Délai estimé' in df.columns:
            df['log_delai'] = np.log1p(np.abs(df['Délai estimé']))

        if 'anciennete_annees' in df.columns:
            df['log_anciennete'] = np.log1p(np.abs(df['anciennete_annees']))

        if 'PNB analytique (vision commerciale) cumulé' in df.columns:
            df['log_pnb'] = np.log1p(np.abs(df['PNB analytique (vision commerciale) cumulé']))

        # 4b. Écart PNB à la médiane de la famille
        if 'Famille Produit' in df.columns and 'PNB analytique (vision commerciale) cumulé' in df.columns and self.family_pnb_medians:
            df['ecart_pnb_mediane_famille'] = df.apply(
                lambda row: (
                    row['PNB analytique (vision commerciale) cumulé'] -
                    self.family_pnb_medians.get(row['Famille Produit'], row['PNB analytique (vision commerciale) cumulé'])
                ) / (self.family_pnb_medians.get(row['Famille Produit'], 1) + 1),
                axis=1
            )

        # 4c. Ratio montant / PNB
        if 'Montant demandé' in df.columns and 'PNB analytique (vision commerciale) cumulé' in df.columns:
            df['ratio_montant_pnb'] = df['Montant demandé'] / (df['PNB analytique (vision commerciale) cumulé'] + 1)

        # 5. Features d'interaction
        if 'Montant demandé' in df.columns and 'anciennete_annees' in df.columns:
            df['montant_x_anciennete'] = df['Montant demandé'] * df['anciennete_annees']

        if 'Délai estimé' in df.columns and 'anciennete_annees' in df.columns:
            df['delai_x_anciennete'] = df['Délai estimé'] * df['anciennete_annees']

        if 'Montant demandé' in df.columns and 'Délai estimé' in df.columns:
            df['montant_x_delai'] = df['Montant demandé'] * df['Délai estimé']

        # 6. Interaction avec taux de fondée
        if 'taux_fondee_famille' in df.columns and 'Montant demandé' in df.columns:
            df['montant_x_taux_famille'] = df['Montant demandé'] * df['taux_fondee_famille']

        # 7. Interactions avec PNB
        if 'PNB analytique (vision commerciale) cumulé' in df.columns and 'anciennete_annees' in df.columns:
            df['pnb_x_anciennete'] = df['PNB analytique (vision commerciale) cumulé'] * df['anciennete_annees']

        if 'PNB analytique (vision commerciale) cumulé' in df.columns and 'taux_fondee_famille' in df.columns:
            df['pnb_x_taux_famille'] = df['PNB analytique (vision commerciale) cumulé'] * df['taux_fondee_famille']

        # Sélectionner colonnes numériques
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        # Garder colonnes nécessaires
        keep_cols = [col for col in numeric_cols if col != 'Fondee']

        df_result = df[keep_cols]

        # Nettoyer les inf et NaN
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

    def get_feature_info(self):
        """Retourne des informations sur les features calculées"""
        info = {
            'n_features': len(self.feature_names_fitted) if self.feature_names_fitted else 0,
            'feature_names': self.feature_names_fitted,
            'family_stats_count': len(self.family_stats.get('taux', {})),
            'category_stats_count': len(self.category_stats.get('taux', {})),
            'subcategory_stats_count': len(self.subcategory_stats.get('taux', {})),
            'segment_stats_count': len(self.segment_stats.get('taux', {})),
            'min_samples_for_stats': self.min_samples_stats
        }
        return info
