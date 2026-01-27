"""
Production Preprocessor - Clean & Optimized
Features available in real-time only
Statistics computed on training data (2024) and reused for inference
"""
import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import RobustScaler, StandardScaler
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class ProductionPreprocessor:
    """
    Production-ready preprocessor with frozen statistics.

    Features:
    - Real-time available columns only
    - Statistical robustness (minimum samples threshold)
    - Frozen statistics from training data
    - Comprehensive feature engineering
    """

    def __init__(self, min_samples_stats: int = 30, scaler_type: str = "robust"):
        """
        Initialize preprocessor.

        Args:
            min_samples_stats: Minimum samples for reliable statistics
            scaler_type: 'robust' or 'standard'
        """
        self.min_samples_stats = min_samples_stats
        self.scaler_type = scaler_type

        # Initialize scaler
        if scaler_type == "robust":
            self.scaler = RobustScaler()
        else:
            self.scaler = StandardScaler()

        # Statistics computed on training data
        self.family_stats: Dict = {}
        self.category_stats: Dict = {}
        self.subcategory_stats: Dict = {}
        self.segment_stats: Dict = {}
        self.family_medians: Dict = {}
        self.family_pnb_medians: Dict = {}
        self.categorical_encodings: Dict = {}

        self.feature_names_fitted: Optional[List[str]] = None
        self._is_fitted = False

    def _clean_numeric_column(self, series: pd.Series) -> pd.Series:
        """
        Clean numeric column that may contain text.

        Examples:
            '500 mad' -> 500.0
            '1 000 DH' -> 1000.0
            '1,500.50' -> 1500.50
            '1.500,50' -> 1500.50
        """
        if series.dtype in ['float64', 'float32', 'int64', 'int32']:
            return pd.to_numeric(series, errors='coerce').fillna(0)

        cleaned_series = series.astype(str)

        def clean_value(val):
            if pd.isna(val) or val in ['', 'nan', 'None', 'NaN']:
                return 0.0

            val_str = str(val).lower().strip()

            # Remove currency words
            val_str = re.sub(
                r'\b(mad|dh|dirham|dirhams|€|euro|euros)\b',
                '',
                val_str,
                flags=re.IGNORECASE
            )

            # Keep only digits, dots, commas, spaces, and signs
            val_str = re.sub(r'[^\d\s\.,\-\+]', '', val_str)

            # Remove spaces (thousand separators)
            val_str = val_str.replace(' ', '')

            # Handle European (1.500,50) vs English (1,500.50) format
            if re.search(r',\d{2}$', val_str):
                # European format
                val_str = val_str.replace('.', '').replace(',', '.')
            else:
                # English format
                val_str = val_str.replace(',', '')

            try:
                return float(val_str) if val_str else 0.0
            except (ValueError, TypeError):
                return 0.0

        cleaned = cleaned_series.apply(clean_value)
        return pd.to_numeric(cleaned, errors='coerce').fillna(0)

    def _compute_rate_stats(self, df: pd.DataFrame, column: str) -> Dict:
        """Compute rate statistics for a categorical column."""
        grouped = df.groupby(column).agg({
            'Fondee': ['mean', 'count']
        })
        grouped.columns = ['taux_fondee', 'count']

        # Filter by minimum samples
        grouped_filtered = grouped[grouped['count'] >= self.min_samples_stats]

        return {
            'taux': grouped_filtered['taux_fondee'].to_dict(),
            'count': grouped['count'].to_dict(),
            'taux_global': df['Fondee'].mean()
        }

    def fit(self, df: pd.DataFrame) -> 'ProductionPreprocessor':
        """
        Fit on training data (2024) - compute all statistics.

        Args:
            df: Training DataFrame with 'Fondee' column

        Returns:
            self
        """
        logger.info("🔧 Fitting Production Preprocessor...")

        X = df.copy()

        if 'Fondee' not in X.columns:
            raise ValueError("Column 'Fondee' is required for training")

        # 1. Clean and convert numeric columns
        logger.info("Converting numeric columns...")
        numeric_columns = [
            'Montant demandé',
            'Délai estimé',
            'anciennete_annees',
            'PNB analytique (vision commerciale) cumulé'
        ]

        for col in numeric_columns:
            if col in X.columns:
                X[col] = self._clean_numeric_column(X[col])
                X[col] = X[col].replace([np.inf, -np.inf], 0).clip(lower=0)

        # 2. Convert categorical to string
        logger.info("Converting categorical columns...")
        categorical_cols = [
            'Marché', 'Segment', 'Famille Produit',
            'Catégorie', 'Sous-catégorie'
        ]

        for col in categorical_cols:
            if col in X.columns:
                X[col] = X[col].astype(str).fillna('UNKNOWN')

        # 3. Compute acceptance rates by category
        logger.info("Computing acceptance rates (statistically robust)...")

        if 'Famille Produit' in X.columns:
            self.family_stats = self._compute_rate_stats(X, 'Famille Produit')
            logger.info(
                f"   ✅ Famille Produit: {len(self.family_stats['taux'])} "
                f"families with robust stats (≥{self.min_samples_stats} cases)"
            )

        if 'Catégorie' in X.columns:
            self.category_stats = self._compute_rate_stats(X, 'Catégorie')
            logger.info(
                f"   ✅ Catégorie: {len(self.category_stats['taux'])} categories"
            )

        if 'Sous-catégorie' in X.columns:
            self.subcategory_stats = self._compute_rate_stats(X, 'Sous-catégorie')
            logger.info(
                f"   ✅ Sous-catégorie: {len(self.subcategory_stats['taux'])} subcategories"
            )

        if 'Segment' in X.columns:
            self.segment_stats = self._compute_rate_stats(X, 'Segment')
            logger.info(
                f"   ✅ Segment: {len(self.segment_stats['taux'])} segments"
            )

        # 4. Compute medians by family
        logger.info("Computing medians by family...")

        if 'Famille Produit' in X.columns and 'Montant demandé' in X.columns:
            self.family_medians = X.groupby('Famille Produit')['Montant demandé'].median().to_dict()

        if 'Famille Produit' in X.columns and 'PNB analytique (vision commerciale) cumulé' in X.columns:
            pnb_data = X[X['PNB analytique (vision commerciale) cumulé'] > 0]
            if len(pnb_data) > 0:
                self.family_pnb_medians = pnb_data.groupby(
                    'Famille Produit'
                )['PNB analytique (vision commerciale) cumulé'].median().to_dict()

        # 5. Encode categorical frequencies
        logger.info("Encoding categorical frequencies...")
        for col in categorical_cols:
            if col in X.columns:
                self.categorical_encodings[col] = X[col].value_counts().to_dict()
                X[f'{col}_freq'] = X[col].map(self.categorical_encodings[col]).fillna(0)

        # 6. Create engineered features
        X = self._create_features(X, fit_mode=True)

        # 7. Final numeric columns
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [c for c in numeric_cols if c != 'Fondee']

        # Sort alphabetically for consistency
        self.feature_names_fitted = sorted(numeric_cols)
        logger.info(f"📋 Final features: {len(self.feature_names_fitted)}")

        # 8. Fit scaler
        X_ordered = X[self.feature_names_fitted]
        self.scaler.fit(X_ordered)

        self._is_fitted = True
        logger.info(f"✅ Preprocessor fitted: {len(self.feature_names_fitted)} features")

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform new data using fitted statistics.

        Args:
            df: DataFrame to transform

        Returns:
            Transformed DataFrame
        """
        if not self._is_fitted:
            raise ValueError("Preprocessor must be fitted before transform")

        X = df.copy()

        # 1. Clean numeric columns
        numeric_columns = [
            'Montant demandé',
            'Délai estimé',
            'anciennete_annees',
            'PNB analytique (vision commerciale) cumulé'
        ]

        for col in numeric_columns:
            if col in X.columns:
                X[col] = self._clean_numeric_column(X[col])
                X[col] = X[col].replace([np.inf, -np.inf], 0).clip(lower=0)

        # 2. Convert categorical
        categorical_cols = [
            'Marché', 'Segment', 'Famille Produit',
            'Catégorie', 'Sous-catégorie'
        ]

        for col in categorical_cols:
            if col in X.columns:
                X[col] = X[col].astype(str).fillna('UNKNOWN')

        # 3. Encode frequencies with training encodings
        for col in categorical_cols:
            if col in X.columns and col in self.categorical_encodings:
                X[f'{col}_freq'] = X[col].map(self.categorical_encodings[col]).fillna(0)

        # 4. Create engineered features
        X = self._create_features(X, fit_mode=False)

        # 5. Handle missing columns
        for col in self.feature_names_fitted:
            if col not in X.columns:
                X[col] = 0

        # 6. Keep only fitted columns in correct order
        X = X[self.feature_names_fitted]

        # 7. Scale
        X[self.feature_names_fitted] = self.scaler.transform(X[self.feature_names_fitted])

        return X

    def _create_features(self, X: pd.DataFrame, fit_mode: bool = True) -> pd.DataFrame:
        """Create engineered features."""
        df = X.copy()

        # 1. Acceptance rates (from training data)
        if 'Famille Produit' in df.columns and self.family_stats:
            df['taux_fondee_famille'] = df['Famille Produit'].map(self.family_stats['taux'])
            df['taux_fondee_famille'] = df['taux_fondee_famille'].fillna(
                self.family_stats['taux_global']
            )
            df['count_famille'] = df['Famille Produit'].map(
                self.family_stats['count']
            ).fillna(0)

        if 'Catégorie' in df.columns and self.category_stats:
            df['taux_fondee_categorie'] = df['Catégorie'].map(self.category_stats['taux'])
            df['taux_fondee_categorie'] = df['taux_fondee_categorie'].fillna(
                self.category_stats['taux_global']
            )

        if 'Sous-catégorie' in df.columns and self.subcategory_stats:
            df['taux_fondee_souscategorie'] = df['Sous-catégorie'].map(
                self.subcategory_stats['taux']
            )
            df['taux_fondee_souscategorie'] = df['taux_fondee_souscategorie'].fillna(
                self.subcategory_stats['taux_global']
            )

        if 'Segment' in df.columns and self.segment_stats:
            df['taux_fondee_segment'] = df['Segment'].map(self.segment_stats['taux'])
            df['taux_fondee_segment'] = df['taux_fondee_segment'].fillna(
                self.segment_stats['taux_global']
            )

        # 2. Deviation from family median
        if (
            'Famille Produit' in df.columns
            and 'Montant demandé' in df.columns
            and self.family_medians
        ):
            df['ecart_mediane_famille'] = df.apply(
                lambda row: (
                    row['Montant demandé']
                    - self.family_medians.get(row['Famille Produit'], row['Montant demandé'])
                ) / (self.family_medians.get(row['Famille Produit'], 1) + 1),
                axis=1
            )

        # 3. Ratios
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
            df['log_pnb'] = np.log1p(
                np.abs(df['PNB analytique (vision commerciale) cumulé'])
            )

        # 5. PNB deviation from family median
        if (
            'Famille Produit' in df.columns
            and 'PNB analytique (vision commerciale) cumulé' in df.columns
            and self.family_pnb_medians
        ):
            df['ecart_pnb_mediane_famille'] = df.apply(
                lambda row: (
                    row['PNB analytique (vision commerciale) cumulé']
                    - self.family_pnb_medians.get(
                        row['Famille Produit'],
                        row['PNB analytique (vision commerciale) cumulé']
                    )
                ) / (self.family_pnb_medians.get(row['Famille Produit'], 1) + 1),
                axis=1
            )

        # 6. Amount / PNB ratio
        if (
            'Montant demandé' in df.columns
            and 'PNB analytique (vision commerciale) cumulé' in df.columns
        ):
            df['ratio_montant_pnb'] = df['Montant demandé'] / (
                df['PNB analytique (vision commerciale) cumulé'] + 1
            )

        # 7. Interaction features
        if 'Montant demandé' in df.columns and 'anciennete_annees' in df.columns:
            df['montant_x_anciennete'] = df['Montant demandé'] * df['anciennete_annees']

        if 'Délai estimé' in df.columns and 'anciennete_annees' in df.columns:
            df['delai_x_anciennete'] = df['Délai estimé'] * df['anciennete_annees']

        if 'Montant demandé' in df.columns and 'Délai estimé' in df.columns:
            df['montant_x_delai'] = df['Montant demandé'] * df['Délai estimé']

        if 'taux_fondee_famille' in df.columns and 'Montant demandé' in df.columns:
            df['montant_x_taux_famille'] = (
                df['Montant demandé'] * df['taux_fondee_famille']
            )

        if (
            'PNB analytique (vision commerciale) cumulé' in df.columns
            and 'anciennete_annees' in df.columns
        ):
            df['pnb_x_anciennete'] = (
                df['PNB analytique (vision commerciale) cumulé'] * df['anciennete_annees']
            )

        if (
            'PNB analytique (vision commerciale) cumulé' in df.columns
            and 'taux_fondee_famille' in df.columns
        ):
            df['pnb_x_taux_famille'] = (
                df['PNB analytique (vision commerciale) cumulé'] * df['taux_fondee_famille']
            )

        # Select numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        keep_cols = [col for col in numeric_cols if col != 'Fondee']

        df_result = df[keep_cols]

        # Clean inf and NaN
        df_result = self._clean_numeric_data(df_result)

        return df_result

    def _clean_numeric_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean NaN and inf in numeric columns."""
        df_clean = df.copy()

        for col in df_clean.columns:
            if df_clean[col].dtype in [np.float64, np.float32, np.int64, np.int32]:
                # Replace inf with NaN
                df_clean[col] = df_clean[col].replace([np.inf, -np.inf], np.nan)

                # Replace NaN with median or 0
                median_val = df_clean[col].median()
                if pd.isna(median_val):
                    median_val = 0.0
                df_clean[col] = df_clean[col].fillna(median_val)

        return df_clean

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform in one call."""
        self.fit(df)
        return self.transform(df)

    def get_feature_info(self) -> Dict:
        """Get information about computed features."""
        return {
            'n_features': len(self.feature_names_fitted) if self.feature_names_fitted else 0,
            'feature_names': self.feature_names_fitted,
            'family_stats_count': len(self.family_stats.get('taux', {})),
            'category_stats_count': len(self.category_stats.get('taux', {})),
            'subcategory_stats_count': len(self.subcategory_stats.get('taux', {})),
            'segment_stats_count': len(self.segment_stats.get('taux', {})),
            'min_samples_for_stats': self.min_samples_stats,
            'scaler_type': self.scaler_type,
            'is_fitted': self._is_fitted
        }
