"""
Sélection de features robuste avec plusieurs méthodes
"""
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
import warnings
warnings.filterwarnings('ignore')


class FeatureSelector:
    """
    Sélection de features multi-critères:
    1. Élimination des features avec trop de NaN
    2. Élimination des features à variance quasi-nulle
    3. Élimination des features trop corrélées
    4. Feature importance (Permutation + Native model + SHAP)
    """

    def __init__(
        self,
        missing_threshold=0.5,
        variance_threshold=0.01,
        correlation_threshold=0.95,
        importance_methods=['permutation', 'native']
    ):
        self.missing_threshold = missing_threshold
        self.variance_threshold = variance_threshold
        self.correlation_threshold = correlation_threshold
        self.importance_methods = importance_methods

        self.selected_features = []
        self.eliminated_features = {
            'missing': [],
            'variance': [],
            'correlation': [],
            'low_importance': []
        }
        self.feature_importance_scores = {}

    def _remove_missing_features(self, X):
        """Élimine les features avec trop de valeurs manquantes"""
        missing_ratios = X.isnull().sum() / len(X)
        features_to_remove = missing_ratios[missing_ratios > self.missing_threshold].index.tolist()

        self.eliminated_features['missing'] = features_to_remove

        if features_to_remove:
            print(f"  ❌ {len(features_to_remove)} features éliminées (>{self.missing_threshold*100:.0f}% NaN): {features_to_remove[:5]}...")

        return X.drop(columns=features_to_remove)

    def _remove_low_variance_features(self, X):
        """Élimine les features à variance quasi-nulle"""
        selector = VarianceThreshold(threshold=self.variance_threshold)
        selector.fit(X)

        low_var_mask = ~selector.get_support()
        features_to_remove = X.columns[low_var_mask].tolist()

        self.eliminated_features['variance'] = features_to_remove

        if features_to_remove:
            print(f"  ❌ {len(features_to_remove)} features éliminées (variance < {self.variance_threshold}): {features_to_remove[:5]}...")

        return X[X.columns[selector.get_support()]]

    def _remove_correlated_features(self, X):
        """Élimine les features trop corrélées entre elles"""
        corr_matrix = X.corr().abs()

        # Matrice triangulaire supérieure
        upper_tri = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )

        # Features à supprimer
        features_to_remove = [
            column for column in upper_tri.columns
            if any(upper_tri[column] > self.correlation_threshold)
        ]

        self.eliminated_features['correlation'] = features_to_remove

        if features_to_remove:
            print(f"  ❌ {len(features_to_remove)} features éliminées (corrélation > {self.correlation_threshold}): {features_to_remove[:5]}...")

        return X.drop(columns=features_to_remove)

    def _calculate_permutation_importance(self, X, y, n_iterations=10):
        """Calcule l'importance par permutation"""
        print("  🔄 Calcul de l'importance par permutation...")

        # Modèle simple et rapide pour l'importance
        model = RandomForestClassifier(
            n_estimators=50,
            max_depth=8,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X, y)

        # Permutation importance
        perm_importance = permutation_importance(
            model, X, y,
            n_repeats=n_iterations,
            random_state=42,
            n_jobs=-1
        )

        importance_df = pd.DataFrame({
            'feature': X.columns,
            'importance_mean': perm_importance.importances_mean,
            'importance_std': perm_importance.importances_std
        }).sort_values('importance_mean', ascending=False)

        return importance_df

    def _calculate_native_importance(self, X, y):
        """Calcule l'importance native du Random Forest"""
        print("  🔄 Calcul de l'importance native RF...")

        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X, y)

        importance_df = pd.DataFrame({
            'feature': X.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)

        return importance_df

    def _calculate_shap_importance(self, X, y, max_samples=1000):
        """Calcule l'importance SHAP (si disponible)"""
        try:
            import shap
            print("  🔄 Calcul de l'importance SHAP...")

            # Sous-échantillon pour accélérer
            if len(X) > max_samples:
                sample_idx = np.random.choice(len(X), max_samples, replace=False)
                X_sample = X.iloc[sample_idx]
                y_sample = y.iloc[sample_idx]
            else:
                X_sample = X
                y_sample = y

            model = RandomForestClassifier(
                n_estimators=50,
                max_depth=8,
                random_state=42,
                n_jobs=-1
            )
            model.fit(X_sample, y_sample)

            # SHAP TreeExplainer
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_sample)

            # Moyenne absolue des SHAP values
            if isinstance(shap_values, list):  # Classification binaire
                shap_values = shap_values[1]

            importance_df = pd.DataFrame({
                'feature': X.columns,
                'importance': np.abs(shap_values).mean(axis=0)
            }).sort_values('importance', ascending=False)

            return importance_df

        except ImportError:
            print("  ⚠️  SHAP non disponible, skipping...")
            return None

    def fit(self, X, y, top_k_features=None):
        """
        Sélectionne les features importantes

        Args:
            X: Features
            y: Target
            top_k_features: Nombre de features à garder (None = toutes les importantes)
        """
        print("\n🔍 SÉLECTION DE FEATURES")
        print("=" * 60)

        # 1. Éliminer features avec trop de NaN
        print("\n1️⃣  Élimination des features avec trop de NaN...")
        X_clean = self._remove_missing_features(X)

        # 2. Éliminer features à variance nulle
        print("\n2️⃣  Élimination des features à variance quasi-nulle...")
        X_clean = self._remove_low_variance_features(X_clean)

        # 3. Éliminer features trop corrélées
        print("\n3️⃣  Élimination des features trop corrélées...")
        X_clean = self._remove_correlated_features(X_clean)

        # 4. Feature importance
        print("\n4️⃣  Calcul de l'importance des features...")

        importance_scores = {}

        if 'permutation' in self.importance_methods:
            perm_imp = self._calculate_permutation_importance(X_clean, y)
            importance_scores['permutation'] = perm_imp.set_index('feature')['importance_mean']

        if 'native' in self.importance_methods:
            native_imp = self._calculate_native_importance(X_clean, y)
            importance_scores['native'] = native_imp.set_index('feature')['importance']

        if 'shap' in self.importance_methods:
            shap_imp = self._calculate_shap_importance(X_clean, y)
            if shap_imp is not None:
                importance_scores['shap'] = shap_imp.set_index('feature')['importance']

        # Combiner les scores (moyenne normalisée)
        combined_importance = pd.DataFrame(importance_scores)

        # Normaliser chaque méthode entre 0 et 1
        for col in combined_importance.columns:
            combined_importance[col] = (
                (combined_importance[col] - combined_importance[col].min()) /
                (combined_importance[col].max() - combined_importance[col].min())
            )

        # Score moyen
        combined_importance['mean_importance'] = combined_importance.mean(axis=1)
        combined_importance = combined_importance.sort_values('mean_importance', ascending=False)

        self.feature_importance_scores = combined_importance

        # 5. Sélection finale
        if top_k_features:
            self.selected_features = combined_importance.head(top_k_features).index.tolist()
            eliminated = combined_importance.iloc[top_k_features:].index.tolist()
            self.eliminated_features['low_importance'] = eliminated
        else:
            # Garder les features qui ont une importance > seuil dans au moins 2 méthodes
            threshold = 0.1
            important_mask = (combined_importance.drop('mean_importance', axis=1) > threshold).sum(axis=1) >= 2
            self.selected_features = combined_importance[important_mask].index.tolist()
            self.eliminated_features['low_importance'] = combined_importance[~important_mask].index.tolist()

        print(f"\n✅ Sélection terminée:")
        print(f"   - Features conservées: {len(self.selected_features)}")
        print(f"   - Features éliminées: {sum(len(v) for v in self.eliminated_features.values())}")
        print(f"\n📊 Top 10 features importantes:")
        for i, (feat, score) in enumerate(combined_importance.head(10)['mean_importance'].items(), 1):
            print(f"   {i:2d}. {feat:40s} {score:.4f}")

        return self

    def transform(self, X):
        """Applique la sélection de features"""
        return X[self.selected_features]

    def fit_transform(self, X, y, top_k_features=None):
        """Fit et transform"""
        self.fit(X, y, top_k_features)
        return self.transform(X)

    def get_feature_importance(self):
        """Retourne le DataFrame d'importance"""
        return self.feature_importance_scores
