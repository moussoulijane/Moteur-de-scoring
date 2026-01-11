"""
Analyse de drift entre datasets 2024 et 2025
Tests statistiques pour détecter les changements de distribution
"""
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


class DriftAnalyzer:
    """
    Détecte le drift entre deux datasets
    """

    def __init__(self, significance_level=0.05):
        self.significance_level = significance_level
        self.drift_report = {}

    def analyze_drift(self, df_reference, df_current, numerical_cols=None, categorical_cols=None):
        """
        Analyse le drift entre deux datasets

        Args:
            df_reference: Dataset de référence (2024)
            df_current: Dataset actuel (2025)
            numerical_cols: Colonnes numériques à tester
            categorical_cols: Colonnes catégorielles à tester
        """
        print(f"\n🔍 ANALYSE DE DRIFT")
        print("=" * 60)
        print(f"Référence: {len(df_reference)} samples")
        print(f"Actuel:    {len(df_current)} samples")
        print(f"Seuil p-value: {self.significance_level}")

        # Auto-détection si non spécifié
        if numerical_cols is None:
            numerical_cols = df_reference.select_dtypes(include=[np.number]).columns.tolist()

        if categorical_cols is None:
            categorical_cols = df_reference.select_dtypes(include=['object', 'category']).columns.tolist()

        # Test KS pour numériques
        print(f"\n📊 Test Kolmogorov-Smirnov (features numériques):")
        print("-" * 60)
        numerical_drift = self._test_numerical_drift(df_reference, df_current, numerical_cols)

        # Test Chi² pour catégorielles
        print(f"\n📊 Test Chi² (features catégorielles):")
        print("-" * 60)
        categorical_drift = self._test_categorical_drift(df_reference, df_current, categorical_cols)

        # Résumé
        self._print_drift_summary(numerical_drift, categorical_drift)

        self.drift_report = {
            'numerical': numerical_drift,
            'categorical': categorical_drift
        }

        return self.drift_report

    def _test_numerical_drift(self, df_ref, df_curr, cols):
        """Test Kolmogorov-Smirnov pour features numériques"""
        results = []

        for col in cols:
            if col not in df_ref.columns or col not in df_curr.columns:
                continue

            # Supprimer NaN
            ref_values = df_ref[col].dropna()
            curr_values = df_curr[col].dropna()

            if len(ref_values) == 0 or len(curr_values) == 0:
                continue

            # Test KS
            statistic, p_value = stats.ks_2samp(ref_values, curr_values)

            # Statistiques descriptives
            ref_mean = ref_values.mean()
            curr_mean = curr_values.mean()
            mean_shift = ((curr_mean - ref_mean) / ref_mean) * 100 if ref_mean != 0 else 0

            drift_detected = p_value < self.significance_level

            results.append({
                'feature': col,
                'ks_statistic': statistic,
                'p_value': p_value,
                'drift_detected': drift_detected,
                'ref_mean': ref_mean,
                'curr_mean': curr_mean,
                'mean_shift_pct': mean_shift
            })

            # Affichage
            status = "🚨 DRIFT" if drift_detected else "✅ OK"
            print(f"  {col:<40s} | KS={statistic:.4f} | p={p_value:.4f} | Δ={mean_shift:+6.2f}% | {status}")

        return pd.DataFrame(results)

    def _test_categorical_drift(self, df_ref, df_curr, cols):
        """Test Chi² pour features catégorielles"""
        results = []

        for col in cols:
            if col not in df_ref.columns or col not in df_curr.columns:
                continue

            # Contingency table
            ref_counts = df_ref[col].value_counts()
            curr_counts = df_curr[col].value_counts()

            # Aligner les catégories
            all_categories = set(ref_counts.index) | set(curr_counts.index)
            ref_aligned = ref_counts.reindex(all_categories, fill_value=0)
            curr_aligned = curr_counts.reindex(all_categories, fill_value=0)

            # Test Chi²
            try:
                statistic, p_value = stats.chisquare(curr_aligned + 1, ref_aligned + 1)  # +1 pour éviter zéros

                drift_detected = p_value < self.significance_level

                results.append({
                    'feature': col,
                    'chi2_statistic': statistic,
                    'p_value': p_value,
                    'drift_detected': drift_detected,
                    'n_categories': len(all_categories)
                })

                # Affichage
                status = "🚨 DRIFT" if drift_detected else "✅ OK"
                print(f"  {col:<40s} | χ²={statistic:.4f} | p={p_value:.4f} | {status}")

            except Exception as e:
                print(f"  {col:<40s} | ⚠️  Erreur: {e}")

        return pd.DataFrame(results)

    def _print_drift_summary(self, numerical_drift, categorical_drift):
        """Affiche le résumé de drift"""
        print(f"\n📋 RÉSUMÉ DU DRIFT")
        print("=" * 60)

        # Numériques
        if not numerical_drift.empty:
            n_drift_num = numerical_drift['drift_detected'].sum()
            n_total_num = len(numerical_drift)
            pct_drift_num = (n_drift_num / n_total_num) * 100 if n_total_num > 0 else 0

            print(f"Features numériques:")
            print(f"  - Total: {n_total_num}")
            print(f"  - Drift détecté: {n_drift_num} ({pct_drift_num:.1f}%)")

            if n_drift_num > 0:
                print(f"  - Features avec drift:")
                for feat in numerical_drift[numerical_drift['drift_detected']]['feature'].tolist()[:10]:
                    print(f"      • {feat}")

        # Catégorielles
        if not categorical_drift.empty:
            n_drift_cat = categorical_drift['drift_detected'].sum()
            n_total_cat = len(categorical_drift)
            pct_drift_cat = (n_drift_cat / n_total_cat) * 100 if n_total_cat > 0 else 0

            print(f"\nFeatures catégorielles:")
            print(f"  - Total: {n_total_cat}")
            print(f"  - Drift détecté: {n_drift_cat} ({pct_drift_cat:.1f}%)")

            if n_drift_cat > 0:
                print(f"  - Features avec drift:")
                for feat in categorical_drift[categorical_drift['drift_detected']]['feature'].tolist()[:10]:
                    print(f"      • {feat}")

        # Recommandation
        total_drift = (
            (numerical_drift['drift_detected'].sum() if not numerical_drift.empty else 0) +
            (categorical_drift['drift_detected'].sum() if not categorical_drift.empty else 0)
        )

        print(f"\n🎯 RECOMMANDATION:")
        if total_drift == 0:
            print("  ✅ Pas de drift significatif détecté")
            print("  → Le modèle peut être déployé en production")
        elif total_drift <= 3:
            print("  ⚠️  Drift léger détecté sur quelques features")
            print("  → Monitoring renforcé recommandé")
        else:
            print("  🚨 Drift significatif détecté!")
            print("  → Réentraînement du modèle fortement recommandé")

    def plot_distribution_comparison(self, df_ref, df_curr, feature, save_path=None):
        """
        Compare visuellement les distributions d'une feature

        Args:
            df_ref: Dataset référence
            df_curr: Dataset actuel
            feature: Feature à comparer
            save_path: Chemin pour sauvegarder
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Distribution référence
        if df_ref[feature].dtype in [np.float64, np.int64]:
            axes[0].hist(df_ref[feature].dropna(), bins=50, alpha=0.7, label='2024', color='blue', edgecolor='black')
            axes[1].hist(df_curr[feature].dropna(), bins=50, alpha=0.7, label='2025', color='orange', edgecolor='black')

            axes[0].set_xlabel(feature)
            axes[0].set_ylabel('Fréquence')
            axes[0].set_title('Distribution 2024')
            axes[0].legend()

            axes[1].set_xlabel(feature)
            axes[1].set_ylabel('Fréquence')
            axes[1].set_title('Distribution 2025')
            axes[1].legend()

        else:
            # Catégorielle
            ref_counts = df_ref[feature].value_counts()
            curr_counts = df_curr[feature].value_counts()

            axes[0].bar(range(len(ref_counts)), ref_counts.values, color='blue', alpha=0.7)
            axes[0].set_xticks(range(len(ref_counts)))
            axes[0].set_xticklabels(ref_counts.index, rotation=45, ha='right')
            axes[0].set_title('Distribution 2024')

            axes[1].bar(range(len(curr_counts)), curr_counts.values, color='orange', alpha=0.7)
            axes[1].set_xticks(range(len(curr_counts)))
            axes[1].set_xticklabels(curr_counts.index, rotation=45, ha='right')
            axes[1].set_title('Distribution 2025')

        plt.suptitle(f'Comparaison de distribution: {feature}', fontsize=14)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.close()

    def compare_prediction_distributions(self, y_prob_ref, y_prob_curr, save_path=None):
        """
        Compare les distributions des probabilités prédites

        Args:
            y_prob_ref: Probabilités 2024
            y_prob_curr: Probabilités 2025
            save_path: Chemin pour sauvegarder
        """
        # Test KS
        statistic, p_value = stats.ks_2samp(y_prob_ref, y_prob_curr)

        print(f"\n📊 Comparaison des probabilités prédites:")
        print(f"  KS Statistic: {statistic:.4f}")
        print(f"  P-value: {p_value:.4f}")

        if p_value < self.significance_level:
            print(f"  🚨 DRIFT détecté dans les prédictions!")
        else:
            print(f"  ✅ Pas de drift significatif dans les prédictions")

        # Graphique
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.hist(y_prob_ref, bins=50, alpha=0.7, label='2024', color='blue', edgecolor='black')
        plt.hist(y_prob_curr, bins=50, alpha=0.7, label='2025', color='orange', edgecolor='black')
        plt.xlabel('Probabilité prédite')
        plt.ylabel('Fréquence')
        plt.title('Distribution des probabilités')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.boxplot([y_prob_ref, y_prob_curr], labels=['2024', '2025'])
        plt.ylabel('Probabilité prédite')
        plt.title('Boxplot des probabilités')

        plt.suptitle(f'Comparaison des prédictions (KS={statistic:.4f}, p={p_value:.4f})', fontsize=14)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.close()
