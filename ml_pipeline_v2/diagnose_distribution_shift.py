"""
DIAGNOSTIC DE DISTRIBUTION SHIFT
Analyse les différences entre les données d'entraînement (2024) et nouvelles données (ex: 2023)
pour identifier pourquoi le modèle prédit des probabilités anormales

Usage:
    python ml_pipeline_v2/diagnose_distribution_shift.py --reference_file data/raw/reclamations_2024.xlsx --new_file data/raw/reclamations_2023.xlsx
"""
import sys
sys.path.append('src')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from preprocessor_v2 import ProductionPreprocessorV2

# Configuration
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (20, 12)


class DistributionShiftDiagnostic:
    """Diagnostic de distribution shift entre données de référence et nouvelles données"""

    def __init__(self, reference_file, new_file):
        self.reference_file = reference_file
        self.new_file = new_file
        self.output_dir = Path('outputs/diagnostic_shift')
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Colonnes importantes
        self.numeric_cols = ['Montant demandé', 'Délai estimé', 'anciennete_annees']
        self.categorical_cols = ['Famille Produit', 'Catégorie', 'Sous-catégorie', 'Segment', 'Marché']

    def load_data(self):
        """Charger les données"""
        print("\n" + "="*80)
        print("📂 CHARGEMENT DES DONNÉES")
        print("="*80)

        self.df_ref = pd.read_excel(self.reference_file)
        self.df_new = pd.read_excel(self.new_file)

        print(f"✅ Données de référence (train): {len(self.df_ref)} lignes")
        print(f"✅ Nouvelles données (inférence): {len(self.df_new)} lignes")

        # Nettoyer les colonnes numériques
        from preprocessor_v2 import ProductionPreprocessorV2
        preprocessor = ProductionPreprocessorV2()
        for df in [self.df_ref, self.df_new]:
            for col in self.numeric_cols:
                if col in df.columns:
                    df[col] = preprocessor._clean_numeric_column(df[col])

    def analyze_numeric_distributions(self):
        """Analyser les distributions des variables numériques"""
        print("\n" + "="*80)
        print("📊 ANALYSE DES DISTRIBUTIONS NUMÉRIQUES")
        print("="*80)

        fig, axes = plt.subplots(len(self.numeric_cols), 3, figsize=(20, 6*len(self.numeric_cols)))
        fig.suptitle('COMPARAISON DISTRIBUTIONS NUMÉRIQUES: Référence vs Nouvelles Données',
                     fontsize=16, fontweight='bold', y=0.995)

        results = []

        for idx, col in enumerate(self.numeric_cols):
            if col not in self.df_ref.columns or col not in self.df_new.columns:
                continue

            # Données
            ref_data = self.df_ref[col][self.df_ref[col] > 0]
            new_data = self.df_new[col][self.df_new[col] > 0]

            # Statistiques
            ref_stats = {
                'mean': ref_data.mean(),
                'median': ref_data.median(),
                'std': ref_data.std(),
                'min': ref_data.min(),
                'max': ref_data.max(),
                'q25': ref_data.quantile(0.25),
                'q75': ref_data.quantile(0.75),
                'zeros': (self.df_ref[col] == 0).sum() / len(self.df_ref)
            }

            new_stats = {
                'mean': new_data.mean(),
                'median': new_data.median(),
                'std': new_data.std(),
                'min': new_data.min(),
                'max': new_data.max(),
                'q25': new_data.quantile(0.25),
                'q75': new_data.quantile(0.75),
                'zeros': (self.df_new[col] == 0).sum() / len(self.df_new)
            }

            # Calcul des écarts
            mean_diff_pct = ((new_stats['mean'] - ref_stats['mean']) / ref_stats['mean'] * 100)
            median_diff_pct = ((new_stats['median'] - ref_stats['median']) / ref_stats['median'] * 100)

            results.append({
                'Variable': col,
                'Ref_Mean': ref_stats['mean'],
                'New_Mean': new_stats['mean'],
                'Mean_Diff_%': mean_diff_pct,
                'Ref_Median': ref_stats['median'],
                'New_Median': new_stats['median'],
                'Median_Diff_%': median_diff_pct,
                'Ref_Zeros_%': ref_stats['zeros'] * 100,
                'New_Zeros_%': new_stats['zeros'] * 100
            })

            print(f"\n📊 {col}:")
            print(f"   Référence - Mean: {ref_stats['mean']:,.2f}, Median: {ref_stats['median']:,.2f}")
            print(f"   Nouvelles - Mean: {new_stats['mean']:,.2f}, Median: {new_stats['median']:,.2f}")
            print(f"   ⚠️  Différence: Mean {mean_diff_pct:+.1f}%, Median {median_diff_pct:+.1f}%")
            if abs(mean_diff_pct) > 20 or abs(median_diff_pct) > 20:
                print(f"   🚨 ALERTE: Différence > 20% !")

            # Plot 1: Histogrammes superposés
            ax = axes[idx, 0]
            # Limiter aux percentiles pour meilleure visualisation
            ref_plot = ref_data[ref_data <= ref_data.quantile(0.95)]
            new_plot = new_data[new_data <= new_data.quantile(0.95)]

            ax.hist(ref_plot, bins=50, alpha=0.5, label='Référence (train)', color='blue', density=True)
            ax.hist(new_plot, bins=50, alpha=0.5, label='Nouvelles données', color='red', density=True)
            ax.axvline(ref_stats['median'], color='blue', linestyle='--', linewidth=2, label=f'Médiane Ref: {ref_stats["median"]:.0f}')
            ax.axvline(new_stats['median'], color='red', linestyle='--', linewidth=2, label=f'Médiane New: {new_stats["median"]:.0f}')
            ax.set_xlabel(col, fontweight='bold')
            ax.set_ylabel('Densité', fontweight='bold')
            ax.set_title(f'Distribution: {col}', fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)

            # Plot 2: Box plots comparatifs
            ax = axes[idx, 1]
            box_data = [ref_plot, new_plot]
            bp = ax.boxplot(box_data, labels=['Référence', 'Nouvelles'], patch_artist=True)
            bp['boxes'][0].set_facecolor('blue')
            bp['boxes'][1].set_facecolor('red')
            for box in bp['boxes']:
                box.set_alpha(0.6)
            ax.set_ylabel(col, fontweight='bold')
            ax.set_title(f'Box Plot: {col}', fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')

            # Plot 3: Statistiques textuelles
            ax = axes[idx, 2]
            ax.axis('off')

            stats_text = f"""
STATISTIQUES COMPARATIVES

Référence (Entraînement):
  Mean     : {ref_stats['mean']:>12,.2f}
  Median   : {ref_stats['median']:>12,.2f}
  Std      : {ref_stats['std']:>12,.2f}
  Q25-Q75  : {ref_stats['q25']:>12,.2f} - {ref_stats['q75']:,.2f}
  Zeros    : {ref_stats['zeros']:>12.1%}

Nouvelles Données:
  Mean     : {new_stats['mean']:>12,.2f}
  Median   : {new_stats['median']:>12,.2f}
  Std      : {new_stats['std']:>12,.2f}
  Q25-Q75  : {new_stats['q25']:>12,.2f} - {new_stats['q75']:,.2f}
  Zeros    : {new_stats['zeros']:>12.1%}

ÉCARTS:
  Mean     : {mean_diff_pct:>12.1f}%
  Median   : {median_diff_pct:>12.1f}%
"""

            if abs(mean_diff_pct) > 20 or abs(median_diff_pct) > 20:
                stats_text += "\n🚨 ALERTE: Shift significatif!"

            ax.text(0.1, 0.5, stats_text, transform=ax.transAxes,
                   fontsize=11, verticalalignment='center', family='monospace',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()
        output_path = self.output_dir / '01_numeric_distributions.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\n✅ Graphique sauvegardé: {output_path}")
        plt.close()

        return pd.DataFrame(results)

    def analyze_categorical_distributions(self):
        """Analyser les distributions catégorielles"""
        print("\n" + "="*80)
        print("📊 ANALYSE DES DISTRIBUTIONS CATÉGORIELLES")
        print("="*80)

        results = []

        for col in self.categorical_cols:
            if col not in self.df_ref.columns or col not in self.df_new.columns:
                continue

            print(f"\n📊 {col}:")

            # Distributions
            ref_dist = self.df_ref[col].value_counts(normalize=True)
            new_dist = self.df_new[col].value_counts(normalize=True)

            # Valeurs communes et nouvelles
            ref_values = set(ref_dist.index)
            new_values = set(new_dist.index)

            only_in_ref = ref_values - new_values
            only_in_new = new_values - ref_values
            common_values = ref_values & new_values

            print(f"   Valeurs dans Référence: {len(ref_values)}")
            print(f"   Valeurs dans Nouvelles: {len(new_values)}")
            print(f"   Valeurs communes: {len(common_values)}")

            if only_in_new:
                print(f"   🚨 NOUVELLES VALEURS (absentes du train): {len(only_in_new)}")
                if len(only_in_new) <= 10:
                    for val in list(only_in_new)[:10]:
                        count = (self.df_new[col] == val).sum()
                        pct = count / len(self.df_new) * 100
                        print(f"      - {val}: {count} cas ({pct:.1f}%)")

            # Top 10 valeurs - comparaison
            print(f"\n   Top 10 valeurs - Comparaison:")
            top_ref = ref_dist.head(10)
            for val, ref_pct in top_ref.items():
                new_pct = new_dist.get(val, 0)
                diff_pct = (new_pct - ref_pct) * 100
                print(f"      {val[:40]:40s}: Ref {ref_pct:5.1%} | New {new_pct:5.1%} | Diff {diff_pct:+5.1f}pp")

            results.append({
                'Variable': col,
                'Ref_Unique': len(ref_values),
                'New_Unique': len(new_values),
                'Common': len(common_values),
                'Only_in_New': len(only_in_new),
                'Only_in_Ref': len(only_in_ref),
                'Coverage_%': len(common_values) / len(new_values) * 100 if len(new_values) > 0 else 0
            })

        return pd.DataFrame(results)

    def analyze_engineered_features(self):
        """Analyser les features engineered (taux de fondée, etc.)"""
        print("\n" + "="*80)
        print("📊 ANALYSE DES FEATURES ENGINEERED")
        print("="*80)

        # Créer preprocessor et fit sur référence
        preprocessor = ProductionPreprocessorV2(min_samples_stats=30)

        if 'Fondee' not in self.df_ref.columns:
            print("⚠️  Colonne 'Fondee' manquante dans données de référence")
            print("   Impossible d'analyser les taux de fondée")
            return None

        preprocessor.fit(self.df_ref)

        # Analyser la couverture des taux de fondée
        print("\n📊 Couverture des statistiques robustes (min 30 cas sur train):")

        results = []

        # Familles
        if 'Famille Produit' in self.df_new.columns:
            new_families = self.df_new['Famille Produit'].unique()
            families_with_stats = set(preprocessor.family_stats['taux'].keys())
            coverage = sum([f in families_with_stats for f in new_families]) / len(new_families) * 100

            missing_families = [f for f in new_families if f not in families_with_stats]
            missing_count = sum([self.df_new['Famille Produit'] == f for f in missing_families]).sum()

            print(f"\n   Famille Produit:")
            print(f"      Familles dans nouvelles données: {len(new_families)}")
            print(f"      Familles avec stats robustes (train): {len(families_with_stats)}")
            print(f"      Couverture: {coverage:.1f}%")
            print(f"      🚨 Familles sans stats: {len(missing_families)} ({missing_count} cas)")

            if missing_families and len(missing_families) <= 20:
                print(f"      Familles manquantes:")
                for fam in missing_families[:20]:
                    count = (self.df_new['Famille Produit'] == fam).sum()
                    pct = count / len(self.df_new) * 100
                    print(f"         - {fam}: {count} cas ({pct:.1f}%)")

            results.append({
                'Feature': 'taux_fondee_famille',
                'Total_New_Values': len(new_families),
                'With_Stats': len([f for f in new_families if f in families_with_stats]),
                'Coverage_%': coverage,
                'Missing_Cases': missing_count,
                'Missing_%': missing_count / len(self.df_new) * 100
            })

        # Catégories
        if 'Catégorie' in self.df_new.columns:
            new_cats = self.df_new['Catégorie'].unique()
            cats_with_stats = set(preprocessor.category_stats['taux'].keys())
            coverage = sum([c in cats_with_stats for c in new_cats]) / len(new_cats) * 100
            missing_count = sum([self.df_new['Catégorie'] == c for c in new_cats if c not in cats_with_stats]).sum()

            print(f"\n   Catégorie:")
            print(f"      Couverture: {coverage:.1f}%")
            print(f"      🚨 Cas sans stats: {missing_count} ({missing_count/len(self.df_new)*100:.1f}%)")

            results.append({
                'Feature': 'taux_fondee_categorie',
                'Total_New_Values': len(new_cats),
                'With_Stats': len([c for c in new_cats if c in cats_with_stats]),
                'Coverage_%': coverage,
                'Missing_Cases': missing_count,
                'Missing_%': missing_count / len(self.df_new) * 100
            })

        # Sous-catégories
        if 'Sous-catégorie' in self.df_new.columns:
            new_subcats = self.df_new['Sous-catégorie'].unique()
            subcats_with_stats = set(preprocessor.subcategory_stats['taux'].keys())
            coverage = sum([s in subcats_with_stats for s in new_subcats]) / len(new_subcats) * 100
            missing_count = sum([self.df_new['Sous-catégorie'] == s for s in new_subcats if s not in subcats_with_stats]).sum()

            print(f"\n   Sous-catégorie:")
            print(f"      Couverture: {coverage:.1f}%")
            print(f"      🚨 Cas sans stats: {missing_count} ({missing_count/len(self.df_new)*100:.1f}%)")

            results.append({
                'Feature': 'taux_fondee_souscategorie',
                'Total_New_Values': len(new_subcats),
                'With_Stats': len([s for s in new_subcats if s in subcats_with_stats]),
                'Coverage_%': coverage,
                'Missing_Cases': missing_count,
                'Missing_%': missing_count / len(self.df_new) * 100
            })

        # Segments
        if 'Segment' in self.df_new.columns:
            new_segs = self.df_new['Segment'].unique()
            segs_with_stats = set(preprocessor.segment_stats['taux'].keys())
            coverage = sum([s in segs_with_stats for s in new_segs]) / len(new_segs) * 100
            missing_count = sum([self.df_new['Segment'] == s for s in new_segs if s not in segs_with_stats]).sum()

            print(f"\n   Segment:")
            print(f"      Couverture: {coverage:.1f}%")
            print(f"      🚨 Cas sans stats: {missing_count} ({missing_count/len(self.df_new)*100:.1f}%)")

            results.append({
                'Feature': 'taux_fondee_segment',
                'Total_New_Values': len(new_segs),
                'With_Stats': len([s for s in new_segs if s in segs_with_stats]),
                'Coverage_%': coverage,
                'Missing_Cases': missing_count,
                'Missing_%': missing_count / len(self.df_new) * 100
            })

        return pd.DataFrame(results)

    def generate_summary_report(self, numeric_results, categorical_results, engineered_results):
        """Générer rapport récapitulatif"""
        print("\n" + "="*80)
        print("📄 GÉNÉRATION DU RAPPORT DE DIAGNOSTIC")
        print("="*80)

        report_path = self.output_dir / f'rapport_diagnostic_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt'

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("RAPPORT DE DIAGNOSTIC - DISTRIBUTION SHIFT\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*80 + "\n\n")

            f.write(f"Données de référence (train): {self.reference_file}\n")
            f.write(f"Nouvelles données (inférence): {self.new_file}\n\n")

            f.write(f"Nombre de lignes:\n")
            f.write(f"  Référence: {len(self.df_ref)}\n")
            f.write(f"  Nouvelles: {len(self.df_new)}\n\n")

            # Résumé numériques
            f.write("="*80 + "\n")
            f.write("1. VARIABLES NUMÉRIQUES - SHIFT DÉTECTÉ\n")
            f.write("="*80 + "\n\n")

            if numeric_results is not None and len(numeric_results) > 0:
                # Identifier les shifts importants
                major_shifts = numeric_results[
                    (abs(numeric_results['Mean_Diff_%']) > 20) |
                    (abs(numeric_results['Median_Diff_%']) > 20)
                ]

                if len(major_shifts) > 0:
                    f.write("🚨 ALERTES: Variables avec shift > 20%:\n\n")
                    for _, row in major_shifts.iterrows():
                        f.write(f"  {row['Variable']}:\n")
                        f.write(f"    Mean shift: {row['Mean_Diff_%']:+.1f}%\n")
                        f.write(f"    Median shift: {row['Median_Diff_%']:+.1f}%\n")
                        f.write(f"    Référence Mean: {row['Ref_Mean']:,.2f}\n")
                        f.write(f"    Nouvelles Mean: {row['New_Mean']:,.2f}\n\n")
                else:
                    f.write("✅ Pas de shift majeur détecté (< 20%)\n\n")

                f.write("\nDétails complets:\n")
                f.write(numeric_results.to_string())
                f.write("\n\n")

            # Résumé catégorielles
            f.write("="*80 + "\n")
            f.write("2. VARIABLES CATÉGORIELLES - NOUVELLES VALEURS\n")
            f.write("="*80 + "\n\n")

            if categorical_results is not None and len(categorical_results) > 0:
                # Identifier les problèmes
                low_coverage = categorical_results[categorical_results['Coverage_%'] < 80]

                if len(low_coverage) > 0:
                    f.write("🚨 ALERTES: Variables avec faible couverture (< 80%):\n\n")
                    for _, row in low_coverage.iterrows():
                        f.write(f"  {row['Variable']}:\n")
                        f.write(f"    Couverture: {row['Coverage_%']:.1f}%\n")
                        f.write(f"    Nouvelles valeurs absentes du train: {row['Only_in_New']}\n\n")
                else:
                    f.write("✅ Bonne couverture catégorielle (> 80%)\n\n")

                f.write("\nDétails complets:\n")
                f.write(categorical_results.to_string())
                f.write("\n\n")

            # Résumé features engineered
            f.write("="*80 + "\n")
            f.write("3. FEATURES ENGINEERED - TAUX DE FONDÉE\n")
            f.write("="*80 + "\n\n")

            if engineered_results is not None and len(engineered_results) > 0:
                # Identifier les problèmes
                low_coverage = engineered_results[engineered_results['Coverage_%'] < 80]

                if len(low_coverage) > 0:
                    f.write("🚨 ALERTES: Features avec stats manquantes:\n\n")
                    for _, row in low_coverage.iterrows():
                        f.write(f"  {row['Feature']}:\n")
                        f.write(f"    Couverture: {row['Coverage_%']:.1f}%\n")
                        f.write(f"    Cas sans stats: {row['Missing_Cases']} ({row['Missing_%']:.1f}%)\n")
                        f.write(f"    → Ces cas utilisent le taux global (fallback)\n\n")
                else:
                    f.write("✅ Bonne couverture des stats (> 80%)\n\n")

                f.write("\nDétails complets:\n")
                f.write(engineered_results.to_string())
                f.write("\n\n")

            # Recommandations
            f.write("="*80 + "\n")
            f.write("4. RECOMMANDATIONS\n")
            f.write("="*80 + "\n\n")

            recommendations = []

            # Basé sur les résultats
            if numeric_results is not None and len(numeric_results) > 0:
                major_shifts = numeric_results[
                    (abs(numeric_results['Mean_Diff_%']) > 20) |
                    (abs(numeric_results['Median_Diff_%']) > 20)
                ]
                if len(major_shifts) > 0:
                    recommendations.append(
                        f"1. SHIFT NUMÉRIQUE MAJEUR détecté sur {len(major_shifts)} variable(s):\n"
                        f"   - Les distributions de {', '.join(major_shifts['Variable'].tolist())} ont changé significativement\n"
                        f"   - Le modèle a été entraîné sur des données avec des distributions différentes\n"
                        f"   - Solution: Ré-entraîner le modèle en incluant les données 2023 dans le train\n"
                    )

            if categorical_results is not None and len(categorical_results) > 0:
                new_values_total = categorical_results['Only_in_New'].sum()
                if new_values_total > 0:
                    recommendations.append(
                        f"2. NOUVELLES VALEURS CATÉGORIELLES: {new_values_total} nouvelles valeurs au total\n"
                        f"   - Ces valeurs n'existaient pas dans les données d'entraînement\n"
                        f"   - Le modèle les traite avec des fréquences = 0\n"
                        f"   - Solution: Ré-entraîner en incluant 2023 pour capturer ces nouvelles valeurs\n"
                    )

            if engineered_results is not None and len(engineered_results) > 0:
                missing_stats_total = engineered_results['Missing_Cases'].sum()
                if missing_stats_total > 100:
                    recommendations.append(
                        f"3. TAUX DE FONDÉE MANQUANTS: {missing_stats_total} cas utilisent le fallback\n"
                        f"   - Ces cas n'ont pas de taux de fondée spécifique (< 30 échantillons dans train)\n"
                        f"   - Ils utilisent le taux global, moins précis\n"
                        f"   - Impact: Prédictions moins fiables pour ces cas\n"
                        f"   - Solution: Ré-entraîner avec 2023 pour enrichir les statistiques\n"
                    )

            if not recommendations:
                recommendations.append("✅ Pas de problème majeur détecté. Les distributions sont similaires.")

            for rec in recommendations:
                f.write(rec + "\n")

            f.write("\n" + "="*80 + "\n")
            f.write("CONCLUSION:\n")
            f.write("="*80 + "\n\n")

            if recommendations and len(recommendations) > 1:
                f.write("Les données de 2023 sont SIGNIFICATIVEMENT DIFFÉRENTES des données 2024/2025.\n")
                f.write("Cela explique les probabilités faibles observées lors de l'inférence.\n\n")
                f.write("SOLUTION RECOMMANDÉE:\n")
                f.write("  1. Ré-entraîner le modèle en incluant 2023 dans les données d'entraînement\n")
                f.write("  2. Ou utiliser uniquement 2023 comme train si vous prédisez sur 2023\n")
                f.write("  3. Recalculer les taux de fondée sur la période appropriée (2023 ou 2023+2024)\n\n")
            else:
                f.write("Les distributions semblent similaires. Le problème peut venir d'ailleurs.\n\n")

        print(f"✅ Rapport sauvegardé: {report_path}")

        return report_path

    def run(self):
        """Exécuter le diagnostic complet"""
        self.load_data()

        numeric_results = self.analyze_numeric_distributions()
        categorical_results = self.analyze_categorical_distributions()
        engineered_results = self.analyze_engineered_features()

        report_path = self.generate_summary_report(numeric_results, categorical_results, engineered_results)

        print("\n" + "="*80)
        print("✅ DIAGNOSTIC TERMINÉ")
        print("="*80)
        print(f"\n📂 Résultats dans: {self.output_dir}")
        print(f"📄 Rapport: {report_path}")
        print(f"📊 Graphiques: 01_numeric_distributions.png")


def main():
    parser = argparse.ArgumentParser(description='Diagnostic de distribution shift')
    parser.add_argument('--reference_file', type=str, required=True,
                       help='Fichier de référence (données d\'entraînement, ex: 2024)')
    parser.add_argument('--new_file', type=str, required=True,
                       help='Nouvelles données (inférence, ex: 2023)')

    args = parser.parse_args()

    diagnostic = DistributionShiftDiagnostic(args.reference_file, args.new_file)
    diagnostic.run()


if __name__ == '__main__':
    main()
