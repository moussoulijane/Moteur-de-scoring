"""
GÉNÉRATION DES GRAPHIQUES DE MONITORING DES RÈGLES MÉTIER
Analyse l'impact des 2 règles métier sur les décisions

Usage:
    python ml_pipeline_v2/generate_monitoring_regles.py \
        --data_2023 predictions_2023_avec_regles.xlsx \
        --data_2025 predictions_2025_avec_regles.xlsx
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

# Configuration
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (18, 12)

PRIX_UNITAIRE = 169  # DH


class MonitoringReglesGenerator:
    """Générateur des graphiques de monitoring des règles métier"""

    def __init__(self, data_2023=None, data_2025=None):
        self.data_2023 = data_2023
        self.data_2025 = data_2025
        self.output_dir = Path('outputs/presentation_final')
        self.output_dir.mkdir(parents=True, exist_ok=True)

        print("\n" + "="*80)
        print("📊 GÉNÉRATEUR DE MONITORING DES RÈGLES MÉTIER")
        print("="*80)

    def clean_numeric_column(self, df, col):
        """Nettoyer colonne numérique"""
        import re

        def clean_value(val):
            if pd.isna(val):
                return np.nan
            if isinstance(val, (int, float)):
                return float(val)

            val_str = str(val).strip().upper()
            val_str = re.sub(r'(MAD|DH|DHs?|EUR|€|\$)', '', val_str, flags=re.IGNORECASE)
            val_str = val_str.strip()

            if not val_str:
                return np.nan

            val_str = val_str.replace(' ', '')

            if ',' in val_str and '.' in val_str:
                comma_pos = val_str.rfind(',')
                dot_pos = val_str.rfind('.')
                if comma_pos > dot_pos:
                    val_str = val_str.replace('.', '').replace(',', '.')
                else:
                    val_str = val_str.replace(',', '')
            elif ',' in val_str:
                parts = val_str.split(',')
                if len(parts[-1]) == 2:
                    val_str = val_str.replace(',', '.')
                else:
                    val_str = val_str.replace(',', '')

            try:
                return float(val_str)
            except:
                return np.nan

        return df[col].apply(clean_value)

    def load_data(self):
        """Charger les données"""
        print("\n📂 Chargement des données...")

        self.df_2023 = None
        self.df_2025 = None

        if self.data_2023:
            self.df_2023 = pd.read_excel(self.data_2023)
            print(f"✅ 2023: {len(self.df_2023)} réclamations")

        if self.data_2025:
            self.df_2025 = pd.read_excel(self.data_2025)
            print(f"✅ 2025: {len(self.df_2025)} réclamations")

        # Nettoyer colonnes numériques
        print("\n🔄 Nettoyage des colonnes numériques...")
        numeric_cols = ['Montant demandé']

        for df, year in [(self.df_2023, 2023), (self.df_2025, 2025)]:
            if df is not None:
                for col in numeric_cols:
                    if col in df.columns:
                        df[col] = self.clean_numeric_column(df, col)
                print(f"   ✅ {year}: colonnes nettoyées")

    def plot_resultats_et_monitoring(self):
        """Graphique combiné: Résultats + Monitoring"""
        print("\n📊 Génération du graphique résultats et monitoring...")

        fig = plt.figure(figsize=(20, 14))
        fig.suptitle('RÉSULTATS ET MONITORING DES RÈGLES MÉTIER',
                     fontsize=20, fontweight='bold', y=0.98)

        # === SECTION 1: MÉTRIQUES 2023 ET 2025 ===
        ax1 = plt.subplot(3, 3, 1)
        ax1.set_title('Métriques de Performance', fontweight='bold', fontsize=13)

        years_data = []
        for df, year in [(self.df_2023, 2023), (self.df_2025, 2025)]:
            if df is not None and 'Decision_Modele' in df.columns and 'Fondée' in df.columns:
                # Calculer métriques
                df_temp = df.copy()
                df_temp['Fondee_bool'] = df_temp['Fondée'].apply(
                    lambda x: 1 if x in ['Oui', 1, True] else 0
                )
                df_temp['Prediction_bool'] = df_temp['Decision_Modele'].apply(
                    lambda x: 1 if x == 'Validation Auto' else 0
                )

                vp = ((df_temp['Fondee_bool'] == 1) & (df_temp['Prediction_bool'] == 1)).sum()
                vn = ((df_temp['Fondee_bool'] == 0) & (df_temp['Prediction_bool'] == 0)).sum()
                fp = ((df_temp['Fondee_bool'] == 0) & (df_temp['Prediction_bool'] == 1)).sum()
                fn = ((df_temp['Fondee_bool'] == 1) & (df_temp['Prediction_bool'] == 0)).sum()

                precision = vp / (vp + fp) if (vp + fp) > 0 else 0
                recall = vp / (vp + fn) if (vp + fn) > 0 else 0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

                years_data.append({
                    'year': year,
                    'precision': precision * 100,
                    'recall': recall * 100,
                    'f1': f1 * 100
                })

        if years_data:
            x = np.arange(len(years_data))
            width = 0.25

            metrics = ['precision', 'recall', 'f1']
            labels = ['Précision', 'Rappel', 'F1-Score']
            colors = ['#3498DB', '#E67E22', '#2ECC71']

            for i, (metric, label, color) in enumerate(zip(metrics, labels, colors)):
                values = [d[metric] for d in years_data]
                bars = ax1.bar(x + i*width, values, width, label=label,
                              color=color, alpha=0.8, edgecolor='black', linewidth=1)

                for bar, val in zip(bars, values):
                    height = bar.get_height()
                    ax1.text(bar.get_x() + bar.get_width()/2., height,
                            f'{val:.1f}%', ha='center', va='bottom',
                            fontsize=9, fontweight='bold')

            ax1.set_ylabel('Score (%)', fontweight='bold', fontsize=11)
            ax1.set_xticks(x + width)
            ax1.set_xticklabels([str(d['year']) for d in years_data], fontsize=11)
            ax1.legend(fontsize=10)
            ax1.grid(True, alpha=0.3, axis='y')
            ax1.set_ylim(0, 110)

        # === SECTION 2: GAIN NET PAR ANNÉE ===
        ax2 = plt.subplot(3, 3, 2)
        ax2.set_title('GAIN NET par Année', fontweight='bold', fontsize=13)

        gains_data = []
        for df, year in [(self.df_2023, 2023), (self.df_2025, 2025)]:
            if df is not None and 'Decision_Modele' in df.columns:
                n_rejet = (df['Decision_Modele'] == 'Rejet Auto').sum()
                n_validation = (df['Decision_Modele'] == 'Validation Auto').sum()
                n_auto = n_rejet + n_validation

                gain_brut = n_auto * PRIX_UNITAIRE
                perte_fp = 0
                perte_fn = 0

                if 'Fondée' in df.columns and 'Montant demandé' in df.columns:
                    df_temp = df.copy()
                    df_temp['Fondee_bool'] = df_temp['Fondée'].apply(
                        lambda x: 1 if x in ['Oui', 1, True] else 0
                    )
                    df_temp['Prediction_bool'] = df_temp['Decision_Modele'].apply(
                        lambda x: 1 if x == 'Validation Auto' else 0
                    )

                    mask_auto = (df_temp['Decision_Modele'] == 'Rejet Auto') | (df_temp['Decision_Modele'] == 'Validation Auto')
                    df_auto = df_temp[mask_auto]

                    fp_mask = (df_auto['Fondee_bool'] == 0) & (df_auto['Prediction_bool'] == 1)
                    fn_mask = (df_auto['Fondee_bool'] == 1) & (df_auto['Prediction_bool'] == 0)

                    montants_auto = df_auto['Montant demandé'].values
                    montants_clean = np.nan_to_num(montants_auto, nan=0.0)
                    montants_clean = np.clip(montants_clean, 0,
                                            np.percentile(montants_clean[montants_clean > 0], 99)
                                            if (montants_clean > 0).any() else 0)

                    perte_fp = montants_clean[fp_mask.values].sum()
                    perte_fn = 2 * montants_clean[fn_mask.values].sum()

                gain_net = gain_brut - perte_fp - perte_fn

                gains_data.append({
                    'year': year,
                    'gain_brut': gain_brut / 1e6,
                    'perte_fp': perte_fp / 1e6,
                    'perte_fn': perte_fn / 1e6,
                    'gain_net': gain_net / 1e6
                })

        if gains_data:
            years = [d['year'] for d in gains_data]
            gain_brut = [d['gain_brut'] for d in gains_data]
            gain_net = [d['gain_net'] for d in gains_data]

            x = np.arange(len(years))
            width = 0.35

            bars1 = ax2.bar(x - width/2, gain_brut, width, label='Gain Brut',
                           color='#2ECC71', alpha=0.8, edgecolor='black', linewidth=2)
            bars2 = ax2.bar(x + width/2, gain_net, width, label='Gain NET',
                           color='#27AE60', alpha=0.8, edgecolor='black', linewidth=2)

            for bars in [bars1, bars2]:
                for bar in bars:
                    height = bar.get_height()
                    ax2.text(bar.get_x() + bar.get_width()/2., height,
                            f'{height:.2f}M', ha='center', va='bottom',
                            fontsize=10, fontweight='bold')

            ax2.set_ylabel('Millions DH', fontweight='bold', fontsize=11)
            ax2.set_xticks(x)
            ax2.set_xticklabels(years, fontsize=11)
            ax2.legend(fontsize=10)
            ax2.grid(True, alpha=0.3, axis='y')

        # === SECTION 3: TAUX AUTOMATISATION ===
        ax3 = plt.subplot(3, 3, 3)
        ax3.set_title('Taux d\'Automatisation', fontweight='bold', fontsize=13)

        auto_data = []
        for df, year in [(self.df_2023, 2023), (self.df_2025, 2025)]:
            if df is not None and 'Decision_Modele' in df.columns:
                n_total = len(df)
                n_rejet = (df['Decision_Modele'] == 'Rejet Auto').sum()
                n_audit = (df['Decision_Modele'] == 'Audit Humain').sum()
                n_validation = (df['Decision_Modele'] == 'Validation Auto').sum()
                taux_auto = 100 * (n_rejet + n_validation) / n_total

                auto_data.append({
                    'year': year,
                    'taux_auto': taux_auto
                })

        if auto_data:
            years = [d['year'] for d in auto_data]
            taux = [d['taux_auto'] for d in auto_data]

            bars = ax3.bar(years, taux, color='#3498DB', alpha=0.8,
                          edgecolor='black', linewidth=2, width=0.6)

            for bar, val in zip(bars, taux):
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height,
                        f'{val:.1f}%', ha='center', va='bottom',
                        fontsize=11, fontweight='bold')

            ax3.set_ylabel('Pourcentage (%)', fontweight='bold', fontsize=11)
            ax3.set_xlabel('Année', fontweight='bold', fontsize=11)
            ax3.grid(True, alpha=0.3, axis='y')
            ax3.set_ylim(0, 110)

        # === SECTION 4-6: IMPACT DES RÈGLES MÉTIER ===
        for idx, (df, year) in enumerate([(self.df_2023, 2023), (self.df_2025, 2025)]):
            if df is None:
                continue

            ax_impact = plt.subplot(3, 3, 4 + idx*3)
            ax_impact.set_title(f'Impact Règles Métier - {year}', fontweight='bold', fontsize=13)

            # Vérifier si on a la colonne Raison_Audit
            if 'Raison_Audit' in df.columns and 'Decision_Modele' in df.columns:
                # Cas convertis par les règles
                mask_rule1 = df['Raison_Audit'].str.contains('Règle #1', na=False)
                mask_rule2 = df['Raison_Audit'].str.contains('Règle #2', na=False)

                n_rule1 = mask_rule1.sum()
                n_rule2 = mask_rule2.sum()
                n_no_rule = (df['Decision_Modele'] == 'Validation Auto').sum()

                labels = ['Validation\nAuto\n(directe)', 'Converti\nRègle #1', 'Converti\nRègle #2']
                values = [n_no_rule, n_rule1, n_rule2]
                colors = ['#2ECC71', '#F39C12', '#E74C3C']

                bars = ax_impact.bar(labels, values, color=colors, alpha=0.8,
                                    edgecolor='black', linewidth=2)

                for bar, val in zip(bars, values):
                    height = bar.get_height()
                    ax_impact.text(bar.get_x() + bar.get_width()/2., height,
                                  f'{int(val):,}', ha='center', va='bottom',
                                  fontsize=10, fontweight='bold')

                ax_impact.set_ylabel('Nombre de cas', fontweight='bold', fontsize=11)
                ax_impact.grid(True, alpha=0.3, axis='y')

            # Graphique des montants concernés
            ax_montant = plt.subplot(3, 3, 5 + idx*3)
            ax_montant.set_title(f'Montants Concernés - {year}', fontweight='bold', fontsize=13)

            if 'Raison_Audit' in df.columns and 'Montant demandé' in df.columns:
                mask_rule1 = df['Raison_Audit'].str.contains('Règle #1', na=False)
                mask_rule2 = df['Raison_Audit'].str.contains('Règle #2', na=False)

                mt_rule1 = df[mask_rule1]['Montant demandé'].sum() / 1e6
                mt_rule2 = df[mask_rule2]['Montant demandé'].sum() / 1e6

                labels = ['Règle #1\n(>1 validation/an)', 'Règle #2\n(Montant > PNB)']
                values = [mt_rule1, mt_rule2]
                colors = ['#F39C12', '#E74C3C']

                bars = ax_montant.bar(labels, values, color=colors, alpha=0.8,
                                     edgecolor='black', linewidth=2)

                for bar, val in zip(bars, values):
                    height = bar.get_height()
                    ax_montant.text(bar.get_x() + bar.get_width()/2., height,
                                   f'{val:.2f}M', ha='center', va='bottom',
                                   fontsize=10, fontweight='bold')

                ax_montant.set_ylabel('Montant Total (M DH)', fontweight='bold', fontsize=11)
                ax_montant.grid(True, alpha=0.3, axis='y')

            # Texte récapitulatif
            ax_recap = plt.subplot(3, 3, 6 + idx*3)
            ax_recap.axis('off')

            if 'Raison_Audit' in df.columns and 'Decision_Modele' in df.columns:
                n_total = len(df)
                n_validation_directe = (df['Decision_Modele'] == 'Validation Auto').sum()
                mask_rule1 = df['Raison_Audit'].str.contains('Règle #1', na=False)
                mask_rule2 = df['Raison_Audit'].str.contains('Règle #2', na=False)
                n_rule1 = mask_rule1.sum()
                n_rule2 = mask_rule2.sum()

                if 'Montant demandé' in df.columns:
                    mt_rule1 = df[mask_rule1]['Montant demandé'].sum() / 1e6
                    mt_rule2 = df[mask_rule2]['Montant demandé'].sum() / 1e6

                    recap_text = f"""
📊 RÉCAPITULATIF {year}

DÉCISIONS:
  • Validation Auto directe: {n_validation_directe:,}
  • Converti par Règle #1:   {n_rule1:,}
  • Converti par Règle #2:   {n_rule2:,}

MONTANTS PROTÉGÉS:
  • Règle #1: {mt_rule1:.2f}M DH
  • Règle #2: {mt_rule2:.2f}M DH
  • TOTAL:    {mt_rule1 + mt_rule2:.2f}M DH

TAUX DE CONVERSION:
  • {100*n_rule1/n_total:.2f}% par Règle #1
  • {100*n_rule2/n_total:.2f}% par Règle #2
                    """
                else:
                    recap_text = f"""
📊 RÉCAPITULATIF {year}

DÉCISIONS:
  • Validation Auto: {n_validation_directe:,}
  • Converti Règle #1: {n_rule1:,}
  • Converti Règle #2: {n_rule2:,}
                    """

                ax_recap.text(0.05, 0.95, recap_text, transform=ax_recap.transAxes,
                            fontsize=10, verticalalignment='top', family='monospace',
                            bbox=dict(boxstyle='round', facecolor='#E8F8F5', alpha=0.9,
                                    edgecolor='#16A085', linewidth=2))

        plt.tight_layout()
        output_path = self.output_dir / 'P3_resultats_monitoring.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ Sauvegardé: {output_path}")
        plt.close()

    def run(self):
        """Exécuter la génération"""
        self.load_data()
        self.plot_resultats_et_monitoring()

        print("\n" + "="*80)
        print("✅ GÉNÉRATION TERMINÉE")
        print("="*80)
        print(f"\n📂 Fichiers dans: {self.output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Générer monitoring des règles')
    parser.add_argument('--data_2023', type=str, help='Fichier Excel 2023 avec décisions')
    parser.add_argument('--data_2025', type=str, help='Fichier Excel 2025 avec décisions')

    args = parser.parse_args()

    if not args.data_2023 and not args.data_2025:
        print("❌ ERREUR: Fournissez au moins un fichier avec données scorées")
        parser.print_help()
        return

    generator = MonitoringReglesGenerator(
        data_2023=args.data_2023,
        data_2025=args.data_2025
    )
    generator.run()


if __name__ == '__main__':
    main()
