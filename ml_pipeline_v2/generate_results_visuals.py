"""
GÉNÉRATION DES VISUALISATIONS DES RÉSULTATS ET GAINS
Crée des graphiques détaillés pour analyser les résultats du modèle sur 2025/2023

Usage:
    python ml_pipeline_v2/generate_results_visuals.py --data_2025 predictions_2025.xlsx --data_2023 reclamations_2023.xlsx
"""
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
plt.rcParams['figure.figsize'] = (16, 10)


class ResultsVisualizer:
    """Générateur de visualisations des résultats"""

    def __init__(self, data_2025, data_2023=None):
        self.data_2025 = data_2025
        self.data_2023 = data_2023
        self.output_dir = Path('outputs/results_analysis')
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Paramètres de gain
        self.cout_traitement_manuel = 50  # DH par dossier
        self.temps_traitement_manuel = 30  # minutes
        self.temps_traitement_auto = 1  # minute
        self.heures_annuelles_fte = 1600  # heures/an

        print("\n" + "="*80)
        print("📊 GÉNÉRATEUR DE VISUALISATIONS DES RÉSULTATS")
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
        """Charger et nettoyer les données"""
        print("\n📂 Chargement des données...")

        self.df_2025 = pd.read_excel(self.data_2025)
        print(f"✅ 2025: {len(self.df_2025)} réclamations")

        if self.data_2023:
            self.df_2023 = pd.read_excel(self.data_2023)
            print(f"✅ 2023: {len(self.df_2023)} réclamations")
        else:
            self.df_2023 = None

        # Nettoyer colonnes numériques
        print("\n🔄 Nettoyage des colonnes numériques...")
        numeric_cols = ['Montant demandé', 'Délai estimé', 'anciennete_annees',
                       'PNB analytique (vision commerciale) cumulé']

        for df, year in [(self.df_2025, 2025), (self.df_2023, 2023)]:
            if df is not None:
                for col in numeric_cols:
                    if col in df.columns:
                        df[col] = self.clean_numeric_column(df, col)
                print(f"   ✅ {year}: colonnes nettoyées")

    def plot_decisions_distribution_2025(self):
        """1. Distribution des décisions sur 2025"""
        print("\n📊 Graphique 1: Distribution des décisions 2025...")

        if 'Decision_Modele' not in self.df_2025.columns:
            print("⚠️  Colonne Decision_Modele manquante - Graphique ignoré")
            return

        fig = plt.figure(figsize=(18, 10))
        fig.suptitle('ANALYSE DES DÉCISIONS DU MODÈLE SUR 2025', fontsize=18, fontweight='bold', y=0.98)

        # Compter les décisions
        n_total = len(self.df_2025)
        n_rejet = (self.df_2025['Decision_Modele'] == 'Rejet Auto').sum()
        n_audit = (self.df_2025['Decision_Modele'] == 'Audit Humain').sum()
        n_validation = (self.df_2025['Decision_Modele'] == 'Validation Auto').sum()

        # 1. Pie chart principal
        ax1 = plt.subplot(2, 3, 1)
        sizes = [n_rejet, n_audit, n_validation]
        labels = ['Rejet Auto', 'Audit Humain', 'Validation Auto']
        colors = ['#E74C3C', '#F39C12', '#2ECC71']
        explode = (0.05, 0.05, 0.1)

        wedges, texts, autotexts = ax1.pie(sizes, explode=explode, labels=labels,
                                            autopct='%1.1f%%', colors=colors,
                                            shadow=True, startangle=90,
                                            textprops={'fontsize': 11, 'weight': 'bold'})

        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontsize(12)

        ax1.set_title('Distribution des Décisions', fontweight='bold', fontsize=13)

        # 2. Barres avec nombres absolus
        ax2 = plt.subplot(2, 3, 2)
        bars = ax2.bar(labels, sizes, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
        ax2.set_ylabel('Nombre de réclamations', fontweight='bold', fontsize=11)
        ax2.set_title('Nombre par Décision', fontweight='bold', fontsize=13)
        ax2.grid(True, alpha=0.3, axis='y')

        for bar, count in zip(bars, sizes):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(count):,}', ha='center', va='bottom', fontweight='bold', fontsize=11)

        # 3. Taux d'automatisation
        ax3 = plt.subplot(2, 3, 3)
        taux_auto = 100 * (n_rejet + n_validation) / n_total
        taux_audit = 100 * n_audit / n_total

        bars = ax3.barh(['Automatisé\n(Rejet + Validation)', 'Audit Humain'],
                       [taux_auto, taux_audit],
                       color=['#2ECC71', '#F39C12'], alpha=0.8, edgecolor='black', linewidth=2)

        ax3.set_xlabel('Pourcentage (%)', fontweight='bold', fontsize=11)
        ax3.set_title('Taux d\'Automatisation', fontweight='bold', fontsize=13)
        ax3.grid(True, alpha=0.3, axis='x')

        for bar in bars:
            width = bar.get_width()
            ax3.text(width + 1, bar.get_y() + bar.get_height()/2.,
                    f'{width:.1f}%', ha='left', va='center', fontweight='bold', fontsize=11)

        # 4. Distribution des probabilités par décision
        ax4 = plt.subplot(2, 3, 4)
        if 'Probabilite_Fondee' in self.df_2025.columns:
            for decision, color in [('Rejet Auto', '#E74C3C'),
                                    ('Audit Humain', '#F39C12'),
                                    ('Validation Auto', '#2ECC71')]:
                data = self.df_2025[self.df_2025['Decision_Modele'] == decision]['Probabilite_Fondee']
                if len(data) > 0:
                    ax4.hist(data, bins=30, alpha=0.6, label=decision, color=color)

            ax4.set_xlabel('Probabilité Fondée', fontweight='bold', fontsize=11)
            ax4.set_ylabel('Fréquence', fontweight='bold', fontsize=11)
            ax4.set_title('Distribution des Probabilités', fontweight='bold', fontsize=13)
            ax4.legend()
            ax4.grid(True, alpha=0.3)

        # 5. Montants par décision
        ax5 = plt.subplot(2, 3, 5)
        if 'Montant demandé' in self.df_2025.columns:
            montants = []
            for decision in labels:
                df_dec = self.df_2025[self.df_2025['Decision_Modele'] == decision]
                mt = df_dec['Montant demandé'][df_dec['Montant demandé'] > 0].sum() / 1e6
                montants.append(mt)

            bars = ax5.bar(labels, montants, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
            ax5.set_ylabel('Montant Total (Millions DH)', fontweight='bold', fontsize=11)
            ax5.set_title('Montant par Type de Décision', fontweight='bold', fontsize=13)
            ax5.grid(True, alpha=0.3, axis='y')
            ax5.tick_params(axis='x', rotation=15)

            for bar, mt in zip(bars, montants):
                height = bar.get_height()
                ax5.text(bar.get_x() + bar.get_width()/2., height,
                        f'{mt:.1f}M', ha='center', va='bottom', fontweight='bold', fontsize=10)

        # 6. Statistiques récapitulatives
        ax6 = plt.subplot(2, 3, 6)
        ax6.axis('off')

        stats_text = f"""
📊 STATISTIQUES CLÉS 2025

Total réclamations: {n_total:,}

DÉCISIONS:
  • Rejet Auto:      {n_rejet:,} ({100*n_rejet/n_total:.1f}%)
  • Audit Humain:    {n_audit:,} ({100*n_audit/n_total:.1f}%)
  • Validation Auto: {n_validation:,} ({100*n_validation/n_total:.1f}%)

AUTOMATISATION:
  • Taux automatisation: {taux_auto:.1f}%
  • Gain traitement:     {n_rejet + n_validation:,} dossiers

MONTANTS:
  • Montant total:       {self.df_2025['Montant demandé'].sum()/1e6:.1f}M DH
  • Montant validé:      {montants[2]:.1f}M DH
  • Montant rejeté:      {montants[0]:.1f}M DH
        """

        ax6.text(0.1, 0.9, stats_text, transform=ax6.transAxes,
                fontsize=11, verticalalignment='top', family='monospace',
                bbox=dict(boxstyle='round', facecolor='#ECF0F1', alpha=0.8,
                         edgecolor='black', linewidth=2))

        plt.tight_layout()
        output_path = self.output_dir / 'R1_decisions_distribution_2025.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ Sauvegardé: {output_path}")
        plt.close()

    def plot_comparison_2025_vs_2023(self):
        """2. Comparaison 2025 vs 2023"""
        print("\n📊 Graphique 2: Comparaison 2025 vs 2023...")

        if self.df_2023 is None:
            print("⚠️  Données 2023 manquantes - Graphique ignoré")
            return

        if 'Decision_Modele' not in self.df_2025.columns:
            print("⚠️  Pas de décisions dans 2025 - Graphique ignoré")
            return

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('COMPARAISON 2023 vs 2025 (AVANT/APRÈS MODÈLE)',
                     fontsize=18, fontweight='bold', y=0.98)

        # 1. Volume comparaison
        volumes = [len(self.df_2023), len(self.df_2025)]
        years = ['2023\n(Sans modèle)', '2025\n(Avec modèle)']
        colors = ['#95A5A6', '#3498DB']

        bars = ax1.bar(years, volumes, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
        ax1.set_ylabel('Nombre de réclamations', fontweight='bold', fontsize=12)
        ax1.set_title('Volume de Réclamations', fontweight='bold', fontsize=14)
        ax1.grid(True, alpha=0.3, axis='y')

        for bar, vol in zip(bars, volumes):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(vol):,}', ha='center', va='bottom', fontweight='bold', fontsize=12)

        # 2. Taux de fondée comparaison
        if 'Fondée' in self.df_2023.columns:
            n_fondee_2023 = (self.df_2023['Fondée'] == 'Oui').sum() if 'Oui' in self.df_2023['Fondée'].values else (self.df_2023['Fondée'] == 1).sum()
            taux_fondee_2023 = 100 * n_fondee_2023 / len(self.df_2023)

            n_validation_2025 = (self.df_2025['Decision_Modele'] == 'Validation Auto').sum()
            taux_validation_2025 = 100 * n_validation_2025 / len(self.df_2025)

            categories = ['Taux fondée\n2023', 'Taux validation\n2025']
            values = [taux_fondee_2023, taux_validation_2025]

            bars = ax2.bar(categories, values, color=['#95A5A6', '#2ECC71'],
                          alpha=0.8, edgecolor='black', linewidth=2)
            ax2.set_ylabel('Pourcentage (%)', fontweight='bold', fontsize=12)
            ax2.set_title('Taux de Validation/Fondée', fontweight='bold', fontsize=14)
            ax2.grid(True, alpha=0.3, axis='y')
            ax2.set_ylim(0, max(values) * 1.2)

            for bar, val in zip(bars, values):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                        f'{val:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=12)

        # 3. Montants comparaison
        if 'Montant demandé' in self.df_2023.columns and 'Montant demandé' in self.df_2025.columns:
            mt_2023 = self.df_2023['Montant demandé'][self.df_2023['Montant demandé'] > 0].sum() / 1e6
            mt_2025 = self.df_2025['Montant demandé'][self.df_2025['Montant demandé'] > 0].sum() / 1e6

            bars = ax3.bar(years, [mt_2023, mt_2025], color=colors,
                          alpha=0.8, edgecolor='black', linewidth=2)
            ax3.set_ylabel('Montant Total (Millions DH)', fontweight='bold', fontsize=12)
            ax3.set_title('Montant Total des Réclamations', fontweight='bold', fontsize=14)
            ax3.grid(True, alpha=0.3, axis='y')

            for bar, mt in zip(bars, [mt_2023, mt_2025]):
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height,
                        f'{mt:.1f}M', ha='center', va='bottom', fontweight='bold', fontsize=12)

        # 4. Tableau comparatif
        ax4.axis('off')

        n_2025 = len(self.df_2025)
        n_rejet = (self.df_2025['Decision_Modele'] == 'Rejet Auto').sum()
        n_audit = (self.df_2025['Decision_Modele'] == 'Audit Humain').sum()
        n_validation = (self.df_2025['Decision_Modele'] == 'Validation Auto').sum()
        taux_auto = 100 * (n_rejet + n_validation) / n_2025

        comparison_text = f"""
📊 COMPARAISON DÉTAILLÉE

2023 (Sans modèle):
  • Volume:      {len(self.df_2023):,}
  • Taux fondée: {taux_fondee_2023:.1f}%
  • Traitement:  100% manuel
  • Temps:       {len(self.df_2023) * self.temps_traitement_manuel / 60:,.0f} heures

2025 (Avec modèle):
  • Volume:           {n_2025:,}
  • Validation auto:  {100*n_validation/n_2025:.1f}%
  • Rejet auto:       {100*n_rejet/n_2025:.1f}%
  • Audit humain:     {100*n_audit/n_2025:.1f}%
  • Automatisation:   {taux_auto:.1f}%
  • Temps économisé:  {(n_rejet + n_validation) * (self.temps_traitement_manuel - self.temps_traitement_auto) / 60:,.0f}h

GAIN:
  • {100 - 100*n_audit/n_2025:.1f}% de dossiers automatisés
  • Réduction temps de {100 * (n_rejet + n_validation) * (self.temps_traitement_manuel - self.temps_traitement_auto) / (n_2025 * self.temps_traitement_manuel):.1f}%
        """

        ax4.text(0.05, 0.95, comparison_text, transform=ax4.transAxes,
                fontsize=10, verticalalignment='top', family='monospace',
                bbox=dict(boxstyle='round', facecolor='#E8F8F5', alpha=0.9,
                         edgecolor='#16A085', linewidth=2))

        plt.tight_layout()
        output_path = self.output_dir / 'R2_comparison_2025_vs_2023.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ Sauvegardé: {output_path}")
        plt.close()

    def plot_gain_calculation(self):
        """3. Calcul détaillé des gains"""
        print("\n📊 Graphique 3: Calcul des gains...")

        if 'Decision_Modele' not in self.df_2025.columns:
            print("⚠️  Pas de décisions - Graphique ignoré")
            return

        fig = plt.figure(figsize=(18, 12))
        fig.suptitle('CALCUL DÉTAILLÉ DES GAINS - 2025', fontsize=18, fontweight='bold', y=0.98)

        n_total = len(self.df_2025)
        n_rejet = (self.df_2025['Decision_Modele'] == 'Rejet Auto').sum()
        n_audit = (self.df_2025['Decision_Modele'] == 'Audit Humain').sum()
        n_validation = (self.df_2025['Decision_Modele'] == 'Validation Auto').sum()
        n_auto = n_rejet + n_validation

        # Calculs
        gain_financier = n_auto * self.cout_traitement_manuel
        temps_economise_min = n_auto * (self.temps_traitement_manuel - self.temps_traitement_auto)
        temps_economise_h = temps_economise_min / 60
        etp_libere = temps_economise_h / self.heures_annuelles_fte

        temps_avant = n_total * self.temps_traitement_manuel / 60
        temps_apres = n_audit * self.temps_traitement_manuel / 60 + n_auto * self.temps_traitement_auto / 60
        reduction_temps_pct = 100 * (temps_avant - temps_apres) / temps_avant

        # 1. Gain financier
        ax1 = plt.subplot(2, 3, 1)
        categories = ['Coût\navant', 'Coût\naprès', 'GAIN']
        cout_avant = n_total * self.cout_traitement_manuel / 1e6
        cout_apres = n_audit * self.cout_traitement_manuel / 1e6
        gain_m = gain_financier / 1e6

        bars = ax1.bar(categories, [cout_avant, cout_apres, gain_m],
                      color=['#E74C3C', '#F39C12', '#2ECC71'],
                      alpha=0.8, edgecolor='black', linewidth=2)

        ax1.set_ylabel('Millions DH', fontweight='bold', fontsize=12)
        ax1.set_title('GAIN FINANCIER', fontweight='bold', fontsize=14)
        ax1.grid(True, alpha=0.3, axis='y')

        for bar, val in zip(bars, [cout_avant, cout_apres, gain_m]):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.2f}M', ha='center', va='bottom', fontweight='bold', fontsize=11)

        # 2. Temps économisé
        ax2 = plt.subplot(2, 3, 2)
        temps_data = ['Temps\navant', 'Temps\naprès', 'ÉCONOMIE']
        temps_values = [temps_avant, temps_apres, temps_economise_h]

        bars = ax2.bar(temps_data, temps_values,
                      color=['#E74C3C', '#F39C12', '#2ECC71'],
                      alpha=0.8, edgecolor='black', linewidth=2)

        ax2.set_ylabel('Heures', fontweight='bold', fontsize=12)
        ax2.set_title('GAIN TEMPS', fontweight='bold', fontsize=14)
        ax2.grid(True, alpha=0.3, axis='y')

        for bar, val in zip(bars, temps_values):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:,.0f}h', ha='center', va='bottom', fontweight='bold', fontsize=11)

        # 3. ETP libérés
        ax3 = plt.subplot(2, 3, 3)
        ax3.axis('off')

        etp_text = f"""
👥 ETP LIBÉRÉS

Temps économisé:
  {temps_economise_h:,.0f} heures

Équivalent ETP:
  {etp_libere:.2f} ETP

(1 ETP = {self.heures_annuelles_fte}h/an)

📊 Capacité libérée pour:
  • Tâches à valeur ajoutée
  • Amélioration continue
  • Innovation
        """

        ax3.text(0.1, 0.9, etp_text, transform=ax3.transAxes,
                fontsize=13, verticalalignment='top', family='monospace',
                bbox=dict(boxstyle='round', facecolor='#D5F4E6', alpha=0.9,
                         edgecolor='#2ECC71', linewidth=3))

        # 4. Schéma explicatif du calcul
        ax4 = plt.subplot(2, 3, 4)
        ax4.axis('off')
        ax4.set_xlim(0, 10)
        ax4.set_ylim(0, 10)

        # Titre
        ax4.text(5, 9, 'SCHÉMA DE CALCUL DU GAIN FINANCIER', ha='center',
                fontsize=12, fontweight='bold')

        # Box 1: Dossiers automatisés
        rect1 = plt.Rectangle((1, 6), 3, 2, facecolor='#3498DB',
                              edgecolor='black', linewidth=2)
        ax4.add_patch(rect1)
        ax4.text(2.5, 7.5, 'Dossiers\nautomatisés', ha='center', va='center',
                fontsize=10, fontweight='bold', color='white')
        ax4.text(2.5, 6.7, f'{n_auto:,}', ha='center', va='center',
                fontsize=11, fontweight='bold', color='white')

        # Flèche
        ax4.arrow(4.2, 7, 1, 0, head_width=0.3, head_length=0.3,
                 fc='black', ec='black', linewidth=2)

        # Box 2: Coût unitaire
        rect2 = plt.Rectangle((5.5, 6), 3, 2, facecolor='#E67E22',
                              edgecolor='black', linewidth=2)
        ax4.add_patch(rect2)
        ax4.text(7, 7.5, 'Coût unitaire\névité', ha='center', va='center',
                fontsize=10, fontweight='bold', color='white')
        ax4.text(7, 6.7, f'{self.cout_traitement_manuel} DH', ha='center', va='center',
                fontsize=11, fontweight='bold', color='white')

        # Flèche vers résultat
        ax4.arrow(2.5, 5.8, 0, -1.5, head_width=0.3, head_length=0.2,
                 fc='black', ec='black', linewidth=2)
        ax4.arrow(7, 5.8, 0, -1.5, head_width=0.3, head_length=0.2,
                 fc='black', ec='black', linewidth=2)

        # Opération
        ax4.text(4.7, 5, '×', ha='center', va='center',
                fontsize=24, fontweight='bold')

        # Box résultat
        rect3 = plt.Rectangle((2, 1.5), 6, 2, facecolor='#2ECC71',
                              edgecolor='black', linewidth=3)
        ax4.add_patch(rect3)
        ax4.text(5, 3, 'GAIN FINANCIER TOTAL', ha='center', va='center',
                fontsize=12, fontweight='bold', color='white')
        ax4.text(5, 2.3, f'{gain_financier:,.0f} DH', ha='center', va='center',
                fontsize=14, fontweight='bold', color='white')
        ax4.text(5, 1.8, f'= {gain_financier/1e6:.2f} Millions DH', ha='center', va='center',
                fontsize=11, fontweight='bold', color='white')

        # 5. Schéma explicatif du temps
        ax5 = plt.subplot(2, 3, 5)
        ax5.axis('off')
        ax5.set_xlim(0, 10)
        ax5.set_ylim(0, 10)

        # Titre
        ax5.text(5, 9, 'SCHÉMA DE CALCUL DU GAIN TEMPS', ha='center',
                fontsize=12, fontweight='bold')

        # Box 1: Temps par dossier
        rect1 = plt.Rectangle((1, 6), 3, 2, facecolor='#3498DB',
                              edgecolor='black', linewidth=2)
        ax5.add_patch(rect1)
        ax5.text(2.5, 7.5, 'Temps manuel', ha='center', va='center',
                fontsize=10, fontweight='bold', color='white')
        ax5.text(2.5, 6.7, f'{self.temps_traitement_manuel} min', ha='center', va='center',
                fontsize=11, fontweight='bold', color='white')

        # Box 2: Temps auto
        rect2 = plt.Rectangle((5.5, 6), 3, 2, facecolor='#E67E22',
                              edgecolor='black', linewidth=2)
        ax5.add_patch(rect2)
        ax5.text(7, 7.5, 'Temps auto', ha='center', va='center',
                fontsize=10, fontweight='bold', color='white')
        ax5.text(7, 6.7, f'{self.temps_traitement_auto} min', ha='center', va='center',
                fontsize=11, fontweight='bold', color='white')

        # Gain par dossier
        ax5.text(4.7, 5, '−', ha='center', va='center',
                fontsize=24, fontweight='bold')

        rect_diff = plt.Rectangle((2.5, 4), 5, 0.8, facecolor='#9B59B6',
                                  edgecolor='black', linewidth=2)
        ax5.add_patch(rect_diff)
        ax5.text(5, 4.4, f'Gain par dossier: {self.temps_traitement_manuel - self.temps_traitement_auto} min',
                ha='center', va='center', fontsize=10, fontweight='bold', color='white')

        # Multiplication
        ax5.arrow(5, 3.8, 0, -0.5, head_width=0.3, head_length=0.2,
                 fc='black', ec='black', linewidth=2)

        ax5.text(5, 3, '×', ha='center', va='center',
                fontsize=20, fontweight='bold')

        ax5.text(5, 2.5, f'{n_auto:,} dossiers automatisés', ha='center', va='center',
                fontsize=10, fontweight='bold')

        # Résultat
        rect_res = plt.Rectangle((2, 1), 6, 0.8, facecolor='#2ECC71',
                                 edgecolor='black', linewidth=3)
        ax5.add_patch(rect_res)
        ax5.text(5, 1.4, f'GAIN: {temps_economise_h:,.0f} heures = {etp_libere:.2f} ETP',
                ha='center', va='center', fontsize=11, fontweight='bold', color='white')

        # 6. ROI et productivité
        ax6 = plt.subplot(2, 3, 6)
        ax6.axis('off')

        roi_text = f"""
💰 RETOUR SUR INVESTISSEMENT

GAINS ANNUELS:
  • Financier:    {gain_financier/1e6:.2f}M DH
  • Temps:        {temps_economise_h:,.0f} heures
  • ETP libérés:  {etp_libere:.2f}

PERFORMANCE:
  • Taux automatisation: {100*(n_rejet+n_validation)/n_total:.1f}%
  • Réduction temps:     {reduction_temps_pct:.1f}%
  • Dossiers/jour gagnés: {n_auto/365:.0f}

IMPACT:
  ✓ Délais réduits
  ✓ Cohérence accrue
  ✓ Capacité libérée
  ✓ Satisfaction client ↗
        """

        ax6.text(0.05, 0.95, roi_text, transform=ax6.transAxes,
                fontsize=11, verticalalignment='top', family='monospace',
                bbox=dict(boxstyle='round', facecolor='#FEF9E7', alpha=0.9,
                         edgecolor='#F39C12', linewidth=2))

        plt.tight_layout()
        output_path = self.output_dir / 'R3_gain_calculation_detailed.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ Sauvegardé: {output_path}")
        plt.close()

    def plot_performance_metrics_2025(self):
        """4. Métriques de performance détaillées sur 2025"""
        print("\n📊 Graphique 4: Métriques de performance 2025...")

        if 'Decision_Modele' not in self.df_2025.columns:
            print("⚠️  Pas de décisions - Graphique ignoré")
            return

        fig = plt.figure(figsize=(18, 10))
        fig.suptitle('MÉTRIQUES DE PERFORMANCE DÉTAILLÉES - 2025',
                     fontsize=18, fontweight='bold', y=0.98)

        # 1. Distribution par famille
        ax1 = plt.subplot(2, 3, 1)
        if 'Famille Produit' in self.df_2025.columns:
            top_families = self.df_2025['Famille Produit'].value_counts().head(8)

            # Calculer taux de validation par famille
            validation_rates = []
            for famille in top_families.index:
                df_fam = self.df_2025[self.df_2025['Famille Produit'] == famille]
                n_val = (df_fam['Decision_Modele'] == 'Validation Auto').sum()
                rate = 100 * n_val / len(df_fam)
                validation_rates.append(rate)

            colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(validation_rates)))
            bars = ax1.barh(range(len(top_families)), validation_rates, color=colors,
                           edgecolor='black', linewidth=1)

            ax1.set_yticks(range(len(top_families)))
            ax1.set_yticklabels(top_families.index, fontsize=9)
            ax1.set_xlabel('% Validation Auto', fontweight='bold', fontsize=11)
            ax1.set_title('Taux Validation par Famille', fontweight='bold', fontsize=13)
            ax1.grid(True, alpha=0.3, axis='x')

            for bar, rate in zip(bars, validation_rates):
                width = bar.get_width()
                ax1.text(width + 1, bar.get_y() + bar.get_height()/2.,
                        f'{rate:.1f}%', ha='left', va='center', fontsize=9)

        # 2. Distribution par segment
        ax2 = plt.subplot(2, 3, 2)
        if 'Segment' in self.df_2025.columns:
            segment_counts = self.df_2025['Segment'].value_counts()

            colors = plt.cm.Set3(range(len(segment_counts)))
            wedges, texts, autotexts = ax2.pie(segment_counts.values,
                                                labels=segment_counts.index,
                                                autopct='%1.1f%%', colors=colors,
                                                startangle=90, textprops={'fontsize': 9})

            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')

            ax2.set_title('Répartition par Segment', fontweight='bold', fontsize=13)

        # 3. Montant moyen par décision
        ax3 = plt.subplot(2, 3, 3)
        if 'Montant demandé' in self.df_2025.columns:
            decisions = ['Rejet Auto', 'Audit Humain', 'Validation Auto']
            moyennes = []
            for dec in decisions:
                df_dec = self.df_2025[self.df_2025['Decision_Modele'] == dec]
                moy = df_dec['Montant demandé'][df_dec['Montant demandé'] > 0].mean()
                moyennes.append(moy)

            colors = ['#E74C3C', '#F39C12', '#2ECC71']
            bars = ax3.bar(range(len(decisions)), moyennes, color=colors,
                          alpha=0.8, edgecolor='black', linewidth=2)

            ax3.set_xticks(range(len(decisions)))
            ax3.set_xticklabels(decisions, rotation=15, fontsize=10)
            ax3.set_ylabel('Montant Moyen (DH)', fontweight='bold', fontsize=11)
            ax3.set_title('Montant Moyen par Décision', fontweight='bold', fontsize=13)
            ax3.grid(True, alpha=0.3, axis='y')

            for bar, moy in zip(bars, moyennes):
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height,
                        f'{moy:,.0f}', ha='center', va='bottom', fontweight='bold', fontsize=9)

        # 4. Distribution délais par décision
        ax4 = plt.subplot(2, 3, 4)
        if 'Délai estimé' in self.df_2025.columns:
            for decision, color in [('Rejet Auto', '#E74C3C'),
                                    ('Validation Auto', '#2ECC71')]:
                df_dec = self.df_2025[self.df_2025['Decision_Modele'] == decision]
                data = df_dec['Délai estimé'][df_dec['Délai estimé'] > 0]
                if len(data) > 0:
                    # Limiter aux percentiles
                    data = data[data <= data.quantile(0.95)]
                    ax4.hist(data, bins=30, alpha=0.6, label=decision, color=color)

            ax4.set_xlabel('Délai estimé (jours)', fontweight='bold', fontsize=11)
            ax4.set_ylabel('Fréquence', fontweight='bold', fontsize=11)
            ax4.set_title('Distribution des Délais', fontweight='bold', fontsize=13)
            ax4.legend()
            ax4.grid(True, alpha=0.3)

        # 5. PNB vs Ancienneté (validations)
        ax5 = plt.subplot(2, 3, 5)
        if ('PNB analytique (vision commerciale) cumulé' in self.df_2025.columns and
            'anciennete_annees' in self.df_2025.columns):
            df_val = self.df_2025[self.df_2025['Decision_Modele'] == 'Validation Auto']
            df_val_clean = df_val[
                (df_val['PNB analytique (vision commerciale) cumulé'] > 0) &
                (df_val['anciennete_annees'] > 0)
            ]

            if len(df_val_clean) > 0:
                ax5.scatter(df_val_clean['anciennete_annees'],
                           df_val_clean['PNB analytique (vision commerciale) cumulé'],
                           alpha=0.5, s=30, color='#2ECC71', edgecolor='black', linewidth=0.5)

                ax5.set_xlabel('Ancienneté (années)', fontweight='bold', fontsize=11)
                ax5.set_ylabel('PNB cumulé (DH)', fontweight='bold', fontsize=11)
                ax5.set_title('PNB vs Ancienneté (Validations)', fontweight='bold', fontsize=13)
                ax5.grid(True, alpha=0.3)

        # 6. Matrice de confusion (si Fondée disponible)
        ax6 = plt.subplot(2, 3, 6)
        ax6.axis('off')

        if 'Fondée' in self.df_2025.columns:
            # Calculer matrice de confusion
            df_2025_copy = self.df_2025.copy()
            df_2025_copy['Fondee_bool'] = df_2025_copy['Fondée'].apply(
                lambda x: 1 if x in ['Oui', 1, True] else 0
            )
            df_2025_copy['Validation_bool'] = df_2025_copy['Decision_Modele'].apply(
                lambda x: 1 if x == 'Validation Auto' else 0
            )

            vp = ((df_2025_copy['Fondee_bool'] == 1) & (df_2025_copy['Validation_bool'] == 1)).sum()
            vn = ((df_2025_copy['Fondee_bool'] == 0) & (df_2025_copy['Validation_bool'] == 0)).sum()
            fp = ((df_2025_copy['Fondee_bool'] == 0) & (df_2025_copy['Validation_bool'] == 1)).sum()
            fn = ((df_2025_copy['Fondee_bool'] == 1) & (df_2025_copy['Validation_bool'] == 0)).sum()

            precision = vp / (vp + fp) if (vp + fp) > 0 else 0
            recall = vp / (vp + fn) if (vp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

            metrics_text = f"""
📊 MÉTRIQUES DE PERFORMANCE

Matrice de confusion:
  VP (Vrai Positif):    {vp:,}
  VN (Vrai Négatif):    {vn:,}
  FP (Faux Positif):    {fp:,}
  FN (Faux Négatif):    {fn:,}

Scores:
  Précision: {100*precision:.1f}%
  Rappel:    {100*recall:.1f}%
  F1-Score:  {100*f1:.1f}%

Exactitude:
  {100*(vp+vn)/(vp+vn+fp+fn):.1f}%
            """
        else:
            metrics_text = """
📊 MÉTRIQUES CLÉS

Décisions traitées:
  • Automatiquement
  • Cohérence élevée
  • Traçabilité totale

Avantages:
  ✓ Rapidité
  ✓ Objectivité
  ✓ Scalabilité
            """

        ax6.text(0.05, 0.95, metrics_text, transform=ax6.transAxes,
                fontsize=10, verticalalignment='top', family='monospace',
                bbox=dict(boxstyle='round', facecolor='#EBF5FB', alpha=0.9,
                         edgecolor='#3498DB', linewidth=2))

        plt.tight_layout()
        output_path = self.output_dir / 'R4_performance_metrics_2025.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ Sauvegardé: {output_path}")
        plt.close()

    def generate_summary_report(self):
        """Générer rapport texte"""
        print("\n📄 Génération du rapport récapitulatif...")

        report_path = self.output_dir / f'rapport_resultats_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt'

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("RAPPORT D'ANALYSE DES RÉSULTATS\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*80 + "\n\n")

            # Données 2025
            if 'Decision_Modele' in self.df_2025.columns:
                n_total = len(self.df_2025)
                n_rejet = (self.df_2025['Decision_Modele'] == 'Rejet Auto').sum()
                n_audit = (self.df_2025['Decision_Modele'] == 'Audit Humain').sum()
                n_validation = (self.df_2025['Decision_Modele'] == 'Validation Auto').sum()
                n_auto = n_rejet + n_validation

                f.write("RÉSULTATS 2025:\n")
                f.write("-" * 80 + "\n")
                f.write(f"Total réclamations:    {n_total:,}\n")
                f.write(f"Rejet Auto:            {n_rejet:,} ({100*n_rejet/n_total:.1f}%)\n")
                f.write(f"Audit Humain:          {n_audit:,} ({100*n_audit/n_total:.1f}%)\n")
                f.write(f"Validation Auto:       {n_validation:,} ({100*n_validation/n_total:.1f}%)\n")
                f.write(f"Taux automatisation:   {100*n_auto/n_total:.1f}%\n\n")

                # Gains
                gain_financier = n_auto * self.cout_traitement_manuel
                temps_economise_h = n_auto * (self.temps_traitement_manuel - self.temps_traitement_auto) / 60
                etp_libere = temps_economise_h / self.heures_annuelles_fte

                f.write("GAINS CALCULÉS:\n")
                f.write("-" * 80 + "\n")
                f.write(f"Gain financier:        {gain_financier:,.0f} DH ({gain_financier/1e6:.2f}M DH)\n")
                f.write(f"Temps économisé:       {temps_economise_h:,.0f} heures\n")
                f.write(f"ETP libérés:           {etp_libere:.2f}\n\n")

            f.write("FICHIERS GÉNÉRÉS:\n")
            f.write("-" * 80 + "\n")
            f.write("1. R1_decisions_distribution_2025.png\n")
            f.write("2. R2_comparison_2025_vs_2023.png\n")
            f.write("3. R3_gain_calculation_detailed.png\n")
            f.write("4. R4_performance_metrics_2025.png\n")

        print(f"✅ Rapport sauvegardé: {report_path}")

    def run(self):
        """Exécuter la génération complète"""
        self.load_data()
        self.plot_decisions_distribution_2025()
        if self.df_2023 is not None:
            self.plot_comparison_2025_vs_2023()
        self.plot_gain_calculation()
        self.plot_performance_metrics_2025()
        self.generate_summary_report()

        print("\n" + "="*80)
        print("✅ GÉNÉRATION DES RÉSULTATS TERMINÉE")
        print("="*80)
        print(f"\n📂 Tous les fichiers sont dans: {self.output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Générer visualisations des résultats')
    parser.add_argument('--data_2025', type=str, required=True,
                       help='Fichier Excel 2025 avec inférence (Decision_Modele)')
    parser.add_argument('--data_2023', type=str,
                       help='Fichier Excel 2023 (optionnel, pour comparaison)')

    args = parser.parse_args()

    visualizer = ResultsVisualizer(args.data_2025, args.data_2023)
    visualizer.run()


if __name__ == '__main__':
    main()
