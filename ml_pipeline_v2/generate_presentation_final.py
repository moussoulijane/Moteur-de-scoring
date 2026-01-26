"""
GÉNÉRATION DES GRAPHIQUES FINAUX POUR PRÉSENTATION
Partie 1: État des lieux (répartition par marché regroupée)
Partie 2: Architecture du modèle (claire et simplifiée)
Partie 3: Résultats + Monitoring des règles métier

Usage:
    python ml_pipeline_v2/generate_presentation_final.py \
        --data_2023 data/reclamations_2023.xlsx \
        --data_2024 data/reclamations_2024.xlsx \
        --data_2025 data/reclamations_2025.xlsx
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
plt.rcParams['figure.figsize'] = (16, 10)


class PresentationFinalGenerator:
    """Générateur des graphiques finaux pour présentation"""

    def __init__(self, data_2023=None, data_2024=None, data_2025=None):
        self.data_2023 = data_2023
        self.data_2024 = data_2024
        self.data_2025 = data_2025
        self.output_dir = Path('outputs/presentation_final')
        self.output_dir.mkdir(parents=True, exist_ok=True)

        print("\n" + "="*80)
        print("📊 GÉNÉRATEUR DE PRÉSENTATION FINALE")
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
        """Charger les données des 3 années"""
        print("\n📂 Chargement des données...")

        self.df_2023 = None
        self.df_2024 = None
        self.df_2025 = None

        if self.data_2023:
            self.df_2023 = pd.read_excel(self.data_2023)
            print(f"✅ 2023: {len(self.df_2023)} réclamations")

        if self.data_2024:
            self.df_2024 = pd.read_excel(self.data_2024)
            print(f"✅ 2024: {len(self.df_2024)} réclamations")

        if self.data_2025:
            self.df_2025 = pd.read_excel(self.data_2025)
            print(f"✅ 2025: {len(self.df_2025)} réclamations")

        # Nettoyer colonnes numériques
        print("\n🔄 Nettoyage des colonnes numériques...")
        numeric_cols = ['Montant demandé', 'Délai estimé', 'anciennete_annees',
                       'PNB analytique (vision commerciale) cumulé']

        for df, year in [(self.df_2023, 2023), (self.df_2024, 2024), (self.df_2025, 2025)]:
            if df is not None:
                for col in numeric_cols:
                    if col in df.columns:
                        df[col] = self.clean_numeric_column(df, col)
                print(f"   ✅ {year}: colonnes nettoyées")

    def plot_etat_lieux_marche(self):
        """PARTIE 1: État des lieux - Répartition par marché (regroupée)"""
        print("\n📊 Partie 1: État des lieux - Répartition par marché...")

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 12))
        fig.suptitle('ÉTAT DES LIEUX - RÉPARTITION PAR MARCHÉ (2023-2025)',
                     fontsize=18, fontweight='bold', y=0.98)

        # Préparer les données pour les 3 années
        data_years = []
        for df, year in [(self.df_2023, 2023), (self.df_2024, 2024), (self.df_2025, 2025)]:
            if df is not None and 'Marché' in df.columns:
                df_copy = df.copy()

                # Regrouper Particulier et Professionnel
                df_copy['Marche_Groupe'] = df_copy['Marché'].apply(
                    lambda x: 'Particulier & Professionnel'
                    if str(x).strip() in ['Particulier', 'Professionnel', 'PARTICULIER', 'PROFESSIONNEL']
                    else str(x).strip()
                )

                data_years.append({
                    'year': year,
                    'df': df_copy
                })

        if not data_years:
            print("⚠️  Aucune donnée disponible avec colonne Marché")
            return

        # 1. Répartition en NOMBRE par marché
        ax1.set_title('Répartition en NOMBRE par Marché', fontweight='bold', fontsize=14)

        marchés_all = set()
        for data in data_years:
            marchés_all.update(data['df']['Marche_Groupe'].unique())
        marchés_sorted = sorted(list(marchés_all))

        x = np.arange(len(marchés_sorted))
        width = 0.25
        colors = ['#3498DB', '#E67E22', '#2ECC71']

        for i, data in enumerate(data_years):
            counts = data['df']['Marche_Groupe'].value_counts()
            values = [counts.get(m, 0) for m in marchés_sorted]

            bars = ax1.bar(x + i*width, values, width,
                          label=f"{data['year']}",
                          color=colors[i], alpha=0.8, edgecolor='black', linewidth=1)

            for bar, val in zip(bars, values):
                if val > 0:
                    height = bar.get_height()
                    ax1.text(bar.get_x() + bar.get_width()/2., height,
                            f'{int(val):,}', ha='center', va='bottom',
                            fontsize=9, fontweight='bold')

        ax1.set_ylabel('Nombre de réclamations', fontweight='bold', fontsize=12)
        ax1.set_xticks(x + width)
        ax1.set_xticklabels(marchés_sorted, rotation=45, ha='right', fontsize=10)
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3, axis='y')

        # 2. Répartition en MONTANT par marché
        ax2.set_title('Répartition en MONTANT par Marché (Millions DH)', fontweight='bold', fontsize=14)

        for i, data in enumerate(data_years):
            if 'Montant demandé' in data['df'].columns:
                montants = data['df'].groupby('Marche_Groupe')['Montant demandé'].sum()
                values = [montants.get(m, 0) / 1e6 for m in marchés_sorted]

                bars = ax2.bar(x + i*width, values, width,
                              label=f"{data['year']}",
                              color=colors[i], alpha=0.8, edgecolor='black', linewidth=1)

                for bar, val in zip(bars, values):
                    if val > 0:
                        height = bar.get_height()
                        ax2.text(bar.get_x() + bar.get_width()/2., height,
                                f'{val:.1f}M', ha='center', va='bottom',
                                fontsize=9, fontweight='bold')

        ax2.set_ylabel('Montant Total (Millions DH)', fontweight='bold', fontsize=12)
        ax2.set_xticks(x + width)
        ax2.set_xticklabels(marchés_sorted, rotation=45, ha='right', fontsize=10)
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3, axis='y')

        # 3. Évolution temporelle en NOMBRE
        ax3.set_title('Évolution du Nombre par Marché', fontweight='bold', fontsize=14)

        for marche in marchés_sorted:
            years = []
            counts = []
            for data in data_years:
                years.append(data['year'])
                count = (data['df']['Marche_Groupe'] == marche).sum()
                counts.append(count)

            ax3.plot(years, counts, marker='o', linewidth=2, markersize=8, label=marche)

            # Annotations
            for y, c in zip(years, counts):
                if c > 0:
                    ax3.annotate(f'{int(c):,}', (y, c),
                               textcoords="offset points", xytext=(0,5),
                               ha='center', fontsize=8, fontweight='bold')

        ax3.set_xlabel('Année', fontweight='bold', fontsize=12)
        ax3.set_ylabel('Nombre de réclamations', fontweight='bold', fontsize=12)
        ax3.legend(fontsize=10, loc='best')
        ax3.grid(True, alpha=0.3)
        ax3.set_xticks([2023, 2024, 2025])

        # 4. Évolution temporelle en MONTANT
        ax4.set_title('Évolution du Montant par Marché (Millions DH)', fontweight='bold', fontsize=14)

        for marche in marchés_sorted:
            years = []
            montants = []
            for data in data_years:
                years.append(data['year'])
                if 'Montant demandé' in data['df'].columns:
                    mt = data['df'][data['df']['Marche_Groupe'] == marche]['Montant demandé'].sum() / 1e6
                    montants.append(mt)
                else:
                    montants.append(0)

            ax4.plot(years, montants, marker='o', linewidth=2, markersize=8, label=marche)

            # Annotations
            for y, m in zip(years, montants):
                if m > 0:
                    ax4.annotate(f'{m:.1f}M', (y, m),
                               textcoords="offset points", xytext=(0,5),
                               ha='center', fontsize=8, fontweight='bold')

        ax4.set_xlabel('Année', fontweight='bold', fontsize=12)
        ax4.set_ylabel('Montant Total (Millions DH)', fontweight='bold', fontsize=12)
        ax4.legend(fontsize=10, loc='best')
        ax4.grid(True, alpha=0.3)
        ax4.set_xticks([2023, 2024, 2025])

        plt.tight_layout()
        output_path = self.output_dir / 'P1_etat_lieux_marche.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ Sauvegardé: {output_path}")
        plt.close()

    def plot_architecture_modele(self):
        """PARTIE 2: Architecture du modèle (claire et simplifiée)"""
        print("\n📊 Partie 2: Architecture du modèle...")

        fig = plt.figure(figsize=(16, 10))
        fig.suptitle('ARCHITECTURE DU MODÈLE DE SCORING',
                     fontsize=20, fontweight='bold', y=0.96)

        ax = fig.add_subplot(111)
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis('off')

        # ===== PARTIE 1: LES 3 PILIERS =====
        pilier_y = 7.5
        pilier_height = 1.8
        pilier_width = 2.5

        # Pilier 1: Type Réclamation
        rect1 = plt.Rectangle((0.5, pilier_y), pilier_width, pilier_height,
                              facecolor='#3498DB', edgecolor='black', linewidth=3)
        ax.add_patch(rect1)
        ax.text(0.5 + pilier_width/2, pilier_y + pilier_height/2 + 0.4,
                'PILIER 1', ha='center', va='center',
                fontsize=14, fontweight='bold', color='white')
        ax.text(0.5 + pilier_width/2, pilier_y + pilier_height/2 - 0.1,
                'Type Réclamation', ha='center', va='center',
                fontsize=11, fontweight='bold', color='white')
        ax.text(0.5 + pilier_width/2, pilier_y + pilier_height/2 - 0.6,
                '• Famille\n• Catégorie\n• Sous-catégorie', ha='center', va='center',
                fontsize=9, color='white', linespacing=1.5)

        # Pilier 2: Risque
        rect2 = plt.Rectangle((3.75, pilier_y), pilier_width, pilier_height,
                              facecolor='#E67E22', edgecolor='black', linewidth=3)
        ax.add_patch(rect2)
        ax.text(3.75 + pilier_width/2, pilier_y + pilier_height/2 + 0.4,
                'PILIER 2', ha='center', va='center',
                fontsize=14, fontweight='bold', color='white')
        ax.text(3.75 + pilier_width/2, pilier_y + pilier_height/2 - 0.1,
                'Risque', ha='center', va='center',
                fontsize=11, fontweight='bold', color='white')
        ax.text(3.75 + pilier_width/2, pilier_y + pilier_height/2 - 0.6,
                '• Montant\n• Délai\n• Ratio/PNB', ha='center', va='center',
                fontsize=9, color='white', linespacing=1.5)

        # Pilier 3: Signalétique
        rect3 = plt.Rectangle((7, pilier_y), pilier_width, pilier_height,
                              facecolor='#2ECC71', edgecolor='black', linewidth=3)
        ax.add_patch(rect3)
        ax.text(7 + pilier_width/2, pilier_y + pilier_height/2 + 0.4,
                'PILIER 3', ha='center', va='center',
                fontsize=14, fontweight='bold', color='white')
        ax.text(7 + pilier_width/2, pilier_y + pilier_height/2 - 0.1,
                'Signalétique', ha='center', va='center',
                fontsize=11, fontweight='bold', color='white')
        ax.text(7 + pilier_width/2, pilier_y + pilier_height/2 - 0.6,
                '• PNB\n• Ancienneté\n• Segment/Marché', ha='center', va='center',
                fontsize=9, color='white', linespacing=1.5)

        # Flèches vers la couche analytique
        arrow_y = pilier_y - 0.3
        for x_pos in [1.75, 5, 8.25]:
            ax.arrow(x_pos, arrow_y, 0, -0.8, head_width=0.3, head_length=0.2,
                    fc='black', ec='black', linewidth=2)

        # ===== PARTIE 2: COUCHE ANALYTIQUE (IA) =====
        couche_y = 5
        couche_height = 1.2

        rect_ia = plt.Rectangle((1, couche_y), 8, couche_height,
                                facecolor='#9B59B6', edgecolor='black', linewidth=3)
        ax.add_patch(rect_ia)
        ax.text(5, couche_y + couche_height/2 + 0.3,
                '🤖 COUCHE ANALYTIQUE (Intelligence Artificielle)', ha='center', va='center',
                fontsize=13, fontweight='bold', color='white')
        ax.text(5, couche_y + couche_height/2 - 0.25,
                'Optimisation automatique des poids de chaque pilier', ha='center', va='center',
                fontsize=10, color='white', style='italic')

        # Flèche vers la couche décisionnelle
        ax.arrow(5, couche_y - 0.2, 0, -0.8, head_width=0.4, head_length=0.2,
                fc='black', ec='black', linewidth=3)

        # ===== PARTIE 3: COUCHE DÉCISIONNELLE =====
        decision_y = 2.5
        decision_height = 1.5

        rect_decision = plt.Rectangle((0.5, decision_y), 9, decision_height,
                                      facecolor='#E74C3C', edgecolor='black', linewidth=3)
        ax.add_patch(rect_decision)
        ax.text(5, decision_y + decision_height/2 + 0.5,
                '⚖️ COUCHE DÉCISIONNELLE', ha='center', va='center',
                fontsize=14, fontweight='bold', color='white')
        ax.text(5, decision_y + decision_height/2,
                'Score du Modèle + Règles Métier', ha='center', va='center',
                fontsize=11, fontweight='bold', color='white')

        # Les 2 règles
        rules_text = '''Règle #1: Maximum 1 validation par client par an
Règle #2: Montant validé ≤ PNB de l'année dernière'''
        ax.text(5, decision_y + decision_height/2 - 0.55,
                rules_text, ha='center', va='center',
                fontsize=9, color='white', linespacing=1.6)

        # Flèches vers les 3 décisions
        ax.arrow(2.5, decision_y - 0.2, 0, -0.5, head_width=0.3, head_length=0.15,
                fc='black', ec='black', linewidth=2)
        ax.arrow(5, decision_y - 0.2, 0, -0.5, head_width=0.3, head_length=0.15,
                fc='black', ec='black', linewidth=2)
        ax.arrow(7.5, decision_y - 0.2, 0, -0.5, head_width=0.3, head_length=0.15,
                fc='black', ec='black', linewidth=2)

        # ===== PARTIE 4: LES 3 DÉCISIONS FINALES =====
        decision_final_y = 0.8
        decision_width = 2.5
        decision_height = 0.8

        # Rejet Auto
        rect_rejet = plt.Rectangle((0.5, decision_final_y), decision_width, decision_height,
                                   facecolor='#E74C3C', edgecolor='black', linewidth=2)
        ax.add_patch(rect_rejet)
        ax.text(0.5 + decision_width/2, decision_final_y + decision_height/2,
                '❌ REJET AUTO', ha='center', va='center',
                fontsize=12, fontweight='bold', color='white')

        # Audit Humain
        rect_audit = plt.Rectangle((3.75, decision_final_y), decision_width, decision_height,
                                   facecolor='#F39C12', edgecolor='black', linewidth=2)
        ax.add_patch(rect_audit)
        ax.text(3.75 + decision_width/2, decision_final_y + decision_height/2,
                '🔍 AUDIT HUMAIN', ha='center', va='center',
                fontsize=12, fontweight='bold', color='white')

        # Validation Auto
        rect_validation = plt.Rectangle((7, decision_final_y), decision_width, decision_height,
                                        facecolor='#2ECC71', edgecolor='black', linewidth=2)
        ax.add_patch(rect_validation)
        ax.text(7 + decision_width/2, decision_final_y + decision_height/2,
                '✅ VALIDATION AUTO', ha='center', va='center',
                fontsize=12, fontweight='bold', color='white')

        plt.tight_layout()
        output_path = self.output_dir / 'P2_architecture_modele.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ Sauvegardé: {output_path}")
        plt.close()

    def plot_resultats_monitoring(self):
        """PARTIE 3: Résultats + Monitoring des règles métier"""
        print("\n📊 Partie 3: Résultats et monitoring...")

        # Ce graphique nécessite les données avec les décisions du modèle
        # Il sera créé avec les vraies données

        fig = plt.figure(figsize=(18, 10))
        fig.suptitle('RÉSULTATS ET MONITORING DES RÈGLES MÉTIER',
                     fontsize=18, fontweight='bold', y=0.98)

        # Note: Ce graphique sera complété quand on aura les données avec décisions
        ax = fig.add_subplot(111)
        ax.text(0.5, 0.5,
                'Ce graphique sera généré avec les données scorées\n\n'
                'Il montrera:\n'
                '• Métriques de performance 2023 et 2025\n'
                '• Gain NET pour chaque année\n'
                '• Impact des règles métier:\n'
                '  - Règle #1: Cas convertis de Validation → Audit\n'
                '  - Règle #2: Cas convertis de Validation → Audit\n'
                '• Avant/après application des règles',
                ha='center', va='center', fontsize=14,
                bbox=dict(boxstyle='round', facecolor='#ECF0F1',
                         edgecolor='black', linewidth=2, pad=20))
        ax.axis('off')

        plt.tight_layout()
        output_path = self.output_dir / 'P3_resultats_monitoring_template.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ Template sauvegardé: {output_path}")
        print("   ℹ️  Utilisez generate_results_visuals.py avec données scorées pour les graphiques détaillés")
        plt.close()

    def run(self):
        """Exécuter la génération complète"""
        self.load_data()

        print("\n" + "="*80)
        print("📊 GÉNÉRATION DES GRAPHIQUES")
        print("="*80)

        self.plot_etat_lieux_marche()
        self.plot_architecture_modele()
        self.plot_resultats_monitoring()

        print("\n" + "="*80)
        print("✅ GÉNÉRATION TERMINÉE")
        print("="*80)
        print(f"\n📂 Tous les fichiers sont dans: {self.output_dir}")
        print("\nFichiers générés:")
        print("  - P1: État des lieux - Répartition par marché")
        print("  - P2: Architecture du modèle")
        print("  - P3: Template résultats & monitoring")


def main():
    parser = argparse.ArgumentParser(description='Générer présentation finale')
    parser.add_argument('--data_2023', type=str, help='Fichier Excel 2023')
    parser.add_argument('--data_2024', type=str, help='Fichier Excel 2024')
    parser.add_argument('--data_2025', type=str, help='Fichier Excel 2025')

    args = parser.parse_args()

    if not any([args.data_2023, args.data_2024, args.data_2025]):
        print("❌ ERREUR: Veuillez fournir au moins un fichier de données")
        parser.print_help()
        return

    generator = PresentationFinalGenerator(
        data_2023=args.data_2023,
        data_2024=args.data_2024,
        data_2025=args.data_2025
    )
    generator.run()


if __name__ == '__main__':
    main()
