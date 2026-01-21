"""
GÉNÉRATION DES VISUELS POUR PRÉSENTATION
Crée tous les graphiques nécessaires pour la présentation d'opérationnalisation

Usage:
    python ml_pipeline_v2/generate_presentation_visuals.py --data_2023 chemin/2023.xlsx --data_2024 chemin/2024.xlsx --data_2025 chemin/2025.xlsx
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

# Configuration style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 11


class PresentationGenerator:
    """Génère tous les visuels pour la présentation"""

    def __init__(self, data_2023, data_2024, data_2025):
        self.data_2023 = data_2023
        self.data_2024 = data_2024
        self.data_2025 = data_2025
        self.output_dir = Path('outputs/presentation')
        self.output_dir.mkdir(parents=True, exist_ok=True)

        print("\n" + "="*80)
        print("📊 GÉNÉRATEUR DE VISUELS POUR PRÉSENTATION")
        print("="*80)

    def load_data(self):
        """Charger les données des 3 années"""
        print("\n📂 Chargement des données...")

        self.df_2023 = pd.read_excel(self.data_2023) if self.data_2023 else None
        self.df_2024 = pd.read_excel(self.data_2024) if self.data_2024 else None
        self.df_2025 = pd.read_excel(self.data_2025) if self.data_2025 else None

        if self.df_2023 is not None:
            print(f"✅ 2023: {len(self.df_2023)} réclamations")
        if self.df_2024 is not None:
            print(f"✅ 2024: {len(self.df_2024)} réclamations")
        if self.df_2025 is not None:
            print(f"✅ 2025: {len(self.df_2025)} réclamations")

        # Ajouter année à chaque dataset
        if self.df_2023 is not None:
            self.df_2023['Annee'] = 2023
        if self.df_2024 is not None:
            self.df_2024['Annee'] = 2024
        if self.df_2025 is not None:
            self.df_2025['Annee'] = 2025

    def plot_evolution_volume(self):
        """1. Évolution du nombre de réclamations et montant total"""
        print("\n📊 Graphique 1: Évolution volume et montant...")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle('ÉVOLUTION DES RÉCLAMATIONS FINANCIÈRES (2023-2025)',
                     fontsize=16, fontweight='bold', y=1.02)

        # Préparer les données
        years = []
        counts = []
        amounts = []

        for year, df in [(2023, self.df_2023), (2024, self.df_2024), (2025, self.df_2025)]:
            if df is not None and len(df) > 0:
                years.append(year)
                counts.append(len(df))

                if 'Montant demandé' in df.columns:
                    total_amount = df['Montant demandé'][df['Montant demandé'] > 0].sum()
                    amounts.append(total_amount)
                else:
                    amounts.append(0)

        # Graphique 1: Nombre de réclamations
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        bars1 = ax1.bar(years, counts, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
        ax1.set_xlabel('Année', fontweight='bold', fontsize=12)
        ax1.set_ylabel('Nombre de réclamations', fontweight='bold', fontsize=12)
        ax1.set_title('Évolution du NOMBRE de réclamations', fontweight='bold', fontsize=13)
        ax1.grid(True, alpha=0.3, axis='y')

        # Ajouter valeurs sur les barres
        for bar, count in zip(bars1, counts):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(count):,}',
                    ha='center', va='bottom', fontweight='bold', fontsize=11)

        # Calculer évolution
        if len(counts) >= 2:
            evol_2324 = ((counts[1] - counts[0]) / counts[0] * 100) if counts[0] > 0 else 0
            ax1.text(0.5, 0.95, f'Évolution 2023→2024: {evol_2324:+.1f}%',
                    transform=ax1.transAxes, ha='center', va='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                    fontweight='bold')

        # Graphique 2: Montant total
        bars2 = ax2.bar(years, [a/1e6 for a in amounts], color=colors, alpha=0.7,
                       edgecolor='black', linewidth=1.5)
        ax2.set_xlabel('Année', fontweight='bold', fontsize=12)
        ax2.set_ylabel('Montant total (Millions DH)', fontweight='bold', fontsize=12)
        ax2.set_title('Évolution du MONTANT des réclamations', fontweight='bold', fontsize=13)
        ax2.grid(True, alpha=0.3, axis='y')

        # Ajouter valeurs
        for bar, amount in zip(bars2, amounts):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{amount/1e6:.1f}M',
                    ha='center', va='bottom', fontweight='bold', fontsize=11)

        # Évolution montant
        if len(amounts) >= 2 and amounts[0] > 0:
            evol_amount = ((amounts[1] - amounts[0]) / amounts[0] * 100)
            ax2.text(0.5, 0.95, f'Évolution 2023→2024: {evol_amount:+.1f}%',
                    transform=ax2.transAxes, ha='center', va='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                    fontweight='bold')

        plt.tight_layout()
        output_path = self.output_dir / '01_evolution_volume_montant.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ Sauvegardé: {output_path}")
        plt.close()

    def plot_fondee_vs_non_fondee(self):
        """2. Évolution fondée vs non fondée"""
        print("\n📊 Graphique 2: Fondée vs Non fondée...")

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('ÉVOLUTION RÉCLAMATIONS FONDÉES vs NON FONDÉES (2023-2025)',
                     fontsize=16, fontweight='bold', y=0.995)

        # Préparer données
        years = []
        pct_fondee = []
        pct_non_fondee = []
        montant_fondee = []
        montant_non_fondee = []

        for year, df in [(2023, self.df_2023), (2024, self.df_2024), (2025, self.df_2025)]:
            if df is not None and len(df) > 0 and 'Fondée' in df.columns:
                years.append(year)

                # Nombre
                n_fondee = (df['Fondée'] == 'Oui').sum() if 'Oui' in df['Fondée'].values else df['Fondée'].sum()
                n_total = len(df)
                pct_fondee.append(100 * n_fondee / n_total)
                pct_non_fondee.append(100 * (n_total - n_fondee) / n_total)

                # Montant
                if 'Montant demandé' in df.columns:
                    df_fondee = df[df['Fondée'] == 'Oui'] if 'Oui' in df['Fondée'].values else df[df['Fondée'] == 1]
                    df_non_fondee = df[df['Fondée'] == 'Non'] if 'Non' in df['Fondée'].values else df[df['Fondée'] == 0]

                    mt_fondee = df_fondee['Montant demandé'][df_fondee['Montant demandé'] > 0].sum()
                    mt_non_fondee = df_non_fondee['Montant demandé'][df_non_fondee['Montant demandé'] > 0].sum()

                    montant_fondee.append(mt_fondee / 1e6)
                    montant_non_fondee.append(mt_non_fondee / 1e6)

        # Graphique 1: Pourcentage en nombre
        x = np.arange(len(years))
        width = 0.35

        bars1 = ax1.bar(x - width/2, pct_fondee, width, label='Fondée', color='#2ECC71', alpha=0.8)
        bars2 = ax1.bar(x + width/2, pct_non_fondee, width, label='Non fondée', color='#E74C3C', alpha=0.8)

        ax1.set_ylabel('Pourcentage (%)', fontweight='bold')
        ax1.set_title('Répartition en NOMBRE (%)', fontweight='bold', fontsize=13)
        ax1.set_xticks(x)
        ax1.set_xticklabels(years)
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')

        # Valeurs sur barres
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%', ha='center', va='bottom', fontweight='bold')
        for bar in bars2:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%', ha='center', va='bottom', fontweight='bold')

        # Graphique 2: Montant en millions
        bars3 = ax2.bar(x - width/2, montant_fondee, width, label='Fondée', color='#2ECC71', alpha=0.8)
        bars4 = ax2.bar(x + width/2, montant_non_fondee, width, label='Non fondée', color='#E74C3C', alpha=0.8)

        ax2.set_ylabel('Montant (Millions DH)', fontweight='bold')
        ax2.set_title('Répartition en MONTANT', fontweight='bold', fontsize=13)
        ax2.set_xticks(x)
        ax2.set_xticklabels(years)
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')

        for bar in bars3:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}M', ha='center', va='bottom', fontweight='bold')
        for bar in bars4:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}M', ha='center', va='bottom', fontweight='bold')

        # Graphique 3: Évolution du taux de fondée
        ax3.plot(years, pct_fondee, marker='o', linewidth=3, markersize=10,
                color='#2ECC71', label='Taux fondée')
        ax3.set_xlabel('Année', fontweight='bold')
        ax3.set_ylabel('Taux fondée (%)', fontweight='bold')
        ax3.set_title('ÉVOLUTION du taux de fondée', fontweight='bold', fontsize=13)
        ax3.grid(True, alpha=0.3)
        ax3.legend()

        for year, pct in zip(years, pct_fondee):
            ax3.text(year, pct + 1, f'{pct:.1f}%', ha='center', fontweight='bold')

        # Graphique 4: Tableau récapitulatif
        ax4.axis('off')
        table_data = []
        table_data.append(['Année', 'Fondée (%)', 'Non fondée (%)', 'Montant fondée (M)', 'Montant non fondée (M)'])

        for i, year in enumerate(years):
            table_data.append([
                str(year),
                f'{pct_fondee[i]:.1f}%',
                f'{pct_non_fondee[i]:.1f}%',
                f'{montant_fondee[i]:.1f}M',
                f'{montant_non_fondee[i]:.1f}M'
            ])

        table = ax4.table(cellText=table_data, cellLoc='center', loc='center',
                         bbox=[0, 0, 1, 1])
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1, 2)

        # Style header
        for i in range(5):
            table[(0, i)].set_facecolor('#34495E')
            table[(0, i)].set_text_props(weight='bold', color='white')

        # Couleurs alternées
        for i in range(1, len(table_data)):
            for j in range(5):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#ECF0F1')

        plt.tight_layout()
        output_path = self.output_dir / '02_fondee_vs_non_fondee.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ Sauvegardé: {output_path}")
        plt.close()

    def plot_repartition_famille(self):
        """3. Répartition par famille (pie charts par année)"""
        print("\n📊 Graphique 3: Répartition par famille...")

        fig = plt.figure(figsize=(18, 12))
        fig.suptitle('RÉPARTITION PAR FAMILLE DE PRODUIT', fontsize=16, fontweight='bold', y=0.98)

        datasets = [
            (2023, self.df_2023, 1),
            (2024, self.df_2024, 2),
            (2025, self.df_2025, 3)
        ]

        for year, df, col_offset in datasets:
            if df is None or 'Famille Produit' not in df.columns:
                continue

            # Pie chart nombre (ligne 1)
            ax1 = plt.subplot(2, 3, col_offset)
            famille_counts = df['Famille Produit'].value_counts()
            top_5 = famille_counts.head(5)

            colors = plt.cm.Set3(range(len(top_5)))
            wedges, texts, autotexts = ax1.pie(top_5.values, labels=top_5.index,
                                                autopct='%1.1f%%', colors=colors,
                                                startangle=90, textprops={'fontsize': 9})

            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')

            ax1.set_title(f'{year} - NOMBRE\n(Total: {len(df):,})', fontweight='bold', fontsize=12)

            # Pie chart montant (ligne 2)
            ax2 = plt.subplot(2, 3, col_offset + 3)
            if 'Montant demandé' in df.columns:
                famille_montants = df.groupby('Famille Produit')['Montant demandé'].sum()
                top_5_mt = famille_montants.nlargest(5)

                wedges, texts, autotexts = ax2.pie(top_5_mt.values, labels=top_5_mt.index,
                                                    autopct='%1.1f%%', colors=colors,
                                                    startangle=90, textprops={'fontsize': 9})

                for autotext in autotexts:
                    autotext.set_color('white')
                    autotext.set_fontweight('bold')

                total_mt = df['Montant demandé'][df['Montant demandé'] > 0].sum()
                ax2.set_title(f'{year} - MONTANT\n(Total: {total_mt/1e6:.1f}M DH)',
                             fontweight='bold', fontsize=12)

        plt.tight_layout()
        output_path = self.output_dir / '03_repartition_famille.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ Sauvegardé: {output_path}")
        plt.close()

    def plot_repartition_marche(self):
        """4. Répartition par marché"""
        print("\n📊 Graphique 4: Répartition par marché...")

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('RÉPARTITION PAR MARCHÉ', fontsize=16, fontweight='bold', y=0.98)

        datasets = [
            (2023, self.df_2023, 0),
            (2024, self.df_2024, 1),
            (2025, self.df_2025, 2)
        ]

        for year, df, col_idx in datasets:
            if df is None or 'Marché' not in df.columns:
                continue

            # Nombre (ligne 1)
            ax1 = axes[0, col_idx]
            marche_counts = df['Marché'].value_counts()

            colors = plt.cm.Pastel1(range(len(marche_counts)))
            wedges, texts, autotexts = ax1.pie(marche_counts.values, labels=marche_counts.index,
                                                autopct='%1.1f%%', colors=colors,
                                                startangle=90, textprops={'fontsize': 10})

            for autotext in autotexts:
                autotext.set_color('black')
                autotext.set_fontweight('bold')

            ax1.set_title(f'{year} - NOMBRE', fontweight='bold', fontsize=12)

            # Montant (ligne 2)
            ax2 = axes[1, col_idx]
            if 'Montant demandé' in df.columns:
                marche_montants = df.groupby('Marché')['Montant demandé'].sum()

                wedges, texts, autotexts = ax2.pie(marche_montants.values, labels=marche_montants.index,
                                                    autopct='%1.1f%%', colors=colors,
                                                    startangle=90, textprops={'fontsize': 10})

                for autotext in autotexts:
                    autotext.set_color('black')
                    autotext.set_fontweight('bold')

                ax2.set_title(f'{year} - MONTANT', fontweight='bold', fontsize=12)

        plt.tight_layout()
        output_path = self.output_dir / '04_repartition_marche.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ Sauvegardé: {output_path}")
        plt.close()

    def generate_model_architecture_diagram(self):
        """5. Diagramme de l'architecture du modèle"""
        print("\n📊 Graphique 5: Architecture du modèle...")

        fig, ax = plt.subplots(figsize=(14, 10))
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis('off')

        # Titre
        ax.text(5, 9.5, 'ARCHITECTURE DU MODÈLE DE SCORING', ha='center',
               fontsize=18, fontweight='bold')

        # Pilier 1: Type réclamations (gauche)
        rect1 = plt.Rectangle((0.5, 6), 2.5, 2.5, facecolor='#3498DB', edgecolor='black', linewidth=2)
        ax.add_patch(rect1)
        ax.text(1.75, 7.7, 'PILIER 1', ha='center', fontsize=12, fontweight='bold', color='white')
        ax.text(1.75, 7.3, 'Type de', ha='center', fontsize=11, color='white')
        ax.text(1.75, 7.0, 'Réclamation', ha='center', fontsize=11, color='white')
        ax.text(1.75, 6.5, '• Famille\n• Catégorie\n• Sous-catégorie', ha='center', fontsize=9, color='white')

        # Pilier 2: Risque (centre)
        rect2 = plt.Rectangle((3.5, 6), 2.5, 2.5, facecolor='#E74C3C', edgecolor='black', linewidth=2)
        ax.add_patch(rect2)
        ax.text(4.75, 7.7, 'PILIER 2', ha='center', fontsize=12, fontweight='bold', color='white')
        ax.text(4.75, 7.3, 'Risque', ha='center', fontsize=11, color='white')
        ax.text(4.75, 6.5, '• Montant\n• Délai\n• Ratio/PNB', ha='center', fontsize=9, color='white')

        # Pilier 3: Signalétique (droite)
        rect3 = plt.Rectangle((6.5, 6), 2.5, 2.5, facecolor='#2ECC71', edgecolor='black', linewidth=2)
        ax.add_patch(rect3)
        ax.text(7.75, 7.7, 'PILIER 3', ha='center', fontsize=12, fontweight='bold', color='white')
        ax.text(7.75, 7.3, 'Signalétique', ha='center', fontsize=11, color='white')
        ax.text(7.75, 6.5, '• PNB cumulé\n• Ancienneté\n• Segment/Marché', ha='center', fontsize=9, color='white')

        # Flèches vers couche analytique
        for x in [1.75, 4.75, 7.75]:
            ax.arrow(x, 6, 0, -0.5, head_width=0.2, head_length=0.2, fc='black', ec='black')

        # Couche analytique (IA)
        rect_ia = plt.Rectangle((1.5, 4), 7, 1.5, facecolor='#9B59B6', edgecolor='black', linewidth=2)
        ax.add_patch(rect_ia)
        ax.text(5, 5.1, 'COUCHE ANALYTIQUE - MODÈLES IA', ha='center',
               fontsize=13, fontweight='bold', color='white')
        ax.text(5, 4.6, 'XGBoost / CatBoost - Optimisation Optuna', ha='center',
               fontsize=10, color='white')
        ax.text(5, 4.3, 'Attribution automatique des POIDS optimaux', ha='center',
               fontsize=9, color='white', style='italic')

        # Flèche vers décisionnel
        ax.arrow(5, 4, 0, -0.5, head_width=0.3, head_length=0.2, fc='black', ec='black', linewidth=2)

        # Couche décisionnelle
        rect_decision = plt.Rectangle((1, 1.5), 8, 2, facecolor='#F39C12', edgecolor='black', linewidth=2)
        ax.add_patch(rect_decision)
        ax.text(5, 3.2, 'COUCHE DÉCISIONNELLE', ha='center',
               fontsize=13, fontweight='bold', color='white')

        # Sous-composants
        ax.text(5, 2.7, '1️⃣ Décision Modèle (3 zones)', ha='center', fontsize=10, color='white', fontweight='bold')
        ax.text(5, 2.4, 'Rejet Auto | Audit Humain | Validation Auto', ha='center', fontsize=9, color='white')

        ax.text(5, 2.0, '2️⃣ Règle Métier #1:', ha='center', fontsize=10, color='white', fontweight='bold')
        ax.text(5, 1.75, 'Maximum 1 validation automatique par client par an', ha='center', fontsize=9, color='white')

        ax.text(5, 1.4, '3️⃣ Règle Métier #2:', ha='center', fontsize=10, color='white', fontweight='bold')
        ax.text(5, 1.15, 'Montant validé ≤ PNB année dernière', ha='center', fontsize=9, color='white')

        # Résultat final
        ax.arrow(5, 1.5, 0, -0.5, head_width=0.3, head_length=0.15, fc='black', ec='black', linewidth=2)

        rect_output = plt.Rectangle((2.5, 0.2), 5, 0.7, facecolor='#16A085', edgecolor='black', linewidth=2)
        ax.add_patch(rect_output)
        ax.text(5, 0.55, 'DÉCISION FINALE AUTOMATISÉE', ha='center',
               fontsize=12, fontweight='bold', color='white')

        plt.tight_layout()
        output_path = self.output_dir / '05_architecture_modele.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ Sauvegardé: {output_path}")
        plt.close()

    def calculate_2025_results(self):
        """6. Résultats et gains sur 2025"""
        print("\n📊 Graphique 6: Résultats 2025...")

        if self.df_2025 is None:
            print("⚠️  Données 2025 manquantes")
            return

        # Vérifier si inférence disponible
        if 'Decision_Modele' not in self.df_2025.columns:
            print("⚠️  Pas de décisions modèle dans les données 2025")
            return

        fig = plt.figure(figsize=(16, 10))
        fig.suptitle('RÉSULTATS DU MODÈLE SUR 2025 & CALCUL DU GAIN',
                     fontsize=16, fontweight='bold', y=0.98)

        # Statistiques décisions
        n_total = len(self.df_2025)
        n_rejet = (self.df_2025['Decision_Modele'] == 'Rejet Auto').sum()
        n_audit = (self.df_2025['Decision_Modele'] == 'Audit Humain').sum()
        n_validation = (self.df_2025['Decision_Modele'] == 'Validation Auto').sum()

        # Graphique 1: Pie chart des décisions
        ax1 = plt.subplot(2, 3, 1)
        sizes = [n_rejet, n_audit, n_validation]
        labels = [f'Rejet Auto\n{n_rejet:,}', f'Audit Humain\n{n_audit:,}', f'Validation Auto\n{n_validation:,}']
        colors = ['#E74C3C', '#F39C12', '#2ECC71']
        explode = (0.05, 0.05, 0.1)

        wedges, texts, autotexts = ax1.pie(sizes, explode=explode, labels=labels,
                                            autopct='%1.1f%%', colors=colors,
                                            shadow=True, startangle=90,
                                            textprops={'fontsize': 10})

        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')

        ax1.set_title('Répartition des DÉCISIONS', fontweight='bold', fontsize=12)

        # Graphique 2: Taux d'automatisation
        ax2 = plt.subplot(2, 3, 2)
        taux_auto = 100 * (n_rejet + n_validation) / n_total
        taux_audit = 100 * n_audit / n_total

        bars = ax2.barh(['Automatisé', 'Audit Humain'],
                       [taux_auto, taux_audit],
                       color=['#2ECC71', '#F39C12'], alpha=0.8, edgecolor='black', linewidth=2)

        ax2.set_xlabel('Pourcentage (%)', fontweight='bold')
        ax2.set_title('TAUX D\'AUTOMATISATION', fontweight='bold', fontsize=12)
        ax2.grid(True, alpha=0.3, axis='x')

        for bar in bars:
            width = bar.get_width()
            ax2.text(width + 1, bar.get_y() + bar.get_height()/2.,
                    f'{width:.1f}%', ha='left', va='center', fontweight='bold', fontsize=11)

        # Graphique 3: Montants par décision
        ax3 = plt.subplot(2, 3, 3)
        if 'Montant demandé' in self.df_2025.columns:
            montants = []
            for decision in ['Rejet Auto', 'Audit Humain', 'Validation Auto']:
                df_dec = self.df_2025[self.df_2025['Decision_Modele'] == decision]
                mt = df_dec['Montant demandé'][df_dec['Montant demandé'] > 0].sum() / 1e6
                montants.append(mt)

            bars = ax3.bar(['Rejet\nAuto', 'Audit\nHumain', 'Validation\nAuto'],
                          montants, color=colors, alpha=0.8, edgecolor='black', linewidth=2)

            ax3.set_ylabel('Montant (Millions DH)', fontweight='bold')
            ax3.set_title('MONTANT par type de décision', fontweight='bold', fontsize=12)
            ax3.grid(True, alpha=0.3, axis='y')

            for bar, mt in zip(bars, montants):
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height,
                        f'{mt:.1f}M', ha='center', va='bottom', fontweight='bold')

        # Graphique 4: Calcul du GAIN
        ax4 = plt.subplot(2, 3, 4)
        ax4.axis('off')

        # Hypothèses de gain
        cout_traitement_manuel = 50  # DH par dossier
        temps_moyen_manuel = 30  # minutes
        temps_moyen_auto = 1  # minute

        n_auto = n_rejet + n_validation
        gain_cout = n_auto * cout_traitement_manuel
        gain_temps_heures = n_auto * (temps_moyen_manuel - temps_moyen_auto) / 60

        # Équivalent ETP (1 ETP = 1600h/an)
        etp_libere = gain_temps_heures / 1600

        gain_text = f"""
📊 CALCUL DU GAIN (2025)

💰 GAIN FINANCIER:
   • Dossiers automatisés: {n_auto:,}
   • Coût évité (50 DH/dossier): {gain_cout:,.0f} DH
   • Soit: {gain_cout/1e6:.2f} Millions DH

⏱️ GAIN TEMPS:
   • Temps économisé: {gain_temps_heures:,.0f} heures
   • Équivalent: {etp_libere:.2f} ETP libérés
   • Productivité: +{taux_auto:.1f}%

📈 PERFORMANCE MODÈLE:
   • Taux automatisation: {taux_auto:.1f}%
   • Validations auto: {100*n_validation/n_total:.1f}%
   • Rejets auto: {100*n_rejet/n_total:.1f}%
        """

        ax4.text(0.1, 0.9, gain_text, transform=ax4.transAxes,
                fontsize=11, verticalalignment='top', family='monospace',
                bbox=dict(boxstyle='round', facecolor='#ECF0F1', alpha=0.8, edgecolor='black', linewidth=2))

        # Graphique 5: Évolution avant/après
        ax5 = plt.subplot(2, 3, 5)

        scenarios = ['AVANT\n(Traitement manuel)', 'APRÈS\n(Avec modèle)']
        temps_traitement = [n_total * temps_moyen_manuel / 60, n_audit * temps_moyen_manuel / 60 + n_auto * temps_moyen_auto / 60]

        bars = ax5.bar(scenarios, temps_traitement, color=['#E74C3C', '#2ECC71'],
                      alpha=0.8, edgecolor='black', linewidth=2)

        ax5.set_ylabel('Temps total (heures)', fontweight='bold')
        ax5.set_title('IMPACT sur le temps de traitement', fontweight='bold', fontsize=12)
        ax5.grid(True, alpha=0.3, axis='y')

        for bar, temps in zip(bars, temps_traitement):
            height = bar.get_height()
            ax5.text(bar.get_x() + bar.get_width()/2., height,
                    f'{temps:,.0f}h', ha='center', va='bottom', fontweight='bold', fontsize=11)

        # Flèche de réduction
        reduction_pct = 100 * (temps_traitement[0] - temps_traitement[1]) / temps_traitement[0]
        ax5.text(0.5, 0.5, f'↓ {reduction_pct:.1f}%',
                transform=ax5.transAxes, ha='center', va='center',
                fontsize=24, fontweight='bold', color='#2ECC71',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        # Graphique 6: ROI
        ax6 = plt.subplot(2, 3, 6)
        ax6.axis('off')

        roi_text = f"""
💡 RETOUR SUR INVESTISSEMENT

✅ BÉNÉFICES QUANTIFIABLES:
   • Gain financier: {gain_cout/1e6:.2f}M DH/an
   • Gain temps: {etp_libere:.2f} ETP libérés
   • Réduction délais: {reduction_pct:.1f}%

🎯 BÉNÉFICES QUALITATIFS:
   • Traitement instantané
   • Cohérence des décisions
   • Traçabilité complète
   • Satisfaction client ↗
   • Réduction erreurs humaines

📊 RECOMMANDATIONS:
   • Déploiement en production
   • Monitoring continu
   • Ajustements trimestriels
        """

        ax6.text(0.1, 0.9, roi_text, transform=ax6.transAxes,
                fontsize=10, verticalalignment='top', family='monospace',
                bbox=dict(boxstyle='round', facecolor='#D5F4E6', alpha=0.8,
                         edgecolor='#2ECC71', linewidth=2))

        plt.tight_layout()
        output_path = self.output_dir / '06_resultats_2025_gain.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ Sauvegardé: {output_path}")
        plt.close()

    def generate_summary_report(self):
        """Générer rapport texte récapitulatif"""
        print("\n📄 Génération du rapport récapitulatif...")

        report_path = self.output_dir / f'rapport_presentation_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt'

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("RAPPORT DE PRÉSENTATION - OPÉRATIONNALISATION DU MODÈLE\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*80 + "\n\n")

            f.write("📊 FICHIERS GÉNÉRÉS:\n")
            f.write("-" * 80 + "\n")
            f.write("1. 01_evolution_volume_montant.png - Évolution nombre et montant 2023-2025\n")
            f.write("2. 02_fondee_vs_non_fondee.png - Analyse fondée vs non fondée\n")
            f.write("3. 03_repartition_famille.png - Répartition par famille de produit\n")
            f.write("4. 04_repartition_marche.png - Répartition par marché\n")
            f.write("5. 05_architecture_modele.png - Architecture du modèle (3 piliers)\n")
            f.write("6. 06_resultats_2025_gain.png - Résultats 2025 et calcul du gain\n\n")

            f.write("🎯 STRUCTURE DE LA PRÉSENTATION:\n")
            f.write("-" * 80 + "\n\n")

            f.write("I. ÉTAT DES LIEUX (Slides 1-4)\n")
            f.write("   • Évolution du volume et montant des réclamations\n")
            f.write("   • Analyse fondée vs non fondée\n")
            f.write("   • Répartition par famille\n")
            f.write("   • Répartition par marché\n\n")

            f.write("II. PRÉSENTATION DU MODÈLE (Slide 5)\n")
            f.write("   • Architecture en 3 piliers:\n")
            f.write("     1. Type de réclamation (Famille, Catégorie, Sous-catégorie)\n")
            f.write("     2. Risque (Montant, Délai, Ratios)\n")
            f.write("     3. Signalétique client (PNB, Ancienneté, Segment)\n\n")
            f.write("   • Couche analytique:\n")
            f.write("     - Modèles IA (XGBoost/CatBoost)\n")
            f.write("     - Optimisation automatique des poids (Optuna)\n\n")
            f.write("   • Couche décisionnelle:\n")
            f.write("     - Décision modèle (3 zones)\n")
            f.write("     - Règle métier #1: 1 validation/client/an\n")
            f.write("     - Règle métier #2: Montant ≤ PNB année dernière\n\n")

            f.write("III. RÉSULTATS 2025 & GAINS (Slide 6)\n")
            f.write("   • Performance du modèle\n")
            f.write("   • Calcul du gain financier\n")
            f.write("   • Gain temps (ETP libérés)\n")
            f.write("   • ROI et recommandations\n\n")

            f.write("="*80 + "\n")
            f.write("TOUS LES GRAPHIQUES SONT PRÊTS POUR INTÉGRATION POWERPOINT\n")
            f.write("="*80 + "\n")

        print(f"✅ Rapport sauvegardé: {report_path}")

    def run(self):
        """Exécuter la génération complète"""
        self.load_data()
        self.plot_evolution_volume()
        self.plot_fondee_vs_non_fondee()
        self.plot_repartition_famille()
        self.plot_repartition_marche()
        self.generate_model_architecture_diagram()
        self.calculate_2025_results()
        self.generate_summary_report()

        print("\n" + "="*80)
        print("✅ GÉNÉRATION TERMINÉE")
        print("="*80)
        print(f"\n📂 Tous les fichiers sont dans: {self.output_dir}")
        print("\n💡 Vous pouvez maintenant insérer ces images dans votre présentation PowerPoint")


def main():
    parser = argparse.ArgumentParser(description='Générer visuels pour présentation')
    parser.add_argument('--data_2023', type=str, help='Fichier Excel 2023')
    parser.add_argument('--data_2024', type=str, required=True, help='Fichier Excel 2024')
    parser.add_argument('--data_2025', type=str, required=True, help='Fichier Excel 2025 (avec inférence)')

    args = parser.parse_args()

    generator = PresentationGenerator(args.data_2023, args.data_2024, args.data_2025)
    generator.run()


if __name__ == '__main__':
    main()
