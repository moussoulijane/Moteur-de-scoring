"""
ANALYSE EXPLORATOIRE DES RÉCLAMATIONS
Génère des visualisations pour comprendre les profils des réclamations

Usage:
    python analyze_claims_profile.py --input_file reclamations.xlsx
    python analyze_claims_profile.py --input_file reclamations.xlsx --with_predictions
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from pathlib import Path
from datetime import datetime
import warnings
import re
warnings.filterwarnings('ignore')

# Configuration
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (16, 10)


class ClaimProfileAnalyzer:
    """Analyse exploratoire des profils de réclamations"""

    def __init__(self, input_file, with_predictions=False):
        self.input_file = input_file
        self.with_predictions = with_predictions
        self.output_dir = Path('outputs/profile_analysis')
        self.output_dir.mkdir(parents=True, exist_ok=True)

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

    def load_data(self):
        """Charger les données"""
        print("\n" + "="*80)
        print("📂 CHARGEMENT DES DONNÉES")
        print("="*80)

        self.df = pd.read_excel(self.input_file)
        print(f"✅ {len(self.df)} réclamations chargées")

        # Vérifier si les prédictions sont présentes
        if 'Decision_Modele' in self.df.columns:
            self.with_predictions = True
            print(f"✅ Prédictions détectées dans le fichier")

        # Nettoyer les colonnes numériques (avec traitement texte robuste)
        print("🔧 Conversion des colonnes numériques (texte -> float)...")
        numeric_cols = ['Montant demandé', 'Délai estimé', 'anciennete_annees',
                       'PNB analytique (vision commerciale) cumulé']

        for col in numeric_cols:
            if col in self.df.columns:
                self.df[col] = self._clean_numeric_column(self.df[col])

        print(f"\n📊 Colonnes disponibles: {len(self.df.columns)}")
        print(f"   - Montant demandé: {'✅' if 'Montant demandé' in self.df.columns else '❌'}")
        print(f"   - Délai estimé: {'✅' if 'Délai estimé' in self.df.columns else '❌'}")
        print(f"   - PNB cumulé: {'✅' if 'PNB analytique (vision commerciale) cumulé' in self.df.columns else '❌'}")
        print(f"   - Ancienneté: {'✅' if 'anciennete_annees' in self.df.columns else '❌'}")

    def analyze_distributions(self):
        """Analyser les distributions des variables numériques"""
        print("\n" + "="*80)
        print("📊 ANALYSE DES DISTRIBUTIONS")
        print("="*80)

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('DISTRIBUTIONS DES VARIABLES NUMÉRIQUES', fontsize=16, fontweight='bold', y=0.995)

        # 1. Distribution Montant demandé
        ax = axes[0, 0]
        if 'Montant demandé' in self.df.columns:
            data = self.df['Montant demandé'][self.df['Montant demandé'] > 0]
            ax.hist(data, bins=50, color='#3498db', alpha=0.7, edgecolor='black')
            ax.axvline(data.median(), color='red', linestyle='--', linewidth=2, label=f'Médiane: {data.median():,.0f}')
            ax.axvline(data.mean(), color='green', linestyle='--', linewidth=2, label=f'Moyenne: {data.mean():,.0f}')
            ax.set_xlabel('Montant demandé (DH)', fontweight='bold')
            ax.set_ylabel('Fréquence', fontweight='bold')
            ax.set_title('Distribution Montant Demandé', fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)

        # 2. Distribution Délai estimé
        ax = axes[0, 1]
        if 'Délai estimé' in self.df.columns:
            data = self.df['Délai estimé'][self.df['Délai estimé'] > 0]
            ax.hist(data, bins=50, color='#2ecc71', alpha=0.7, edgecolor='black')
            ax.axvline(data.median(), color='red', linestyle='--', linewidth=2, label=f'Médiane: {data.median():.1f}')
            ax.axvline(data.mean(), color='green', linestyle='--', linewidth=2, label=f'Moyenne: {data.mean():.1f}')
            ax.set_xlabel('Délai estimé', fontweight='bold')
            ax.set_ylabel('Fréquence', fontweight='bold')
            ax.set_title('Distribution Délai Estimé', fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)

        # 3. Distribution Ancienneté
        ax = axes[0, 2]
        if 'anciennete_annees' in self.df.columns:
            data = self.df['anciennete_annees'][self.df['anciennete_annees'] > 0]
            ax.hist(data, bins=30, color='#e74c3c', alpha=0.7, edgecolor='black')
            ax.axvline(data.median(), color='red', linestyle='--', linewidth=2, label=f'Médiane: {data.median():.1f}')
            ax.axvline(data.mean(), color='green', linestyle='--', linewidth=2, label=f'Moyenne: {data.mean():.1f}')
            ax.set_xlabel('Ancienneté (années)', fontweight='bold')
            ax.set_ylabel('Fréquence', fontweight='bold')
            ax.set_title('Distribution Ancienneté Client', fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)

        # 4. Distribution PNB cumulé
        ax = axes[1, 0]
        if 'PNB analytique (vision commerciale) cumulé' in self.df.columns:
            data = self.df['PNB analytique (vision commerciale) cumulé']
            data = data[data > 0]
            if len(data) > 0:
                ax.hist(data, bins=50, color='#f39c12', alpha=0.7, edgecolor='black')
                ax.axvline(data.median(), color='red', linestyle='--', linewidth=2, label=f'Médiane: {data.median():,.0f}')
                ax.axvline(data.mean(), color='green', linestyle='--', linewidth=2, label=f'Moyenne: {data.mean():,.0f}')
                ax.set_xlabel('PNB cumulé (DH)', fontweight='bold')
                ax.set_ylabel('Fréquence', fontweight='bold')
                ax.set_title('Distribution PNB Cumulé', fontweight='bold')
                ax.legend()
                ax.grid(True, alpha=0.3)

        # 5. Ratio Montant/Délai
        ax = axes[1, 1]
        if 'Montant demandé' in self.df.columns and 'Délai estimé' in self.df.columns:
            df_temp = self.df[(self.df['Montant demandé'] > 0) & (self.df['Délai estimé'] > 0)].copy()
            df_temp['ratio_montant_delai'] = df_temp['Montant demandé'] / df_temp['Délai estimé']
            data = df_temp['ratio_montant_delai']
            # Limiter aux percentiles pour éviter les outliers extrêmes
            data = data[(data > data.quantile(0.01)) & (data < data.quantile(0.99))]

            ax.hist(data, bins=50, color='#9b59b6', alpha=0.7, edgecolor='black')
            ax.axvline(data.median(), color='red', linestyle='--', linewidth=2, label=f'Médiane: {data.median():.1f}')
            ax.set_xlabel('Montant / Délai', fontweight='bold')
            ax.set_ylabel('Fréquence', fontweight='bold')
            ax.set_title('Distribution Ratio Montant/Délai', fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)

        # 6. Box plot comparatif
        ax = axes[1, 2]
        if 'Montant demandé' in self.df.columns:
            data_to_plot = []
            labels = []

            if 'Montant demandé' in self.df.columns:
                data_to_plot.append(self.df['Montant demandé'][self.df['Montant demandé'] > 0])
                labels.append('Montant')

            if 'PNB analytique (vision commerciale) cumulé' in self.df.columns:
                pnb_data = self.df['PNB analytique (vision commerciale) cumulé']
                if len(pnb_data[pnb_data > 0]) > 0:
                    data_to_plot.append(pnb_data[pnb_data > 0])
                    labels.append('PNB')

            if data_to_plot:
                bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True)
                for patch, color in zip(bp['boxes'], ['#3498db', '#f39c12']):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.7)

                ax.set_ylabel('Montant (DH)', fontweight='bold')
                ax.set_title('Comparaison Montant vs PNB', fontweight='bold')
                ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        output_path = self.output_dir / '01_distributions.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ Sauvegardé: {output_path}")
        plt.close()

    def analyze_by_family(self):
        """Analyser les métriques par famille de produit"""
        print("\n" + "="*80)
        print("📊 ANALYSE PAR FAMILLE DE PRODUIT")
        print("="*80)

        if 'Famille Produit' not in self.df.columns:
            print("⚠️  Colonne 'Famille Produit' non disponible")
            return

        fig, axes = plt.subplots(2, 2, figsize=(18, 12))
        fig.suptitle('ANALYSE PAR FAMILLE DE PRODUIT', fontsize=16, fontweight='bold', y=0.995)

        # 1. Montant moyen par famille
        ax = axes[0, 0]
        if 'Montant demandé' in self.df.columns:
            family_stats = self.df.groupby('Famille Produit').agg({
                'Montant demandé': ['mean', 'count']
            })
            family_stats.columns = ['montant_moyen', 'count']
            family_stats = family_stats[family_stats['count'] >= 10]  # Au moins 10 cas
            family_stats = family_stats.sort_values('montant_moyen', ascending=False).head(15)

            colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(family_stats)))
            bars = ax.barh(range(len(family_stats)), family_stats['montant_moyen'], color=colors, alpha=0.8)
            ax.set_yticks(range(len(family_stats)))
            ax.set_yticklabels(family_stats.index, fontsize=9)
            ax.set_xlabel('Montant moyen (DH)', fontweight='bold')
            ax.set_title('Top 15 - Montant Moyen par Famille (n≥10)', fontweight='bold')
            ax.grid(True, alpha=0.3, axis='x')

            # Ajouter valeurs
            for i, (idx, row) in enumerate(family_stats.iterrows()):
                ax.text(row['montant_moyen'] + row['montant_moyen']*0.02, i,
                       f"{row['montant_moyen']:,.0f} DH\n(n={int(row['count'])})",
                       va='center', fontsize=8)

        # 2. Volume par famille
        ax = axes[0, 1]
        family_counts = self.df['Famille Produit'].value_counts().head(15)

        colors = plt.cm.plasma(np.linspace(0.3, 0.9, len(family_counts)))
        bars = ax.barh(range(len(family_counts)), family_counts.values, color=colors, alpha=0.8)
        ax.set_yticks(range(len(family_counts)))
        ax.set_yticklabels(family_counts.index, fontsize=9)
        ax.set_xlabel('Nombre de réclamations', fontweight='bold')
        ax.set_title('Top 15 - Volume par Famille', fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')

        # Ajouter valeurs
        for i, val in enumerate(family_counts.values):
            pct = 100 * val / len(self.df)
            ax.text(val + val*0.02, i, f'{val} ({pct:.1f}%)', va='center', fontsize=8, fontweight='bold')

        # 3. PNB moyen par famille
        ax = axes[1, 0]
        if 'PNB analytique (vision commerciale) cumulé' in self.df.columns:
            df_temp = self.df[self.df['PNB analytique (vision commerciale) cumulé'] > 0].copy()
            pnb_by_family = df_temp.groupby('Famille Produit').agg({
                'PNB analytique (vision commerciale) cumulé': ['mean', 'count']
            })
            pnb_by_family.columns = ['pnb_moyen', 'count']
            pnb_by_family = pnb_by_family[pnb_by_family['count'] >= 10]
            pnb_by_family = pnb_by_family.sort_values('pnb_moyen', ascending=False).head(15)

            colors = plt.cm.coolwarm(np.linspace(0.3, 0.9, len(pnb_by_family)))
            bars = ax.barh(range(len(pnb_by_family)), pnb_by_family['pnb_moyen'], color=colors, alpha=0.8)
            ax.set_yticks(range(len(pnb_by_family)))
            ax.set_yticklabels(pnb_by_family.index, fontsize=9)
            ax.set_xlabel('PNB moyen (DH)', fontweight='bold')
            ax.set_title('Top 15 - PNB Moyen par Famille (n≥10)', fontweight='bold')
            ax.grid(True, alpha=0.3, axis='x')

            for i, (idx, row) in enumerate(pnb_by_family.iterrows()):
                ax.text(row['pnb_moyen'] + row['pnb_moyen']*0.02, i,
                       f"{row['pnb_moyen']:,.0f}\n(n={int(row['count'])})",
                       va='center', fontsize=8)

        # 4. Délai moyen par famille
        ax = axes[1, 1]
        if 'Délai estimé' in self.df.columns:
            df_temp = self.df[self.df['Délai estimé'] > 0].copy()
            delai_by_family = df_temp.groupby('Famille Produit').agg({
                'Délai estimé': ['mean', 'count']
            })
            delai_by_family.columns = ['delai_moyen', 'count']
            delai_by_family = delai_by_family[delai_by_family['count'] >= 10]
            delai_by_family = delai_by_family.sort_values('delai_moyen', ascending=False).head(15)

            colors = plt.cm.RdYlGn_r(np.linspace(0.3, 0.9, len(delai_by_family)))
            bars = ax.barh(range(len(delai_by_family)), delai_by_family['delai_moyen'], color=colors, alpha=0.8)
            ax.set_yticks(range(len(delai_by_family)))
            ax.set_yticklabels(delai_by_family.index, fontsize=9)
            ax.set_xlabel('Délai moyen', fontweight='bold')
            ax.set_title('Top 15 - Délai Moyen par Famille (n≥10)', fontweight='bold')
            ax.grid(True, alpha=0.3, axis='x')

            for i, (idx, row) in enumerate(delai_by_family.iterrows()):
                ax.text(row['delai_moyen'] + row['delai_moyen']*0.02, i,
                       f"{row['delai_moyen']:.1f}\n(n={int(row['count'])})",
                       va='center', fontsize=8)

        plt.tight_layout()
        output_path = self.output_dir / '02_analyse_famille.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ Sauvegardé: {output_path}")
        plt.close()

    def analyze_correlations(self):
        """Analyser les corrélations entre variables"""
        print("\n" + "="*80)
        print("📊 ANALYSE DES CORRÉLATIONS")
        print("="*80)

        fig, axes = plt.subplots(2, 2, figsize=(18, 12))
        fig.suptitle('RELATIONS ENTRE VARIABLES', fontsize=16, fontweight='bold', y=0.995)

        # 1. Montant vs Ancienneté
        ax = axes[0, 0]
        if 'Montant demandé' in self.df.columns and 'anciennete_annees' in self.df.columns:
            df_temp = self.df[(self.df['Montant demandé'] > 0) & (self.df['anciennete_annees'] > 0)].copy()

            if len(df_temp) > 10:  # Au moins 10 points pour une corrélation significative
                # Limiter aux percentiles pour meilleure visualisation
                df_temp = df_temp[
                    (df_temp['Montant demandé'] <= df_temp['Montant demandé'].quantile(0.95)) &
                    (df_temp['anciennete_annees'] <= df_temp['anciennete_annees'].quantile(0.95))
                ]

                if len(df_temp) > 10:
                    ax.scatter(df_temp['anciennete_annees'], df_temp['Montant demandé'],
                              alpha=0.3, s=30, color='#3498db')

                    # Ligne de tendance
                    z = np.polyfit(df_temp['anciennete_annees'], df_temp['Montant demandé'], 1)
                    p = np.poly1d(z)
                    x_line = np.linspace(df_temp['anciennete_annees'].min(), df_temp['anciennete_annees'].max(), 100)
                    ax.plot(x_line, p(x_line), "r--", linewidth=2, label='Tendance')

                    # Corrélation
                    corr = df_temp['anciennete_annees'].corr(df_temp['Montant demandé'])
                    ax.text(0.05, 0.95, f'Corrélation: {corr:.3f}',
                           transform=ax.transAxes, fontsize=11, fontweight='bold',
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

                    ax.set_xlabel('Ancienneté (années)', fontweight='bold')
                    ax.set_ylabel('Montant demandé (DH)', fontweight='bold')
                    ax.set_title('Montant vs Ancienneté Client', fontweight='bold')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                else:
                    ax.text(0.5, 0.5, 'Données insuffisantes\n(< 10 points)',
                           transform=ax.transAxes, ha='center', va='center', fontsize=12)
                    ax.set_title('Montant vs Ancienneté Client', fontweight='bold')
            else:
                ax.text(0.5, 0.5, 'Données insuffisantes',
                       transform=ax.transAxes, ha='center', va='center', fontsize=12)
                ax.set_title('Montant vs Ancienneté Client', fontweight='bold')

        # 2. Montant vs PNB
        ax = axes[0, 1]
        if 'Montant demandé' in self.df.columns and 'PNB analytique (vision commerciale) cumulé' in self.df.columns:
            df_temp = self.df[
                (self.df['Montant demandé'] > 0) &
                (self.df['PNB analytique (vision commerciale) cumulé'] > 0)
            ].copy()

            if len(df_temp) > 10:
                # Limiter aux percentiles
                df_temp = df_temp[
                    (df_temp['Montant demandé'] <= df_temp['Montant demandé'].quantile(0.95)) &
                    (df_temp['PNB analytique (vision commerciale) cumulé'] <= df_temp['PNB analytique (vision commerciale) cumulé'].quantile(0.95))
                ]

                if len(df_temp) > 10:
                    ax.scatter(df_temp['PNB analytique (vision commerciale) cumulé'], df_temp['Montant demandé'],
                              alpha=0.3, s=30, color='#2ecc71')

                    # Ligne de tendance
                    z = np.polyfit(df_temp['PNB analytique (vision commerciale) cumulé'], df_temp['Montant demandé'], 1)
                    p = np.poly1d(z)
                    x_line = np.linspace(df_temp['PNB analytique (vision commerciale) cumulé'].min(),
                                       df_temp['PNB analytique (vision commerciale) cumulé'].max(), 100)
                    ax.plot(x_line, p(x_line), "r--", linewidth=2, label='Tendance')

                    # Corrélation
                    corr = df_temp['PNB analytique (vision commerciale) cumulé'].corr(df_temp['Montant demandé'])
                    ax.text(0.05, 0.95, f'Corrélation: {corr:.3f}',
                           transform=ax.transAxes, fontsize=11, fontweight='bold',
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

                    ax.set_xlabel('PNB cumulé (DH)', fontweight='bold')
                    ax.set_ylabel('Montant demandé (DH)', fontweight='bold')
                    ax.set_title('Montant vs PNB Cumulé', fontweight='bold')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                else:
                    ax.text(0.5, 0.5, 'Données insuffisantes\n(< 10 points)',
                           transform=ax.transAxes, ha='center', va='center', fontsize=12)
                    ax.set_title('Montant vs PNB Cumulé', fontweight='bold')
            else:
                ax.text(0.5, 0.5, 'Données insuffisantes',
                       transform=ax.transAxes, ha='center', va='center', fontsize=12)
                ax.set_title('Montant vs PNB Cumulé', fontweight='bold')

        # 3. Délai vs Montant
        ax = axes[1, 0]
        if 'Montant demandé' in self.df.columns and 'Délai estimé' in self.df.columns:
            df_temp = self.df[(self.df['Montant demandé'] > 0) & (self.df['Délai estimé'] > 0)].copy()

            if len(df_temp) > 10:
                # Limiter aux percentiles
                df_temp = df_temp[
                    (df_temp['Montant demandé'] <= df_temp['Montant demandé'].quantile(0.95)) &
                    (df_temp['Délai estimé'] <= df_temp['Délai estimé'].quantile(0.95))
                ]

                if len(df_temp) > 10:
                    ax.scatter(df_temp['Délai estimé'], df_temp['Montant demandé'],
                              alpha=0.3, s=30, color='#e74c3c')

                    # Ligne de tendance
                    z = np.polyfit(df_temp['Délai estimé'], df_temp['Montant demandé'], 1)
                    p = np.poly1d(z)
                    x_line = np.linspace(df_temp['Délai estimé'].min(), df_temp['Délai estimé'].max(), 100)
                    ax.plot(x_line, p(x_line), "r--", linewidth=2, label='Tendance')

                    # Corrélation
                    corr = df_temp['Délai estimé'].corr(df_temp['Montant demandé'])
                    ax.text(0.05, 0.95, f'Corrélation: {corr:.3f}',
                           transform=ax.transAxes, fontsize=11, fontweight='bold',
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

                    ax.set_xlabel('Délai estimé', fontweight='bold')
                    ax.set_ylabel('Montant demandé (DH)', fontweight='bold')
                    ax.set_title('Montant vs Délai Estimé', fontweight='bold')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                else:
                    ax.text(0.5, 0.5, 'Données insuffisantes\n(< 10 points)',
                           transform=ax.transAxes, ha='center', va='center', fontsize=12)
                    ax.set_title('Montant vs Délai Estimé', fontweight='bold')
            else:
                ax.text(0.5, 0.5, 'Données insuffisantes',
                       transform=ax.transAxes, ha='center', va='center', fontsize=12)
                ax.set_title('Montant vs Délai Estimé', fontweight='bold')

        # 4. PNB vs Ancienneté
        ax = axes[1, 1]
        if 'anciennete_annees' in self.df.columns and 'PNB analytique (vision commerciale) cumulé' in self.df.columns:
            df_temp = self.df[
                (self.df['anciennete_annees'] > 0) &
                (self.df['PNB analytique (vision commerciale) cumulé'] > 0)
            ].copy()

            if len(df_temp) > 10:
                # Limiter aux percentiles
                df_temp = df_temp[
                    (df_temp['anciennete_annees'] <= df_temp['anciennete_annees'].quantile(0.95)) &
                    (df_temp['PNB analytique (vision commerciale) cumulé'] <= df_temp['PNB analytique (vision commerciale) cumulé'].quantile(0.95))
                ]

                if len(df_temp) > 10:
                    ax.scatter(df_temp['anciennete_annees'], df_temp['PNB analytique (vision commerciale) cumulé'],
                              alpha=0.3, s=30, color='#f39c12')

                    # Ligne de tendance
                    z = np.polyfit(df_temp['anciennete_annees'], df_temp['PNB analytique (vision commerciale) cumulé'], 1)
                    p = np.poly1d(z)
                    x_line = np.linspace(df_temp['anciennete_annees'].min(), df_temp['anciennete_annees'].max(), 100)
                    ax.plot(x_line, p(x_line), "r--", linewidth=2, label='Tendance')

                    # Corrélation
                    corr = df_temp['anciennete_annees'].corr(df_temp['PNB analytique (vision commerciale) cumulé'])
                    ax.text(0.05, 0.95, f'Corrélation: {corr:.3f}',
                           transform=ax.transAxes, fontsize=11, fontweight='bold',
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

                    ax.set_xlabel('Ancienneté (années)', fontweight='bold')
                    ax.set_ylabel('PNB cumulé (DH)', fontweight='bold')
                    ax.set_title('PNB vs Ancienneté Client', fontweight='bold')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                else:
                    ax.text(0.5, 0.5, 'Données insuffisantes\n(< 10 points)',
                           transform=ax.transAxes, ha='center', va='center', fontsize=12)
                    ax.set_title('PNB vs Ancienneté Client', fontweight='bold')
            else:
                ax.text(0.5, 0.5, 'Données insuffisantes',
                       transform=ax.transAxes, ha='center', va='center', fontsize=12)
                ax.set_title('PNB vs Ancienneté Client', fontweight='bold')

        plt.tight_layout()
        output_path = self.output_dir / '03_correlations.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ Sauvegardé: {output_path}")
        plt.close()

    def analyze_with_predictions(self):
        """Analyser les profils selon les prédictions du modèle"""
        if not self.with_predictions or 'Decision_Modele' not in self.df.columns:
            return

        print("\n" + "="*80)
        print("📊 ANALYSE PAR DÉCISION DU MODÈLE")
        print("="*80)

        fig, axes = plt.subplots(2, 2, figsize=(18, 12))
        fig.suptitle('PROFILS PAR DÉCISION DU MODÈLE', fontsize=16, fontweight='bold', y=0.995)

        # 1. Montant moyen par décision
        ax = axes[0, 0]
        if 'Montant demandé' in self.df.columns:
            decision_stats = self.df.groupby('Decision_Modele').agg({
                'Montant demandé': ['mean', 'median', 'count']
            })
            decision_stats.columns = ['mean', 'median', 'count']

            x = range(len(decision_stats))
            width = 0.35

            bars1 = ax.bar([i - width/2 for i in x], decision_stats['mean'],
                          width, label='Moyenne', color='#3498db', alpha=0.8)
            bars2 = ax.bar([i + width/2 for i in x], decision_stats['median'],
                          width, label='Médiane', color='#2ecc71', alpha=0.8)

            ax.set_ylabel('Montant (DH)', fontweight='bold')
            ax.set_title('Montant par Décision', fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(decision_stats.index, rotation=15, ha='right')
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')

            # Ajouter valeurs
            for bars in [bars1, bars2]:
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:,.0f}', ha='center', va='bottom', fontsize=8)

        # 2. Ancienneté moyenne par décision
        ax = axes[0, 1]
        if 'anciennete_annees' in self.df.columns:
            anc_by_decision = self.df.groupby('Decision_Modele')['anciennete_annees'].agg(['mean', 'count'])

            colors = ['#e74c3c', '#f39c12', '#2ecc71']
            bars = ax.bar(range(len(anc_by_decision)), anc_by_decision['mean'],
                         color=colors[:len(anc_by_decision)], alpha=0.8)

            ax.set_ylabel('Ancienneté moyenne (années)', fontweight='bold')
            ax.set_title('Ancienneté par Décision', fontweight='bold')
            ax.set_xticks(range(len(anc_by_decision)))
            ax.set_xticklabels(anc_by_decision.index, rotation=15, ha='right')
            ax.grid(True, alpha=0.3, axis='y')

            for i, (bar, count) in enumerate(zip(bars, anc_by_decision['count'])):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f}\n(n={int(count)})',
                       ha='center', va='bottom', fontsize=9, fontweight='bold')

        # 3. Distribution des décisions par famille (top 10)
        ax = axes[1, 0]
        if 'Famille Produit' in self.df.columns:
            top_families = self.df['Famille Produit'].value_counts().head(10).index
            df_top = self.df[self.df['Famille Produit'].isin(top_families)]

            decision_counts = pd.crosstab(df_top['Famille Produit'], df_top['Decision_Modele'])
            decision_counts.plot(kind='barh', stacked=True, ax=ax,
                                color=['#e74c3c', '#f39c12', '#2ecc71'], alpha=0.8)

            ax.set_xlabel('Nombre de réclamations', fontweight='bold')
            ax.set_ylabel('Famille Produit', fontweight='bold')
            ax.set_title('Distribution Décisions par Famille (Top 10)', fontweight='bold')
            ax.legend(title='Décision', bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.grid(True, alpha=0.3, axis='x')

        # 4. PNB moyen par décision
        ax = axes[1, 1]
        if 'PNB analytique (vision commerciale) cumulé' in self.df.columns:
            df_pnb = self.df[self.df['PNB analytique (vision commerciale) cumulé'] > 0]
            pnb_by_decision = df_pnb.groupby('Decision_Modele')['PNB analytique (vision commerciale) cumulé'].agg(['mean', 'count'])

            colors = ['#e74c3c', '#f39c12', '#2ecc71']
            bars = ax.bar(range(len(pnb_by_decision)), pnb_by_decision['mean'],
                         color=colors[:len(pnb_by_decision)], alpha=0.8)

            ax.set_ylabel('PNB moyen (DH)', fontweight='bold')
            ax.set_title('PNB par Décision', fontweight='bold')
            ax.set_xticks(range(len(pnb_by_decision)))
            ax.set_xticklabels(pnb_by_decision.index, rotation=15, ha='right')
            ax.grid(True, alpha=0.3, axis='y')

            for i, (bar, count) in enumerate(zip(bars, pnb_by_decision['count'])):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:,.0f}\n(n={int(count)})',
                       ha='center', va='bottom', fontsize=9, fontweight='bold')

        plt.tight_layout()
        output_path = self.output_dir / '04_profils_decisions.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ Sauvegardé: {output_path}")
        plt.close()

    def generate_summary_report(self):
        """Générer rapport récapitulatif"""
        print("\n" + "="*80)
        print("📄 GÉNÉRATION DU RAPPORT")
        print("="*80)

        report_path = self.output_dir / f'rapport_profils_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt'

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("RAPPORT D'ANALYSE DES PROFILS DE RÉCLAMATIONS\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*80 + "\n\n")

            f.write(f"Nombre total de réclamations: {len(self.df)}\n\n")

            # Statistiques globales
            f.write("STATISTIQUES GLOBALES:\n")
            f.write("-" * 80 + "\n")

            if 'Montant demandé' in self.df.columns:
                data = self.df['Montant demandé'][self.df['Montant demandé'] > 0]
                f.write(f"Montant demandé:\n")
                f.write(f"  Moyenne: {data.mean():,.2f} DH\n")
                f.write(f"  Médiane: {data.median():,.2f} DH\n")
                f.write(f"  Min: {data.min():,.2f} DH\n")
                f.write(f"  Max: {data.max():,.2f} DH\n\n")

            if 'anciennete_annees' in self.df.columns:
                data = self.df['anciennete_annees'][self.df['anciennete_annees'] > 0]
                f.write(f"Ancienneté client:\n")
                f.write(f"  Moyenne: {data.mean():.2f} années\n")
                f.write(f"  Médiane: {data.median():.2f} années\n\n")

            # Top familles
            if 'Famille Produit' in self.df.columns:
                f.write("\nTOP 10 FAMILLES DE PRODUITS:\n")
                f.write("-" * 80 + "\n")
                top_families = self.df['Famille Produit'].value_counts().head(10)
                for i, (famille, count) in enumerate(top_families.items(), 1):
                    pct = 100 * count / len(self.df)
                    f.write(f"{i:2d}. {famille:40s}: {count:6d} ({pct:5.1f}%)\n")

            # Statistiques par décision (si disponibles)
            if self.with_predictions and 'Decision_Modele' in self.df.columns:
                f.write("\n\nSTATISTIQUES PAR DÉCISION:\n")
                f.write("-" * 80 + "\n")

                for decision in ['Rejet Auto', 'Audit Humain', 'Validation Auto']:
                    df_dec = self.df[self.df['Decision_Modele'] == decision]
                    if len(df_dec) > 0:
                        f.write(f"\n{decision}:\n")
                        f.write(f"  Nombre: {len(df_dec)}\n")

                        if 'Montant demandé' in df_dec.columns:
                            montants = df_dec['Montant demandé'][df_dec['Montant demandé'] > 0]
                            if len(montants) > 0:
                                f.write(f"  Montant moyen: {montants.mean():,.2f} DH\n")
                                f.write(f"  Montant médian: {montants.median():,.2f} DH\n")

        print(f"✅ Rapport généré: {report_path}")

    def run(self):
        """Exécuter toutes les analyses"""
        self.load_data()
        self.analyze_distributions()
        self.analyze_by_family()
        self.analyze_correlations()

        if self.with_predictions:
            self.analyze_with_predictions()

        self.generate_summary_report()

        print("\n" + "="*80)
        print("✅ ANALYSE DES PROFILS TERMINÉE")
        print("="*80)
        print(f"\n📂 Tous les graphiques sont dans: {self.output_dir}")
        print("\nFichiers générés:")
        print("   - 01_distributions.png       : Distributions des variables")
        print("   - 02_analyse_famille.png     : Analyse par famille")
        print("   - 03_correlations.png        : Corrélations entre variables")
        if self.with_predictions:
            print("   - 04_profils_decisions.png   : Profils par décision modèle")
        print("   - rapport_profils_*.txt      : Rapport récapitulatif")


def main():
    parser = argparse.ArgumentParser(description='Analyse exploratoire des profils de réclamations')
    parser.add_argument('--input_file', type=str, required=True, help='Fichier Excel avec les réclamations')
    parser.add_argument('--with_predictions', action='store_true', help='Le fichier contient les prédictions du modèle')

    args = parser.parse_args()

    analyzer = ClaimProfileAnalyzer(args.input_file, args.with_predictions)
    analyzer.run()


if __name__ == '__main__':
    main()
