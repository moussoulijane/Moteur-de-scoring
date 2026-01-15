"""
Script de Visualisation des Résultats 2025
- Analyse par famille produit (succès)
- Analyse des faux positifs (montants)
- Quantification pertes/gains (169 DH par réclamation automatisée)
"""
import sys
sys.path.append('src')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import joblib
import json

# Configuration
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 10

# Couleurs
COLORS = {
    'success': '#2ecc71',
    'error': '#e74c3c',
    'warning': '#f39c12',
    'info': '#3498db',
    'neutral': '#95a5a6'
}

PRIX_UNITAIRE_DH = 169  # Prix de traitement manuel d'une réclamation


class ResultsVisualizer2025:
    """Visualisation complète des résultats 2025"""

    def __init__(self, data_path, predictions_path=None, model_path=None):
        self.data_path = data_path
        self.predictions_path = predictions_path
        self.model_path = model_path
        self.df = None
        self.y_true = None
        self.y_pred = None
        self.y_prob = None

    def load_data(self):
        """Charge les données et prédictions"""
        print("📂 Chargement des données 2025...")
        self.df = pd.read_excel(self.data_path)
        print(f"✅ Chargé: {len(self.df)} réclamations")

        # Charger les prédictions si fournies
        if self.predictions_path and Path(self.predictions_path).exists():
            preds = joblib.load(self.predictions_path)
            self.y_true = preds['y_true']
            self.y_pred = preds['y_pred']
            self.y_prob = preds['y_prob']
            print(f"✅ Prédictions chargées: {len(self.y_true)} échantillons")
        else:
            # Sinon, utiliser les vraies valeurs seulement
            self.y_true = self.df['Fondee'].values
            print("⚠️  Pas de prédictions - analyse sur vraies valeurs uniquement")

    def analyze_by_family(self):
        """Analyse détaillée par famille produit"""
        print("\n📊 Analyse par Famille Produit...")

        if 'Famille Produit' not in self.df.columns:
            print("❌ Colonne 'Famille Produit' non trouvée")
            return None

        # Créer DataFrame d'analyse
        df_analysis = self.df.copy()
        df_analysis['y_true'] = self.y_true

        if self.y_pred is not None:
            df_analysis['y_pred'] = self.y_pred
            df_analysis['y_prob'] = self.y_prob

            # Calculer métriques par famille
            results = []
            for family in df_analysis['Famille Produit'].unique():
                family_data = df_analysis[df_analysis['Famille Produit'] == family]

                y_t = family_data['y_true'].values
                y_p = family_data['y_pred'].values

                # Métriques
                tp = np.sum((y_t == 1) & (y_p == 1))
                tn = np.sum((y_t == 0) & (y_p == 0))
                fp = np.sum((y_t == 0) & (y_p == 1))
                fn = np.sum((y_t == 1) & (y_p == 0))

                accuracy = (tp + tn) / len(y_t) if len(y_t) > 0 else 0
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

                # Montants
                montant_col = 'Montant demandé' if 'Montant demandé' in family_data.columns else 'Montant'
                if montant_col in family_data.columns:
                    avg_montant = family_data[montant_col].mean()
                else:
                    avg_montant = 0

                results.append({
                    'Famille': family,
                    'Volume': len(family_data),
                    'Accuracy': accuracy,
                    'Precision': precision,
                    'Recall': recall,
                    'F1-Score': f1,
                    'TP': tp,
                    'TN': tn,
                    'FP': fp,
                    'FN': fn,
                    'Montant_Moyen': avg_montant,
                    'Taux_Fondees_Reel': y_t.mean(),
                    'Taux_Fondees_Pred': y_p.mean()
                })

            df_metrics = pd.DataFrame(results)
            df_metrics = df_metrics.sort_values('F1-Score', ascending=False)

        else:
            # Sans prédictions, analyse descriptive seulement
            results = []
            for family in df_analysis['Famille Produit'].unique():
                family_data = df_analysis[df_analysis['Famille Produit'] == family]

                montant_col = 'Montant demandé' if 'Montant demandé' in family_data.columns else 'Montant'
                if montant_col in family_data.columns:
                    avg_montant = family_data[montant_col].mean()
                else:
                    avg_montant = 0

                results.append({
                    'Famille': family,
                    'Volume': len(family_data),
                    'Taux_Fondees_Reel': family_data['y_true'].mean(),
                    'Montant_Moyen': avg_montant
                })

            df_metrics = pd.DataFrame(results)
            df_metrics = df_metrics.sort_values('Volume', ascending=False)

        print(f"\n✅ Analyse terminée pour {len(df_metrics)} familles")
        return df_metrics

    def analyze_false_positives(self):
        """Analyse détaillée des faux positifs (FP = Prédit Fondée mais vraiment Non Fondée)"""
        print("\n📊 Analyse des Faux Positifs...")

        if self.y_pred is None:
            print("❌ Pas de prédictions disponibles")
            return None

        # Identifier les faux positifs
        fp_mask = (self.y_true == 0) & (self.y_pred == 1)
        df_fp = self.df[fp_mask].copy()

        print(f"⚠️  Faux Positifs détectés: {len(df_fp)} / {len(self.df)} ({100*len(df_fp)/len(self.df):.1f}%)")

        # Analyse par montant
        montant_col = 'Montant demandé' if 'Montant demandé' in df_fp.columns else 'Montant'

        if montant_col in df_fp.columns:
            df_fp['Montant'] = df_fp[montant_col]

            fp_analysis = {
                'count': len(df_fp),
                'montant_total': df_fp['Montant'].sum(),
                'montant_moyen': df_fp['Montant'].mean(),
                'montant_median': df_fp['Montant'].median(),
                'montant_std': df_fp['Montant'].std(),
                'montant_min': df_fp['Montant'].min(),
                'montant_max': df_fp['Montant'].max()
            }

            # Distribution par tranches
            df_fp['Tranche_Montant'] = pd.cut(
                df_fp['Montant'],
                bins=[0, 100, 500, 1000, 5000, 10000, np.inf],
                labels=['0-100 DH', '100-500 DH', '500-1k DH', '1k-5k DH', '5k-10k DH', '>10k DH']
            )

            fp_analysis['distribution_tranches'] = df_fp['Tranche_Montant'].value_counts().to_dict()

            # Par famille
            if 'Famille Produit' in df_fp.columns:
                fp_analysis['par_famille'] = df_fp.groupby('Famille Produit').agg({
                    'Montant': ['count', 'sum', 'mean']
                }).to_dict()

        else:
            fp_analysis = {'count': len(df_fp)}

        return df_fp, fp_analysis

    def calculate_financial_impact(self):
        """Calcule l'impact financier : pertes (FN) et gains (automatisation)"""
        print("\n💰 Calcul de l'Impact Financier...")

        if self.y_pred is None:
            print("❌ Pas de prédictions disponibles")
            return None

        # Confusion matrix
        tp = np.sum((self.y_true == 1) & (self.y_pred == 1))
        tn = np.sum((self.y_true == 0) & (self.y_pred == 0))
        fp = np.sum((self.y_true == 0) & (self.y_pred == 1))
        fn = np.sum((self.y_true == 1) & (self.y_pred == 0))

        # PERTES : Faux Négatifs (réclamations fondées non détectées)
        # Client mécontent + risque de réclamation escaladée
        fn_mask = (self.y_true == 1) & (self.y_pred == 0)
        df_fn = self.df[fn_mask].copy()

        montant_col = 'Montant demandé' if 'Montant demandé' in self.df.columns else 'Montant'

        if montant_col in df_fn.columns:
            perte_montant_total = df_fn[montant_col].sum()
            perte_montant_moyen = df_fn[montant_col].mean()
        else:
            perte_montant_total = 0
            perte_montant_moyen = 0

        # GAINS : Réclamations automatisées (TN + TP correctement classifiés)
        # Économie = nombre de réclamations bien traitées × prix unitaire
        reclamations_automatisees = tp + tn
        gain_automatisation = reclamations_automatisees * PRIX_UNITAIRE_DH

        # Coût des erreurs
        # FP : Coût de traitement inutile
        cout_fp = fp * PRIX_UNITAIRE_DH

        # FN : Coût client mécontent + re-traitement (estimé à 2x le coût normal)
        cout_fn = fn * (2 * PRIX_UNITAIRE_DH)

        # Bilan net
        gain_net = gain_automatisation - cout_fp - cout_fn

        impact = {
            'total_reclamations': len(self.df),
            'reclamations_automatisees': reclamations_automatisees,
            'taux_automatisation': reclamations_automatisees / len(self.df),

            # Confusion matrix
            'tp': tp,
            'tn': tn,
            'fp': fp,
            'fn': fn,

            # Pertes (Faux Négatifs)
            'nb_faux_negatifs': fn,
            'perte_montant_total_dh': perte_montant_total,
            'perte_montant_moyen_dh': perte_montant_moyen,
            'cout_fn_dh': cout_fn,

            # Coûts (Faux Positifs)
            'nb_faux_positifs': fp,
            'cout_fp_dh': cout_fp,

            # Gains
            'gain_automatisation_dh': gain_automatisation,
            'gain_net_dh': gain_net,
            'prix_unitaire_dh': PRIX_UNITAIRE_DH,

            # ROI
            'roi_pct': (gain_net / (cout_fp + cout_fn)) * 100 if (cout_fp + cout_fn) > 0 else 0
        }

        print(f"\n💵 Réclamations automatisées: {reclamations_automatisees} ({100*impact['taux_automatisation']:.1f}%)")
        print(f"💰 Gain brut: {gain_automatisation:,.0f} DH")
        print(f"❌ Coût FP: {cout_fp:,.0f} DH")
        print(f"❌ Coût FN: {cout_fn:,.0f} DH")
        print(f"✅ GAIN NET: {gain_net:,.0f} DH")
        print(f"📈 ROI: {impact['roi_pct']:.1f}%")

        return impact

    def plot_family_success(self, df_metrics, save_path):
        """Graphique des familles avec meilleurs succès"""
        print("\n📊 Création graphique succès par famille...")

        if df_metrics is None or len(df_metrics) == 0:
            print("❌ Pas de données à visualiser")
            return

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('🏆 Performance par Famille Produit - 2025', fontsize=16, fontweight='bold')

        # Top familles par F1-Score (si disponible)
        if 'F1-Score' in df_metrics.columns:
            ax = axes[0, 0]
            top_families = df_metrics.nlargest(8, 'F1-Score')
            bars = ax.barh(top_families['Famille'], top_families['F1-Score'], color=COLORS['success'])
            ax.set_xlabel('F1-Score', fontweight='bold')
            ax.set_title('🥇 Top 8 Familles - Meilleur F1-Score', fontweight='bold')
            ax.set_xlim(0, 1)

            # Ajouter valeurs sur barres
            for i, (bar, val) in enumerate(zip(bars, top_families['F1-Score'])):
                ax.text(val + 0.02, bar.get_y() + bar.get_height()/2,
                       f'{val:.3f}', va='center', fontweight='bold')

        # Volume par famille
        ax = axes[0, 1]
        top_volume = df_metrics.nlargest(8, 'Volume')
        bars = ax.barh(top_volume['Famille'], top_volume['Volume'], color=COLORS['info'])
        ax.set_xlabel('Nombre de Réclamations', fontweight='bold')
        ax.set_title('📊 Top 8 Familles - Volume', fontweight='bold')

        for i, (bar, val) in enumerate(zip(bars, top_volume['Volume'])):
            ax.text(val + 10, bar.get_y() + bar.get_height()/2,
                   f'{int(val)}', va='center', fontweight='bold')

        # Taux de fondées réel vs prédit (si disponible)
        ax = axes[1, 0]
        if 'Taux_Fondees_Pred' in df_metrics.columns:
            top_8 = df_metrics.nlargest(8, 'Volume')
            x = np.arange(len(top_8))
            width = 0.35

            ax.bar(x - width/2, top_8['Taux_Fondees_Reel'], width,
                  label='Réel', color=COLORS['warning'], alpha=0.8)
            ax.bar(x + width/2, top_8['Taux_Fondees_Pred'], width,
                  label='Prédit', color=COLORS['info'], alpha=0.8)

            ax.set_ylabel('Taux de Fondées', fontweight='bold')
            ax.set_title('📈 Taux Fondées: Réel vs Prédit', fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(top_8['Famille'], rotation=45, ha='right')
            ax.legend()
            ax.set_ylim(0, 1)
        else:
            top_8 = df_metrics.nlargest(8, 'Volume')
            ax.barh(top_8['Famille'], top_8['Taux_Fondees_Reel'], color=COLORS['warning'])
            ax.set_xlabel('Taux de Fondées', fontweight='bold')
            ax.set_title('📈 Taux de Réclamations Fondées', fontweight='bold')
            ax.set_xlim(0, 1)

        # Montant moyen par famille
        ax = axes[1, 1]
        if 'Montant_Moyen' in df_metrics.columns:
            top_montant = df_metrics.nlargest(8, 'Montant_Moyen')
            bars = ax.barh(top_montant['Famille'], top_montant['Montant_Moyen'],
                          color=COLORS['warning'])
            ax.set_xlabel('Montant Moyen (DH)', fontweight='bold')
            ax.set_title('💵 Montant Moyen par Famille', fontweight='bold')

            for i, (bar, val) in enumerate(zip(bars, top_montant['Montant_Moyen'])):
                ax.text(val + 50, bar.get_y() + bar.get_height()/2,
                       f'{val:,.0f} DH', va='center', fontsize=9)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Graphique sauvegardé: {save_path}")
        plt.close()

    def plot_false_positives_analysis(self, df_fp, fp_analysis, save_path):
        """Graphique analyse des faux positifs"""
        print("\n📊 Création graphique faux positifs...")

        if df_fp is None or len(df_fp) == 0:
            print("❌ Pas de faux positifs à visualiser")
            return

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('⚠️ Analyse des Faux Positifs - 2025', fontsize=16, fontweight='bold')

        # Distribution par tranches de montant
        ax = axes[0, 0]
        if 'Tranche_Montant' in df_fp.columns:
            tranche_counts = df_fp['Tranche_Montant'].value_counts().sort_index()
            bars = ax.bar(range(len(tranche_counts)), tranche_counts.values, color=COLORS['error'])
            ax.set_xticks(range(len(tranche_counts)))
            ax.set_xticklabels(tranche_counts.index, rotation=45, ha='right')
            ax.set_ylabel('Nombre de Faux Positifs', fontweight='bold')
            ax.set_title('📊 Distribution par Tranche de Montant', fontweight='bold')

            for bar, val in zip(bars, tranche_counts.values):
                ax.text(bar.get_x() + bar.get_width()/2, val + 1,
                       f'{int(val)}', ha='center', va='bottom', fontweight='bold')

        # Boxplot des montants
        ax = axes[0, 1]
        if 'Montant' in df_fp.columns:
            bp = ax.boxplot([df_fp['Montant'].dropna()], vert=True, patch_artist=True,
                           labels=['Faux Positifs'])
            bp['boxes'][0].set_facecolor(COLORS['error'])
            ax.set_ylabel('Montant (DH)', fontweight='bold')
            ax.set_title(f'📦 Distribution des Montants\nMoyenne: {fp_analysis["montant_moyen"]:,.0f} DH',
                        fontweight='bold')
            ax.grid(True, alpha=0.3)

        # Top familles avec le plus de FP
        ax = axes[1, 0]
        if 'Famille Produit' in df_fp.columns:
            fp_by_family = df_fp['Famille Produit'].value_counts().head(8)
            bars = ax.barh(range(len(fp_by_family)), fp_by_family.values, color=COLORS['error'])
            ax.set_yticks(range(len(fp_by_family)))
            ax.set_yticklabels(fp_by_family.index)
            ax.set_xlabel('Nombre de Faux Positifs', fontweight='bold')
            ax.set_title('🏢 Top Familles - Faux Positifs', fontweight='bold')

            for i, (bar, val) in enumerate(zip(bars, fp_by_family.values)):
                ax.text(val + 0.5, bar.get_y() + bar.get_height()/2,
                       f'{int(val)}', va='center', fontweight='bold')

        # Montant total par famille
        ax = axes[1, 1]
        if 'Famille Produit' in df_fp.columns and 'Montant' in df_fp.columns:
            montant_by_family = df_fp.groupby('Famille Produit')['Montant'].sum().sort_values(ascending=False).head(8)
            bars = ax.barh(range(len(montant_by_family)), montant_by_family.values, color=COLORS['warning'])
            ax.set_yticks(range(len(montant_by_family)))
            ax.set_yticklabels(montant_by_family.index)
            ax.set_xlabel('Montant Total FP (DH)', fontweight='bold')
            ax.set_title('💰 Impact Financier par Famille', fontweight='bold')

            for i, (bar, val) in enumerate(zip(bars, montant_by_family.values)):
                ax.text(val + 100, bar.get_y() + bar.get_height()/2,
                       f'{val:,.0f}', va='center', fontsize=9)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Graphique sauvegardé: {save_path}")
        plt.close()

    def plot_financial_impact(self, impact, save_path):
        """Graphique de l'impact financier (pertes vs gains)"""
        print("\n📊 Création graphique impact financier...")

        if impact is None:
            print("❌ Pas de données d'impact")
            return

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('💰 Quantification Financière - Pertes vs Gains', fontsize=16, fontweight='bold')

        # Graphique 1: Confusion Matrix visuelle avec valeurs
        ax = axes[0, 0]
        cm_data = [[impact['tn'], impact['fp']], [impact['fn'], impact['tp']]]
        im = ax.imshow(cm_data, cmap='RdYlGn', alpha=0.6)

        # Ajouter texte
        for i in range(2):
            for j in range(2):
                text = ax.text(j, i, cm_data[i][j], ha="center", va="center",
                             color="black", fontsize=20, fontweight='bold')

        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(['Prédit Non Fondée', 'Prédit Fondée'])
        ax.set_yticklabels(['Vrai Non Fondée', 'Vrai Fondée'])
        ax.set_title('📊 Matrice de Confusion', fontweight='bold', fontsize=12)

        # Graphique 2: Coûts vs Gains (barres)
        ax = axes[0, 1]
        categories = ['Gain\nAutomatisation', 'Coût\nFaux Positifs', 'Coût\nFaux Négatifs', 'Gain\nNet']
        values = [impact['gain_automatisation_dh'], -impact['cout_fp_dh'],
                 -impact['cout_fn_dh'], impact['gain_net_dh']]
        colors_bar = [COLORS['success'], COLORS['error'], COLORS['error'],
                     COLORS['success'] if impact['gain_net_dh'] > 0 else COLORS['error']]

        bars = ax.bar(categories, values, color=colors_bar, alpha=0.7)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        ax.set_ylabel('Montant (DH)', fontweight='bold')
        ax.set_title('💵 Bilan Financier', fontweight='bold')

        # Ajouter valeurs
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:,.0f} DH', ha='center',
                   va='bottom' if height > 0 else 'top',
                   fontweight='bold', fontsize=9)

        # Graphique 3: Répartition des réclamations
        ax = axes[1, 0]
        labels = ['Automatisées\nCorrectement', 'Erreurs\n(FP + FN)']
        sizes = [impact['reclamations_automatisees'], impact['fp'] + impact['fn']]
        colors_pie = [COLORS['success'], COLORS['error']]
        explode = (0.05, 0.05)

        wedges, texts, autotexts = ax.pie(sizes, explode=explode, labels=labels,
                                          colors=colors_pie, autopct='%1.1f%%',
                                          shadow=True, startangle=90)
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
            autotext.set_fontsize(12)

        ax.set_title(f'🎯 Taux d\'Automatisation\n{impact["reclamations_automatisees"]}/{impact["total_reclamations"]} réclamations',
                    fontweight='bold')

        # Graphique 4: Métriques clés
        ax = axes[1, 1]
        ax.axis('off')

        metrics_text = f"""
        📊 MÉTRIQUES CLÉS
        {'='*40}

        🎯 Performance:
           • Réclamations traitées: {impact['total_reclamations']:,}
           • Automatisées correctement: {impact['reclamations_automatisees']:,}
           • Taux d'automatisation: {100*impact['taux_automatisation']:.1f}%

        ❌ Erreurs:
           • Faux Positifs (FP): {impact['fp']}
           • Faux Négatifs (FN): {impact['fn']}

        💰 Impact Financier:
           • Prix unitaire: {PRIX_UNITAIRE_DH} DH
           • Gain brut: {impact['gain_automatisation_dh']:,.0f} DH
           • Coût FP: {impact['cout_fp_dh']:,.0f} DH
           • Coût FN: {impact['cout_fn_dh']:,.0f} DH

        ✅ GAIN NET: {impact['gain_net_dh']:,.0f} DH
        📈 ROI: {impact['roi_pct']:.1f}%
        """

        ax.text(0.1, 0.95, metrics_text, transform=ax.transAxes,
               fontsize=11, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3),
               fontfamily='monospace')

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Graphique sauvegardé: {save_path}")
        plt.close()

    def generate_all_visualizations(self, output_dir='outputs/reports/figures'):
        """Génère toutes les visualisations"""
        print("\n" + "="*60)
        print("🎨 GÉNÉRATION DES VISUALISATIONS 2025")
        print("="*60)

        # Créer dossier de sortie
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Charger données
        self.load_data()

        # 1. Analyse par famille
        df_metrics = self.analyze_by_family()
        if df_metrics is not None:
            self.plot_family_success(df_metrics, f'{output_dir}/family_success_2025.png')
            # Sauvegarder CSV
            df_metrics.to_csv(f'{output_dir}/../family_metrics_2025.csv', index=False)
            print(f"✅ Métriques CSV: {output_dir}/../family_metrics_2025.csv")

        # 2. Analyse faux positifs
        if self.y_pred is not None:
            df_fp, fp_analysis = self.analyze_false_positives()
            if df_fp is not None and len(df_fp) > 0:
                self.plot_false_positives_analysis(df_fp, fp_analysis,
                                                   f'{output_dir}/false_positives_analysis_2025.png')
                # Sauvegarder analyse JSON
                with open(f'{output_dir}/../false_positives_analysis_2025.json', 'w') as f:
                    # Convertir numpy types en python types pour JSON
                    fp_analysis_clean = {k: (int(v) if isinstance(v, (np.integer, np.int64))
                                            else float(v) if isinstance(v, (np.floating, np.float64))
                                            else v)
                                        for k, v in fp_analysis.items() if not isinstance(v, dict)}
                    json.dump(fp_analysis_clean, f, indent=2)
                print(f"✅ Analyse FP JSON: {output_dir}/../false_positives_analysis_2025.json")

        # 3. Impact financier
        if self.y_pred is not None:
            impact = self.calculate_financial_impact()
            if impact is not None:
                self.plot_financial_impact(impact, f'{output_dir}/financial_impact_2025.png')
                # Sauvegarder impact JSON
                with open(f'{output_dir}/../financial_impact_2025.json', 'w') as f:
                    impact_clean = {k: (int(v) if isinstance(v, (np.integer, np.int64))
                                       else float(v) if isinstance(v, (np.floating, np.float64))
                                       else v)
                                   for k, v in impact.items()}
                    json.dump(impact_clean, f, indent=2)
                print(f"✅ Impact financier JSON: {output_dir}/../financial_impact_2025.json")

        print("\n" + "="*60)
        print("✅ VISUALISATIONS TERMINÉES")
        print("="*60)


def main():
    """Point d'entrée principal"""
    print("\n" + "="*60)
    print("🎨 SCRIPT DE VISUALISATION DES RÉSULTATS 2025")
    print("="*60)

    # Configuration
    data_path = 'data/raw/reclamations_2025.xlsx'
    predictions_path = 'outputs/models/predictions_2025.pkl'  # Optionnel
    output_dir = 'outputs/reports/figures'

    # Vérifier si fichier existe
    if not Path(data_path).exists():
        print(f"❌ Fichier non trouvé: {data_path}")
        print("\n💡 Assurez-vous d'avoir:")
        print("   1. Exécuté le pipeline principal (main_pipeline.py)")
        print("   2. Ou placé vos données dans data/raw/reclamations_2025.xlsx")
        return

    # Créer visualiseur
    visualizer = ResultsVisualizer2025(
        data_path=data_path,
        predictions_path=predictions_path if Path(predictions_path).exists() else None
    )

    # Générer toutes les visualisations
    visualizer.generate_all_visualizations(output_dir=output_dir)

    print(f"\n📂 Tous les résultats sont dans: {output_dir}/")
    print("\n✅ Script terminé avec succès!")


if __name__ == '__main__':
    main()
