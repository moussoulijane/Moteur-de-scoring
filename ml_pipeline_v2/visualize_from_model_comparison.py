#!/usr/bin/env python3
"""
VISUALISATION À PARTIR DES RÉSULTATS DE MODEL_COMPARISON_V2
Utilise directement les prédictions sauvegardées par model_comparison_v2.py
pour garantir la cohérence des métriques.

Les 3 graphiques générés:
1. Accuracy + Taux d'automatisation + Accuracy par top familles
2. Gain en montant uniquement (GAIN NET)
3. Impact des règles métier (1 validation/an + PNB > montant)

Usage:
    # D'abord exécuter model_comparison_v2.py pour générer les prédictions
    python ml_pipeline_v2/model_comparison_v2.py

    # Puis visualiser les résultats
    python ml_pipeline_v2/visualize_from_model_comparison.py
"""

import sys
sys.path.append('src')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import joblib
import warnings
warnings.filterwarnings('ignore')

# Configuration
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (16, 10)

# Paramètres de gain (identiques à model_comparison_v2.py)
PRIX_UNITAIRE_DH = 169


class VisualizerFromModelComparison:
    """Générateur de visualisations à partir des résultats de model_comparison_v2"""

    def __init__(self):
        self.output_dir = Path('outputs/results_model_comparison')
        self.output_dir.mkdir(parents=True, exist_ok=True)

        print("\n" + "="*80)
        print("📊 VISUALISATION DES RÉSULTATS MODEL_COMPARISON_V2")
        print("="*80)

    def load_predictions_and_data(self):
        """Charger les prédictions et les données 2025"""
        print("\n📂 Chargement des prédictions et données...")

        # Charger les prédictions sauvegardées par model_comparison_v2
        predictions_path = Path('outputs/production_v2/predictions/predictions_2025_v2.pkl')

        if not predictions_path.exists():
            print(f"❌ ERREUR: Fichier de prédictions introuvable: {predictions_path}")
            print("\n💡 Veuillez d'abord exécuter:")
            print("   python ml_pipeline_v2/model_comparison_v2.py")
            sys.exit(1)

        predictions_data = joblib.load(predictions_path)
        print(f"✅ Prédictions chargées depuis {predictions_path}")

        # Extraire les données du meilleur modèle
        best_model_name = predictions_data['best_model']
        print(f"🏆 Meilleur modèle: {best_model_name}")

        self.y_prob = predictions_data[best_model_name]['y_prob']
        self.threshold_low = predictions_data[best_model_name]['threshold_low']
        self.threshold_high = predictions_data[best_model_name]['threshold_high']
        self.y_true = predictions_data['y_true']

        print(f"   Seuil BAS:  {self.threshold_low:.4f}")
        print(f"   Seuil HAUT: {self.threshold_high:.4f}")

        # Charger les données 2025 pour les informations contextuelles
        data_path = Path('data/raw/reclamations_2025.xlsx')
        if not data_path.exists():
            print(f"⚠️  Fichier de données introuvable: {data_path}")
            print("   Tentative avec un autre chemin...")
            data_path = Path('ml_pipeline/data/raw/reclamations_2025.xlsx')

        if not data_path.exists():
            print(f"❌ ERREUR: Fichier de données 2025 introuvable")
            sys.exit(1)

        self.df_2025 = pd.read_excel(data_path)
        print(f"✅ Données 2025 chargées: {len(self.df_2025)} réclamations")

        # Vérifier la cohérence
        if len(self.df_2025) != len(self.y_true):
            print(f"⚠️  ATTENTION: Taille différente entre données ({len(self.df_2025)}) et prédictions ({len(self.y_true)})")

        # Créer les décisions à partir des prédictions optimisées
        self.create_decisions()

        # Ajouter les décisions au DataFrame
        self.df_2025['Probabilite_Fondee'] = self.y_prob
        self.df_2025['Decision_Modele'] = self.decisions
        self.df_2025['Fondee_bool'] = self.y_true

    def create_decisions(self):
        """Créer les décisions à partir des probabilités et seuils optimisés"""
        print("\n📊 Création des décisions avec seuils optimisés...")

        self.decisions = []
        self.decision_codes = []

        for prob in self.y_prob:
            if prob <= self.threshold_low:
                self.decisions.append('Rejet Auto')
                self.decision_codes.append(-1)
            elif prob >= self.threshold_high:
                self.decisions.append('Validation Auto')
                self.decision_codes.append(1)
            else:
                self.decisions.append('Audit Humain')
                self.decision_codes.append(0)

        n_rejet = self.decision_codes.count(-1)
        n_audit = self.decision_codes.count(0)
        n_validation = self.decision_codes.count(1)

        print(f"   Rejet Auto:      {n_rejet:,} ({100*n_rejet/len(self.y_prob):.1f}%)")
        print(f"   Audit Humain:    {n_audit:,} ({100*n_audit/len(self.y_prob):.1f}%)")
        print(f"   Validation Auto: {n_validation:,} ({100*n_validation/len(self.y_prob):.1f}%)")

    def apply_business_rules(self):
        """Appliquer les règles métier"""
        print("\n" + "="*80)
        print("🔧 APPLICATION DES RÈGLES MÉTIER")
        print("="*80)

        df = self.df_2025.copy()

        # Initialiser
        df['Raison_Audit'] = ''
        df['Decision_Finale'] = df['Decision_Modele']

        # Règle #1: Maximum 1 validation par client par an
        print("\n📋 Règle #1: Maximum 1 validation par client par an")

        code_client_col = None
        for col in ['Code Client', 'idtfcl', 'code_client', 'client_id']:
            if col in df.columns:
                code_client_col = col
                break

        n_rule1 = 0
        if code_client_col:
            if 'Date Création réclamation' in df.columns:
                df_sorted = df.sort_values('Date Création réclamation')
            else:
                df_sorted = df.copy()

            validation_mask = df_sorted['Decision_Modele'] == 'Validation Auto'
            df_sorted['validation_count'] = df_sorted.groupby(code_client_col)['Decision_Modele'].transform(
                lambda x: (x == 'Validation Auto').cumsum()
            )

            rule1_mask = validation_mask & (df_sorted['validation_count'] > 1)
            n_rule1 = rule1_mask.sum()

            df_sorted.loc[rule1_mask, 'Decision_Finale'] = 'Audit Humain'
            df_sorted.loc[rule1_mask, 'Raison_Audit'] = 'Règle #1: >1 validation/client/an'

            print(f"   ✅ {n_rule1:,} cas convertis en Audit")
            df = df_sorted.drop(columns=['validation_count'])
        else:
            print("   ⚠️  Colonne client manquante - Règle #1 ignorée")

        # Règle #2: Montant > PNB (ignorer PNB NaN)
        print("\n📋 Règle #2: Montant demandé > PNB")

        n_rule2 = 0
        if 'Montant demandé' in df.columns and 'PNB analytique (vision commerciale) cumulé' in df.columns:
            validation_mask = df['Decision_Finale'] == 'Validation Auto'
            pnb_valid_mask = df['PNB analytique (vision commerciale) cumulé'].notna()
            montant_valid_mask = df['Montant demandé'].notna()

            rule2_mask = (
                validation_mask &
                pnb_valid_mask &
                montant_valid_mask &
                (df['Montant demandé'] > df['PNB analytique (vision commerciale) cumulé'])
            )

            n_rule2 = rule2_mask.sum()

            df.loc[rule2_mask, 'Decision_Finale'] = 'Audit Humain'
            df.loc[rule2_mask & (df['Raison_Audit'] != ''), 'Raison_Audit'] += ' + '
            df.loc[rule2_mask, 'Raison_Audit'] += 'Règle #2: Montant > PNB'

            print(f"   ✅ {n_rule2:,} cas convertis en Audit")

            n_pnb_nan = df['PNB analytique (vision commerciale) cumulé'].isna().sum()
            print(f"   ℹ️  {n_pnb_nan:,} cas avec PNB NaN (décision conservée)")
        else:
            print("   ⚠️  Colonnes manquantes - Règle #2 ignorée")

        # Stats finales
        print("\n📊 Décisions finales après règles métier:")
        n_rejet_final = (df['Decision_Finale'] == 'Rejet Auto').sum()
        n_audit_final = (df['Decision_Finale'] == 'Audit Humain').sum()
        n_validation_final = (df['Decision_Finale'] == 'Validation Auto').sum()

        print(f"   Rejet Auto:      {n_rejet_final:,} ({100*n_rejet_final/len(df):.1f}%)")
        print(f"   Audit Humain:    {n_audit_final:,} ({100*n_audit_final/len(df):.1f}%)")
        print(f"   Validation Auto: {n_validation_final:,} ({100*n_validation_final/len(df):.1f}%)")

        self.df_2025 = df

    def plot_accuracy_automation_families(self):
        """Graphique 1: Accuracy + Automatisation + Top Familles"""
        print("\n📊 Graphique 1: Performance globale et par famille...")

        fig = plt.figure(figsize=(20, 10))
        fig.suptitle('PERFORMANCE GLOBALE ET PAR FAMILLE - 2025 (Model Comparison V2)',
                     fontsize=20, fontweight='bold', y=0.98)

        # Préparer les données
        df_copy = self.df_2025.copy()
        df_copy['Validation_bool'] = df_copy['Decision_Finale'].apply(
            lambda x: 1 if x == 'Validation Auto' else 0
        )

        # Calculs globaux
        n_total = len(df_copy)
        n_rejet = (df_copy['Decision_Finale'] == 'Rejet Auto').sum()
        n_validation = (df_copy['Decision_Finale'] == 'Validation Auto').sum()
        n_auto = n_rejet + n_validation
        taux_auto = 100 * n_auto / n_total

        # Accuracy globale
        vp = ((df_copy['Fondee_bool'] == 1) & (df_copy['Validation_bool'] == 1)).sum()
        vn = ((df_copy['Fondee_bool'] == 0) & (df_copy['Validation_bool'] == 0)).sum()
        fp = ((df_copy['Fondee_bool'] == 0) & (df_copy['Validation_bool'] == 1)).sum()
        fn = ((df_copy['Fondee_bool'] == 1) & (df_copy['Validation_bool'] == 0)).sum()

        accuracy_globale = 100 * (vp + vn) / (vp + vn + fp + fn) if (vp + vn + fp + fn) > 0 else 0
        precision = 100 * vp / (vp + fp) if (vp + fp) > 0 else 0
        recall = 100 * vp / (vp + fn) if (vp + fn) > 0 else 0

        # 1. Métriques globales
        ax1 = plt.subplot(2, 3, 1)
        metrics = ['Accuracy', 'Précision', 'Rappel']
        values = [accuracy_globale, precision, recall]
        colors = ['#27AE60', '#3498DB', '#E67E22']

        bars = ax1.bar(metrics, values, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
        ax1.set_ylabel('Pourcentage (%)', fontweight='bold', fontsize=13)
        ax1.set_title('Métriques de Performance Globales', fontweight='bold', fontsize=15)
        ax1.set_ylim(0, 100)
        ax1.grid(True, alpha=0.3, axis='y')

        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 2,
                    f'{val:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=12)

        # 2. Taux d'automatisation
        ax2 = plt.subplot(2, 3, 2)
        categories = ['Automatisé\n(Rejet + Validation)', 'Audit\nHumain']
        sizes = [n_auto, n_total - n_auto]
        colors_pie = ['#2ECC71', '#F39C12']
        explode = (0.1, 0)

        wedges, texts, autotexts = ax2.pie(sizes, explode=explode, labels=categories,
                                            autopct='%1.1f%%', colors=colors_pie,
                                            shadow=True, startangle=90,
                                            textprops={'fontsize': 11, 'weight': 'bold'})

        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontsize(13)

        ax2.set_title('Taux d\'Automatisation', fontweight='bold', fontsize=15)

        # 3. Stats clés
        ax3 = plt.subplot(2, 3, 3)
        ax3.axis('off')

        stats_text = f"""
📊 STATISTIQUES CLÉS

Total réclamations:  {n_total:,}

AUTOMATISATION:
  • Taux:            {taux_auto:.1f}%
  • Dossiers auto:   {n_auto:,}
  • Audit humain:    {n_total - n_auto:,}

PERFORMANCE:
  • Accuracy:        {accuracy_globale:.1f}%
  • Précision:       {precision:.1f}%
  • Rappel:          {recall:.1f}%

CONFUSION:
  • VP: {vp:,}    VN: {vn:,}
  • FP: {fp:,}    FN: {fn:,}

SEUILS OPTIMISÉS:
  • Bas:  {self.threshold_low:.4f}
  • Haut: {self.threshold_high:.4f}
        """

        ax3.text(0.05, 0.95, stats_text, transform=ax3.transAxes,
                fontsize=10, verticalalignment='top', family='monospace',
                bbox=dict(boxstyle='round', facecolor='#E8F8F5', alpha=0.9,
                         edgecolor='#16A085', linewidth=2))

        # 4-6. Accuracy par top familles
        if 'Famille Produit' in df_copy.columns:
            top_families = df_copy['Famille Produit'].value_counts().head(10)

            accuracies = []
            volumes = []
            precisions = []

            for famille in top_families.index:
                df_fam = df_copy[df_copy['Famille Produit'] == famille]
                n_fam = len(df_fam)

                vp_fam = ((df_fam['Fondee_bool'] == 1) & (df_fam['Validation_bool'] == 1)).sum()
                vn_fam = ((df_fam['Fondee_bool'] == 0) & (df_fam['Validation_bool'] == 0)).sum()
                fp_fam = ((df_fam['Fondee_bool'] == 0) & (df_fam['Validation_bool'] == 1)).sum()
                fn_fam = ((df_fam['Fondee_bool'] == 1) & (df_fam['Validation_bool'] == 0)).sum()

                total_fam = vp_fam + vn_fam + fp_fam + fn_fam
                acc_fam = 100 * (vp_fam + vn_fam) / total_fam if total_fam > 0 else 0
                prec_fam = 100 * vp_fam / (vp_fam + fp_fam) if (vp_fam + fp_fam) > 0 else 0

                accuracies.append(acc_fam)
                volumes.append(n_fam)
                precisions.append(prec_fam)

            # Graph 4: Accuracy par famille
            ax4 = plt.subplot(2, 3, 4)
            colors_fam = plt.cm.RdYlGn(np.array(accuracies) / 100)
            bars = ax4.barh(range(len(top_families)), accuracies, color=colors_fam,
                           edgecolor='black', linewidth=1.5)

            ax4.set_yticks(range(len(top_families)))
            ax4.set_yticklabels([f[:30] for f in top_families.index], fontsize=10)
            ax4.set_xlabel('Accuracy (%)', fontweight='bold', fontsize=12)
            ax4.set_title('Accuracy par Famille Produit (Top 10)', fontweight='bold', fontsize=14)
            ax4.set_xlim(0, 100)
            ax4.grid(True, alpha=0.3, axis='x')

            for bar, acc in zip(bars, accuracies):
                width = bar.get_width()
                ax4.text(width + 2, bar.get_y() + bar.get_height()/2.,
                        f'{acc:.1f}%', ha='left', va='center', fontweight='bold', fontsize=9)

            # Graph 5: Volume par famille
            ax5 = plt.subplot(2, 3, 5)
            bars = ax5.barh(range(len(top_families)), volumes, color='#3498DB',
                           alpha=0.8, edgecolor='black', linewidth=1.5)

            ax5.set_yticks(range(len(top_families)))
            ax5.set_yticklabels([f[:30] for f in top_families.index], fontsize=10)
            ax5.set_xlabel('Nombre de réclamations', fontweight='bold', fontsize=12)
            ax5.set_title('Volume par Famille Produit (Top 10)', fontweight='bold', fontsize=14)
            ax5.grid(True, alpha=0.3, axis='x')

            for bar, vol in zip(bars, volumes):
                width = bar.get_width()
                ax5.text(width + max(volumes)*0.02, bar.get_y() + bar.get_height()/2.,
                        f'{int(vol):,}', ha='left', va='center', fontweight='bold', fontsize=9)

            # Graph 6: Précision par famille
            ax6 = plt.subplot(2, 3, 6)
            colors_prec = plt.cm.RdYlGn(np.array(precisions) / 100)
            bars = ax6.barh(range(len(top_families)), precisions, color=colors_prec,
                           edgecolor='black', linewidth=1.5)

            ax6.set_yticks(range(len(top_families)))
            ax6.set_yticklabels([f[:30] for f in top_families.index], fontsize=10)
            ax6.set_xlabel('Précision (%)', fontweight='bold', fontsize=12)
            ax6.set_title('Précision par Famille Produit (Top 10)', fontweight='bold', fontsize=14)
            ax6.set_xlim(0, 100)
            ax6.grid(True, alpha=0.3, axis='x')

            for bar, prec in zip(bars, precisions):
                width = bar.get_width()
                ax6.text(width + 2, bar.get_y() + bar.get_height()/2.,
                        f'{prec:.1f}%', ha='left', va='center', fontweight='bold', fontsize=9)

        plt.tight_layout()
        output_path = self.output_dir / 'G1_accuracy_automation_families_v2.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ Sauvegardé: {output_path}")
        plt.close()

    def plot_gain_montant_only(self):
        """Graphique 2: Gain en montant (GAIN NET)"""
        print("\n📊 Graphique 2: Gain financier NET...")

        if 'Montant demandé' not in self.df_2025.columns:
            print("⚠️  Colonne 'Montant demandé' manquante - Graphique ignoré")
            return

        fig = plt.figure(figsize=(20, 10))
        fig.suptitle('GAIN FINANCIER NET - 2025 (Model Comparison V2)',
                     fontsize=20, fontweight='bold', y=0.98)

        # Préparer les données
        df_copy = self.df_2025.copy()
        df_copy['Validation_bool'] = df_copy['Decision_Finale'].apply(
            lambda x: 1 if x == 'Validation Auto' else 0
        )

        # Calculs
        n_total = len(df_copy)
        n_rejet = (df_copy['Decision_Finale'] == 'Rejet Auto').sum()
        n_validation = (df_copy['Decision_Finale'] == 'Validation Auto').sum()
        n_auto = n_rejet + n_validation

        # GAIN BRUT
        gain_brut = n_auto * PRIX_UNITAIRE_DH

        # Masques pour cas automatisés
        mask_auto = (df_copy['Decision_Finale'] == 'Rejet Auto') | (df_copy['Decision_Finale'] == 'Validation Auto')
        df_auto = df_copy[mask_auto]

        # FP et FN
        fp_mask = (df_auto['Fondee_bool'] == 0) & (df_auto['Validation_bool'] == 1)
        fn_mask = (df_auto['Fondee_bool'] == 1) & (df_auto['Validation_bool'] == 0)

        n_fp = fp_mask.sum()
        n_fn = fn_mask.sum()

        # Nettoyer montants
        montants_auto = df_auto['Montant demandé'].values
        montants_clean = np.nan_to_num(montants_auto, nan=0.0, posinf=0.0, neginf=0.0)
        montants_clean = np.clip(montants_clean, 0,
                                np.percentile(montants_clean[montants_clean > 0], 99)
                                if (montants_clean > 0).any() else 0)

        # Calcul pertes
        perte_fp = montants_clean[fp_mask.values].sum()
        perte_fn = 2 * montants_clean[fn_mask.values].sum()

        # GAIN NET
        gain_net = gain_brut - perte_fp - perte_fn

        # 1. Cascade du gain
        ax1 = plt.subplot(2, 3, 1)
        categories = ['Gain\nBrut', 'Perte\nFP', 'Perte\nFN', 'GAIN\nNET']
        values = [gain_brut, -perte_fp, -perte_fn, gain_net]
        values_m = [v / 1e6 for v in values]
        colors_bars = ['#2ECC71', '#E74C3C', '#E67E22', '#27AE60']

        bars = ax1.bar(categories, values_m, color=colors_bars,
                      alpha=0.8, edgecolor='black', linewidth=2)

        ax1.set_ylabel('Millions DH', fontweight='bold', fontsize=13)
        ax1.set_title('Cascade du Gain Net', fontweight='bold', fontsize=15)
        ax1.grid(True, alpha=0.3, axis='y')
        ax1.axhline(y=0, color='black', linestyle='-', linewidth=1.5)

        for bar, val_m, val in zip(bars, values_m, values):
            height = bar.get_height()
            va = 'bottom' if val_m >= 0 else 'top'
            y_pos = height if val_m >= 0 else height
            ax1.text(bar.get_x() + bar.get_width()/2., y_pos,
                    f'{abs(val_m):.2f}M\n({abs(val):,.0f} DH)',
                    ha='center', va=va, fontweight='bold', fontsize=10)

        # 2. Répartition du gain brut
        ax2 = plt.subplot(2, 3, 2)
        parts = ['Rejet Auto\n(économie)', 'Validation Auto\n(économie)']
        gains_parts = [n_rejet * PRIX_UNITAIRE_DH, n_validation * PRIX_UNITAIRE_DH]
        colors_parts = ['#E74C3C', '#2ECC71']

        wedges, texts, autotexts = ax2.pie(gains_parts, labels=parts,
                                            autopct='%1.1f%%', colors=colors_parts,
                                            shadow=True, startangle=90,
                                            textprops={'fontsize': 11, 'weight': 'bold'})

        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontsize(12)

        ax2.set_title(f'Composition du Gain Brut\n({gain_brut/1e6:.2f}M DH)',
                     fontweight='bold', fontsize=15)

        # 3. Détails des pertes
        ax3 = plt.subplot(2, 3, 3)
        if n_fp + n_fn > 0:
            perte_categories = ['FP\n(accordé à tort)', 'FN\n(refusé à tort × 2)']
            perte_values = [perte_fp / 1e6, perte_fn / 1e6]
            perte_colors = ['#E74C3C', '#E67E22']

            bars = ax3.bar(perte_categories, perte_values, color=perte_colors,
                          alpha=0.8, edgecolor='black', linewidth=2)

            ax3.set_ylabel('Millions DH', fontweight='bold', fontsize=13)
            ax3.set_title('Détail des Pertes', fontweight='bold', fontsize=15)
            ax3.grid(True, alpha=0.3, axis='y')

            for bar, val in zip(bars, perte_values):
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height,
                        f'{val:.2f}M', ha='center', va='bottom', fontweight='bold', fontsize=11)
        else:
            ax3.text(0.5, 0.5, 'Aucune perte\n(Performance parfaite!)',
                    ha='center', va='center', transform=ax3.transAxes,
                    fontsize=16, fontweight='bold', color='#2ECC71')
            ax3.axis('off')

        # 4. Schéma de calcul
        ax4 = plt.subplot(2, 3, 4)
        ax4.axis('off')
        ax4.set_xlim(0, 10)
        ax4.set_ylim(0, 10)

        ax4.text(5, 9.5, 'FORMULE DU GAIN NET', ha='center',
                fontsize=14, fontweight='bold')

        # Box 1
        rect1 = plt.Rectangle((1, 7), 8, 1.2, facecolor='#2ECC71',
                              edgecolor='black', linewidth=2)
        ax4.add_patch(rect1)
        ax4.text(5, 7.6, f'GAIN BRUT = {n_auto:,} dossiers × {PRIX_UNITAIRE_DH} DH = {gain_brut:,.0f} DH',
                ha='center', va='center', fontsize=11, fontweight='bold', color='white')

        ax4.text(5, 6.3, '−', ha='center', va='center', fontsize=30, fontweight='bold')

        # Box 2
        rect2 = plt.Rectangle((1, 5), 8, 0.9, facecolor='#E74C3C',
                              edgecolor='black', linewidth=2)
        ax4.add_patch(rect2)
        ax4.text(5, 5.45, f'PERTE FP = {n_fp:,} cas × montants = {perte_fp:,.0f} DH',
                ha='center', va='center', fontsize=10, fontweight='bold', color='white')

        ax4.text(5, 4.3, '−', ha='center', va='center', fontsize=30, fontweight='bold')

        # Box 3
        rect3 = plt.Rectangle((1, 3), 8, 0.9, facecolor='#E67E22',
                              edgecolor='black', linewidth=2)
        ax4.add_patch(rect3)
        ax4.text(5, 3.45, f'PERTE FN = {n_fn:,} cas × 2 × montants = {perte_fn:,.0f} DH',
                ha='center', va='center', fontsize=10, fontweight='bold', color='white')

        ax4.text(5, 2.3, '=', ha='center', va='center', fontsize=30, fontweight='bold')

        # Résultat
        rect_final = plt.Rectangle((1, 0.5), 8, 1.3, facecolor='#27AE60',
                                   edgecolor='black', linewidth=3)
        ax4.add_patch(rect_final)
        ax4.text(5, 1.3, 'GAIN NET FINAL', ha='center', va='center',
                fontsize=14, fontweight='bold', color='white')
        ax4.text(5, 0.8, f'{gain_net:,.0f} DH = {gain_net/1e6:.2f} Millions DH',
                ha='center', va='center', fontsize=13, fontweight='bold', color='white')

        # 5. Stats récap
        ax5 = plt.subplot(2, 3, 5)
        ax5.axis('off')

        stats_text = f"""
💰 RÉCAPITULATIF FINANCIER

DOSSIERS:
  • Total:              {n_total:,}
  • Automatisés:        {n_auto:,} ({100*n_auto/n_total:.1f}%)
    - Rejet:            {n_rejet:,}
    - Validation:       {n_validation:,}

GAINS:
  • Gain Brut:          {gain_brut:,.0f} DH
                        = {gain_brut/1e6:.2f} M DH

PERTES:
  • FP ({n_fp:,} cas):      {perte_fp:,.0f} DH
                        = {perte_fp/1e6:.2f} M DH
  • FN ({n_fn:,} cas):      {perte_fn:,.0f} DH
                        = {perte_fn/1e6:.2f} M DH
  • Total pertes:       {perte_fp + perte_fn:,.0f} DH

RÉSULTAT:
  • GAIN NET:           {gain_net:,.0f} DH
                        = {gain_net/1e6:.2f} M DH
        """

        ax5.text(0.05, 0.95, stats_text, transform=ax5.transAxes,
                fontsize=10, verticalalignment='top', family='monospace',
                bbox=dict(boxstyle='round', facecolor='#D5F4E6', alpha=0.9,
                         edgecolor='#2ECC71', linewidth=3))

        # 6. Comparaison
        ax6 = plt.subplot(2, 3, 6)
        comparison_data = ['Gain\nBrut', 'Pertes\nTotales', 'GAIN\nNET']
        comparison_values = [gain_brut/1e6, (perte_fp + perte_fn)/1e6, gain_net/1e6]
        comparison_colors = ['#2ECC71', '#E74C3C', '#27AE60']

        bars = ax6.bar(comparison_data, comparison_values, color=comparison_colors,
                      alpha=0.8, edgecolor='black', linewidth=2)

        ax6.set_ylabel('Millions DH', fontweight='bold', fontsize=13)
        ax6.set_title('Comparaison Gains vs Pertes', fontweight='bold', fontsize=15)
        ax6.grid(True, alpha=0.3, axis='y')

        for bar, val in zip(bars, comparison_values):
            height = bar.get_height()
            ax6.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.2f}M', ha='center', va='bottom', fontweight='bold', fontsize=12)

        plt.tight_layout(rect=[0, 0, 0.98, 0.96])
        output_path = self.output_dir / 'G2_gain_montant_net_v2.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ Sauvegardé: {output_path}")
        plt.close()

    def plot_business_rules_impact(self):
        """Graphique 3: Impact des règles métier"""
        print("\n📊 Graphique 3: Impact des règles métier...")

        fig = plt.figure(figsize=(20, 10))
        fig.suptitle('IMPACT DES RÈGLES MÉTIER - 2025 (Model Comparison V2)',
                     fontsize=20, fontweight='bold', y=0.98)

        # Compter les cas convertis
        n_total = len(self.df_2025)
        n_regle1 = self.df_2025['Raison_Audit'].str.contains('Règle #1', na=False).sum()
        n_regle2 = self.df_2025['Raison_Audit'].str.contains('Règle #2', na=False).sum()
        n_both = self.df_2025['Raison_Audit'].str.contains('\\+', na=False, regex=True).sum()

        # Décisions avant et après
        n_validation_avant = (self.df_2025['Decision_Modele'] == 'Validation Auto').sum()
        n_validation_apres = (self.df_2025['Decision_Finale'] == 'Validation Auto').sum()
        n_converties = n_validation_avant - n_validation_apres

        # 1. Nombre de cas convertis
        ax1 = plt.subplot(2, 3, 1)
        rules = ['Règle #1\n(>1 validation/an)', 'Règle #2\n(Montant > PNB)', 'Les 2 règles']
        counts = [n_regle1 - n_both, n_regle2 - n_both, n_both]
        colors_rules = ['#3498DB', '#E67E22', '#9B59B6']

        bars = ax1.bar(rules, counts, color=colors_rules,
                      alpha=0.8, edgecolor='black', linewidth=2)

        ax1.set_ylabel('Nombre de cas', fontweight='bold', fontsize=13)
        ax1.set_title('Cas Convertis par Règle', fontweight='bold', fontsize=15)
        ax1.grid(True, alpha=0.3, axis='y')

        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(count):,}', ha='center', va='bottom', fontweight='bold', fontsize=11)

        # 2. Avant / Après
        ax2 = plt.subplot(2, 3, 2)
        decisions_avant = ['Validation\nAuto', 'Autres']
        values_avant = [n_validation_avant, n_total - n_validation_avant]
        values_apres = [n_validation_apres, n_total - n_validation_apres]

        x = np.arange(len(decisions_avant))
        width = 0.35

        bars1 = ax2.bar(x - width/2, values_avant, width, label='Avant règles',
                       color='#95A5A6', alpha=0.8, edgecolor='black', linewidth=1.5)
        bars2 = ax2.bar(x + width/2, values_apres, width, label='Après règles',
                       color='#2ECC71', alpha=0.8, edgecolor='black', linewidth=1.5)

        ax2.set_ylabel('Nombre de réclamations', fontweight='bold', fontsize=13)
        ax2.set_title('Avant / Après Règles Métier', fontweight='bold', fontsize=15)
        ax2.set_xticks(x)
        ax2.set_xticklabels(decisions_avant)
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3, axis='y')

        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height):,}', ha='center', va='bottom', fontweight='bold', fontsize=10)

        # 3. Pourcentage de conversion
        ax3 = plt.subplot(2, 3, 3)
        pct_conversion = 100 * n_converties / n_validation_avant if n_validation_avant > 0 else 0
        pct_conserve = 100 - pct_conversion

        sizes = [pct_conserve, pct_conversion]
        labels = ['Conservées', 'Converties\nen Audit']
        colors_conv = ['#2ECC71', '#E67E22']
        explode = (0, 0.1)

        wedges, texts, autotexts = ax3.pie(sizes, explode=explode, labels=labels,
                                            autopct='%1.1f%%', colors=colors_conv,
                                            shadow=True, startangle=90,
                                            textprops={'fontsize': 12, 'weight': 'bold'})

        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontsize(13)

        ax3.set_title(f'Impact sur Validations Auto\n({n_validation_avant:,} validations initiales)',
                     fontweight='bold', fontsize=15)

        # 4. Stats détaillées
        ax4 = plt.subplot(2, 3, 4)
        ax4.axis('off')

        stats_text = f"""
📋 STATISTIQUES DES RÈGLES

AVANT RÈGLES MÉTIER:
  • Validation Auto:    {n_validation_avant:,}
  • Autres décisions:   {n_total - n_validation_avant:,}

CONVERSIONS:
  • Règle #1 seule:     {n_regle1 - n_both:,}
  • Règle #2 seule:     {n_regle2 - n_both:,}
  • Les 2 règles:       {n_both:,}
  • TOTAL converti:     {n_converties:,}

APRÈS RÈGLES MÉTIER:
  • Validation Auto:    {n_validation_apres:,}
  • Audit Humain:       {n_total - n_validation_apres:,}

TAUX:
  • Conversion:         {pct_conversion:.1f}%
  • Conservation:       {pct_conserve:.1f}%
        """

        ax4.text(0.05, 0.95, stats_text, transform=ax4.transAxes,
                fontsize=11, verticalalignment='top', family='monospace',
                bbox=dict(boxstyle='round', facecolor='#FEF9E7', alpha=0.9,
                         edgecolor='#F39C12', linewidth=2))

        # 5. Montants protégés
        ax5 = plt.subplot(2, 3, 5)
        if 'Montant demandé' in self.df_2025.columns:
            mask_r1 = self.df_2025['Raison_Audit'].str.contains('Règle #1', na=False)
            mask_r2 = self.df_2025['Raison_Audit'].str.contains('Règle #2', na=False)
            mask_both = self.df_2025['Raison_Audit'].str.contains('\\+', na=False, regex=True)

            montant_r1 = self.df_2025[mask_r1 & ~mask_both]['Montant demandé'].sum() / 1e6
            montant_r2 = self.df_2025[mask_r2 & ~mask_both]['Montant demandé'].sum() / 1e6
            montant_both = self.df_2025[mask_both]['Montant demandé'].sum() / 1e6

            montants = [montant_r1, montant_r2, montant_both]

            bars = ax5.bar(rules, montants, color=colors_rules,
                          alpha=0.8, edgecolor='black', linewidth=2)

            ax5.set_ylabel('Montant (Millions DH)', fontweight='bold', fontsize=13)
            ax5.set_title('Montants Protégés par Règle', fontweight='bold', fontsize=15)
            ax5.grid(True, alpha=0.3, axis='y')

            for bar, mt in zip(bars, montants):
                height = bar.get_height()
                ax5.text(bar.get_x() + bar.get_width()/2., height,
                        f'{mt:.2f}M', ha='center', va='bottom', fontweight='bold', fontsize=11)

        # 6. Flowchart
        ax6 = plt.subplot(2, 3, 6)
        ax6.axis('off')
        ax6.set_xlim(0, 10)
        ax6.set_ylim(0, 10)

        ax6.text(5, 9.5, 'LOGIQUE DES RÈGLES MÉTIER', ha='center',
                fontsize=13, fontweight='bold')

        # Box initial
        rect_init = plt.Rectangle((2, 7.5), 6, 1, facecolor='#95A5A6',
                                  edgecolor='black', linewidth=2)
        ax6.add_patch(rect_init)
        ax6.text(5, 8, f'VALIDATION AUTO (Modèle)\n{n_validation_avant:,} cas',
                ha='center', va='center', fontsize=10, fontweight='bold', color='white')

        ax6.arrow(5, 7.3, 0, -0.5, head_width=0.3, head_length=0.2,
                 fc='black', ec='black', linewidth=2)

        # Box règle 1
        rect_r1 = plt.Rectangle((0.5, 5.5), 4, 1, facecolor='#3498DB',
                               edgecolor='black', linewidth=2)
        ax6.add_patch(rect_r1)
        ax6.text(2.5, 6.2, 'RÈGLE #1', ha='center', va='center',
                fontsize=10, fontweight='bold', color='white')
        ax6.text(2.5, 5.8, f'>1 validation/an\n{n_regle1:,} cas',
                ha='center', va='center', fontsize=9, fontweight='bold', color='white')

        # Box règle 2
        rect_r2 = plt.Rectangle((5.5, 5.5), 4, 1, facecolor='#E67E22',
                               edgecolor='black', linewidth=2)
        ax6.add_patch(rect_r2)
        ax6.text(7.5, 6.2, 'RÈGLE #2', ha='center', va='center',
                fontsize=10, fontweight='bold', color='white')
        ax6.text(7.5, 5.8, f'Montant > PNB\n{n_regle2:,} cas',
                ha='center', va='center', fontsize=9, fontweight='bold', color='white')

        # Flèches convergentes
        ax6.arrow(2.5, 5.3, 1.5, -1, head_width=0.2, head_length=0.15,
                 fc='black', ec='black', linewidth=1.5)
        ax6.arrow(7.5, 5.3, -1.5, -1, head_width=0.2, head_length=0.15,
                 fc='black', ec='black', linewidth=1.5)

        # Box conversion
        rect_conv = plt.Rectangle((2.5, 2.5), 5, 1, facecolor='#E67E22',
                                  edgecolor='black', linewidth=2)
        ax6.add_patch(rect_conv)
        ax6.text(5, 3, f'CONVERTI EN AUDIT\n{n_converties:,} cas',
                ha='center', va='center', fontsize=10, fontweight='bold', color='white')

        ax6.arrow(5, 2.3, 0, -0.5, head_width=0.3, head_length=0.2,
                 fc='black', ec='black', linewidth=2)

        # Box finale
        rect_final = plt.Rectangle((2, 0.5), 6, 1, facecolor='#2ECC71',
                                   edgecolor='black', linewidth=3)
        ax6.add_patch(rect_final)
        ax6.text(5, 1.2, 'VALIDATION AUTO FINALE', ha='center', va='center',
                fontsize=11, fontweight='bold', color='white')
        ax6.text(5, 0.8, f'{n_validation_apres:,} cas ({100*n_validation_apres/n_total:.1f}%)',
                ha='center', va='center', fontsize=10, fontweight='bold', color='white')

        plt.tight_layout(rect=[0, 0, 0.98, 0.96])
        output_path = self.output_dir / 'G3_business_rules_impact_v2.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ Sauvegardé: {output_path}")
        plt.close()

    def run(self):
        """Exécuter la génération complète"""
        self.load_predictions_and_data()
        self.apply_business_rules()
        self.plot_accuracy_automation_families()
        self.plot_gain_montant_only()
        self.plot_business_rules_impact()

        print("\n" + "="*80)
        print("✅ GÉNÉRATION TERMINÉE")
        print("="*80)
        print(f"\n📂 Fichiers dans: {self.output_dir}")
        print("\nGraphiques générés:")
        print("  - G1: Accuracy + Automatisation + Top Familles")
        print("  - G2: Gain en Montant (GAIN NET)")
        print("  - G3: Impact des Règles Métier")
        print()


def main():
    visualizer = VisualizerFromModelComparison()
    visualizer.run()


if __name__ == '__main__':
    main()
