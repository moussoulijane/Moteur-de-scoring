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

        # Sauvegarder l'état AVANT règles métier pour les graphiques
        self.df_before_rules = df.copy()

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
        """Graphique 1: Accuracy + Automatisation + Top Familles (AVANT règles métier)"""
        print("\n📊 Graphique 1: Performance globale et par famille (AVANT règles métier)...")

        fig = plt.figure(figsize=(20, 10))
        fig.suptitle('PERFORMANCE GLOBALE ET PAR FAMILLE - 2025 (AVANT Règles Métier)',
                     fontsize=20, fontweight='bold', y=0.98)

        # Utiliser les données AVANT l'application des règles métier
        df_copy = self.df_before_rules.copy()
        df_copy['Validation_bool'] = df_copy['Decision_Modele'].apply(
            lambda x: 1 if x == 'Validation Auto' else 0
        )

        # Calculs globaux (AVANT règles métier)
        n_total = len(df_copy)
        n_rejet = (df_copy['Decision_Modele'] == 'Rejet Auto').sum()
        n_validation = (df_copy['Decision_Modele'] == 'Validation Auto').sum()
        n_auto = n_rejet + n_validation
        taux_auto = 100 * n_auto / n_total

        # IMPORTANT: Calculer les métriques UNIQUEMENT sur les cas AUTOMATISÉS
        # (même logique que model_comparison_v2.py lignes 330-338)
        mask_auto = (df_copy['Decision_Modele'] == 'Rejet Auto') | (df_copy['Decision_Modele'] == 'Validation Auto')
        df_auto = df_copy[mask_auto].copy()

        if len(df_auto) > 0:
            # Accuracy globale (sur cas automatisés seulement)
            vp = ((df_auto['Fondee_bool'] == 1) & (df_auto['Validation_bool'] == 1)).sum()
            vn = ((df_auto['Fondee_bool'] == 0) & (df_auto['Validation_bool'] == 0)).sum()
            fp = ((df_auto['Fondee_bool'] == 0) & (df_auto['Validation_bool'] == 1)).sum()
            fn = ((df_auto['Fondee_bool'] == 1) & (df_auto['Validation_bool'] == 0)).sum()

            accuracy_globale = 100 * (vp + vn) / (vp + vn + fp + fn) if (vp + vn + fp + fn) > 0 else 0
            precision = 100 * vp / (vp + fp) if (vp + fp) > 0 else 0
            recall = 100 * vp / (vp + fn) if (vp + fn) > 0 else 0
        else:
            accuracy_globale = precision = recall = 0
            vp = vn = fp = fn = 0

        # 1. Métriques globales
        ax1 = plt.subplot(2, 3, 1)
        metrics = ['Accuracy', 'Précision', 'Rappel']
        values = [accuracy_globale, precision, recall]
        colors = ['#27AE60', '#3498DB', '#E67E22']

        bars = ax1.bar(metrics, values, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
        ax1.set_ylabel('Pourcentage (%)', fontweight='bold', fontsize=13)
        ax1.set_title('Métriques sur Cas Automatisés', fontweight='bold', fontsize=15)
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

PERFORMANCE (cas auto uniquement):
  • Accuracy:        {accuracy_globale:.1f}%
  • Précision:       {precision:.1f}%
  • Rappel:          {recall:.1f}%

CONFUSION (cas auto):
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

        # 4-6. Accuracy par top familles (calculé sur cas automatisés uniquement)
        if 'Famille Produit' in df_copy.columns:
            # Top familles basées sur le volume total
            top_families = df_copy['Famille Produit'].value_counts().head(10)

            accuracies = []
            volumes = []
            precisions = []

            for famille in top_families.index:
                df_fam = df_copy[df_copy['Famille Produit'] == famille]
                # FILTRER pour ne garder que les cas automatisés
                df_fam_auto = df_fam[
                    (df_fam['Decision_Modele'] == 'Rejet Auto') |
                    (df_fam['Decision_Modele'] == 'Validation Auto')
                ].copy()

                n_fam = len(df_fam)  # Volume total pour le graphique

                if len(df_fam_auto) > 0:
                    vp_fam = ((df_fam_auto['Fondee_bool'] == 1) & (df_fam_auto['Validation_bool'] == 1)).sum()
                    vn_fam = ((df_fam_auto['Fondee_bool'] == 0) & (df_fam_auto['Validation_bool'] == 0)).sum()
                    fp_fam = ((df_fam_auto['Fondee_bool'] == 0) & (df_fam_auto['Validation_bool'] == 1)).sum()
                    fn_fam = ((df_fam_auto['Fondee_bool'] == 1) & (df_fam_auto['Validation_bool'] == 0)).sum()

                    total_fam = vp_fam + vn_fam + fp_fam + fn_fam
                    acc_fam = 100 * (vp_fam + vn_fam) / total_fam if total_fam > 0 else 0
                    prec_fam = 100 * vp_fam / (vp_fam + fp_fam) if (vp_fam + fp_fam) > 0 else 0
                else:
                    acc_fam = 0
                    prec_fam = 0

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
        """Graphique 2: Comparaison Simple Gain AVANT vs APRÈS Règles Métier"""
        print("\n📊 Graphique 2: Gain AVANT vs APRÈS règles métier...")

        if 'Montant demandé' not in self.df_2025.columns:
            print("⚠️  Colonne 'Montant demandé' manquante - Graphique ignoré")
            return

        def calculer_gain_net(df, decision_col):
            """Calculer le gain NET pour un DataFrame"""
            df_work = df.copy()
            df_work['Validation_bool'] = df_work[decision_col].apply(
                lambda x: 1 if x == 'Validation Auto' else 0
            )

            # Nombre de cas automatisés
            n_auto = ((df_work[decision_col] == 'Rejet Auto') |
                     (df_work[decision_col] == 'Validation Auto')).sum()

            # GAIN BRUT
            gain_brut = n_auto * PRIX_UNITAIRE_DH

            # Masque pour cas automatisés
            mask_auto = ((df_work[decision_col] == 'Rejet Auto') |
                        (df_work[decision_col] == 'Validation Auto'))
            df_auto = df_work[mask_auto]

            if len(df_auto) == 0:
                return gain_brut, 0, 0, gain_brut, n_auto

            # FP et FN
            fp_mask = (df_auto['Fondee_bool'] == 0) & (df_auto['Validation_bool'] == 1)
            fn_mask = (df_auto['Fondee_bool'] == 1) & (df_auto['Validation_bool'] == 0)

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

            return gain_brut, perte_fp, perte_fn, gain_net, n_auto

        # Calculer AVANT règles métier
        gain_brut_avant, perte_fp_avant, perte_fn_avant, gain_net_avant, n_auto_avant = calculer_gain_net(
            self.df_before_rules, 'Decision_Modele'
        )

        # Calculer APRÈS règles métier
        gain_brut_apres, perte_fp_apres, perte_fn_apres, gain_net_apres, n_auto_apres = calculer_gain_net(
            self.df_2025, 'Decision_Finale'
        )

        # Différence (impact des règles)
        diff_gain_net = gain_net_apres - gain_net_avant

        # Créer le graphique SIMPLIFIÉ (2 éléments seulement)
        fig = plt.figure(figsize=(16, 8))
        fig.suptitle('GAIN NET: AVANT vs APRÈS Règles Métier',
                     fontsize=22, fontweight='bold', y=0.98)

        # 1. Comparaison AVANT vs APRÈS (graphique principal)
        ax1 = plt.subplot(1, 2, 1)
        categories = ['AVANT Règles Métier', 'APRÈS Règles Métier']
        values = [gain_net_avant / 1e6, gain_net_apres / 1e6]
        colors = ['#3498DB', '#27AE60'] if diff_gain_net >= 0 else ['#3498DB', '#E74C3C']

        bars = ax1.bar(categories, values, color=colors,
                      alpha=0.85, edgecolor='black', linewidth=3, width=0.5)

        ax1.set_ylabel('Gain NET (Millions DH)', fontweight='bold', fontsize=16)
        ax1.set_title('Comparaison du Gain NET', fontweight='bold', fontsize=18)
        ax1.grid(True, alpha=0.3, axis='y')
        ax1.axhline(y=0, color='black', linestyle='-', linewidth=1.5)

        # Ajouter les valeurs sur les barres
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + max(values)*0.03,
                    f'{val:.2f}M DH\n({val*1e6:,.0f} DH)',
                    ha='center', va='bottom',
                    fontweight='bold', fontsize=13)

        # Ajouter la différence
        ax1.text(0.5, max(values)*0.5,
                f'Différence: {diff_gain_net/1e6:+.2f}M DH\n({diff_gain_net:+,.0f} DH)',
                ha='center', va='center',
                fontsize=14, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.8', facecolor='yellow', alpha=0.7,
                         edgecolor='black', linewidth=2))

        # 2. Statistiques détaillées
        ax2 = plt.subplot(1, 2, 2)
        ax2.axis('off')

        stats_text = f"""
╔════════════════════════════════════════════════════╗
║          STATISTIQUES DÉTAILLÉES                   ║
╚════════════════════════════════════════════════════╝

📊 AVANT Règles Métier:
   • Cas automatisés:     {n_auto_avant:,}
   • Gain BRUT:           {gain_brut_avant:,.0f} DH ({gain_brut_avant/1e6:.2f}M)
   • Perte FP:            {perte_fp_avant:,.0f} DH ({perte_fp_avant/1e6:.2f}M)
   • Perte FN:            {perte_fn_avant:,.0f} DH ({perte_fn_avant/1e6:.2f}M)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   • GAIN NET:            {gain_net_avant:,.0f} DH
                          {gain_net_avant/1e6:.2f} MILLIONS DH

📊 APRÈS Règles Métier:
   • Cas automatisés:     {n_auto_apres:,}
   • Gain BRUT:           {gain_brut_apres:,.0f} DH ({gain_brut_apres/1e6:.2f}M)
   • Perte FP:            {perte_fp_apres:,.0f} DH ({perte_fp_apres/1e6:.2f}M)
   • Perte FN:            {perte_fn_apres:,.0f} DH ({perte_fn_apres/1e6:.2f}M)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   • GAIN NET:            {gain_net_apres:,.0f} DH
                          {gain_net_apres/1e6:.2f} MILLIONS DH

🎯 IMPACT DES RÈGLES MÉTIER:
   • Variation Gain NET:  {diff_gain_net:+,.0f} DH
                          {diff_gain_net/1e6:+.2f} MILLIONS DH
   • Variation Cas Auto:  {n_auto_apres - n_auto_avant:+,} cas
   • Statut:              {"✅ AMÉLIORATION" if diff_gain_net >= 0 else "⚠️ RÉDUCTION"}
        """

        ax2.text(0.5, 0.5, stats_text, transform=ax2.transAxes,
                fontsize=11, verticalalignment='center', horizontalalignment='center',
                family='monospace',
                bbox=dict(boxstyle='round,pad=1',
                         facecolor='#E8F8F5' if diff_gain_net >= 0 else '#FADBD8',
                         alpha=0.9,
                         edgecolor='#16A085' if diff_gain_net >= 0 else '#E74C3C',
                         linewidth=3))

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        output_path = self.output_dir / 'G2_gain_montant_net_v2.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ Sauvegardé: {output_path}")
        plt.close()

    def plot_business_rules_impact(self):
        """Graphique 3: Impact des règles métier par Marché et par Famille"""
        print("\n📊 Graphique 3: Impact des règles métier par Marché et Famille...")

        fig = plt.figure(figsize=(20, 12))
        fig.suptitle('IMPACT DES RÈGLES MÉTIER PAR MARCHÉ ET PAR FAMILLE - 2025',
                     fontsize=20, fontweight='bold', y=0.98)

        # Identifier les cas convertis
        df = self.df_2025.copy()
        df['Converti'] = (df['Decision_Modele'] == 'Validation Auto') & (df['Decision_Finale'] == 'Audit Humain')

        n_total = len(df)
        n_converties = df['Converti'].sum()
        n_validation_avant = (df['Decision_Modele'] == 'Validation Auto').sum()

        # 1. Impact par FAMILLE PRODUIT (Top 10)
        ax1 = plt.subplot(2, 3, 1)
        if 'Famille Produit' in df.columns:
            # Analyser par famille
            famille_stats = df.groupby('Famille Produit').agg({
                'Converti': 'sum',
                'Decision_Modele': lambda x: (x == 'Validation Auto').sum()
            }).rename(columns={'Decision_Modele': 'Validations_Initiales'})

            famille_stats = famille_stats[famille_stats['Validations_Initiales'] > 0].copy()
            famille_stats['Taux_Conversion'] = 100 * famille_stats['Converti'] / famille_stats['Validations_Initiales']
            famille_stats = famille_stats.sort_values('Converti', ascending=True).tail(10)

            y_pos = np.arange(len(famille_stats))
            colors_fam = plt.cm.RdYlGn_r(famille_stats['Taux_Conversion'] / 100)

            bars = ax1.barh(y_pos, famille_stats['Converti'], color=colors_fam,
                           edgecolor='black', linewidth=1.5)

            ax1.set_yticks(y_pos)
            ax1.set_yticklabels([f[:30] for f in famille_stats.index], fontsize=10)
            ax1.set_xlabel('Nombre de cas convertis', fontweight='bold', fontsize=12)
            ax1.set_title('Impact par Famille Produit (Top 10)', fontweight='bold', fontsize=14)
            ax1.grid(True, alpha=0.3, axis='x')

            for i, (bar, count, taux) in enumerate(zip(bars, famille_stats['Converti'], famille_stats['Taux_Conversion'])):
                width = bar.get_width()
                ax1.text(width + max(famille_stats['Converti'])*0.02, bar.get_y() + bar.get_height()/2.,
                        f'{int(count)} ({taux:.1f}%)', ha='left', va='center',
                        fontweight='bold', fontsize=9)
        else:
            ax1.text(0.5, 0.5, 'Colonne "Famille Produit"\nnon disponible',
                    ha='center', va='center', transform=ax1.transAxes,
                    fontsize=12, fontweight='bold')
            ax1.axis('off')

        # 2. Impact par MARCHÉ
        ax2 = plt.subplot(2, 3, 2)
        marche_col = None
        for col in ['Marché', 'Marche', 'marche', 'market']:
            if col in df.columns:
                marche_col = col
                break

        if marche_col:
            # Analyser par marché
            marche_stats = df.groupby(marche_col).agg({
                'Converti': 'sum',
                'Decision_Modele': lambda x: (x == 'Validation Auto').sum()
            }).rename(columns={'Decision_Modele': 'Validations_Initiales'})

            marche_stats = marche_stats[marche_stats['Validations_Initiales'] > 0].copy()
            marche_stats['Taux_Conversion'] = 100 * marche_stats['Converti'] / marche_stats['Validations_Initiales']
            marche_stats = marche_stats.sort_values('Converti', ascending=True)

            y_pos = np.arange(len(marche_stats))
            colors_mar = plt.cm.RdYlGn_r(marche_stats['Taux_Conversion'] / 100)

            bars = ax2.barh(y_pos, marche_stats['Converti'], color=colors_mar,
                           edgecolor='black', linewidth=1.5)

            ax2.set_yticks(y_pos)
            ax2.set_yticklabels([str(m)[:30] for m in marche_stats.index], fontsize=10)
            ax2.set_xlabel('Nombre de cas convertis', fontweight='bold', fontsize=12)
            ax2.set_title('Impact par Marché', fontweight='bold', fontsize=14)
            ax2.grid(True, alpha=0.3, axis='x')

            for bar, count, taux in zip(bars, marche_stats['Converti'], marche_stats['Taux_Conversion']):
                width = bar.get_width()
                ax2.text(width + max(marche_stats['Converti'])*0.02, bar.get_y() + bar.get_height()/2.,
                        f'{int(count)} ({taux:.1f}%)', ha='left', va='center',
                        fontweight='bold', fontsize=9)
        else:
            ax2.text(0.5, 0.5, 'Colonne "Marché"\nnon disponible',
                    ha='center', va='center', transform=ax2.transAxes,
                    fontsize=12, fontweight='bold')
            ax2.axis('off')

        # 3. Montants impactés par FAMILLE
        ax3 = plt.subplot(2, 3, 3)
        if 'Famille Produit' in df.columns and 'Montant demandé' in df.columns:
            df_conv = df[df['Converti']].copy()
            famille_montants = df_conv.groupby('Famille Produit')['Montant demandé'].sum() / 1e6
            famille_montants = famille_montants.sort_values(ascending=True).tail(10)

            y_pos = np.arange(len(famille_montants))
            bars = ax3.barh(y_pos, famille_montants, color='#E67E22',
                           alpha=0.8, edgecolor='black', linewidth=1.5)

            ax3.set_yticks(y_pos)
            ax3.set_yticklabels([f[:30] for f in famille_montants.index], fontsize=10)
            ax3.set_xlabel('Montant (Millions DH)', fontweight='bold', fontsize=12)
            ax3.set_title('Montants Protégés par Famille (Top 10)', fontweight='bold', fontsize=14)
            ax3.grid(True, alpha=0.3, axis='x')

            for bar, val in zip(bars, famille_montants):
                width = bar.get_width()
                ax3.text(width + max(famille_montants)*0.02, bar.get_y() + bar.get_height()/2.,
                        f'{val:.2f}M', ha='left', va='center', fontweight='bold', fontsize=9)
        else:
            ax3.text(0.5, 0.5, 'Données montants\nnon disponibles',
                    ha='center', va='center', transform=ax3.transAxes,
                    fontsize=12, fontweight='bold')
            ax3.axis('off')

        # 4. Montants impactés par MARCHÉ
        ax4 = plt.subplot(2, 3, 4)
        if marche_col and 'Montant demandé' in df.columns:
            df_conv = df[df['Converti']].copy()
            marche_montants = df_conv.groupby(marche_col)['Montant demandé'].sum() / 1e6
            marche_montants = marche_montants.sort_values(ascending=True)

            y_pos = np.arange(len(marche_montants))
            bars = ax4.barh(y_pos, marche_montants, color='#9B59B6',
                           alpha=0.8, edgecolor='black', linewidth=1.5)

            ax4.set_yticks(y_pos)
            ax4.set_yticklabels([str(m)[:30] for m in marche_montants.index], fontsize=10)
            ax4.set_xlabel('Montant (Millions DH)', fontweight='bold', fontsize=12)
            ax4.set_title('Montants Protégés par Marché', fontweight='bold', fontsize=14)
            ax4.grid(True, alpha=0.3, axis='x')

            for bar, val in zip(bars, marche_montants):
                width = bar.get_width()
                ax4.text(width + max(marche_montants)*0.02, bar.get_y() + bar.get_height()/2.,
                        f'{val:.2f}M', ha='left', va='center', fontweight='bold', fontsize=9)
        else:
            ax4.text(0.5, 0.5, 'Données montants\nnon disponibles',
                    ha='center', va='center', transform=ax4.transAxes,
                    fontsize=12, fontweight='bold')
            ax4.axis('off')

        # 5. Répartition des règles appliquées
        ax5 = plt.subplot(2, 3, 5)
        n_regle1 = df['Raison_Audit'].str.contains('Règle #1', na=False).sum()
        n_regle2 = df['Raison_Audit'].str.contains('Règle #2', na=False).sum()
        n_both = df['Raison_Audit'].str.contains('\\+', na=False, regex=True).sum()

        rules = ['Règle #1\n(>1 validation/an)', 'Règle #2\n(Montant > PNB)', 'Les 2 règles']
        counts = [n_regle1 - n_both, n_regle2 - n_both, n_both]
        colors_rules = ['#3498DB', '#E67E22', '#9B59B6']

        bars = ax5.bar(rules, counts, color=colors_rules,
                      alpha=0.8, edgecolor='black', linewidth=2)

        ax5.set_ylabel('Nombre de cas', fontweight='bold', fontsize=13)
        ax5.set_title('Répartition par Type de Règle', fontweight='bold', fontsize=15)
        ax5.grid(True, alpha=0.3, axis='y')

        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax5.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(count):,}', ha='center', va='bottom', fontweight='bold', fontsize=11)

        # 6. Statistiques globales
        ax6 = plt.subplot(2, 3, 6)
        ax6.axis('off')

        pct_conversion = 100 * n_converties / n_validation_avant if n_validation_avant > 0 else 0
        total_montant_protege = df[df['Converti']]['Montant demandé'].sum() if 'Montant demandé' in df.columns else 0

        stats_text = f"""
╔════════════════════════════════════════════╗
║       STATISTIQUES GLOBALES                ║
╚════════════════════════════════════════════╝

📊 IMPACT GLOBAL:
   • Total réclamations:       {n_total:,}
   • Validations initiales:    {n_validation_avant:,}
   • Cas convertis en audit:   {n_converties:,}
   • Taux de conversion:       {pct_conversion:.1f}%

📋 RÉPARTITION DES RÈGLES:
   • Règle #1 seule:           {n_regle1 - n_both:,}
   • Règle #2 seule:           {n_regle2 - n_both:,}
   • Les 2 règles:             {n_both:,}
   • TOTAL:                    {n_converties:,}

💰 MONTANTS PROTÉGÉS:
   • Total protégé:            {total_montant_protege:,.0f} DH
                               {total_montant_protege/1e6:.2f} Millions DH

🎯 RÉSULTAT:
   Les règles métier ont converti {pct_conversion:.1f}%
   des validations automatiques en audit humain,
   protégeant ainsi {total_montant_protege/1e6:.2f}M DH
   de risque potentiel.
        """

        ax6.text(0.5, 0.5, stats_text, transform=ax6.transAxes,
                fontsize=10, verticalalignment='center', horizontalalignment='center',
                family='monospace',
                bbox=dict(boxstyle='round,pad=1', facecolor='#FEF9E7', alpha=0.9,
                         edgecolor='#F39C12', linewidth=3))

        plt.tight_layout(rect=[0, 0, 1, 0.96])
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
