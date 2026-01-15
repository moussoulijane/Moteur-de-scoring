"""
Script de Visualisation des Résultats 2025 - VERSION ANALYSE APPROFONDIE
Focus : Succès par famille + Analyse des erreurs pour améliorer le modèle
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
plt.rcParams['figure.figsize'] = (16, 10)
plt.rcParams['font.size'] = 10

COLORS = {
    'success': '#2ecc71',
    'error': '#e74c3c',
    'warning': '#f39c12',
    'info': '#3498db',
    'neutral': '#95a5a6',
    'fp': '#e74c3c',
    'fn': '#e67e22'
}

PRIX_UNITAIRE_DH = 169


class ResultsAnalyzer2025:
    """Analyse approfondie des résultats pour amélioration du modèle"""

    def __init__(self, data_path, predictions_path=None):
        self.data_path = data_path
        self.predictions_path = predictions_path
        self.df = None
        self.y_true = None
        self.y_pred = None
        self.y_prob = None

    def load_data(self):
        """Charge les données et prédictions"""
        print("📂 Chargement des données 2025...")
        self.df = pd.read_excel(self.data_path)
        print(f"✅ Chargé: {len(self.df)} réclamations")

        if self.predictions_path and Path(self.predictions_path).exists():
            preds = joblib.load(self.predictions_path)
            self.y_true = preds['y_true']
            self.y_pred = preds['y_pred']
            self.y_prob = preds['y_prob']
            print(f"✅ Prédictions chargées")
        else:
            self.y_true = self.df['Fondee'].values
            self.y_pred = None
            self.y_prob = None
            print("⚠️  Mode analyse descriptive (pas de prédictions)")

    def analyze_family_success(self):
        """Analyse du succès par famille : volume, %, pertes"""
        print("\n📊 Analyse Succès par Famille...")

        df_analysis = self.df.copy()
        df_analysis['y_true'] = self.y_true

        if self.y_pred is not None:
            df_analysis['y_pred'] = self.y_pred
            df_analysis['y_prob'] = self.y_prob
            df_analysis['is_error'] = (df_analysis['y_true'] != df_analysis['y_pred']).astype(int)
            df_analysis['is_fp'] = ((df_analysis['y_true'] == 0) & (df_analysis['y_pred'] == 1)).astype(int)
            df_analysis['is_fn'] = ((df_analysis['y_true'] == 1) & (df_analysis['y_pred'] == 0)).astype(int)
        else:
            df_analysis['is_error'] = 0
            df_analysis['is_fp'] = 0
            df_analysis['is_fn'] = 0

        # Montant
        montant_col = 'Montant demandé' if 'Montant demandé' in df_analysis.columns else 'Montant'
        if montant_col in df_analysis.columns:
            df_analysis['Montant'] = df_analysis[montant_col]
        else:
            df_analysis['Montant'] = 0

        # Analyse par famille
        results = []
        for family in df_analysis['Famille Produit'].unique():
            family_data = df_analysis[df_analysis['Famille Produit'] == family]

            volume = len(family_data)
            pct_volume = 100 * volume / len(df_analysis)

            # Métriques de performance
            if self.y_pred is not None:
                nb_errors = family_data['is_error'].sum()
                nb_fp = family_data['is_fp'].sum()
                nb_fn = family_data['is_fn'].sum()
                taux_erreur = 100 * nb_errors / volume if volume > 0 else 0
                taux_succes = 100 - taux_erreur

                # Calcul pertes financières
                perte_fp = nb_fp * PRIX_UNITAIRE_DH
                perte_fn = nb_fn * 2 * PRIX_UNITAIRE_DH
                perte_totale = perte_fp + perte_fn
            else:
                nb_errors = 0
                nb_fp = 0
                nb_fn = 0
                taux_erreur = 0
                taux_succes = 100
                perte_fp = 0
                perte_fn = 0
                perte_totale = 0

            # Montants
            montant_total = family_data['Montant'].sum()
            montant_moyen = family_data['Montant'].mean()

            # Taux fondées
            taux_fondees = 100 * family_data['y_true'].mean()

            results.append({
                'Famille': family,
                'Volume': volume,
                'Pct_Volume': pct_volume,
                'Taux_Succes': taux_succes,
                'Taux_Erreur': taux_erreur,
                'Nb_Erreurs': nb_errors,
                'Nb_FP': nb_fp,
                'Nb_FN': nb_fn,
                'Perte_FP_DH': perte_fp,
                'Perte_FN_DH': perte_fn,
                'Perte_Totale_DH': perte_totale,
                'Montant_Total': montant_total,
                'Montant_Moyen': montant_moyen,
                'Taux_Fondees': taux_fondees
            })

        df_metrics = pd.DataFrame(results)
        df_metrics = df_metrics.sort_values('Volume', ascending=False)

        return df_metrics

    def analyze_errors_deep(self):
        """Analyse approfondie des erreurs pour identifier patterns"""
        print("\n📊 Analyse Approfondie des Erreurs...")

        if self.y_pred is None:
            print("❌ Pas de prédictions disponibles")
            return None

        df_errors = self.df.copy()
        df_errors['y_true'] = self.y_true
        df_errors['y_pred'] = self.y_pred
        df_errors['y_prob'] = self.y_prob

        # Identifier les erreurs
        df_errors['Type_Erreur'] = 'Correct'
        df_errors.loc[(df_errors['y_true'] == 0) & (df_errors['y_pred'] == 1), 'Type_Erreur'] = 'Faux Positif'
        df_errors.loc[(df_errors['y_true'] == 1) & (df_errors['y_pred'] == 0), 'Type_Erreur'] = 'Faux Négatif'

        # Extraire uniquement les erreurs
        df_only_errors = df_errors[df_errors['Type_Erreur'] != 'Correct'].copy()

        # Montant
        montant_col = 'Montant demandé' if 'Montant demandé' in df_only_errors.columns else 'Montant'
        if montant_col in df_only_errors.columns:
            df_only_errors['Montant'] = df_only_errors[montant_col]

        print(f"   FP: {(df_errors['Type_Erreur'] == 'Faux Positif').sum()}")
        print(f"   FN: {(df_errors['Type_Erreur'] == 'Faux Négatif').sum()}")

        return df_errors, df_only_errors

    def plot_family_success_analysis(self, df_metrics, save_path):
        """Graphiques : Volume, %, Pertes par famille"""
        print("\n📊 Création graphiques succès famille...")

        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('🏆 ANALYSE DE SUCCÈS PAR FAMILLE PRODUIT - 2025',
                     fontsize=18, fontweight='bold', y=0.995)

        # 1. Volume absolu (Top 10)
        ax = axes[0, 0]
        top_volume = df_metrics.nlargest(10, 'Volume')
        bars = ax.barh(range(len(top_volume)), top_volume['Volume'], color=COLORS['info'])
        ax.set_yticks(range(len(top_volume)))
        ax.set_yticklabels(top_volume['Famille'])
        ax.set_xlabel('Nombre de Réclamations', fontweight='bold', fontsize=11)
        ax.set_title('📊 Top 10 Familles - Volume Absolu', fontweight='bold', fontsize=12)
        ax.invert_yaxis()

        for i, (bar, val) in enumerate(zip(bars, top_volume['Volume'])):
            ax.text(val + 5, i, f'{int(val)}', va='center', fontweight='bold')

        # 2. Pourcentage du volume total
        ax = axes[0, 1]
        top_pct = df_metrics.nlargest(10, 'Pct_Volume')
        bars = ax.barh(range(len(top_pct)), top_pct['Pct_Volume'], color=COLORS['success'])
        ax.set_yticks(range(len(top_pct)))
        ax.set_yticklabels(top_pct['Famille'])
        ax.set_xlabel('% du Volume Total', fontweight='bold', fontsize=11)
        ax.set_title('📈 Top 10 Familles - Part du Volume (%)', fontweight='bold', fontsize=12)
        ax.invert_yaxis()

        for i, (bar, val) in enumerate(zip(bars, top_pct['Pct_Volume'])):
            ax.text(val + 0.5, i, f'{val:.1f}%', va='center', fontweight='bold')

        # 3. Taux de succès (Top 10)
        ax = axes[0, 2]
        if 'Taux_Succes' in df_metrics.columns:
            # Filtrer seulement les familles avec volume > 10 pour éviter biais
            df_significant = df_metrics[df_metrics['Volume'] >= 10].copy()
            top_success = df_significant.nlargest(10, 'Taux_Succes')

            bars = ax.barh(range(len(top_success)), top_success['Taux_Succes'],
                          color=COLORS['success'], alpha=0.8)
            ax.set_yticks(range(len(top_success)))
            ax.set_yticklabels(top_success['Famille'])
            ax.set_xlabel('Taux de Succès (%)', fontweight='bold', fontsize=11)
            ax.set_title('🎯 Top 10 Familles - Taux de Succès\n(Volume ≥ 10)',
                        fontweight='bold', fontsize=12)
            ax.set_xlim(0, 105)
            ax.invert_yaxis()

            for i, (bar, val, vol) in enumerate(zip(bars, top_success['Taux_Succes'],
                                                     top_success['Volume'])):
                ax.text(val + 1, i, f'{val:.1f}% (n={int(vol)})',
                       va='center', fontweight='bold', fontsize=9)

        # 4. Pertes totales (Top 10 - pires familles)
        ax = axes[1, 0]
        if 'Perte_Totale_DH' in df_metrics.columns:
            top_loss = df_metrics.nlargest(10, 'Perte_Totale_DH')
            bars = ax.barh(range(len(top_loss)), top_loss['Perte_Totale_DH'],
                          color=COLORS['error'], alpha=0.8)
            ax.set_yticks(range(len(top_loss)))
            ax.set_yticklabels(top_loss['Famille'])
            ax.set_xlabel('Perte Totale (DH)', fontweight='bold', fontsize=11)
            ax.set_title('💸 Top 10 Familles - Pertes Financières\n(FP + FN)',
                        fontweight='bold', fontsize=12)
            ax.invert_yaxis()

            for i, (bar, val) in enumerate(zip(bars, top_loss['Perte_Totale_DH'])):
                ax.text(val + 100, i, f'{val:,.0f} DH', va='center', fontsize=9)

        # 5. Répartition FP vs FN par famille (Top 10 erreurs)
        ax = axes[1, 1]
        if 'Nb_FP' in df_metrics.columns and 'Nb_FN' in df_metrics.columns:
            df_metrics['Total_Erreurs'] = df_metrics['Nb_FP'] + df_metrics['Nb_FN']
            top_errors = df_metrics[df_metrics['Total_Erreurs'] > 0].nlargest(10, 'Total_Erreurs')

            x = np.arange(len(top_errors))
            width = 0.35

            ax.barh(x - width/2, top_errors['Nb_FP'], width, label='Faux Positifs',
                   color=COLORS['fp'], alpha=0.8)
            ax.barh(x + width/2, top_errors['Nb_FN'], width, label='Faux Négatifs',
                   color=COLORS['fn'], alpha=0.8)

            ax.set_yticks(x)
            ax.set_yticklabels(top_errors['Famille'])
            ax.set_xlabel('Nombre d\'Erreurs', fontweight='bold', fontsize=11)
            ax.set_title('⚠️ Top 10 Familles - FP vs FN', fontweight='bold', fontsize=12)
            ax.legend(loc='lower right')
            ax.invert_yaxis()

        # 6. Scatter : Volume vs Taux de Succès
        ax = axes[1, 2]
        if 'Taux_Succes' in df_metrics.columns:
            scatter = ax.scatter(df_metrics['Volume'], df_metrics['Taux_Succes'],
                               s=df_metrics['Perte_Totale_DH']/50 if 'Perte_Totale_DH' in df_metrics.columns else 100,
                               c=df_metrics['Taux_Succes'], cmap='RdYlGn',
                               alpha=0.6, edgecolors='black', linewidth=1)

            # Annoter les familles importantes
            for idx, row in df_metrics.iterrows():
                if row['Volume'] > df_metrics['Volume'].quantile(0.7) or \
                   row['Taux_Succes'] < df_metrics['Taux_Succes'].quantile(0.3):
                    ax.annotate(row['Famille'][:15],
                               (row['Volume'], row['Taux_Succes']),
                               fontsize=8, alpha=0.7)

            ax.set_xlabel('Volume de Réclamations', fontweight='bold', fontsize=11)
            ax.set_ylabel('Taux de Succès (%)', fontweight='bold', fontsize=11)
            ax.set_title('📊 Volume vs Succès\n(Taille bulle = Perte)',
                        fontweight='bold', fontsize=12)
            ax.grid(True, alpha=0.3)

            # Colorbar
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label('Taux Succès (%)', fontweight='bold')

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Sauvegardé: {save_path}")
        plt.close()

    def plot_error_analysis(self, df_errors, df_only_errors, save_path):
        """Analyse approfondie des erreurs pour amélioration modèle"""
        print("\n📊 Création graphiques analyse erreurs...")

        fig = plt.figure(figsize=(20, 16))
        gs = fig.add_gridspec(4, 3, hspace=0.35, wspace=0.3)

        fig.suptitle('🔍 ANALYSE APPROFONDIE DES ERREURS - Où améliorer le modèle ?',
                     fontsize=18, fontweight='bold', y=0.995)

        # 1. Distribution par type d'erreur
        ax1 = fig.add_subplot(gs[0, 0])
        error_counts = df_errors['Type_Erreur'].value_counts()
        colors_pie = [COLORS['success'], COLORS['fp'], COLORS['fn']]
        wedges, texts, autotexts = ax1.pie(error_counts.values,
                                            labels=error_counts.index,
                                            autopct='%1.1f%%',
                                            colors=colors_pie,
                                            startangle=90,
                                            explode=(0, 0.05, 0.05))
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        ax1.set_title('📊 Répartition Globale', fontweight='bold', fontsize=12)

        # 2. Erreurs par Famille Produit
        ax2 = fig.add_subplot(gs[0, 1])
        if 'Famille Produit' in df_only_errors.columns:
            error_by_family = df_only_errors['Famille Produit'].value_counts().head(10)
            bars = ax2.barh(range(len(error_by_family)), error_by_family.values,
                           color=COLORS['error'])
            ax2.set_yticks(range(len(error_by_family)))
            ax2.set_yticklabels(error_by_family.index)
            ax2.set_xlabel('Nombre d\'Erreurs', fontweight='bold')
            ax2.set_title('🏢 Top 10 Familles - Total Erreurs', fontweight='bold', fontsize=12)
            ax2.invert_yaxis()

        # 3. FP vs FN par famille
        ax3 = fig.add_subplot(gs[0, 2])
        if 'Famille Produit' in df_only_errors.columns:
            fp_fn_family = df_only_errors.groupby(['Famille Produit', 'Type_Erreur']).size().unstack(fill_value=0)
            top_families = (fp_fn_family.sum(axis=1)).nlargest(10).index
            fp_fn_top = fp_fn_family.loc[top_families]

            x = np.arange(len(fp_fn_top))
            width = 0.35

            if 'Faux Positif' in fp_fn_top.columns:
                ax3.barh(x - width/2, fp_fn_top['Faux Positif'], width,
                        label='FP', color=COLORS['fp'], alpha=0.8)
            if 'Faux Négatif' in fp_fn_top.columns:
                ax3.barh(x + width/2, fp_fn_top['Faux Négatif'], width,
                        label='FN', color=COLORS['fn'], alpha=0.8)

            ax3.set_yticks(x)
            ax3.set_yticklabels(fp_fn_top.index)
            ax3.set_xlabel('Nombre', fontweight='bold')
            ax3.set_title('⚖️ FP vs FN - Top 10 Familles', fontweight='bold', fontsize=12)
            ax3.legend()
            ax3.invert_yaxis()

        # 4. Distribution des montants : Correct vs FP vs FN
        ax4 = fig.add_subplot(gs[1, 0])
        montant_col = 'Montant demandé' if 'Montant demandé' in df_errors.columns else 'Montant'
        if montant_col in df_errors.columns:
            data_correct = df_errors[df_errors['Type_Erreur'] == 'Correct'][montant_col].dropna()
            data_fp = df_errors[df_errors['Type_Erreur'] == 'Faux Positif'][montant_col].dropna()
            data_fn = df_errors[df_errors['Type_Erreur'] == 'Faux Négatif'][montant_col].dropna()

            ax4.hist([data_correct, data_fp, data_fn], bins=30, label=['Correct', 'FP', 'FN'],
                    color=[COLORS['success'], COLORS['fp'], COLORS['fn']], alpha=0.6)
            ax4.set_xlabel('Montant (DH)', fontweight='bold')
            ax4.set_ylabel('Fréquence', fontweight='bold')
            ax4.set_title('💰 Distribution Montants: Correct vs Erreurs', fontweight='bold', fontsize=12)
            ax4.legend()
            ax4.grid(True, alpha=0.3)

        # 5. Boxplot montants par type
        ax5 = fig.add_subplot(gs[1, 1])
        if montant_col in df_errors.columns:
            data_to_plot = [data_correct, data_fp, data_fn]
            bp = ax5.boxplot(data_to_plot, labels=['Correct', 'FP', 'FN'],
                            patch_artist=True)
            for patch, color in zip(bp['boxes'], [COLORS['success'], COLORS['fp'], COLORS['fn']]):
                patch.set_facecolor(color)
                patch.set_alpha(0.6)
            ax5.set_ylabel('Montant (DH)', fontweight='bold')
            ax5.set_title('📦 Montants - Comparaison', fontweight='bold', fontsize=12)
            ax5.grid(True, alpha=0.3, axis='y')

        # 6. Erreurs par Segment Client
        ax6 = fig.add_subplot(gs[1, 2])
        if 'Segment' in df_only_errors.columns:
            error_by_segment = df_only_errors.groupby(['Segment', 'Type_Erreur']).size().unstack(fill_value=0)

            x = np.arange(len(error_by_segment))
            width = 0.35

            if 'Faux Positif' in error_by_segment.columns:
                ax6.bar(x - width/2, error_by_segment['Faux Positif'], width,
                       label='FP', color=COLORS['fp'], alpha=0.8)
            if 'Faux Négatif' in error_by_segment.columns:
                ax6.bar(x + width/2, error_by_segment['Faux Négatif'], width,
                       label='FN', color=COLORS['fn'], alpha=0.8)

            ax6.set_xticks(x)
            ax6.set_xticklabels(error_by_segment.index, rotation=45, ha='right')
            ax6.set_ylabel('Nombre d\'Erreurs', fontweight='bold')
            ax6.set_title('👥 Erreurs par Segment Client', fontweight='bold', fontsize=12)
            ax6.legend()
            ax6.grid(True, alpha=0.3, axis='y')

        # 7. Probabilités des erreurs
        ax7 = fig.add_subplot(gs[2, 0])
        if 'y_prob' in df_errors.columns:
            prob_correct = df_errors[df_errors['Type_Erreur'] == 'Correct']['y_prob']
            prob_fp = df_errors[df_errors['Type_Erreur'] == 'Faux Positif']['y_prob']
            prob_fn = df_errors[df_errors['Type_Erreur'] == 'Faux Négatif']['y_prob']

            ax7.hist([prob_correct, prob_fp, prob_fn], bins=20,
                    label=['Correct', 'FP', 'FN'],
                    color=[COLORS['success'], COLORS['fp'], COLORS['fn']], alpha=0.6)
            ax7.set_xlabel('Probabilité Prédite', fontweight='bold')
            ax7.set_ylabel('Fréquence', fontweight='bold')
            ax7.set_title('🎲 Distribution des Probabilités\n(Identifier la zone d\'incertitude)',
                         fontweight='bold', fontsize=12)
            ax7.legend()
            ax7.axvline(x=0.5, color='black', linestyle='--', alpha=0.5, label='Seuil 0.5')
            ax7.grid(True, alpha=0.3)

        # 8. Erreurs par Canal de Réception
        ax8 = fig.add_subplot(gs[2, 1])
        if 'Canal de Réception' in df_only_errors.columns:
            error_by_canal = df_only_errors['Canal de Réception'].value_counts().head(8)
            bars = ax8.barh(range(len(error_by_canal)), error_by_canal.values,
                           color=COLORS['warning'])
            ax8.set_yticks(range(len(error_by_canal)))
            ax8.set_yticklabels(error_by_canal.index)
            ax8.set_xlabel('Nombre d\'Erreurs', fontweight='bold')
            ax8.set_title('📞 Erreurs par Canal de Réception', fontweight='bold', fontsize=12)
            ax8.invert_yaxis()

        # 9. Erreurs par Priorité Client
        ax9 = fig.add_subplot(gs[2, 2])
        if 'Priorité Client' in df_only_errors.columns:
            error_by_priority = df_only_errors.groupby(['Priorité Client', 'Type_Erreur']).size().unstack(fill_value=0)
            error_by_priority.plot(kind='bar', stacked=False, ax=ax9,
                                  color=[COLORS['fp'], COLORS['fn']], alpha=0.8)
            ax9.set_xlabel('Priorité Client', fontweight='bold')
            ax9.set_ylabel('Nombre d\'Erreurs', fontweight='bold')
            ax9.set_title('⭐ Erreurs par Priorité Client', fontweight='bold', fontsize=12)
            ax9.legend(['FP', 'FN'])
            ax9.tick_params(axis='x', rotation=45)
            ax9.grid(True, alpha=0.3, axis='y')

        # 10. Heatmap: Catégorie × Type d'erreur
        ax10 = fig.add_subplot(gs[3, 0])
        if 'Catégorie' in df_only_errors.columns:
            cat_error = pd.crosstab(df_only_errors['Catégorie'],
                                    df_only_errors['Type_Erreur'])
            # Top 10 catégories
            top_cats = cat_error.sum(axis=1).nlargest(10).index
            cat_error_top = cat_error.loc[top_cats]

            sns.heatmap(cat_error_top, annot=True, fmt='d', cmap='YlOrRd', ax=ax10,
                       cbar_kws={'label': 'Nombre d\'Erreurs'})
            ax10.set_title('🔥 Heatmap: Top 10 Catégories × Type Erreur',
                          fontweight='bold', fontsize=12)
            ax10.set_xlabel('Type d\'Erreur', fontweight='bold')
            ax10.set_ylabel('Catégorie', fontweight='bold')

        # 11. PNB vs Erreurs
        ax11 = fig.add_subplot(gs[3, 1])
        pnb_col = 'PNB analytique (vision commerciale) cumulé'
        if pnb_col in df_errors.columns:
            pnb_correct = df_errors[df_errors['Type_Erreur'] == 'Correct'][pnb_col].dropna()
            pnb_fp = df_errors[df_errors['Type_Erreur'] == 'Faux Positif'][pnb_col].dropna()
            pnb_fn = df_errors[df_errors['Type_Erreur'] == 'Faux Négatif'][pnb_col].dropna()

            data_to_plot = [pnb_correct, pnb_fp, pnb_fn]
            bp = ax11.boxplot(data_to_plot, labels=['Correct', 'FP', 'FN'],
                             patch_artist=True)
            for patch, color in zip(bp['boxes'], [COLORS['success'], COLORS['fp'], COLORS['fn']]):
                patch.set_facecolor(color)
                patch.set_alpha(0.6)
            ax11.set_ylabel('PNB Cumulé', fontweight='bold')
            ax11.set_title('💼 PNB Client vs Type d\'Erreur', fontweight='bold', fontsize=12)
            ax11.grid(True, alpha=0.3, axis='y')

        # 12. Recommandations Features
        ax12 = fig.add_subplot(gs[3, 2])
        ax12.axis('off')

        # Calculer stats pour recommandations
        total_errors = len(df_only_errors)
        pct_fp = 100 * (df_errors['Type_Erreur'] == 'Faux Positif').sum() / len(df_errors)
        pct_fn = 100 * (df_errors['Type_Erreur'] == 'Faux Négatif').sum() / len(df_errors)

        # Top famille avec erreurs
        if 'Famille Produit' in df_only_errors.columns:
            top_error_family = df_only_errors['Famille Produit'].value_counts().index[0]
        else:
            top_error_family = 'N/A'

        recommendations = f"""
        🎯 RECOMMANDATIONS POUR AMÉLIORER LE MODÈLE
        {'='*45}

        📊 Statistiques:
           • Total erreurs: {total_errors}
           • Faux Positifs: {pct_fp:.1f}%
           • Faux Négatifs: {pct_fn:.1f}%
           • Famille problématique: {top_error_family}

        💡 FEATURES À AJOUTER:

        1. Features Temporelles:
           ✓ Historique client (nb réclamations passées)
           ✓ Délai depuis dernière réclamation
           ✓ Saisonnalité (mois/trimestre)

        2. Features Comportementales:
           ✓ Ratio montant/PNB
           ✓ Ancienneté × Segment
           ✓ Volatilité des montants

        3. Features Contextuelles:
           ✓ Taux fondées par famille (rolling)
           ✓ Complexité catégorie (entropie)
           ✓ Score canal (fiabilité)

        4. Features d'Interaction:
           ✓ Montant × Priorité
           ✓ PNB × Canal
           ✓ Segment × Catégorie

        🔍 Focus Analyse:
           → Analyser zone proba 0.4-0.6 (incertitude)
           → Investiguer {top_error_family}
           → Comparer montants FP vs FN
        """

        ax12.text(0.05, 0.95, recommendations, transform=ax12.transAxes,
                 fontsize=10, verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Sauvegardé: {save_path}")
        plt.close()

    def generate_analysis_report(self, df_metrics, df_errors, save_path):
        """Génère un rapport texte d'analyse"""
        print("\n📄 Génération rapport d'analyse...")

        lines = []
        lines.append("="*80)
        lines.append("RAPPORT D'ANALYSE APPROFONDIE - RÉSULTATS 2025")
        lines.append("="*80)
        lines.append("")

        # 1. Succès par famille
        lines.append("="*80)
        lines.append("1. ANALYSE SUCCÈS PAR FAMILLE")
        lines.append("="*80)
        lines.append("")

        lines.append("Top 5 Familles par Volume:")
        lines.append("-" * 80)
        for idx, row in df_metrics.head(5).iterrows():
            lines.append(f"  {row['Famille']:30s} | Vol: {int(row['Volume']):5d} ({row['Pct_Volume']:5.1f}%) | "
                        f"Succès: {row['Taux_Succes']:5.1f}% | Perte: {row['Perte_Totale_DH']:10,.0f} DH")
        lines.append("")

        if 'Taux_Succes' in df_metrics.columns:
            df_significant = df_metrics[df_metrics['Volume'] >= 10]
            lines.append("Top 5 Familles par Taux de Succès (Volume ≥ 10):")
            lines.append("-" * 80)
            for idx, row in df_significant.nlargest(5, 'Taux_Succes').iterrows():
                lines.append(f"  {row['Famille']:30s} | Succès: {row['Taux_Succes']:5.1f}% | "
                            f"Vol: {int(row['Volume']):5d} | Erreurs: FP={int(row['Nb_FP'])}, FN={int(row['Nb_FN'])}")
            lines.append("")

            lines.append("Top 5 Familles avec Plus Grandes Pertes:")
            lines.append("-" * 80)
            for idx, row in df_metrics.nlargest(5, 'Perte_Totale_DH').iterrows():
                lines.append(f"  {row['Famille']:30s} | Perte: {row['Perte_Totale_DH']:10,.0f} DH | "
                            f"FP: {row['Perte_FP_DH']:8,.0f} DH | FN: {row['Perte_FN_DH']:8,.0f} DH")
            lines.append("")

        # 2. Analyse des erreurs
        if df_errors is not None:
            lines.append("="*80)
            lines.append("2. ANALYSE DES ERREURS")
            lines.append("="*80)
            lines.append("")

            total = len(df_errors)
            nb_correct = (df_errors['Type_Erreur'] == 'Correct').sum()
            nb_fp = (df_errors['Type_Erreur'] == 'Faux Positif').sum()
            nb_fn = (df_errors['Type_Erreur'] == 'Faux Négatif').sum()

            lines.append(f"Total réclamations: {total}")
            lines.append(f"  Correctes:       {nb_correct:5d} ({100*nb_correct/total:5.1f}%)")
            lines.append(f"  Faux Positifs:   {nb_fp:5d} ({100*nb_fp/total:5.1f}%)")
            lines.append(f"  Faux Négatifs:   {nb_fn:5d} ({100*nb_fn/total:5.1f}%)")
            lines.append("")

            # Montants moyens
            montant_col = 'Montant demandé' if 'Montant demandé' in df_errors.columns else 'Montant'
            if montant_col in df_errors.columns:
                montant_correct = df_errors[df_errors['Type_Erreur'] == 'Correct'][montant_col].mean()
                montant_fp = df_errors[df_errors['Type_Erreur'] == 'Faux Positif'][montant_col].mean()
                montant_fn = df_errors[df_errors['Type_Erreur'] == 'Faux Négatif'][montant_col].mean()

                lines.append("Montants Moyens:")
                lines.append(f"  Correct:         {montant_correct:10,.2f} DH")
                lines.append(f"  Faux Positifs:   {montant_fp:10,.2f} DH")
                lines.append(f"  Faux Négatifs:   {montant_fn:10,.2f} DH")
                lines.append("")

        # 3. Recommandations
        lines.append("="*80)
        lines.append("3. RECOMMANDATIONS")
        lines.append("="*80)
        lines.append("")
        lines.append("Actions Prioritaires:")
        lines.append("  1. Ajouter features temporelles (historique réclamations)")
        lines.append("  2. Créer features d'interaction (Montant × Priorité, PNB × Segment)")
        lines.append("  3. Analyser zone d'incertitude (probabilités 0.4-0.6)")
        lines.append("  4. Investiguer les familles avec pertes élevées")
        lines.append("  5. Ajuster seuil de décision si FN >> FP")
        lines.append("")

        # Sauvegarder
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))

        print(f"✅ Rapport sauvegardé: {save_path}")

    def run_complete_analysis(self, output_dir='outputs/analysis_2025'):
        """Exécute l'analyse complète"""
        print("\n" + "="*80)
        print("🔍 ANALYSE APPROFONDIE DES RÉSULTATS 2025")
        print("="*80)

        Path(output_dir).mkdir(parents=True, exist_ok=True)
        Path(f"{output_dir}/figures").mkdir(parents=True, exist_ok=True)

        # Charger données
        self.load_data()

        # 1. Analyse succès famille
        df_metrics = self.analyze_family_success()
        self.plot_family_success_analysis(
            df_metrics,
            f"{output_dir}/figures/family_success_detailed_2025.png"
        )
        df_metrics.to_csv(f"{output_dir}/family_success_metrics_2025.csv", index=False)
        print(f"✅ CSV sauvegardé: {output_dir}/family_success_metrics_2025.csv")

        # 2. Analyse erreurs
        if self.y_pred is not None:
            df_errors, df_only_errors = self.analyze_errors_deep()
            self.plot_error_analysis(
                df_errors,
                df_only_errors,
                f"{output_dir}/figures/error_deep_analysis_2025.png"
            )

            # Rapport texte
            self.generate_analysis_report(
                df_metrics,
                df_errors,
                f"{output_dir}/analysis_report_2025.txt"
            )
        else:
            print("⚠️  Analyse erreurs ignorée (pas de prédictions)")

        print("\n" + "="*80)
        print("✅ ANALYSE TERMINÉE")
        print("="*80)
        print(f"\n📂 Résultats dans: {output_dir}/")


def main():
    """Point d'entrée"""
    print("\n" + "="*80)
    print("🔍 ANALYSE APPROFONDIE RÉSULTATS 2025")
    print("="*80)

    data_path = 'data/raw/reclamations_2025.xlsx'
    predictions_path = 'outputs/models/predictions_2025.pkl'

    if not Path(data_path).exists():
        print(f"❌ Fichier non trouvé: {data_path}")
        print("\n💡 Exécutez d'abord: python main_pipeline.py")
        return

    analyzer = ResultsAnalyzer2025(
        data_path=data_path,
        predictions_path=predictions_path if Path(predictions_path).exists() else None
    )

    analyzer.run_complete_analysis(output_dir='outputs/analysis_2025')

    print("\n✅ Analyse terminée!")
    print("\n📂 Tous les résultats dans: outputs/analysis_2025/")
    print("\n📊 Fichiers générés:")
    print("   📈 Visualisations (PNG):")
    print("      • figures/family_success_detailed_2025.png")
    print("      • figures/error_deep_analysis_2025.png")
    print("   📊 Données (CSV/TXT):")
    print("      • family_success_metrics_2025.csv")
    print("      • analysis_report_2025.txt")


if __name__ == '__main__':
    main()
