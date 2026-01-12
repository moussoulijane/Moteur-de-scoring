"""
Analyse et visualisation des résultats par Famille Produit
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')


class FamilyProductAnalyzer:
    """Analyse les performances par famille produit"""

    def __init__(self):
        self.family_metrics = {}
        self.df_analysis = None

    def analyze_by_family(self, df_original, y_true, y_pred, y_prob):
        """
        Analyse les métriques par famille produit

        Args:
            df_original: DataFrame original avec 'Famille Produit'
            y_true: Vraies étiquettes
            y_pred: Prédictions
            y_prob: Probabilités prédites
        """
        results = []

        df_temp = df_original.copy()
        df_temp['y_true'] = y_true
        df_temp['y_pred'] = y_pred
        df_temp['y_prob'] = y_prob

        families = df_temp['Famille Produit'].unique()

        print(f"\n📊 ANALYSE PAR FAMILLE PRODUIT")
        print("=" * 80)

        for family in families:
            # Filtrer par famille
            mask = df_temp['Famille Produit'] == family
            df_fam = df_temp[mask]

            if len(df_fam) < 10:  # Skip si trop peu de données
                continue

            y_t = df_fam['y_true']
            y_p = df_fam['y_pred']
            y_pr = df_fam['y_prob']

            # Calculer métriques
            accuracy = accuracy_score(y_t, y_p)
            precision = precision_score(y_t, y_p, zero_division=0)
            recall = recall_score(y_t, y_p, zero_division=0)
            f1 = f1_score(y_t, y_p, zero_division=0)

            # Statistiques
            n_samples = len(df_fam)
            taux_fondees_reel = y_t.mean()
            taux_fondees_pred = y_p.mean()
            montant_moyen = df_fam['Montant demandé'].mean() if 'Montant demandé' in df_fam.columns else 0

            # Confusion matrix
            cm = confusion_matrix(y_t, y_p)
            if cm.shape == (2, 2):
                tn, fp, fn, tp = cm.ravel()
            else:
                tn, fp, fn, tp = 0, 0, 0, 0

            results.append({
                'Famille': family,
                'N_samples': n_samples,
                'Accuracy': accuracy,
                'Precision': precision,
                'Recall': recall,
                'F1_Score': f1,
                'Taux_Fondees_Reel': taux_fondees_reel,
                'Taux_Fondees_Pred': taux_fondees_pred,
                'Montant_Moyen': montant_moyen,
                'TN': tn,
                'FP': fp,
                'FN': fn,
                'TP': tp
            })

        self.df_analysis = pd.DataFrame(results).sort_values('F1_Score', ascending=False)
        self.family_metrics = self.df_analysis.to_dict('records')

        # Afficher résumé
        print(f"\n{'Famille':<25s} {'N':>6s} {'Acc':>7s} {'Prec':>7s} {'Rec':>7s} {'F1':>7s} {'Fond%':>7s}")
        print("-" * 80)
        for _, row in self.df_analysis.iterrows():
            print(f"{row['Famille']:<25s} {row['N_samples']:>6.0f} {row['Accuracy']:>7.1%} "
                  f"{row['Precision']:>7.1%} {row['Recall']:>7.1%} {row['F1_Score']:>7.1%} "
                  f"{row['Taux_Fondees_Reel']:>7.1%}")

        return self.df_analysis

    def plot_family_comparison(self, save_path=None, year='2025'):
        """
        Crée une visualisation complète des résultats par famille

        Args:
            save_path: Chemin pour sauvegarder le graphique
            year: Année des données (pour le titre)
        """
        if self.df_analysis is None or len(self.df_analysis) == 0:
            print("⚠️  Pas de données d'analyse disponibles")
            return

        df = self.df_analysis.copy()

        # Créer une figure avec plusieurs sous-graphiques
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

        # Couleurs
        colors = sns.color_palette("husl", len(df))

        # === 1. Barplot des métriques principales ===
        ax1 = fig.add_subplot(gs[0, :])
        x = np.arange(len(df))
        width = 0.2

        ax1.bar(x - 1.5*width, df['Accuracy'], width, label='Accuracy', alpha=0.8, color='skyblue')
        ax1.bar(x - 0.5*width, df['Precision'], width, label='Precision', alpha=0.8, color='lightcoral')
        ax1.bar(x + 0.5*width, df['Recall'], width, label='Recall', alpha=0.8, color='lightgreen')
        ax1.bar(x + 1.5*width, df['F1_Score'], width, label='F1-Score', alpha=0.8, color='gold')

        ax1.set_xlabel('Famille Produit', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax1.set_title(f'Performance par Famille Produit - {year}', fontsize=14, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(df['Famille'], rotation=45, ha='right')
        ax1.legend(loc='upper right')
        ax1.grid(axis='y', alpha=0.3)
        ax1.set_ylim(0, 1.0)

        # === 2. Nombre d'échantillons par famille ===
        ax2 = fig.add_subplot(gs[1, 0])
        ax2.barh(df['Famille'], df['N_samples'], color=colors, alpha=0.7)
        ax2.set_xlabel('Nombre de réclamations', fontsize=11, fontweight='bold')
        ax2.set_title('Volume par Famille', fontsize=12, fontweight='bold')
        ax2.grid(axis='x', alpha=0.3)

        # === 3. Taux de fondement (Réel vs Prédit) ===
        ax3 = fig.add_subplot(gs[1, 1])
        x_pos = np.arange(len(df))
        width = 0.35

        ax3.bar(x_pos - width/2, df['Taux_Fondees_Reel'], width, label='Réel', alpha=0.8, color='steelblue')
        ax3.bar(x_pos + width/2, df['Taux_Fondees_Pred'], width, label='Prédit', alpha=0.8, color='orange')

        ax3.set_xlabel('Famille Produit', fontsize=11, fontweight='bold')
        ax3.set_ylabel('Taux de fondement', fontsize=11, fontweight='bold')
        ax3.set_title('Taux Fondées: Réel vs Prédit', fontsize=12, fontweight='bold')
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels(df['Famille'], rotation=45, ha='right')
        ax3.legend()
        ax3.grid(axis='y', alpha=0.3)

        # === 4. Montant moyen par famille ===
        ax4 = fig.add_subplot(gs[1, 2])
        ax4.bar(df['Famille'], df['Montant_Moyen'], color=colors, alpha=0.7)
        ax4.set_xlabel('Famille Produit', fontsize=11, fontweight='bold')
        ax4.set_ylabel('Montant moyen (MAD)', fontsize=11, fontweight='bold')
        ax4.set_title('Montant Moyen par Famille', fontsize=12, fontweight='bold')
        ax4.tick_params(axis='x', rotation=45)
        ax4.grid(axis='y', alpha=0.3)

        # === 5. Confusion Matrix - Famille avec meilleur F1 ===
        ax5 = fig.add_subplot(gs[2, 0])
        best_family = df.iloc[0]
        cm_best = np.array([[best_family['TN'], best_family['FP']],
                           [best_family['FN'], best_family['TP']]])
        sns.heatmap(cm_best, annot=True, fmt='d', cmap='Blues', ax=ax5,
                   xticklabels=['Non Fondée', 'Fondée'],
                   yticklabels=['Non Fondée', 'Fondée'])
        ax5.set_title(f'Meilleure: {best_family["Famille"]}\n(F1={best_family["F1_Score"]:.1%})',
                     fontsize=12, fontweight='bold')
        ax5.set_xlabel('Prédiction', fontsize=11)
        ax5.set_ylabel('Vérité', fontsize=11)

        # === 6. Confusion Matrix - Famille avec pire F1 ===
        ax6 = fig.add_subplot(gs[2, 1])
        worst_family = df.iloc[-1]
        cm_worst = np.array([[worst_family['TN'], worst_family['FP']],
                            [worst_family['FN'], worst_family['TP']]])
        sns.heatmap(cm_worst, annot=True, fmt='d', cmap='Reds', ax=ax6,
                   xticklabels=['Non Fondée', 'Fondée'],
                   yticklabels=['Non Fondée', 'Fondée'])
        ax6.set_title(f'Moins Bonne: {worst_family["Famille"]}\n(F1={worst_family["F1_Score"]:.1%})',
                     fontsize=12, fontweight='bold')
        ax6.set_xlabel('Prédiction', fontsize=11)
        ax6.set_ylabel('Vérité', fontsize=11)

        # === 7. Heatmap des métriques ===
        ax7 = fig.add_subplot(gs[2, 2])
        metrics_data = df[['Accuracy', 'Precision', 'Recall', 'F1_Score']].values
        sns.heatmap(metrics_data.T, annot=True, fmt='.2f', cmap='YlGnBu', ax=ax7,
                   xticklabels=df['Famille'], yticklabels=['Accuracy', 'Precision', 'Recall', 'F1'],
                   cbar_kws={'label': 'Score'})
        ax7.set_title('Heatmap des Métriques', fontsize=12, fontweight='bold')
        ax7.tick_params(axis='x', rotation=45)

        plt.suptitle(f'📊 ANALYSE COMPLÈTE DES RÉSULTATS PAR FAMILLE PRODUIT - {year}',
                    fontsize=16, fontweight='bold', y=0.995)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\n✅ Visualisation sauvegardée: {save_path}")

        plt.close()

    def generate_family_report(self, save_path=None):
        """Génère un rapport texte détaillé par famille"""
        if self.df_analysis is None:
            return

        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("RAPPORT DÉTAILLÉ PAR FAMILLE PRODUIT")
        report_lines.append("=" * 80)

        for idx, row in self.df_analysis.iterrows():
            report_lines.append(f"\n{'='*80}")
            report_lines.append(f"FAMILLE: {row['Famille']}")
            report_lines.append(f"{'='*80}")
            report_lines.append(f"Volume:              {row['N_samples']:.0f} réclamations")
            report_lines.append(f"\nMétriques de Performance:")
            report_lines.append(f"  - Accuracy:        {row['Accuracy']:.2%}")
            report_lines.append(f"  - Precision:       {row['Precision']:.2%}")
            report_lines.append(f"  - Recall:          {row['Recall']:.2%}")
            report_lines.append(f"  - F1-Score:        {row['F1_Score']:.2%}")
            report_lines.append(f"\nTaux de Fondement:")
            report_lines.append(f"  - Réel:            {row['Taux_Fondees_Reel']:.2%}")
            report_lines.append(f"  - Prédit:          {row['Taux_Fondees_Pred']:.2%}")
            report_lines.append(f"  - Écart:           {abs(row['Taux_Fondees_Reel'] - row['Taux_Fondees_Pred']):.2%}")
            report_lines.append(f"\nMontant:")
            report_lines.append(f"  - Montant moyen:   {row['Montant_Moyen']:.2f} MAD")
            report_lines.append(f"\nMatrice de Confusion:")
            report_lines.append(f"  - True Negatives:  {row['TN']:.0f}")
            report_lines.append(f"  - False Positives: {row['FP']:.0f}")
            report_lines.append(f"  - False Negatives: {row['FN']:.0f}")
            report_lines.append(f"  - True Positives:  {row['TP']:.0f}")

        report_lines.append(f"\n{'='*80}")
        report_lines.append("RÉSUMÉ")
        report_lines.append(f"{'='*80}")
        report_lines.append(f"Meilleure famille (F1): {self.df_analysis.iloc[0]['Famille']} "
                          f"({self.df_analysis.iloc[0]['F1_Score']:.2%})")
        report_lines.append(f"Moins bonne famille (F1): {self.df_analysis.iloc[-1]['Famille']} "
                          f"({self.df_analysis.iloc[-1]['F1_Score']:.2%})")
        report_lines.append(f"F1 moyen global: {self.df_analysis['F1_Score'].mean():.2%}")
        report_lines.append(f"Écart-type F1: {self.df_analysis['F1_Score'].std():.2%}")

        report_text = '\n'.join(report_lines)

        if save_path:
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(report_text)
            print(f"✅ Rapport famille sauvegardé: {save_path}")

        return report_text

    def get_summary_stats(self):
        """Retourne les statistiques récapitulatives"""
        if self.df_analysis is None:
            return None

        return {
            'best_family': {
                'name': self.df_analysis.iloc[0]['Famille'],
                'f1_score': self.df_analysis.iloc[0]['F1_Score']
            },
            'worst_family': {
                'name': self.df_analysis.iloc[-1]['Famille'],
                'f1_score': self.df_analysis.iloc[-1]['F1_Score']
            },
            'mean_f1': self.df_analysis['F1_Score'].mean(),
            'std_f1': self.df_analysis['F1_Score'].std(),
            'total_samples': self.df_analysis['N_samples'].sum()
        }
