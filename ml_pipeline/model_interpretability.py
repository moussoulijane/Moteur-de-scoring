"""
ANALYSE D'INTERPRÉTABILITÉ DU MODÈLE
Comprendre comment le modèle CatBoost fait ses prédictions
- Feature importance
- SHAP values
- Analyse des erreurs
- Impact des métriques calculées
"""
import sys
sys.path.append('src')

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from preprocessor import ProductionPreprocessor

# Configuration
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (16, 10)

PRIX_UNITAIRE_DH = 169


class ModelInterpreter:
    """Analyse d'interprétabilité du modèle CatBoost"""

    def __init__(self):
        self.output_dir = Path('outputs/interpretability')
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def load_data_and_model(self):
        """Charger données, modèle et preprocessor"""
        print("\n" + "="*80)
        print("📂 CHARGEMENT DES DONNÉES ET MODÈLE")
        print("="*80)

        # Charger données
        self.df_2024 = pd.read_excel('data/raw/reclamations_2024.xlsx')
        self.df_2025 = pd.read_excel('data/raw/reclamations_2025.xlsx')

        print(f"✅ 2024: {len(self.df_2024)} réclamations")
        print(f"✅ 2025: {len(self.df_2025)} réclamations")

        # Charger modèle et preprocessor
        self.model = joblib.load('outputs/production/models/catboost_model.pkl')
        self.preprocessor = joblib.load('outputs/production/models/preprocessor.pkl')

        print(f"✅ Modèle CatBoost chargé")
        print(f"✅ Preprocessor chargé")

        # Charger prédictions
        predictions_data = joblib.load('outputs/production/predictions/predictions_2025.pkl')
        self.y_true = predictions_data['y_true']
        self.y_pred = predictions_data['CatBoost']['y_pred']
        self.y_prob = predictions_data['CatBoost']['y_prob']

        print(f"✅ Prédictions chargées")

        # Preprocesser les données
        print("\n🔧 Preprocessing des données...")
        self.X_train = self.preprocessor.fit_transform(self.df_2024)
        self.y_train = self.df_2024['Fondee'].values

        self.X_test = self.preprocessor.transform(self.df_2025)
        self.y_test = self.df_2025['Fondee'].values

        print(f"✅ Shape train: {self.X_train.shape}")
        print(f"✅ Shape test: {self.X_test.shape}")

    def analyze_feature_importance(self):
        """Analyser l'importance des features selon CatBoost"""
        print("\n" + "="*80)
        print("🎯 ANALYSE DE L'IMPORTANCE DES FEATURES")
        print("="*80)

        # Obtenir feature importance
        feature_importance = self.model.get_feature_importance()

        # Obtenir les noms de features - essayer plusieurs méthodes
        if hasattr(self.model, 'feature_names_'):
            feature_names = self.model.feature_names_
        elif hasattr(self.model, 'get_feature_names'):
            feature_names = self.model.get_feature_names()
        else:
            feature_names = self.preprocessor.feature_names_fitted

        # Vérifier les longueurs
        print(f"\n🔍 Vérification:")
        print(f"   Nombre de noms de features: {len(feature_names)}")
        print(f"   Nombre de valeurs d'importance: {len(feature_importance)}")

        # S'assurer que les longueurs correspondent
        if len(feature_importance) != len(feature_names):
            print(f"\n⚠️  Ajustement: longueurs différentes")
            # Si l'importance est plus courte, utiliser seulement les premières features
            # Si l'importance est plus longue, tronquer
            min_len = min(len(feature_importance), len(feature_names))
            feature_importance = feature_importance[:min_len]
            feature_names = list(feature_names[:min_len])
            print(f"   ✅ Ajusté à {min_len} features")
        else:
            print(f"   ✅ Longueurs cohérentes")

        # Créer DataFrame
        df_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': feature_importance
        }).sort_values('importance', ascending=False)

        # Afficher top 20
        print("\n📊 Top 20 features les plus importantes:")
        for i, row in df_importance.head(20).iterrows():
            print(f"   {row['feature']:40s} : {row['importance']:8.2f}")

        # Visualisation
        fig, axes = plt.subplots(2, 2, figsize=(18, 14))
        fig.suptitle('ANALYSE DE L\'IMPORTANCE DES FEATURES - CatBoost',
                     fontsize=16, fontweight='bold', y=0.995)

        # 1. Top 30 features
        ax = axes[0, 0]
        top_30 = df_importance.head(30).iloc[::-1]
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(top_30)))
        ax.barh(range(len(top_30)), top_30['importance'], color=colors, alpha=0.8)
        ax.set_yticks(range(len(top_30)))
        ax.set_yticklabels(top_30['feature'], fontsize=8)
        ax.set_xlabel('Importance', fontweight='bold')
        ax.set_title('Top 30 Features les Plus Importantes', fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')

        # 2. Distribution de l'importance
        ax = axes[0, 1]
        ax.hist(df_importance['importance'], bins=50, color='#3498db', alpha=0.7, edgecolor='black')
        ax.axvline(df_importance['importance'].median(), color='red',
                   linestyle='--', linewidth=2, label=f'Médiane: {df_importance["importance"].median():.2f}')
        ax.axvline(df_importance['importance'].mean(), color='green',
                   linestyle='--', linewidth=2, label=f'Moyenne: {df_importance["importance"].mean():.2f}')
        ax.set_xlabel('Importance', fontweight='bold')
        ax.set_ylabel('Nombre de features', fontweight='bold')
        ax.set_title('Distribution de l\'Importance', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 3. Importance cumulée
        ax = axes[1, 0]
        df_importance_sorted = df_importance.sort_values('importance', ascending=False).reset_index(drop=True)
        cumsum = df_importance_sorted['importance'].cumsum()
        cumsum_pct = 100 * cumsum / cumsum.iloc[-1]

        ax.plot(range(len(cumsum_pct)), cumsum_pct, linewidth=2, color='#2ecc71')
        ax.axhline(y=80, color='red', linestyle='--', alpha=0.7, label='80% de l\'importance')
        ax.axhline(y=90, color='orange', linestyle='--', alpha=0.7, label='90% de l\'importance')

        # Trouver nombre de features pour 80% et 90%
        n_80 = (cumsum_pct >= 80).idxmax()
        n_90 = (cumsum_pct >= 90).idxmax()

        ax.axvline(x=n_80, color='red', linestyle='--', alpha=0.5)
        ax.axvline(x=n_90, color='orange', linestyle='--', alpha=0.5)

        ax.text(n_80, 85, f'{n_80} features\n(80%)', ha='center', fontweight='bold', fontsize=9)
        ax.text(n_90, 95, f'{n_90} features\n(90%)', ha='center', fontweight='bold', fontsize=9)

        ax.set_xlabel('Nombre de features', fontweight='bold')
        ax.set_ylabel('Importance cumulée (%)', fontweight='bold')
        ax.set_title('Importance Cumulée des Features', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 4. Catégories de features
        ax = axes[1, 1]

        # Identifier les catégories
        categories = {
            'Fréquences': [],
            'Log transforms': [],
            'Interactions': [],
            'Ratios': [],
            'Autres': []
        }

        for feat in feature_names:
            if '_freq' in feat:
                categories['Fréquences'].append(feat)
            elif 'log_' in feat:
                categories['Log transforms'].append(feat)
            elif '_x_' in feat:
                categories['Interactions'].append(feat)
            elif 'ratio' in feat or 'ecart' in feat:
                categories['Ratios'].append(feat)
            else:
                categories['Autres'].append(feat)

        # Calculer importance moyenne par catégorie
        cat_importance = {}
        for cat, feats in categories.items():
            if feats:
                imp = df_importance[df_importance['feature'].isin(feats)]['importance'].sum()
                cat_importance[cat] = imp

        # Plot
        cats = list(cat_importance.keys())
        imps = list(cat_importance.values())
        colors_cat = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12', '#9b59b6']

        bars = ax.bar(cats, imps, color=colors_cat, alpha=0.7, edgecolor='black', linewidth=1.5)
        ax.set_ylabel('Importance totale', fontweight='bold')
        ax.set_title('Importance par Catégorie de Features', fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

        # Ajouter valeurs
        for bar, val in zip(bars, imps):
            ax.text(bar.get_x() + bar.get_width()/2, val + max(imps)*0.02,
                   f'{val:.1f}', ha='center', fontweight='bold', fontsize=10)

        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        output_path = self.output_dir / '01_feature_importance.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\n✅ Graphique sauvegardé: {output_path}")
        plt.close()

        # Sauvegarder CSV
        csv_path = self.output_dir / 'feature_importance.csv'
        df_importance.to_csv(csv_path, index=False)
        print(f"✅ CSV sauvegardé: {csv_path}")

        return df_importance

    def analyze_feature_distributions(self, df_importance):
        """Analyser la distribution des features importantes selon la classe"""
        print("\n" + "="*80)
        print("📊 ANALYSE DES DISTRIBUTIONS PAR CLASSE")
        print("="*80)

        # Top 12 features
        top_features = df_importance.head(12)['feature'].tolist()

        # Créer DataFrame avec features brutes
        df_analysis = pd.DataFrame(self.X_test, columns=self.preprocessor.feature_names_fitted)
        df_analysis['Fondee'] = self.y_test
        df_analysis['Prediction'] = self.y_pred

        fig, axes = plt.subplots(4, 3, figsize=(18, 16))
        fig.suptitle('DISTRIBUTION DES TOP 12 FEATURES PAR CLASSE',
                     fontsize=16, fontweight='bold', y=0.995)
        axes = axes.ravel()

        for idx, feature in enumerate(top_features):
            ax = axes[idx]

            # Séparer par classe réelle
            data_0 = df_analysis[df_analysis['Fondee'] == 0][feature]
            data_1 = df_analysis[df_analysis['Fondee'] == 1][feature]

            # Histogrammes
            ax.hist(data_0, bins=30, alpha=0.6, label='Non Fondée (0)',
                   color='#e74c3c', edgecolor='black')
            ax.hist(data_1, bins=30, alpha=0.6, label='Fondée (1)',
                   color='#2ecc71', edgecolor='black')

            # Lignes médianes
            ax.axvline(data_0.median(), color='#e74c3c', linestyle='--', linewidth=2)
            ax.axvline(data_1.median(), color='#2ecc71', linestyle='--', linewidth=2)

            ax.set_xlabel(feature, fontsize=8, fontweight='bold')
            ax.set_ylabel('Fréquence', fontsize=8, fontweight='bold')
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)

            # Statistiques
            ax.text(0.02, 0.98,
                   f'Médiane 0: {data_0.median():.2f}\nMédiane 1: {data_1.median():.2f}',
                   transform=ax.transAxes, fontsize=7,
                   verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()

        output_path = self.output_dir / '02_feature_distributions.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\n✅ Graphique sauvegardé: {output_path}")
        plt.close()

    def analyze_errors(self):
        """Analyser les erreurs du modèle pour comprendre où il se trompe"""
        print("\n" + "="*80)
        print("❌ ANALYSE DES ERREURS")
        print("="*80)

        # Créer DataFrame d'analyse
        df_analysis = self.df_2025.copy()
        df_analysis['y_true'] = self.y_test
        df_analysis['y_pred'] = self.y_pred
        df_analysis['y_prob'] = self.y_prob

        # Identifier les erreurs
        df_analysis['correct'] = df_analysis['y_true'] == df_analysis['y_pred']
        df_analysis['error_type'] = 'Correct'
        df_analysis.loc[(df_analysis['y_true'] == 0) & (df_analysis['y_pred'] == 1), 'error_type'] = 'FP (Faux Positif)'
        df_analysis.loc[(df_analysis['y_true'] == 1) & (df_analysis['y_pred'] == 0), 'error_type'] = 'FN (Faux Négatif)'

        # Statistiques
        n_total = len(df_analysis)
        n_correct = df_analysis['correct'].sum()
        n_fp = (df_analysis['error_type'] == 'FP (Faux Positif)').sum()
        n_fn = (df_analysis['error_type'] == 'FN (Faux Négatif)').sum()

        print(f"\n📊 Statistiques globales:")
        print(f"   Total: {n_total}")
        print(f"   Correct: {n_correct} ({100*n_correct/n_total:.2f}%)")
        print(f"   Faux Positifs (FP): {n_fp} ({100*n_fp/n_total:.2f}%)")
        print(f"   Faux Négatifs (FN): {n_fn} ({100*n_fn/n_total:.2f}%)")

        # Visualisations
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('ANALYSE DES ERREURS DU MODÈLE',
                     fontsize=16, fontweight='bold', y=0.995)

        # 1. Distribution des probabilités par type d'erreur
        ax = axes[0, 0]
        for error_type in ['Correct', 'FP (Faux Positif)', 'FN (Faux Négatif)']:
            data = df_analysis[df_analysis['error_type'] == error_type]['y_prob']
            if len(data) > 0:
                ax.hist(data, bins=30, alpha=0.6, label=error_type, edgecolor='black')

        ax.set_xlabel('Probabilité prédite', fontweight='bold')
        ax.set_ylabel('Fréquence', fontweight='bold')
        ax.set_title('Distribution des Probabilités par Type', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 2. Montants demandés par type d'erreur
        ax = axes[0, 1]
        error_types = ['Correct', 'FP (Faux Positif)', 'FN (Faux Négatif)']
        medians = [df_analysis[df_analysis['error_type'] == et]['Montant demandé'].median()
                   for et in error_types]
        colors = ['#2ecc71', '#e74c3c', '#f39c12']

        bars = ax.bar(error_types, medians, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
        ax.set_ylabel('Montant médian demandé (DH)', fontweight='bold')
        ax.set_title('Montant Demandé par Type d\'Erreur', fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

        for bar, val in zip(bars, medians):
            ax.text(bar.get_x() + bar.get_width()/2, val + max(medians)*0.02,
                   f'{val:,.0f}', ha='center', fontweight='bold')

        plt.setp(ax.xaxis.get_majorticklabels(), rotation=15, ha='right')

        # 3. Erreurs par famille de produit
        ax = axes[0, 2]
        errors_by_family = df_analysis[~df_analysis['correct']].groupby('Famille Produit').size()
        errors_by_family = errors_by_family.sort_values(ascending=False).head(15)

        ax.barh(range(len(errors_by_family)), errors_by_family.values, color='#e74c3c', alpha=0.7)
        ax.set_yticks(range(len(errors_by_family)))
        ax.set_yticklabels(errors_by_family.index, fontsize=8)
        ax.set_xlabel('Nombre d\'erreurs', fontweight='bold')
        ax.set_title('Top 15 Familles avec le Plus d\'Erreurs', fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')

        # 4. Taux d'erreur par famille
        ax = axes[1, 0]
        family_stats = df_analysis.groupby('Famille Produit').agg({
            'correct': ['sum', 'count']
        })
        family_stats.columns = ['correct', 'total']
        family_stats['error_rate'] = 100 * (1 - family_stats['correct'] / family_stats['total'])
        family_stats = family_stats[family_stats['total'] >= 50]  # Au moins 50 cas
        family_stats = family_stats.sort_values('error_rate', ascending=False).head(15)

        colors_err = ['#e74c3c' if x > 5 else '#f39c12' if x > 2 else '#2ecc71'
                      for x in family_stats['error_rate']]

        ax.barh(range(len(family_stats)), family_stats['error_rate'], color=colors_err, alpha=0.7)
        ax.set_yticks(range(len(family_stats)))
        ax.set_yticklabels(family_stats.index, fontsize=8)
        ax.set_xlabel('Taux d\'erreur (%)', fontweight='bold')
        ax.set_title('Top 15 Familles avec Taux d\'Erreur le Plus Élevé (n≥50)', fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')

        # 5. Impact financier des erreurs
        ax = axes[1, 1]

        df_fp = df_analysis[df_analysis['error_type'] == 'FP (Faux Positif)']
        df_fn = df_analysis[df_analysis['error_type'] == 'FN (Faux Négatif)']

        perte_fp = df_fp['Montant demandé'].sum()
        perte_fn = 2 * df_fn['Montant demandé'].sum()  # Pénalité x2

        data_impact = {
            f'FP\n({n_fp} cas)': perte_fp,
            f'FN\n({n_fn} cas)': perte_fn
        }

        bars = ax.bar(data_impact.keys(), data_impact.values(),
                     color=['#e74c3c', '#f39c12'], alpha=0.7, edgecolor='black', linewidth=1.5)
        ax.set_ylabel('Perte financière (DH)', fontweight='bold')
        ax.set_title('Impact Financier des Erreurs', fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        ax.ticklabel_format(style='plain', axis='y')

        for bar, val in zip(bars, data_impact.values()):
            ax.text(bar.get_x() + bar.get_width()/2, val + max(data_impact.values())*0.02,
                   f'{val:,.0f} DH', ha='center', fontweight='bold', fontsize=9)

        # 6. Confiance du modèle sur les erreurs
        ax = axes[1, 2]

        # Séparer par type d'erreur et niveau de confiance
        fp_high = df_fp[df_fp['y_prob'] >= 0.8]
        fp_low = df_fp[df_fp['y_prob'] < 0.8]
        fn_high = df_fn[df_fn['y_prob'] <= 0.2]
        fn_low = df_fn[df_fn['y_prob'] > 0.2]

        data_confidence = {
            'FP\nHaute confiance\n(prob≥0.8)': len(fp_high),
            'FP\nBasse confiance\n(prob<0.8)': len(fp_low),
            'FN\nHaute confiance\n(prob≤0.2)': len(fn_high),
            'FN\nBasse confiance\n(prob>0.2)': len(fn_low)
        }

        colors_conf = ['#c0392b', '#e74c3c', '#d68910', '#f39c12']
        bars = ax.bar(data_confidence.keys(), data_confidence.values(),
                     color=colors_conf, alpha=0.7, edgecolor='black', linewidth=1.5)
        ax.set_ylabel('Nombre de cas', fontweight='bold')
        ax.set_title('Niveau de Confiance sur les Erreurs', fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

        for bar, val in zip(bars, data_confidence.values()):
            ax.text(bar.get_x() + bar.get_width()/2, val + max(data_confidence.values())*0.02,
                   f'{val}', ha='center', fontweight='bold', fontsize=9)

        plt.setp(ax.xaxis.get_majorticklabels(), rotation=0, ha='center', fontsize=8)

        plt.tight_layout()

        output_path = self.output_dir / '03_error_analysis.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\n✅ Graphique sauvegardé: {output_path}")
        plt.close()

        # Export des cas d'erreurs pour analyse détaillée
        errors_df = df_analysis[~df_analysis['correct']].copy()
        errors_df = errors_df.sort_values('y_prob', ascending=False)

        csv_path = self.output_dir / 'errors_detail.csv'
        errors_df.to_csv(csv_path, index=False)
        print(f"✅ Détail des erreurs sauvegardé: {csv_path}")

    def generate_recommendations(self, df_importance):
        """Générer des recommandations d'amélioration"""
        print("\n" + "="*80)
        print("💡 RECOMMANDATIONS D'AMÉLIORATION")
        print("="*80)

        report_path = self.output_dir / 'recommendations.txt'

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("RECOMMANDATIONS POUR AMÉLIORER LE MODÈLE\n")
            f.write("="*80 + "\n\n")

            # 1. Features importantes
            f.write("1. FEATURES LES PLUS IMPORTANTES\n")
            f.write("-" * 80 + "\n")
            top_10 = df_importance.head(10)
            f.write("Top 10 features qui pilotent les prédictions:\n\n")
            for i, row in top_10.iterrows():
                f.write(f"   {i+1:2d}. {row['feature']:40s} - Importance: {row['importance']:8.2f}\n")

            f.write("\n💡 Recommandation:\n")
            f.write("   - Vérifiez la qualité de ces features dans vos données sources\n")
            f.write("   - Assurez-vous qu'elles sont bien renseignées et sans erreurs\n")
            f.write("   - Ces features sont critiques pour la performance du modèle\n\n")

            # 2. Features peu importantes
            f.write("\n2. FEATURES PEU IMPORTANTES\n")
            f.write("-" * 80 + "\n")
            bottom_features = df_importance[df_importance['importance'] < 1.0]
            f.write(f"Nombre de features avec importance < 1.0: {len(bottom_features)}\n\n")

            f.write("💡 Recommandation:\n")
            f.write("   - Considérez supprimer ces features pour simplifier le modèle\n")
            f.write("   - Cela peut réduire l'overfitting et améliorer la généralisation\n")
            f.write("   - Testez la performance après suppression\n\n")

            # 3. Catégories de features
            f.write("\n3. TYPES DE FEATURES\n")
            f.write("-" * 80 + "\n")

            freq_features = df_importance[df_importance['feature'].str.contains('_freq', na=False)]
            log_features = df_importance[df_importance['feature'].str.contains('log_', na=False)]
            interaction_features = df_importance[df_importance['feature'].str.contains('_x_', na=False)]
            ratio_features = df_importance[df_importance['feature'].str.contains('ratio|ecart', na=False)]

            f.write(f"Features de fréquence (_freq):     {len(freq_features):3d} - Importance totale: {freq_features['importance'].sum():.1f}\n")
            f.write(f"Features log (log_):                {len(log_features):3d} - Importance totale: {log_features['importance'].sum():.1f}\n")
            f.write(f"Features d'interaction (_x_):       {len(interaction_features):3d} - Importance totale: {interaction_features['importance'].sum():.1f}\n")
            f.write(f"Features de ratio/écart:            {len(ratio_features):3d} - Importance totale: {ratio_features['importance'].sum():.1f}\n")

            f.write("\n💡 Recommandation:\n")
            if freq_features['importance'].sum() > df_importance['importance'].sum() * 0.3:
                f.write("   - Les fréquences catégorielles sont très importantes\n")
                f.write("   - Considérez ajouter plus d'encodages sophistiqués (target encoding, etc.)\n")

            if interaction_features['importance'].sum() < 10:
                f.write("   - Les interactions actuelles apportent peu\n")
                f.write("   - Testez de nouvelles interactions entre features importantes\n")

            f.write("\n")

            # 4. Pistes d'amélioration
            f.write("\n4. PISTES D'AMÉLIORATION PRIORITAIRES\n")
            f.write("-" * 80 + "\n\n")

            f.write("A. Ingénierie de features:\n")
            f.write("   □ Créer des features temporelles (jour de la semaine, mois, trimestre)\n")
            f.write("   □ Ajouter des statistiques agrégées par client (nombre réclamations, montant moyen)\n")
            f.write("   □ Créer des ratios entre features importantes\n")
            f.write("   □ Tester des encodages catégoriels avancés (target encoding, CatBoost encoding)\n")
            f.write("   □ Ajouter des features de délai/ancienneté\n\n")

            f.write("B. Qualité des données:\n")
            f.write("   □ Vérifier les valeurs manquantes sur les top features\n")
            f.write("   □ Détecter et corriger les outliers\n")
            f.write("   □ Harmoniser les formats de données catégorielles\n")
            f.write("   □ Enrichir avec des données externes si possible\n\n")

            f.write("C. Optimisation du modèle:\n")
            f.write("   □ Ré-optimiser les hyperparamètres avec plus d'essais Optuna\n")
            f.write("   □ Tester différentes profondeurs d'arbres\n")
            f.write("   □ Ajuster le learning rate\n")
            f.write("   □ Expérimenter avec class_weights pour gérer le déséquilibre\n\n")

            f.write("D. Analyse des erreurs:\n")
            f.write("   □ Analyser en détail les FP et FN (voir errors_detail.csv)\n")
            f.write("   □ Identifier des patterns dans les erreurs\n")
            f.write("   □ Créer des features spécifiques pour corriger ces erreurs\n")
            f.write("   □ Considérer un modèle en deux étapes pour les cas difficiles\n\n")

            f.write("\n" + "="*80 + "\n")

        print(f"✅ Recommandations sauvegardées: {report_path}")

        # Afficher aussi à l'écran
        with open(report_path, 'r', encoding='utf-8') as f:
            print("\n" + f.read())

    def run(self):
        """Exécution complète de l'analyse"""
        self.load_data_and_model()

        # 1. Feature importance
        df_importance = self.analyze_feature_importance()

        # 2. Distributions
        self.analyze_feature_distributions(df_importance)

        # 3. Analyse des erreurs
        self.analyze_errors()

        # 4. Recommandations
        self.generate_recommendations(df_importance)

        print("\n" + "="*80)
        print("✅ ANALYSE D'INTERPRÉTABILITÉ TERMINÉE")
        print("="*80)
        print(f"\n📂 Tous les résultats sont dans: {self.output_dir}")
        print("\nFichiers générés:")
        print("   - 01_feature_importance.png      : Importance des features")
        print("   - 02_feature_distributions.png   : Distributions par classe")
        print("   - 03_error_analysis.png          : Analyse des erreurs")
        print("   - feature_importance.csv         : Détail importance features")
        print("   - errors_detail.csv              : Détail de chaque erreur")
        print("   - recommendations.txt            : Recommandations d'amélioration")


if __name__ == '__main__':
    interpreter = ModelInterpreter()
    interpreter.run()
