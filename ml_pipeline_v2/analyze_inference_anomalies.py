"""
ANALYSE DES ANOMALIES POST-INFÉRENCE
Identifie les décisions suspectes après inférence pour détecter les incohérences

Usage:
    python ml_pipeline_v2/analyze_inference_anomalies.py --input_file predictions.xlsx
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
plt.rcParams['figure.figsize'] = (20, 12)


class InferenceAnomalyAnalyzer:
    """Analyse les anomalies dans les décisions d'inférence"""

    def __init__(self, input_file):
        self.input_file = input_file
        self.output_dir = Path('outputs/anomaly_analysis')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.anomalies = []

    def load_data(self):
        """Charger les résultats d'inférence"""
        print("\n" + "="*80)
        print("📂 CHARGEMENT DES DONNÉES D'INFÉRENCE")
        print("="*80)

        self.df = pd.read_excel(self.input_file)
        print(f"✅ {len(self.df)} réclamations chargées")

        # Vérifier colonnes nécessaires
        required_cols = ['Decision_Modele', 'Probabilite_Fondee']
        missing = [c for c in required_cols if c not in self.df.columns]
        if missing:
            raise ValueError(f"Colonnes manquantes: {missing}")

        # Statistiques initiales
        print("\n📊 Distribution des décisions AVANT règle métier:")
        for decision in ['Rejet Auto', 'Audit Humain', 'Validation Auto']:
            count = (self.df['Decision_Modele'] == decision).sum()
            pct = count / len(self.df) * 100
            print(f"   {decision:20s}: {count:6d} ({pct:5.1f}%)")

    def apply_business_rule(self):
        """Appliquer la règle métier: 1 validation par client par an"""
        print("\n" + "="*80)
        print("📋 APPLICATION RÈGLE MÉTIER (1 validation/client/an)")
        print("="*80)

        if 'Date de Qualification' not in self.df.columns:
            print("⚠️  Colonne 'Date de Qualification' manquante, règle non applicable")
            self.df_after_rule = self.df.copy()
            return

        df = self.df.copy()

        # Convertir dates
        df['Date de Qualification'] = pd.to_datetime(df['Date de Qualification'], errors='coerce')

        # Identifier colonne client
        client_col = None
        for possible_col in ['Identifiant client', 'ID Client', 'Client ID']:
            if possible_col in df.columns:
                client_col = possible_col
                break

        if client_col is None:
            print("⚠️  Colonne identifiant client manquante")
            self.df_after_rule = self.df.copy()
            return

        # Extraire année
        df['annee'] = df['Date de Qualification'].dt.year

        # Trier par date (garder la première validation par client/année)
        df = df.sort_values(['annee', client_col, 'Date de Qualification'])

        # Numéroter les validations par client/année
        df['validation_number'] = df.groupby([client_col, 'annee']).cumcount() + 1

        # Identifier les cas à convertir
        mask_to_convert = (df['Decision_Modele'] == 'Validation Auto') & (df['validation_number'] > 1)
        n_converted = mask_to_convert.sum()

        if n_converted > 0:
            print(f"✅ {n_converted} validations converties en Audit Humain")
            df.loc[mask_to_convert, 'Decision_Modele'] = 'Audit Humain'
            df.loc[mask_to_convert, 'Decision_Code'] = -1
            df.loc[mask_to_convert, 'Raison_Conversion'] = 'Règle métier: >1 validation/client/an'

        # Statistiques après règle
        print("\n📊 Distribution des décisions APRÈS règle métier:")
        for decision in ['Rejet Auto', 'Audit Humain', 'Validation Auto']:
            count = (df['Decision_Modele'] == decision).sum()
            pct = count / len(df) * 100
            print(f"   {decision:20s}: {count:6d} ({pct:5.1f}%)")

        # Supprimer colonnes temporaires
        df = df.drop(['annee', 'validation_number'], axis=1, errors='ignore')

        self.df_after_rule = df

    def analyze_validation_profiles(self):
        """Analyser les profils des validations automatiques"""
        print("\n" + "="*80)
        print("🔍 ANALYSE DES PROFILS DE VALIDATION AUTO")
        print("="*80)

        df = self.df_after_rule.copy()

        # Séparer par décision
        df_validation = df[df['Decision_Modele'] == 'Validation Auto'].copy()
        df_rejet = df[df['Decision_Modele'] == 'Rejet Auto'].copy()

        print(f"\n📊 Validations Auto: {len(df_validation)}")
        print(f"📊 Rejets Auto: {len(df_rejet)}")

        # Analyser les métriques numériques
        numeric_cols = ['Montant demandé', 'Délai estimé', 'anciennete_annees',
                       'PNB analytique (vision commerciale) cumulé', 'Probabilite_Fondee']

        stats_comparison = []

        for col in numeric_cols:
            if col not in df.columns:
                continue

            val_data = df_validation[col][df_validation[col] > 0] if col != 'Probabilite_Fondee' else df_validation[col]
            rej_data = df_rejet[col][df_rejet[col] > 0] if col != 'Probabilite_Fondee' else df_rejet[col]

            if len(val_data) > 0 and len(rej_data) > 0:
                stats_comparison.append({
                    'Variable': col,
                    'Validation_Mean': val_data.mean(),
                    'Validation_Median': val_data.median(),
                    'Rejet_Mean': rej_data.mean(),
                    'Rejet_Median': rej_data.median(),
                    'Diff_Mean_%': ((val_data.mean() - rej_data.mean()) / rej_data.mean() * 100) if rej_data.mean() != 0 else 0
                })

        if stats_comparison:
            df_stats = pd.DataFrame(stats_comparison)
            print("\n📊 Comparaison Validation vs Rejet:")
            print(df_stats.to_string(index=False))

        return df_stats if stats_comparison else None

    def detect_anomalies(self):
        """Détecter les anomalies dans les validations"""
        print("\n" + "="*80)
        print("🚨 DÉTECTION DES ANOMALIES")
        print("="*80)

        df = self.df_after_rule.copy()
        df_validation = df[df['Decision_Modele'] == 'Validation Auto'].copy()

        if len(df_validation) == 0:
            print("⚠️  Aucune validation automatique à analyser")
            return

        anomalies = []

        # Calculer les médianes globales
        medians = {}
        for col in ['Montant demandé', 'Délai estimé', 'anciennete_annees',
                   'PNB analytique (vision commerciale) cumulé']:
            if col in df.columns:
                data = df[col][df[col] > 0]
                if len(data) > 0:
                    medians[col] = data.median()

        print(f"\n📊 Médianes globales:")
        for col, val in medians.items():
            print(f"   {col}: {val:,.2f}")

        # Anomalie 1: Montant très élevé avec validation
        if 'Montant demandé' in df_validation.columns and 'Montant demandé' in medians:
            threshold_high = medians['Montant demandé'] * 3  # 3x la médiane
            mask_high_amount = df_validation['Montant demandé'] > threshold_high

            if mask_high_amount.sum() > 0:
                anomaly_type = f"Montant > {threshold_high:,.0f} DH (3x médiane)"
                print(f"\n🚨 ANOMALIE 1: {anomaly_type}")
                print(f"   {mask_high_amount.sum()} cas détectés")

                for idx in df_validation[mask_high_amount].index:
                    anomalies.append({
                        'Index': idx,
                        'Type': 'Montant élevé',
                        'Montant': df_validation.loc[idx, 'Montant demandé'],
                        'Probabilite': df_validation.loc[idx, 'Probabilite_Fondee'],
                        'Raison': f"Montant {df_validation.loc[idx, 'Montant demandé']:,.0f} > seuil {threshold_high:,.0f}"
                    })

        # Anomalie 2: Client récent avec PNB faible
        if 'anciennete_annees' in df_validation.columns and 'PNB analytique (vision commerciale) cumulé' in df_validation.columns:
            mask_recent_low_pnb = (
                (df_validation['anciennete_annees'] < 2) &  # Client récent
                (df_validation['PNB analytique (vision commerciale) cumulé'] < medians.get('PNB analytique (vision commerciale) cumulé', 0) * 0.5)  # PNB faible
            )

            if mask_recent_low_pnb.sum() > 0:
                print(f"\n🚨 ANOMALIE 2: Client récent (<2 ans) avec PNB faible")
                print(f"   {mask_recent_low_pnb.sum()} cas détectés")

                for idx in df_validation[mask_recent_low_pnb].index:
                    anomalies.append({
                        'Index': idx,
                        'Type': 'Client récent, PNB faible',
                        'Anciennete': df_validation.loc[idx, 'anciennete_annees'],
                        'PNB': df_validation.loc[idx, 'PNB analytique (vision commerciale) cumulé'],
                        'Probabilite': df_validation.loc[idx, 'Probabilite_Fondee'],
                        'Raison': f"Ancien {df_validation.loc[idx, 'anciennete_annees']:.1f} ans, PNB {df_validation.loc[idx, 'PNB analytique (vision commerciale) cumulé']:,.0f}"
                    })

        # Anomalie 3: Écart extrême à la médiane du montant
        if 'Montant demandé' in df_validation.columns and 'Famille Produit' in df_validation.columns:
            # Calculer médiane par famille
            family_medians = df.groupby('Famille Produit')['Montant demandé'].median().to_dict()

            df_validation['ecart_famille'] = df_validation.apply(
                lambda row: abs(row['Montant demandé'] - family_medians.get(row['Famille Produit'], row['Montant demandé']))
                / (family_medians.get(row['Famille Produit'], 1) + 1),
                axis=1
            )

            mask_extreme_deviation = df_validation['ecart_famille'] > 5  # 5x l'écart

            if mask_extreme_deviation.sum() > 0:
                print(f"\n🚨 ANOMALIE 3: Écart extrême à la médiane famille")
                print(f"   {mask_extreme_deviation.sum()} cas détectés")

                for idx in df_validation[mask_extreme_deviation].index:
                    anomalies.append({
                        'Index': idx,
                        'Type': 'Écart extrême médiane',
                        'Montant': df_validation.loc[idx, 'Montant demandé'],
                        'Ecart': df_validation.loc[idx, 'ecart_famille'],
                        'Probabilite': df_validation.loc[idx, 'Probabilite_Fondee'],
                        'Raison': f"Écart {df_validation.loc[idx, 'ecart_famille']:.1f}x la médiane"
                    })

        # Anomalie 4: Probabilité marginale (proche du seuil)
        if 'Probabilite_Fondee' in df_validation.columns:
            # Identifier les cas avec probabilité dans les 5% au-dessus du seuil
            prob_values = df_validation['Probabilite_Fondee'].values
            if len(prob_values) > 0:
                threshold_high = prob_values.min() + 0.05  # 5% au-dessus du min

                mask_marginal = df_validation['Probabilite_Fondee'] < threshold_high

                if mask_marginal.sum() > 0:
                    print(f"\n🚨 ANOMALIE 4: Probabilité marginale (proche du seuil)")
                    print(f"   {mask_marginal.sum()} cas détectés")

                    for idx in df_validation[mask_marginal].index:
                        anomalies.append({
                            'Index': idx,
                            'Type': 'Probabilité marginale',
                            'Probabilite': df_validation.loc[idx, 'Probabilite_Fondee'],
                            'Raison': f"Prob {df_validation.loc[idx, 'Probabilite_Fondee']:.4f} proche seuil"
                        })

        # Anomalie 5: Délai très élevé avec validation
        if 'Délai estimé' in df_validation.columns and 'Délai estimé' in medians:
            threshold_high_delay = medians['Délai estimé'] * 3

            mask_high_delay = df_validation['Délai estimé'] > threshold_high_delay

            if mask_high_delay.sum() > 0:
                print(f"\n🚨 ANOMALIE 5: Délai très élevé")
                print(f"   {mask_high_delay.sum()} cas détectés")

                for idx in df_validation[mask_high_delay].index:
                    anomalies.append({
                        'Index': idx,
                        'Type': 'Délai élevé',
                        'Delai': df_validation.loc[idx, 'Délai estimé'],
                        'Probabilite': df_validation.loc[idx, 'Probabilite_Fondee'],
                        'Raison': f"Délai {df_validation.loc[idx, 'Délai estimé']:.0f} > {threshold_high_delay:.0f}"
                    })

        # Stocker les anomalies
        self.anomalies = anomalies

        print(f"\n📊 TOTAL ANOMALIES DÉTECTÉES: {len(anomalies)}")

        if len(anomalies) > 0:
            # Compter par type
            anomaly_types = {}
            for a in anomalies:
                anomaly_types[a['Type']] = anomaly_types.get(a['Type'], 0) + 1

            print("\n📊 Répartition par type:")
            for atype, count in sorted(anomaly_types.items(), key=lambda x: x[1], reverse=True):
                print(f"   {atype:30s}: {count:5d}")

    def visualize_anomalies(self):
        """Créer visualisations des anomalies"""
        print("\n" + "="*80)
        print("📊 GÉNÉRATION DES VISUALISATIONS")
        print("="*80)

        df = self.df_after_rule.copy()

        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('ANALYSE DES ANOMALIES - VALIDATIONS AUTOMATIQUES',
                     fontsize=16, fontweight='bold', y=0.995)

        # 1. Distribution des probabilités par décision
        ax = axes[0, 0]
        for decision, color in [('Rejet Auto', 'red'), ('Audit Humain', 'orange'), ('Validation Auto', 'green')]:
            data = df[df['Decision_Modele'] == decision]['Probabilite_Fondee']
            if len(data) > 0:
                ax.hist(data, bins=30, alpha=0.5, label=decision, color=color)

        ax.set_xlabel('Probabilité Fondée', fontweight='bold')
        ax.set_ylabel('Fréquence', fontweight='bold')
        ax.set_title('Distribution des Probabilités par Décision', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 2. Montant par décision
        ax = axes[0, 1]
        if 'Montant demandé' in df.columns:
            decisions_order = ['Rejet Auto', 'Audit Humain', 'Validation Auto']
            data_to_plot = []
            labels = []

            for decision in decisions_order:
                data = df[df['Decision_Modele'] == decision]['Montant demandé']
                data = data[data > 0]
                if len(data) > 0:
                    # Limiter aux percentiles
                    data = data[data <= data.quantile(0.95)]
                    data_to_plot.append(data)
                    labels.append(decision)

            bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True)
            colors = ['red', 'orange', 'green']
            for patch, color in zip(bp['boxes'], colors[:len(data_to_plot)]):
                patch.set_facecolor(color)
                patch.set_alpha(0.5)

            ax.set_ylabel('Montant demandé (DH)', fontweight='bold')
            ax.set_title('Montant par Décision', fontweight='bold')
            ax.tick_params(axis='x', rotation=15)
            ax.grid(True, alpha=0.3, axis='y')

        # 3. PNB vs Ancienneté (validation auto)
        ax = axes[0, 2]
        if 'PNB analytique (vision commerciale) cumulé' in df.columns and 'anciennete_annees' in df.columns:
            df_val = df[df['Decision_Modele'] == 'Validation Auto']
            df_val_clean = df_val[
                (df_val['PNB analytique (vision commerciale) cumulé'] > 0) &
                (df_val['anciennete_annees'] > 0)
            ]

            if len(df_val_clean) > 0:
                ax.scatter(df_val_clean['anciennete_annees'],
                          df_val_clean['PNB analytique (vision commerciale) cumulé'],
                          alpha=0.5, s=50, color='green')

                ax.set_xlabel('Ancienneté (années)', fontweight='bold')
                ax.set_ylabel('PNB cumulé (DH)', fontweight='bold')
                ax.set_title('PNB vs Ancienneté (Validations Auto)', fontweight='bold')
                ax.grid(True, alpha=0.3)

                # Identifier zone suspecte
                median_anc = df_val_clean['anciennete_annees'].median()
                median_pnb = df_val_clean['PNB analytique (vision commerciale) cumulé'].median()

                ax.axvline(median_anc, color='blue', linestyle='--', alpha=0.5, label=f'Médiane anc: {median_anc:.1f}')
                ax.axhline(median_pnb, color='blue', linestyle='--', alpha=0.5, label=f'Médiane PNB: {median_pnb:,.0f}')
                ax.legend()

        # 4. Distribution par famille (top 10)
        ax = axes[1, 0]
        if 'Famille Produit' in df.columns:
            top_families = df['Famille Produit'].value_counts().head(10).index
            df_top = df[df['Famille Produit'].isin(top_families)]

            decision_counts = pd.crosstab(df_top['Famille Produit'], df_top['Decision_Modele'])

            if 'Validation Auto' in decision_counts.columns:
                decision_counts['Pct_Validation'] = (
                    decision_counts['Validation Auto'] / decision_counts.sum(axis=1) * 100
                )
                decision_counts = decision_counts.sort_values('Pct_Validation', ascending=False)

                ax.barh(range(len(decision_counts)), decision_counts['Pct_Validation'], color='green', alpha=0.7)
                ax.set_yticks(range(len(decision_counts)))
                ax.set_yticklabels(decision_counts.index, fontsize=9)
                ax.set_xlabel('% Validation Auto', fontweight='bold')
                ax.set_title('Taux Validation Auto par Famille (Top 10)', fontweight='bold')
                ax.grid(True, alpha=0.3, axis='x')

        # 5. Ratio Montant/PNB par décision
        ax = axes[1, 1]
        if 'Montant demandé' in df.columns and 'PNB analytique (vision commerciale) cumulé' in df.columns:
            df_temp = df[
                (df['Montant demandé'] > 0) &
                (df['PNB analytique (vision commerciale) cumulé'] > 0)
            ].copy()

            df_temp['ratio_montant_pnb'] = df_temp['Montant demandé'] / df_temp['PNB analytique (vision commerciale) cumulé']
            # Limiter aux percentiles
            df_temp = df_temp[
                (df_temp['ratio_montant_pnb'] > df_temp['ratio_montant_pnb'].quantile(0.01)) &
                (df_temp['ratio_montant_pnb'] < df_temp['ratio_montant_pnb'].quantile(0.99))
            ]

            for decision, color in [('Rejet Auto', 'red'), ('Validation Auto', 'green')]:
                data = df_temp[df_temp['Decision_Modele'] == decision]['ratio_montant_pnb']
                if len(data) > 0:
                    ax.hist(data, bins=30, alpha=0.5, label=decision, color=color)

            ax.set_xlabel('Ratio Montant/PNB', fontweight='bold')
            ax.set_ylabel('Fréquence', fontweight='bold')
            ax.set_title('Distribution Ratio Montant/PNB', fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)

        # 6. Anomalies par type
        ax = axes[1, 2]
        if len(self.anomalies) > 0:
            anomaly_counts = {}
            for a in self.anomalies:
                anomaly_counts[a['Type']] = anomaly_counts.get(a['Type'], 0) + 1

            types = list(anomaly_counts.keys())
            counts = list(anomaly_counts.values())

            colors_map = plt.cm.Reds(np.linspace(0.4, 0.9, len(types)))
            ax.barh(range(len(types)), counts, color=colors_map)
            ax.set_yticks(range(len(types)))
            ax.set_yticklabels(types, fontsize=9)
            ax.set_xlabel('Nombre d\'anomalies', fontweight='bold')
            ax.set_title('Anomalies Détectées par Type', fontweight='bold')
            ax.grid(True, alpha=0.3, axis='x')

            for i, count in enumerate(counts):
                ax.text(count + count*0.02, i, f'{count}', va='center', fontweight='bold')

        plt.tight_layout()
        output_path = self.output_dir / 'anomaly_analysis.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ Graphique sauvegardé: {output_path}")
        plt.close()

    def generate_report(self, stats_comparison):
        """Générer rapport d'anomalies"""
        print("\n" + "="*80)
        print("📄 GÉNÉRATION DU RAPPORT")
        print("="*80)

        report_path = self.output_dir / f'rapport_anomalies_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt'

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("RAPPORT D'ANALYSE DES ANOMALIES POST-INFÉRENCE\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*80 + "\n\n")

            f.write(f"Fichier analysé: {self.input_file}\n")
            f.write(f"Nombre total de réclamations: {len(self.df)}\n\n")

            # Distribution finale
            f.write("DISTRIBUTION DES DÉCISIONS (après règle métier):\n")
            f.write("-" * 80 + "\n")
            for decision in ['Rejet Auto', 'Audit Humain', 'Validation Auto']:
                count = (self.df_after_rule['Decision_Modele'] == decision).sum()
                pct = count / len(self.df_after_rule) * 100
                f.write(f"  {decision:20s}: {count:6d} ({pct:5.1f}%)\n")

            # Comparaison profils
            if stats_comparison is not None:
                f.write("\n\nCOMPARAISON VALIDATION vs REJET:\n")
                f.write("-" * 80 + "\n")
                f.write(stats_comparison.to_string(index=False))
                f.write("\n")

            # Anomalies détectées
            f.write("\n\nANOMALIES DÉTECTÉES:\n")
            f.write("-" * 80 + "\n")
            f.write(f"Total: {len(self.anomalies)} anomalies\n\n")

            if len(self.anomalies) > 0:
                # Par type
                anomaly_types = {}
                for a in self.anomalies:
                    anomaly_types[a['Type']] = anomaly_types.get(a['Type'], 0) + 1

                f.write("Répartition par type:\n")
                for atype, count in sorted(anomaly_types.items(), key=lambda x: x[1], reverse=True):
                    pct = count / len(self.anomalies) * 100
                    f.write(f"  {atype:30s}: {count:5d} ({pct:5.1f}%)\n")

                # Top 20 anomalies
                f.write("\n\nTOP 20 ANOMALIES (par index):\n")
                f.write("-" * 80 + "\n")
                for i, anomaly in enumerate(self.anomalies[:20], 1):
                    f.write(f"\n{i}. Index {anomaly['Index']}:\n")
                    f.write(f"   Type: {anomaly['Type']}\n")
                    f.write(f"   Raison: {anomaly['Raison']}\n")
                    if 'Probabilite' in anomaly:
                        f.write(f"   Probabilité: {anomaly['Probabilite']:.4f}\n")

            # Recommandations
            f.write("\n\n" + "="*80 + "\n")
            f.write("RECOMMANDATIONS:\n")
            f.write("="*80 + "\n\n")

            val_pct = (self.df_after_rule['Decision_Modele'] == 'Validation Auto').sum() / len(self.df_after_rule) * 100

            if val_pct > 40:
                f.write(f"1. TAUX DE VALIDATION ÉLEVÉ ({val_pct:.1f}%):\n")
                f.write("   - Vérifier si le modèle est bien calibré\n")
                f.write("   - Examiner les seuils de décision\n")
                f.write("   - Analyser si les données d'inférence sont similaires aux données d'entraînement\n\n")

            if len(self.anomalies) > len(self.df_after_rule) * 0.1:
                f.write(f"2. NOMBRE D'ANOMALIES ÉLEVÉ ({len(self.anomalies)}, {len(self.anomalies)/len(self.df_after_rule)*100:.1f}%):\n")
                f.write("   - Revoir les critères de validation\n")
                f.write("   - Considérer un audit manuel des cas suspects\n")
                f.write("   - Vérifier la qualité des données d'entrée\n\n")

            f.write("3. ACTIONS SUGGÉRÉES:\n")
            f.write("   - Audit manuel des validations avec anomalies\n")
            f.write("   - Analyse approfondie des familles avec taux élevé de validation\n")
            f.write("   - Vérification de la cohérence des données (montants, PNB, ancienneté)\n")

        print(f"✅ Rapport sauvegardé: {report_path}")

        # Export liste anomalies
        if len(self.anomalies) > 0:
            anomalies_path = self.output_dir / 'anomalies_list.xlsx'
            df_anomalies = pd.DataFrame(self.anomalies)
            df_anomalies.to_excel(anomalies_path, index=False)
            print(f"✅ Liste anomalies: {anomalies_path}")

    def run(self):
        """Exécuter l'analyse complète"""
        self.load_data()
        self.apply_business_rule()
        stats_comparison = self.analyze_validation_profiles()
        self.detect_anomalies()
        self.visualize_anomalies()
        self.generate_report(stats_comparison)

        print("\n" + "="*80)
        print("✅ ANALYSE DES ANOMALIES TERMINÉE")
        print("="*80)
        print(f"\n📂 Résultats dans: {self.output_dir}")
        print(f"📊 Anomalies détectées: {len(self.anomalies)}")


def main():
    parser = argparse.ArgumentParser(description='Analyse des anomalies post-inférence')
    parser.add_argument('--input_file', type=str, required=True,
                       help='Fichier Excel avec résultats d\'inférence (doit contenir Decision_Modele)')

    args = parser.parse_args()

    analyzer = InferenceAnomalyAnalyzer(args.input_file)
    analyzer.run()


if __name__ == '__main__':
    main()
