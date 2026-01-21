"""
ANALYSE DES ANOMALIES POST-INFÉRENCE
Identifie les décisions suspectes après inférence pour détecter les incohérences

Usage:
    python ml_pipeline_v2/analyze_inference_anomalies.py --input_file predictions.xlsx

    Si le fichier n'a pas encore été scoré, le script fera automatiquement l'inférence
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
import joblib
warnings.filterwarnings('ignore')

# Import preprocessing
from preprocessor_v2 import ProductionPreprocessorV2

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
        self.model = None
        self.preprocessor = None
        self.feature_importances = None
        self.global_stats = {}

    def run_inference_if_needed(self):
        """Faire l'inférence si les colonnes de décision sont manquantes"""
        required_cols = ['Decision_Modele', 'Probabilite_Fondee']
        missing = [c for c in required_cols if c not in self.df.columns]

        if not missing:
            return  # Les colonnes existent déjà

        print("\n" + "="*80)
        print("🔮 INFÉRENCE AUTOMATIQUE (colonnes manquantes)")
        print("="*80)
        print(f"   Colonnes manquantes: {missing}")
        print("   Chargement du modèle et exécution de l'inférence...")

        # Charger modèle et preprocessor
        model_path = Path('outputs/production_v2/models/best_model_v2.pkl')
        preprocessor_path = Path('outputs/production_v2/models/preprocessor_v2.pkl')
        predictions_path = Path('outputs/production_v2/predictions/predictions_2025_v2.pkl')

        if not model_path.exists():
            raise FileNotFoundError(
                f"Modèle non trouvé: {model_path}\n"
                "Exécutez d'abord: python ml_pipeline_v2/model_comparison_v2.py"
            )

        if not preprocessor_path.exists():
            raise FileNotFoundError(
                f"Preprocessor non trouvé: {preprocessor_path}\n"
                "Exécutez d'abord: python ml_pipeline_v2/model_comparison_v2.py"
            )

        # Charger
        self.model = joblib.load(model_path)
        self.preprocessor = joblib.load(preprocessor_path)
        print("✅ Modèle et preprocessor chargés")

        # Charger seuils
        if predictions_path.exists():
            predictions_data = joblib.load(predictions_path)
            if 'best_model' in predictions_data:
                best_name = predictions_data['best_model']
                threshold_low = predictions_data[best_name]['threshold_low']
                threshold_high = predictions_data[best_name]['threshold_high']
            else:
                threshold_low = 0.3
                threshold_high = 0.7
        else:
            threshold_low = 0.3
            threshold_high = 0.7

        print(f"✅ Seuils: BAS={threshold_low:.4f}, HAUT={threshold_high:.4f}")

        # Préprocessing
        print("\n🔄 Préprocessing des données...")
        X = self.preprocessor.transform(self.df)
        print(f"✅ {X.shape[1]} features générées")

        # Prédiction
        print("\n🎯 Prédiction...")
        y_prob = self.model.predict_proba(X)[:, 1]

        # Décisions
        decisions = []
        decision_codes = []

        for prob in y_prob:
            if prob <= threshold_low:
                decisions.append('Rejet Auto')
                decision_codes.append(-1)
            elif prob >= threshold_high:
                decisions.append('Validation Auto')
                decision_codes.append(1)
            else:
                decisions.append('Audit Humain')
                decision_codes.append(0)

        self.df['Probabilite_Fondee'] = y_prob
        self.df['Decision_Modele'] = decisions
        self.df['Decision_Code'] = decision_codes

        print("✅ Inférence terminée")

    def load_model_if_needed(self):
        """Charger le modèle et preprocessor si pas encore fait"""
        if self.model is not None and self.preprocessor is not None:
            return  # Déjà chargés

        print("\n📂 Chargement du modèle pour explicabilité...")

        model_path = Path('outputs/production_v2/models/best_model_v2.pkl')
        preprocessor_path = Path('outputs/production_v2/models/preprocessor_v2.pkl')

        if not model_path.exists() or not preprocessor_path.exists():
            print("⚠️  Modèle non trouvé - explicabilité désactivée")
            return

        self.model = joblib.load(model_path)
        self.preprocessor = joblib.load(preprocessor_path)
        print("✅ Modèle chargé pour analyse d'explicabilité")

    def clean_numeric_columns(self):
        """Nettoyer les colonnes numériques (convertir texte -> float)"""
        import re

        numeric_cols = ['Montant demandé', 'Délai estimé', 'anciennete_annees',
                       'PNB analytique (vision commerciale) cumulé']

        def clean_numeric_value(val):
            """Convertir une valeur texte en float"""
            if pd.isna(val):
                return np.nan

            if isinstance(val, (int, float)):
                return float(val)

            # Convertir en string
            val_str = str(val).strip().upper()

            # Supprimer currency symbols et texte
            val_str = re.sub(r'(MAD|DH|DHs?|EUR|€|\$)', '', val_str, flags=re.IGNORECASE)
            val_str = val_str.strip()

            if not val_str or val_str == '':
                return np.nan

            # Remplacer les espaces (1 000 -> 1000)
            val_str = val_str.replace(' ', '')

            # Gérer format européen (1.500,50) vs anglais (1,500.50)
            if ',' in val_str and '.' in val_str:
                # Les deux présents - déterminer le format
                comma_pos = val_str.rfind(',')
                dot_pos = val_str.rfind('.')

                if comma_pos > dot_pos:
                    # Format européen: 1.500,50
                    val_str = val_str.replace('.', '').replace(',', '.')
                else:
                    # Format anglais: 1,500.50
                    val_str = val_str.replace(',', '')
            elif ',' in val_str:
                # Seule virgule présente
                # Si 2 chiffres après -> décimal (1,50)
                # Sinon -> séparateur milliers (1,500)
                parts = val_str.split(',')
                if len(parts[-1]) == 2:
                    val_str = val_str.replace(',', '.')
                else:
                    val_str = val_str.replace(',', '')

            try:
                return float(val_str)
            except:
                return np.nan

        print("\n🔄 Nettoyage des colonnes numériques...")
        for col in numeric_cols:
            if col in self.df.columns:
                original_type = self.df[col].dtype
                self.df[col] = self.df[col].apply(clean_numeric_value)
                non_null = self.df[col].notna().sum()
                print(f"   {col}: {non_null} valeurs valides (était {original_type})")

    def load_data(self):
        """Charger les résultats d'inférence"""
        print("\n" + "="*80)
        print("📂 CHARGEMENT DES DONNÉES")
        print("="*80)

        self.df = pd.read_excel(self.input_file)
        print(f"✅ {len(self.df)} réclamations chargées")

        # Nettoyer les colonnes numériques AVANT inférence
        self.clean_numeric_columns()

        # Vérifier et faire inférence si nécessaire
        self.run_inference_if_needed()

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

    def explain_anomalies(self):
        """Expliquer POURQUOI chaque anomalie a une probabilité élevée"""
        print("\n" + "="*80)
        print("🔍 EXPLICABILITÉ DES ANOMALIES")
        print("="*80)

        # Charger modèle si nécessaire
        self.load_model_if_needed()

        if self.model is None or self.preprocessor is None:
            print("⚠️  Modèle non disponible - explicabilité impossible")
            return

        if len(self.anomalies) == 0:
            print("⚠️  Aucune anomalie à expliquer")
            return

        # Obtenir feature importances
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            # Les noms des features correspondent à la sortie du preprocessor
            # On doit transformer une observation pour obtenir les noms
            sample = self.df_after_rule.iloc[[0]].copy()
            X_sample = self.preprocessor.transform(sample)
            feature_names = X_sample.columns.tolist() if hasattr(X_sample, 'columns') else [f'feature_{i}' for i in range(X_sample.shape[1])]

            # Créer dict importance
            feat_importance = dict(zip(feature_names, importances))
            # Trier par importance
            top_features = sorted(feat_importance.items(), key=lambda x: x[1], reverse=True)[:15]

            print(f"\n📊 Top 15 features importantes du modèle:")
            for feat, imp in top_features[:10]:
                print(f"   {feat:40s}: {imp:.4f}")
        else:
            print("⚠️  Feature importances non disponibles")
            top_features = []

        # Calculer statistiques globales des validations vs rejets
        df = self.df_after_rule.copy()
        df_validation = df[df['Decision_Modele'] == 'Validation Auto'].copy()
        df_rejet = df[df['Decision_Modele'] == 'Rejet Auto'].copy()

        # Statistiques de base
        base_cols = ['Montant demandé', 'Délai estimé', 'anciennete_annees',
                    'PNB analytique (vision commerciale) cumulé']

        stats_validation = {}
        stats_rejet = {}

        for col in base_cols:
            if col in df.columns:
                val_data = df_validation[col][df_validation[col] > 0] if len(df_validation) > 0 else pd.Series()
                rej_data = df_rejet[col][df_rejet[col] > 0] if len(df_rejet) > 0 else pd.Series()

                if len(val_data) > 0:
                    stats_validation[col] = {
                        'median': val_data.median(),
                        'mean': val_data.mean(),
                        'q25': val_data.quantile(0.25),
                        'q75': val_data.quantile(0.75)
                    }

                if len(rej_data) > 0:
                    stats_rejet[col] = {
                        'median': rej_data.median(),
                        'mean': rej_data.mean(),
                        'q25': rej_data.quantile(0.25),
                        'q75': rej_data.quantile(0.75)
                    }

        # Expliquer chaque anomalie
        print(f"\n🔍 Analyse détaillée des {len(self.anomalies)} anomalies...")

        explanations = []

        for i, anomaly in enumerate(self.anomalies):
            idx = anomaly['Index']

            if idx not in self.df_after_rule.index:
                continue

            row = self.df_after_rule.loc[idx]

            # Créer explication
            explanation = {
                'Index': idx,
                'Type_Anomalie': anomaly['Type'],
                'Probabilite': anomaly.get('Probabilite', row.get('Probabilite_Fondee', 0)),
            }

            # Ajouter données brutes
            for col in base_cols:
                if col in row.index:
                    explanation[col] = row[col]

            # Ajouter colonnes catégorielles
            for col in ['Famille Produit', 'Catégorie', 'Sous-catégorie', 'Segment', 'Marché']:
                if col in row.index:
                    explanation[col] = row[col]

            # Analyser les features engineered si possible
            try:
                X_row = self.preprocessor.transform(pd.DataFrame([row]))

                # Identifier les features avec valeurs élevées
                if hasattr(X_row, 'values'):
                    row_values = X_row.values[0] if len(X_row.values) > 0 else []
                    feature_names_list = X_row.columns.tolist() if hasattr(X_row, 'columns') else []

                    # Pour les top features importantes, montrer les valeurs
                    top_feat_values = []
                    for feat_name, _ in top_features[:10]:
                        if feat_name in feature_names_list:
                            feat_idx = feature_names_list.index(feat_name)
                            if feat_idx < len(row_values):
                                top_feat_values.append(f"{feat_name}={row_values[feat_idx]:.2f}")

                    explanation['Top_Features_Values'] = '; '.join(top_feat_values[:5])
            except Exception as e:
                explanation['Top_Features_Values'] = f"Erreur: {str(e)}"

            # Construire explication textuelle
            reasons = []

            # Comparer avec médiane des validations
            for col in base_cols:
                if col in row.index and pd.notna(row[col]) and row[col] > 0:
                    val = row[col]

                    if col in stats_validation:
                        median_val = stats_validation[col]['median']
                        diff_pct = ((val - median_val) / median_val * 100) if median_val > 0 else 0

                        if abs(diff_pct) > 50:
                            direction = "supérieur" if diff_pct > 0 else "inférieur"
                            reasons.append(f"{col} {direction} de {abs(diff_pct):.0f}% vs médiane validations ({median_val:,.0f})")

                    # Comparer avec rejets
                    if col in stats_rejet:
                        median_rej = stats_rejet[col]['median']
                        if col == 'Montant demandé' and val < median_rej * 0.5:
                            reasons.append(f"Montant très faible vs rejets (médiane rejets: {median_rej:,.0f})")
                        elif col == 'PNB analytique (vision commerciale) cumulé' and val > median_rej * 2:
                            reasons.append(f"PNB élevé vs rejets (médiane rejets: {median_rej:,.0f})")

            # Raisons spécifiques basées sur les taux de fondée (si catégories connues)
            if 'Famille Produit' in row.index and pd.notna(row['Famille Produit']):
                famille = row['Famille Produit']
                reasons.append(f"Famille: {famille}")

            if 'Catégorie' in row.index and pd.notna(row['Catégorie']):
                categorie = row['Catégorie']
                reasons.append(f"Catégorie: {categorie}")

            explanation['Explication_Detaillee'] = ' | '.join(reasons) if reasons else "Analyse en cours"

            explanations.append(explanation)

        # Stocker les explications
        self.anomaly_explanations = explanations

        print(f"✅ {len(explanations)} anomalies expliquées")

        # Afficher quelques exemples
        if len(explanations) > 0:
            print(f"\n📋 Exemple d'explications (5 premières):")
            for i, exp in enumerate(explanations[:5], 1):
                print(f"\n{i}. Index {exp['Index']} - Type: {exp['Type_Anomalie']}")
                print(f"   Probabilité: {exp['Probabilite']:.4f}")
                if 'Montant demandé' in exp:
                    print(f"   Montant: {exp['Montant demandé']:,.0f} DH")
                if 'PNB analytique (vision commerciale) cumulé' in exp:
                    print(f"   PNB: {exp['PNB analytique (vision commerciale) cumulé']:,.0f} DH")
                if 'anciennete_annees' in exp:
                    print(f"   Ancienneté: {exp['anciennete_annees']:.1f} ans")
                print(f"   Explication: {exp['Explication_Detaillee'][:150]}")

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
            print(f"✅ Liste anomalies basique: {anomalies_path}")

        # Export DÉTAILLÉ avec explications
        if hasattr(self, 'anomaly_explanations') and len(self.anomaly_explanations) > 0:
            explanations_path = self.output_dir / f'anomalies_detaillees_{datetime.now().strftime("%Y%m%d_%H%M%S")}.xlsx'
            df_explanations = pd.DataFrame(self.anomaly_explanations)

            # Créer fichier Excel avec formatage
            with pd.ExcelWriter(explanations_path, engine='openpyxl') as writer:
                df_explanations.to_excel(writer, sheet_name='Anomalies Détaillées', index=False)

                # Ajuster les largeurs de colonnes
                worksheet = writer.sheets['Anomalies Détaillées']
                for idx, col in enumerate(df_explanations.columns):
                    max_length = max(
                        df_explanations[col].astype(str).map(len).max(),
                        len(col)
                    )
                    adjusted_width = min(max_length + 2, 80)
                    worksheet.column_dimensions[chr(65 + idx)].width = adjusted_width

            print(f"✅ Liste DÉTAILLÉE avec explications: {explanations_path}")
            print(f"   📋 Contient: Type anomalie, Probabilité, Montant, PNB, Ancienneté, Famille, Catégorie, Explication détaillée")

    def run(self):
        """Exécuter l'analyse complète"""
        self.load_data()
        self.apply_business_rule()
        stats_comparison = self.analyze_validation_profiles()
        self.detect_anomalies()
        self.explain_anomalies()  # NOUVEAU: Explicabilité détaillée
        self.visualize_anomalies()
        self.generate_report(stats_comparison)

        print("\n" + "="*80)
        print("✅ ANALYSE DES ANOMALIES TERMINÉE")
        print("="*80)
        print(f"\n📂 Résultats dans: {self.output_dir}")
        print(f"📊 Anomalies détectées: {len(self.anomalies)}")
        if hasattr(self, 'anomaly_explanations'):
            print(f"📋 Explications générées: {len(self.anomaly_explanations)}")
            print(f"\n💡 Consultez le fichier 'anomalies_detaillees_*.xlsx' pour voir les explications détaillées")


def main():
    parser = argparse.ArgumentParser(description='Analyse des anomalies post-inférence')
    parser.add_argument('--input_file', type=str, required=True,
                       help='Fichier Excel (avec ou sans inférence - sera scoré automatiquement si nécessaire)')

    args = parser.parse_args()

    analyzer = InferenceAnomalyAnalyzer(args.input_file)
    analyzer.run()


if __name__ == '__main__':
    main()
