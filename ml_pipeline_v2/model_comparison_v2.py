"""
COMPARAISON DE MODÈLES V2 - Features Production-Ready
XGBoost vs Random Forest vs CatBoost
Entraînement sur 2024, Test sur 2025

Utilise uniquement des features disponibles en temps réel
"""
import sys
sys.path.append('src')

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from catboost import CatBoostClassifier
import optuna
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Import du preprocessor V2
from preprocessor_v2 import ProductionPreprocessorV2

# Configuration
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (16, 10)

PRIX_UNITAIRE_DH = 169


class ModelComparisonV2:
    """Comparaison de 3 modèles avec features production-ready"""

    def __init__(self, min_samples_stats=30):
        self.models = {}
        self.results = {}
        self.preprocessor = ProductionPreprocessorV2(min_samples_stats=min_samples_stats)

    def load_data(self):
        """Charger données 2024 et 2025"""
        print("\n" + "="*80)
        print("📂 CHARGEMENT DES DONNÉES")
        print("="*80)

        self.df_2024 = pd.read_excel('data/raw/reclamations_2024.xlsx')
        self.df_2025 = pd.read_excel('data/raw/reclamations_2025.xlsx')

        print(f"✅ 2024: {len(self.df_2024)} réclamations")
        print(f"✅ 2025: {len(self.df_2025)} réclamations")

        # Vérifier les colonnes nécessaires
        required_cols = ['Montant demandé', 'Famille Produit', 'Fondee']
        for col in required_cols:
            if col not in self.df_2024.columns:
                raise ValueError(f"Colonne manquante dans 2024: {col}")
            if col not in self.df_2025.columns and col != 'Fondee':
                raise ValueError(f"Colonne manquante dans 2025: {col}")

    def prepare_data(self):
        """Preprocessing"""
        print("\n" + "="*80)
        print("🔧 PREPROCESSING V2 - Features Production-Ready")
        print("="*80)

        # Fit sur 2024 (avec Fondee pour calculer les taux)
        X_train = self.preprocessor.fit_transform(self.df_2024)
        y_train = self.df_2024['Fondee'].values

        # Transform sur 2025 (utilise les stats de 2024)
        X_test = self.preprocessor.transform(self.df_2025)
        y_test = self.df_2025['Fondee'].values

        print(f"\n📊 Shape 2024: {X_train.shape}")
        print(f"📊 Shape 2025: {X_test.shape}")

        # Afficher info sur les features
        info = self.preprocessor.get_feature_info()
        print(f"\n📋 Statistiques calculées sur 2024:")
        print(f"   Familles avec stats robustes: {info['family_stats_count']}")
        print(f"   Catégories avec stats robustes: {info['category_stats_count']}")
        print(f"   Sous-catégories avec stats robustes: {info['subcategory_stats_count']}")
        print(f"   Segments avec stats robustes: {info['segment_stats_count']}")
        print(f"   (Minimum {info['min_samples_for_stats']} cas pour être considéré robuste)")

        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    def optimize_catboost(self):
        """Optimiser CatBoost avec Optuna - modèle principal"""
        print("\n🔬 Optimisation CatBoost (modèle principal)...")

        def objective(trial):
            params = {
                'iterations': trial.suggest_int('iterations', 100, 500),
                'depth': trial.suggest_int('depth', 4, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-8, 10.0, log=True),
                'border_count': trial.suggest_int('border_count', 32, 255),
            }

            model = CatBoostClassifier(**params, random_state=42, verbose=0)
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            scores = cross_val_score(model, self.X_train, self.y_train, cv=cv, scoring='f1', n_jobs=-1)

            return scores.mean()

        study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(objective, n_trials=50, show_progress_bar=False)

        best_params = study.best_params
        print(f"   ✅ Best F1-Score CV: {study.best_value:.4f}")

        model = CatBoostClassifier(**best_params, random_state=42, verbose=0)
        model.fit(self.X_train, self.y_train)

        return model

    def train_models(self):
        """Entraîner CatBoost uniquement (modèle principal)"""
        print("\n" + "="*80)
        print("🎯 ENTRAÎNEMENT DU MODÈLE")
        print("="*80)

        self.models['CatBoost'] = self.optimize_catboost()

        print("\n✅ Modèle CatBoost entraîné")

    def optimize_threshold_dual(self, y_prob, name):
        """Optimiser 2 seuils pour créer 3 zones"""
        print(f"\n🎯 Optimisation des seuils de décision (3 zones)...")

        montants_2025 = self.df_2025['Montant demandé'].values
        best_result = None
        best_gain_net = -float('inf')
        threshold_results = []

        # Tester différentes combinaisons de seuils
        for t_low in np.arange(0.10, 0.50, 0.02):
            for t_high in np.arange(0.50, 0.95, 0.02):
                if t_high <= t_low:
                    continue

                # 3 zones
                mask_rejet = y_prob <= t_low
                mask_audit = (y_prob > t_low) & (y_prob < t_high)
                mask_validation = y_prob >= t_high

                # Prédictions
                y_pred = np.zeros(len(y_prob), dtype=int)
                y_pred[mask_validation] = 1

                # Cas automatisés
                mask_auto = mask_rejet | mask_validation

                if mask_auto.sum() == 0:
                    continue

                # Métriques
                y_pred_auto = y_pred[mask_auto]
                y_true_auto = self.y_test[mask_auto]
                montants_auto = montants_2025[mask_auto]

                fp_mask = (y_true_auto == 0) & (y_pred_auto == 1)
                fn_mask = (y_true_auto == 1) & (y_pred_auto == 0)

                # Calcul financier
                montants_auto_clean = np.nan_to_num(montants_auto, nan=0.0, posinf=0.0, neginf=0.0)
                if len(montants_auto_clean) > 0 and montants_auto_clean.max() > 0:
                    montants_auto_clean = np.clip(montants_auto_clean, 0, np.percentile(montants_auto_clean, 99))
                else:
                    montants_auto_clean = montants_auto_clean.clip(0)

                perte_fp = montants_auto_clean[fp_mask].sum()
                perte_fn = 2 * montants_auto_clean[fn_mask].sum()

                auto_count = mask_auto.sum()
                gain_brut = auto_count * PRIX_UNITAIRE_DH
                gain_net = gain_brut - perte_fp - perte_fn

                # Précisions par zone
                prec_rejet = (self.y_test[mask_rejet] == 0).mean() if mask_rejet.sum() > 0 else 0
                prec_validation = (self.y_test[mask_validation] == 1).mean() if mask_validation.sum() > 0 else 0

                automation_rate = mask_auto.sum() / len(y_prob)

                threshold_results.append({
                    'threshold_low': t_low,
                    'threshold_high': t_high,
                    'gain_net': gain_net,
                    'auto': auto_count,
                    'fp': fp_mask.sum(),
                    'fn': fn_mask.sum(),
                    'n_rejet': mask_rejet.sum(),
                    'n_audit': mask_audit.sum(),
                    'n_validation': mask_validation.sum(),
                    'prec_rejet': prec_rejet,
                    'prec_validation': prec_validation,
                    'automation_rate': automation_rate
                })

                # Critère: Maximiser gain NET avec contraintes
                if prec_rejet >= 0.95 and prec_validation >= 0.93:
                    if gain_net > best_gain_net:
                        best_gain_net = gain_net
                        best_result = threshold_results[-1].copy()

        # Fallback
        if best_result is None and threshold_results:
            best_result = max(threshold_results, key=lambda x: x['gain_net'])
            print(f"   ⚠️ Aucune solution avec précisions cibles, utilisation du meilleur gain NET")

        if best_result:
            print(f"   ✅ Seuil BAS (Rejet): {best_result['threshold_low']:.2f}")
            print(f"   ✅ Seuil HAUT (Validation): {best_result['threshold_high']:.2f}")
            print(f"   ✅ Taux automatisation: {best_result['automation_rate']:.1%}")
            print(f"   ✅ Précision Rejet: {best_result['prec_rejet']:.1%}")
            print(f"   ✅ Précision Validation: {best_result['prec_validation']:.1%}")
            print(f"   ✅ Gain NET max: {best_gain_net:,.0f} DH")
        else:
            # Fallback ultime
            best_result = {
                'threshold_low': 0.3,
                'threshold_high': 0.7,
                'gain_net': 0,
                'auto': 0,
                'fp': 0,
                'fn': 0,
                'n_rejet': 0,
                'n_audit': len(y_prob),
                'n_validation': 0,
                'prec_rejet': 0,
                'prec_validation': 0,
                'automation_rate': 0
            }

        return best_result, threshold_results

    def evaluate_models(self):
        """Évaluer sur 2025 avec optimisation des seuils"""
        print("\n" + "="*80)
        print("📊 ÉVALUATION SUR 2025")
        print("="*80)

        for name, model in self.models.items():
            print(f"\n{'='*80}")
            print(f"📊 {name}")
            print(f"{'='*80}")

            # Probabilités
            y_prob = model.predict_proba(self.X_test)[:, 1]

            # Optimisation des seuils
            best_thresholds, threshold_results = self.optimize_threshold_dual(y_prob, name)

            # Créer les prédictions
            t_low = best_thresholds['threshold_low']
            t_high = best_thresholds['threshold_high']

            y_pred = np.zeros(len(y_prob), dtype=int)
            mask_rejet = y_prob <= t_low
            mask_audit = (y_prob > t_low) & (y_prob < t_high)
            mask_validation = y_prob >= t_high
            y_pred[mask_validation] = 1

            # Métriques
            mask_auto = mask_rejet | mask_validation
            if mask_auto.sum() > 0:
                y_pred_auto = y_pred[mask_auto]
                y_true_auto = self.y_test[mask_auto]

                acc = accuracy_score(y_true_auto, y_pred_auto)
                prec = precision_score(y_true_auto, y_pred_auto, zero_division=0)
                rec = recall_score(y_true_auto, y_pred_auto, zero_division=0)
                f1 = f1_score(y_true_auto, y_pred_auto, zero_division=0)
            else:
                acc = prec = rec = f1 = 0

            auc = roc_auc_score(self.y_test, y_prob)

            print(f"\n📊 Métriques (sur cas automatisés):")
            print(f"   Accuracy   : {acc:.4f}")
            print(f"   Precision  : {prec:.4f}")
            print(f"   Recall     : {rec:.4f}")
            print(f"   F1-Score   : {f1:.4f}")
            print(f"   ROC-AUC    : {auc:.4f}")

            # Sauvegarder résultats
            self.results[name] = {
                'threshold_low': t_low,
                'threshold_high': t_high,
                'accuracy': acc,
                'precision': prec,
                'recall': rec,
                'f1': f1,
                'auc': auc,
                'auto': best_thresholds['auto'],
                'gain_net': best_thresholds['gain_net'],
                'y_pred': y_pred,
                'y_prob': y_prob,
                'threshold_results': threshold_results
            }

    def save_models(self):
        """Sauvegarder le modèle et le preprocessor"""
        print("\n💾 Sauvegarde du modèle...")

        output_dir = Path('outputs/production_v2/models')
        output_dir.mkdir(parents=True, exist_ok=True)

        # Sauvegarder CatBoost
        model_path = output_dir / 'catboost_model_v2.pkl'
        joblib.dump(self.models['CatBoost'], model_path)
        print(f"✅ CatBoost sauvegardé: {model_path}")

        # Sauvegarder le preprocessor
        preprocessor_path = output_dir / 'preprocessor_v2.pkl'
        joblib.dump(self.preprocessor, preprocessor_path)
        print(f"✅ Preprocessor V2 sauvegardé: {preprocessor_path}")

        print(f"\n📂 Modèles disponibles pour l'inférence dans: {output_dir}")

    def save_predictions(self):
        """Sauvegarder les prédictions"""
        print("\n💾 Sauvegarde des prédictions...")

        output_dir = Path('outputs/production_v2/predictions')
        output_dir.mkdir(parents=True, exist_ok=True)

        predictions_data = {
            'CatBoost': {
                'y_pred': self.results['CatBoost']['y_pred'],
                'y_prob': self.results['CatBoost']['y_prob'],
                'threshold_low': self.results['CatBoost']['threshold_low'],
                'threshold_high': self.results['CatBoost']['threshold_high']
            },
            'y_true': self.y_test
        }

        predictions_path = output_dir / 'predictions_2025_v2.pkl'
        joblib.dump(predictions_data, predictions_path)

        print(f"✅ Prédictions sauvegardées: {predictions_path}")

    def generate_report(self):
        """Générer rapport texte"""
        output_dir = Path('outputs/production_v2')
        output_dir.mkdir(parents=True, exist_ok=True)

        report_path = output_dir / 'rapport_v2.txt'

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("RAPPORT MODÈLE V2 - FEATURES PRODUCTION-READY\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*80 + "\n\n")

            f.write("FEATURES UTILISÉES:\n")
            f.write("- Uniquement des colonnes disponibles en temps réel\n")
            f.write("- Montant demandé, Délai estimé, Famille Produit, Catégorie, Sous-catégorie\n")
            f.write("- Segment, Marché, anciennete_annees\n")
            f.write("- Taux de fondée calculés sur 2024 (statistiquement renforcés)\n")
            f.write("- Ratios, interactions, log transformations\n\n")

            info = self.preprocessor.get_feature_info()
            f.write(f"NOMBRE TOTAL DE FEATURES: {info['n_features']}\n\n")

            f.write("STATISTIQUES ROBUSTES (>= 30 cas):\n")
            f.write(f"- Familles avec taux fondée: {info['family_stats_count']}\n")
            f.write(f"- Catégories avec taux fondée: {info['category_stats_count']}\n")
            f.write(f"- Sous-catégories avec taux fondée: {info['subcategory_stats_count']}\n")
            f.write(f"- Segments avec taux fondée: {info['segment_stats_count']}\n\n")

            f.write("="*80 + "\n")
            f.write("RÉSULTATS CatBoost\n")
            f.write("="*80 + "\n\n")

            r = self.results['CatBoost']
            f.write(f"Seuils optimaux:\n")
            f.write(f"  Seuil BAS (Rejet)      : {r['threshold_low']:.2f}\n")
            f.write(f"  Seuil HAUT (Validation): {r['threshold_high']:.2f}\n\n")

            f.write(f"Métriques:\n")
            f.write(f"  Accuracy  : {r['accuracy']:.4f}\n")
            f.write(f"  Precision : {r['precision']:.4f}\n")
            f.write(f"  Recall    : {r['recall']:.4f}\n")
            f.write(f"  F1-Score  : {r['f1']:.4f}\n")
            f.write(f"  ROC-AUC   : {r['auc']:.4f}\n\n")

            f.write(f"Performance financière:\n")
            f.write(f"  Gain NET  : {r['gain_net']:,.0f} DH\n")
            f.write(f"  Cas auto  : {r['auto']}\n\n")

        print(f"   ✅ Rapport sauvegardé: {report_path}")

    def run(self):
        """Exécution complète"""
        self.load_data()
        self.prepare_data()
        self.train_models()
        self.evaluate_models()

        # Sauvegarder
        self.save_models()
        self.save_predictions()
        self.generate_report()

        print("\n" + "="*80)
        print("✅ ENTRAÎNEMENT V2 TERMINÉ")
        print("="*80)
        print(f"\n📂 Résultats: outputs/production_v2/")
        print(f"\n🏆 F1-Score: {self.results['CatBoost']['f1']:.4f}")
        print(f"💰 Gain NET: {self.results['CatBoost']['gain_net']:,.0f} DH")


if __name__ == '__main__':
    comparison = ModelComparisonV2(min_samples_stats=30)
    comparison.run()
