"""
PIPELINE ML PRODUCTION - CLASSIFICATION DES RÉCLAMATIONS BANCAIRES
VERSION AVEC LES VRAIES COLONNES DE PRODUCTION

Pipeline complet incluant:
1. Preprocessing robuste avec feature engineering (vraies colonnes)
2. Sélection de features multi-critères
3. Optimisation Optuna (XGBoost/LightGBM/CatBoost)
4. Calibration des probabilités
5. Validation sur données 2025
6. Analyse de drift
7. NOUVEAU: Analyse détaillée par Famille Produit
8. Rapport complet
"""
import sys
import os
sys.path.append('src')

import pandas as pd
import numpy as np
import joblib
import json
from datetime import datetime
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Import des modules avec vraies colonnes
from preprocessing.preprocessor import RobustPreprocessor
from feature_selection.selector import FeatureSelector
from modeling.optuna_optimizer import OptunaOptimizer
from evaluation.calibrator import ProbabilityCalibrator
from evaluation.metrics import MetricsCalculator
from evaluation.drift_analyzer import DriftAnalyzer
from evaluation.family_analysis import FamilyProductAnalyzer

# Matplotlib pour les graphiques
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')


class MLPipeline:
    """Pipeline ML complet de production pour classification des réclamations"""

    def __init__(self, config=None):
        self.config = config or self._default_config()
        self.preprocessor = None
        self.feature_selector = None
        self.model = None
        self.calibrator = None
        self.metrics_2024 = {}
        self.metrics_2025 = {}
        self.drift_report = {}
        self.family_analyzer = FamilyProductAnalyzer()

        # Créer dossiers de sortie
        self._create_output_dirs()

    def _default_config(self):
        """Configuration par défaut"""
        return {
            'data_path_2024': 'data/raw/reclamations_2024.xlsx',
            'data_path_2025': 'data/raw/reclamations_2025.xlsx',
            'target_col': 'Fondee',
            'optuna_trials': 30,  # Réduit pour la démo
            'cv_folds': 5,
            'model_type': 'xgboost',
            'calibration_method': 'isotonic',
            'random_state': 42,
            'output_dir': 'outputs'
        }

    def _create_output_dirs(self):
        """Créer les dossiers de sortie"""
        dirs = ['models', 'preprocessors', 'reports', 'reports/figures']
        for d in dirs:
            Path(f"{self.config['output_dir']}/{d}").mkdir(parents=True, exist_ok=True)

    def load_data(self):
        """Charge les données 2024 et 2025"""
        print("\n" + "=" * 80)
        print("📂 CHARGEMENT DES DONNÉES (VRAIES COLONNES)")
        print("=" * 80)

        # Charger 2024
        print(f"\nChargement données 2024: {self.config['data_path_2024']}")
        self.df_2024 = pd.read_excel(self.config['data_path_2024'])
        print(f"  ✅ {len(self.df_2024)} réclamations chargées")
        print(f"  - Colonnes: {len(self.df_2024.columns)}")
        print(f"  - Taux fondées: {self.df_2024['Fondee'].mean():.2%}")
        print(f"  - Familles: {self.df_2024['Famille Produit'].nunique()}")

        # Charger 2025
        print(f"\nChargement données 2025: {self.config['data_path_2025']}")
        self.df_2025 = pd.read_excel(self.config['data_path_2025'])
        print(f"  ✅ {len(self.df_2025)} réclamations chargées")
        print(f"  - Colonnes: {len(self.df_2025.columns)}")
        print(f"  - Taux fondées: {self.df_2025['Fondee'].mean():.2%}")
        print(f"  - Familles: {self.df_2025['Famille Produit'].nunique()}")

        return self

    def preprocess(self):
        """Preprocessing avec feature engineering"""
        print("\n" + "=" * 80)
        print("⚙️  PREPROCESSING & FEATURE ENGINEERING (VRAIES COLONNES)")
        print("=" * 80)

        # Initialiser le preprocessor
        self.preprocessor = RobustPreprocessor(target_col='Fondee')

        # Fit sur 2024
        print("\n🔧 Configuration du preprocessing sur données 2024...")
        self.preprocessor.fit(self.df_2024)

        # Transform 2024
        print("\n🔄 Transformation des données 2024...")
        self.X_train = self.preprocessor.transform(self.df_2024)
        self.y_train = self.df_2024['Fondee'].values

        print(f"  ✅ Données 2024 transformées: {self.X_train.shape}")

        # Transform 2025
        print("\n🔄 Transformation des données 2025...")
        self.X_test_2025 = self.preprocessor.transform(self.df_2025)
        self.y_test_2025 = self.df_2025['Fondee'].values

        print(f"  ✅ Données 2025 transformées: {self.X_test_2025.shape}")

        # Sauvegarder le preprocessor
        preprocessor_path = f"{self.config['output_dir']}/preprocessors/preprocessor_real_cols.pkl"
        joblib.dump(self.preprocessor, preprocessor_path)
        print(f"\n💾 Preprocessor sauvegardé: {preprocessor_path}")

        return self

    def select_features(self):
        """Sélection de features"""
        print("\n" + "=" * 80)
        print("🎯 SÉLECTION DE FEATURES")
        print("=" * 80)

        self.feature_selector = FeatureSelector(
            missing_threshold=0.5,
            variance_threshold=0.01,
            correlation_threshold=0.95,
            importance_methods=['permutation', 'native']
        )

        # Fit et transform
        self.X_train_selected = self.feature_selector.fit_transform(
            self.X_train,
            self.y_train,
            top_k_features=None
        )

        # Transform 2025
        self.X_test_2025_selected = self.feature_selector.transform(self.X_test_2025)

        print(f"\n✅ Features sélectionnées: {self.X_train_selected.shape[1]} / {self.X_train.shape[1]}")

        # Sauvegarder
        selector_path = f"{self.config['output_dir']}/preprocessors/feature_selector_real_cols.pkl"
        joblib.dump(self.feature_selector, selector_path)
        print(f"💾 Feature selector sauvegardé: {selector_path}")

        # Sauvegarder l'importance
        importance_df = self.feature_selector.get_feature_importance()
        importance_path = f"{self.config['output_dir']}/reports/feature_importance_real_cols.csv"
        importance_df.to_csv(importance_path)
        print(f"💾 Feature importance sauvegardée: {importance_path}")

        return self

    def optimize_model(self):
        """Optimisation avec Optuna"""
        print("\n" + "=" * 80)
        print(f"🚀 OPTIMISATION {self.config['model_type'].upper()} avec OPTUNA")
        print("=" * 80)

        optimizer = OptunaOptimizer(
            n_trials=self.config['optuna_trials'],
            cv_folds=self.config['cv_folds'],
            random_state=self.config['random_state'],
            optimization_metric='f1'
        )

        self.model, self.best_params, self.best_cv_score = optimizer.optimize(
            self.X_train_selected,
            self.y_train,
            model_type=self.config['model_type']
        )

        # Sauvegarder l'historique
        history_df = optimizer.get_optimization_history()
        history_path = f"{self.config['output_dir']}/reports/optuna_history_real_cols.csv"
        history_df.to_csv(history_path, index=False)
        print(f"\n💾 Historique Optuna sauvegardé: {history_path}")

        # Sauvegarder les hyperparamètres
        params_path = f"{self.config['output_dir']}/models/best_hyperparameters_real_cols.json"
        with open(params_path, 'w') as f:
            json.dump(self.best_params, f, indent=2)
        print(f"💾 Hyperparamètres sauvegardés: {params_path}")

        return self

    def calibrate_model(self):
        """Calibration des probabilités"""
        print("\n" + "=" * 80)
        print("🎯 CALIBRATION DES PROBABILITÉS")
        print("=" * 80)

        self.calibrator = ProbabilityCalibrator(
            method=self.config['calibration_method'],
            cv_folds=3
        )

        # Calibrer sur un subset
        from sklearn.model_selection import train_test_split
        X_cal, X_val_cal, y_cal, y_val_cal = train_test_split(
            self.X_train_selected,
            self.y_train,
            test_size=0.2,
            random_state=self.config['random_state'],
            stratify=self.y_train
        )

        self.calibrator.fit(self.model, X_cal, y_cal)

        # Évaluer
        calib_metrics = self.calibrator.evaluate_calibration(
            self.calibrator.calibrated_model,
            X_val_cal,
            y_val_cal,
            plot=True,
            save_path=f"{self.config['output_dir']}/reports/figures/calibration_curve_real_cols.png"
        )

        # Sauvegarder
        calibrator_path = f"{self.config['output_dir']}/models/calibrator_real_cols.pkl"
        joblib.dump(self.calibrator, calibrator_path)
        print(f"\n💾 Calibrateur sauvegardé: {calibrator_path}")

        return self

    def evaluate_2024(self):
        """Évaluation sur 2024"""
        print("\n" + "=" * 80)
        print("📊 ÉVALUATION SUR DONNÉES 2024")
        print("=" * 80)

        y_pred_2024 = self.model.predict(self.X_train_selected)
        y_prob_2024 = self.model.predict_proba(self.X_train_selected)[:, 1]

        metrics_calc = MetricsCalculator()
        self.metrics_2024 = metrics_calc.calculate_all_metrics(
            self.y_train,
            y_pred_2024,
            y_prob_2024
        )

        metrics_calc.print_metrics("MÉTRIQUES 2024")

        # Graphiques
        metrics_calc.plot_confusion_matrix(
            save_path=f"{self.config['output_dir']}/reports/figures/confusion_matrix_2024_real_cols.png"
        )
        metrics_calc.plot_roc_curve(
            self.y_train,
            y_prob_2024,
            save_path=f"{self.config['output_dir']}/reports/figures/roc_curve_2024_real_cols.png"
        )
        metrics_calc.plot_precision_recall_curve(
            self.y_train,
            y_prob_2024,
            save_path=f"{self.config['output_dir']}/reports/figures/pr_curve_2024_real_cols.png"
        )

        # Sauvegarder
        metrics_path = f"{self.config['output_dir']}/reports/metrics_2024_real_cols.json"
        with open(metrics_path, 'w') as f:
            json.dump(self.metrics_2024, f, indent=2)

        return self

    def evaluate_2025(self):
        """Évaluation sur 2025"""
        print("\n" + "=" * 80)
        print("📊 ÉVALUATION SUR DONNÉES 2025")
        print("=" * 80)

        y_pred_2025 = self.model.predict(self.X_test_2025_selected)
        y_prob_2025 = self.model.predict_proba(self.X_test_2025_selected)[:, 1]

        metrics_calc = MetricsCalculator()
        self.metrics_2025 = metrics_calc.calculate_all_metrics(
            self.y_test_2025,
            y_pred_2025,
            y_prob_2025
        )

        metrics_calc.print_metrics("MÉTRIQUES 2025")

        # Graphiques
        metrics_calc.plot_confusion_matrix(
            save_path=f"{self.config['output_dir']}/reports/figures/confusion_matrix_2025_real_cols.png"
        )
        metrics_calc.plot_roc_curve(
            self.y_test_2025,
            y_prob_2025,
            save_path=f"{self.config['output_dir']}/reports/figures/roc_curve_2025_real_cols.png"
        )
        metrics_calc.plot_precision_recall_curve(
            self.y_test_2025,
            y_prob_2025,
            save_path=f"{self.config['output_dir']}/reports/figures/pr_curve_2025_real_cols.png"
        )

        # Comparaison
        metrics_calc.compare_metrics(self.metrics_2024, self.metrics_2025)

        # Sauvegarder
        metrics_path = f"{self.config['output_dir']}/reports/metrics_2025_real_cols.json"
        with open(metrics_path, 'w') as f:
            json.dump(self.metrics_2025, f, indent=2)

        # Sauvegarder prédictions pour analyse famille
        self.y_pred_2025 = y_pred_2025
        self.y_prob_2025 = y_prob_2025

        return self

    def analyze_by_family(self):
        """NOUVEAU: Analyse par famille produit"""
        print("\n" + "=" * 80)
        print("👨‍👩‍👧‍👦 ANALYSE DÉTAILLÉE PAR FAMILLE PRODUIT 2025")
        print("=" * 80)

        # Analyse par famille
        df_family_analysis = self.family_analyzer.analyze_by_family(
            self.df_2025,
            self.y_test_2025,
            self.y_pred_2025,
            self.y_prob_2025
        )

        # Visualisation
        self.family_analyzer.plot_family_comparison(
            save_path=f"{self.config['output_dir']}/reports/figures/family_analysis_2025_real_cols.png",
            year='2025'
        )

        # Rapport texte
        family_report = self.family_analyzer.generate_family_report(
            save_path=f"{self.config['output_dir']}/reports/family_analysis_2025_real_cols.txt"
        )

        # Sauvegarder CSV
        df_family_analysis.to_csv(
            f"{self.config['output_dir']}/reports/family_metrics_2025_real_cols.csv",
            index=False
        )

        print(f"\n✅ Analyse par famille terminée: {len(df_family_analysis)} familles")

        return self

    def analyze_drift(self):
        """Analyse de drift"""
        print("\n" + "=" * 80)
        print("🔍 ANALYSE DE DRIFT 2024 → 2025")
        print("=" * 80)

        drift_analyzer = DriftAnalyzer(significance_level=0.05)

        numerical_cols = [
            'Montant demandé',
            'PNB analytique (vision commerciale) cumulé',
            'anciennete_annees',
            'Délai Estimé (j)'
        ]
        numerical_cols = [c for c in numerical_cols if c in self.df_2024.columns]

        categorical_cols = ['Famille Produit', 'Catégorie', 'Segment', 'Canal de Réception']
        categorical_cols = [c for c in categorical_cols if c in self.df_2024.columns]

        self.drift_report = drift_analyzer.analyze_drift(
            self.df_2024,
            self.df_2025,
            numerical_cols=numerical_cols,
            categorical_cols=categorical_cols
        )

        # Sauvegarder
        drift_path = f"{self.config['output_dir']}/reports/drift_report_real_cols.csv"
        if 'numerical' in self.drift_report:
            self.drift_report['numerical'].to_csv(drift_path.replace('.csv', '_numerical.csv'), index=False)
        if 'categorical' in self.drift_report:
            self.drift_report['categorical'].to_csv(drift_path.replace('.csv', '_categorical.csv'), index=False)

        # Comparer distributions de probabilités
        y_prob_2024 = self.model.predict_proba(self.X_train_selected)[:, 1]

        drift_analyzer.compare_prediction_distributions(
            y_prob_2024,
            self.y_prob_2025,
            save_path=f"{self.config['output_dir']}/reports/figures/prob_distribution_comparison_real_cols.png"
        )

        return self

    def save_model(self):
        """Sauvegarde le modèle"""
        print("\n" + "=" * 80)
        print("💾 SAUVEGARDE DU MODÈLE ET ARTEFACTS")
        print("=" * 80)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        model_path = f"{self.config['output_dir']}/models/model_{self.config['model_type']}_real_cols_{timestamp}.pkl"
        joblib.dump(self.model, model_path)
        print(f"✅ Modèle sauvegardé: {model_path}")

        # Metadata
        family_stats = self.family_analyzer.get_summary_stats()

        metadata = {
            'timestamp': timestamp,
            'model_type': self.config['model_type'],
            'n_features': self.X_train_selected.shape[1],
            'n_samples_train': len(self.y_train),
            'n_samples_test': len(self.y_test_2025),
            'best_cv_score': float(self.best_cv_score),
            'best_hyperparameters': self.best_params,
            'metrics_2024': {k: float(v) if isinstance(v, (np.integer, np.floating)) else v
                           for k, v in self.metrics_2024.items()},
            'metrics_2025': {k: float(v) if isinstance(v, (np.integer, np.floating)) else v
                           for k, v in self.metrics_2025.items()},
            'family_analysis': family_stats,
            'selected_features': self.feature_selector.selected_features
        }

        metadata_path = f"{self.config['output_dir']}/models/metadata_real_cols_{timestamp}.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"✅ Metadata sauvegardées: {metadata_path}")

        return self

    def generate_report(self):
        """Génère le rapport final"""
        print("\n" + "=" * 80)
        print("📄 GÉNÉRATION DU RAPPORT FINAL")
        print("=" * 80)

        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("RAPPORT FINAL - PIPELINE ML AVEC VRAIES COLONNES DE PRODUCTION")
        report_lines.append("=" * 80)
        report_lines.append(f"\nDate: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"Modèle: {self.config['model_type'].upper()}")

        # Données
        report_lines.append("\n" + "-" * 80)
        report_lines.append("DONNÉES")
        report_lines.append("-" * 80)
        report_lines.append(f"Entraînement 2024: {len(self.y_train)} réclamations")
        report_lines.append(f"Test 2025:         {len(self.y_test_2025)} réclamations")
        report_lines.append(f"Features sélectionnées: {self.X_train_selected.shape[1]}")

        # Métriques 2024
        report_lines.append("\n" + "-" * 80)
        report_lines.append("PERFORMANCE 2024")
        report_lines.append("-" * 80)
        for key, value in self.metrics_2024.items():
            if isinstance(value, float):
                report_lines.append(f"{key:25s}: {value:.4f}")

        # Métriques 2025
        report_lines.append("\n" + "-" * 80)
        report_lines.append("PERFORMANCE 2025")
        report_lines.append("-" * 80)
        for key, value in self.metrics_2025.items():
            if isinstance(value, float):
                report_lines.append(f"{key:25s}: {value:.4f}")

        # Comparaison
        report_lines.append("\n" + "-" * 80)
        report_lines.append("COMPARAISON 2024 vs 2025")
        report_lines.append("-" * 80)
        degradation = ((self.metrics_2025['accuracy'] - self.metrics_2024['accuracy']) /
                      self.metrics_2024['accuracy']) * 100
        report_lines.append(f"Dégradation Accuracy: {degradation:+.2f}%")

        if abs(degradation) < 2:
            report_lines.append("✅ EXCELLENT: Stabilité < 2%")
        elif abs(degradation) < 5:
            report_lines.append("✅ ACCEPTABLE: Stabilité < 5%")
        else:
            report_lines.append("❌ ALERTE: Dégradation > 5%")

        # Analyse par famille
        report_lines.append("\n" + "-" * 80)
        report_lines.append("ANALYSE PAR FAMILLE PRODUIT 2025")
        report_lines.append("-" * 80)
        family_stats = self.family_analyzer.get_summary_stats()
        if family_stats:
            report_lines.append(f"Meilleure famille: {family_stats['best_family']['name']} "
                              f"(F1={family_stats['best_family']['f1_score']:.2%})")
            report_lines.append(f"Moins bonne famille: {family_stats['worst_family']['name']} "
                              f"(F1={family_stats['worst_family']['f1_score']:.2%})")
            report_lines.append(f"F1 moyen: {family_stats['mean_f1']:.2%}")

        # Features importantes
        report_lines.append("\n" + "-" * 80)
        report_lines.append("TOP 15 FEATURES IMPORTANTES")
        report_lines.append("-" * 80)
        importance_df = self.feature_selector.get_feature_importance()
        for i, (feat, row) in enumerate(importance_df.head(15).iterrows(), 1):
            report_lines.append(f"{i:2d}. {feat:40s} {row['mean_importance']:.4f}")

        # Recommandation
        report_lines.append("\n" + "-" * 80)
        report_lines.append("RECOMMANDATION FINALE")
        report_lines.append("-" * 80)

        if abs(degradation) < 5 and self.metrics_2025['f1_score'] > 0.70:
            report_lines.append("✅ GO POUR PRODUCTION")
            report_lines.append("Le modèle est stable et performant")
        elif abs(degradation) < 5:
            report_lines.append("⚠️  MONITORING RENFORCÉ")
            report_lines.append("Modèle acceptable mais nécessite un suivi")
        else:
            report_lines.append("❌ NO-GO - RÉENTRAÎNEMENT NÉCESSAIRE")
            report_lines.append("Dégradation trop importante sur 2025")

        report_lines.append("\n" + "=" * 80)

        # Sauvegarder
        report_path = f"{self.config['output_dir']}/reports/RAPPORT_FINAL_REAL_COLS.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))

        print(f"✅ Rapport sauvegardé: {report_path}")
        print("\n" + '\n'.join(report_lines))

        return self

    def run(self):
        """Exécute le pipeline complet"""
        print("\n")
        print("🚀" * 40)
        print("PIPELINE ML PRODUCTION - VRAIES COLONNES")
        print("🚀" * 40)

        try:
            self.load_data()
            self.preprocess()
            self.select_features()
            self.optimize_model()
            self.calibrate_model()
            self.evaluate_2024()
            self.evaluate_2025()
            self.analyze_by_family()  # NOUVEAU!
            self.analyze_drift()
            self.save_model()
            self.generate_report()

            print("\n")
            print("✅" * 40)
            print("PIPELINE TERMINÉ AVEC SUCCÈS!")
            print("✅" * 40)
            print(f"\nRésultats: {self.config['output_dir']}/")

        except Exception as e:
            print(f"\n❌ ERREUR: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    config = {
        'data_path_2024': 'data/raw/reclamations_2024.xlsx',
        'data_path_2025': 'data/raw/reclamations_2025.xlsx',
        'target_col': 'Fondee',
        'optuna_trials': 30,  # Réduit pour démo
        'cv_folds': 5,
        'model_type': 'xgboost',
        'calibration_method': 'isotonic',
        'random_state': 42,
        'output_dir': 'outputs'
    }

    pipeline = MLPipeline(config)
    pipeline.run()
