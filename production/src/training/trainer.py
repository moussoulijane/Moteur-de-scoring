"""
Model training module with hyperparameter optimization.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, Optional, Any
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score
)
import xgboost as xgb
from catboost import CatBoostClassifier
import optuna
import joblib
import logging
from datetime import datetime

from ..preprocessing import ProductionPreprocessor
from .optimizer import ThresholdOptimizer

logger = logging.getLogger(__name__)


class ModelTrainer:
    """
    Model trainer with hyperparameter optimization using Optuna.

    Supports:
    - XGBoost
    - CatBoost
    - Automatic hyperparameter tuning
    - Cross-validation
    - Threshold optimization
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize trainer.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.preprocessor = ProductionPreprocessor(
            min_samples_stats=config.get('preprocessing', {}).get('min_samples_stats', 30),
            scaler_type=config.get('preprocessing', {}).get('scaler_type', 'robust')
        )

        self.models = {}
        self.results = {}
        self.best_thresholds = {}

        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.df_test = None

    def load_data(
        self,
        train_file: str,
        test_file: Optional[str] = None
    ) -> None:
        """
        Load training and test data.

        Args:
            train_file: Path to training data Excel file
            test_file: Path to test data Excel file (optional)
        """
        logger.info("="*80)
        logger.info("📂 LOADING DATA")
        logger.info("="*80)

        self.df_train = pd.read_excel(train_file)
        logger.info(f"✅ Training data: {len(self.df_train)} complaints")

        if test_file:
            self.df_test = pd.read_excel(test_file)
            logger.info(f"✅ Test data: {len(self.df_test)} complaints")

        # Verify required columns
        required_cols = self.config.get('data', {}).get('required_columns', [])
        for col in required_cols:
            if col not in self.df_train.columns:
                raise ValueError(f"Missing required column in training data: {col}")
            if test_file and col != 'Fondee' and col not in self.df_test.columns:
                raise ValueError(f"Missing required column in test data: {col}")

    def prepare_data(self) -> None:
        """Preprocess training and test data."""
        logger.info("="*80)
        logger.info("🔧 PREPROCESSING DATA")
        logger.info("="*80)

        # Fit on training data
        self.X_train = self.preprocessor.fit_transform(self.df_train)
        self.y_train = self.df_train['Fondee'].values

        logger.info(f"✅ Training shape: {self.X_train.shape}")

        # Transform test data if available
        if self.df_test is not None:
            self.X_test = self.preprocessor.transform(self.df_test)
            self.y_test = self.df_test['Fondee'].values
            logger.info(f"✅ Test shape: {self.X_test.shape}")

        # Display feature info
        info = self.preprocessor.get_feature_info()
        logger.info(f"\n📊 Features computed on training data:")
        logger.info(f"   Total features: {info['n_features']}")
        logger.info(f"   Families with robust stats: {info['family_stats_count']}")
        logger.info(f"   Categories with robust stats: {info['category_stats_count']}")

    def _optimize_xgboost(self, n_trials: int = 50) -> xgb.XGBClassifier:
        """Optimize XGBoost hyperparameters using Optuna."""
        logger.info("\n🔬 Optimizing XGBoost...")

        param_space = self.config.get('models', {}).get('xgboost', {}).get('param_space', {})

        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int(
                    'n_estimators',
                    *param_space.get('n_estimators', [100, 500])
                ),
                'max_depth': trial.suggest_int(
                    'max_depth',
                    *param_space.get('max_depth', [3, 10])
                ),
                'learning_rate': trial.suggest_float(
                    'learning_rate',
                    *param_space.get('learning_rate', [0.01, 0.3]),
                    log=True
                ),
                'subsample': trial.suggest_float(
                    'subsample',
                    *param_space.get('subsample', [0.6, 1.0])
                ),
                'colsample_bytree': trial.suggest_float(
                    'colsample_bytree',
                    *param_space.get('colsample_bytree', [0.6, 1.0])
                ),
                'reg_alpha': trial.suggest_float(
                    'reg_alpha',
                    *param_space.get('reg_alpha', [1e-8, 10.0]),
                    log=True
                ),
                'reg_lambda': trial.suggest_float(
                    'reg_lambda',
                    *param_space.get('reg_lambda', [1e-8, 10.0]),
                    log=True
                ),
            }

            model = xgb.XGBClassifier(
                **params,
                random_state=self.config.get('metrics', {}).get('random_state', 42),
                n_jobs=-1,
                eval_metric='logloss'
            )

            cv = StratifiedKFold(
                n_splits=self.config.get('metrics', {}).get('cv_folds', 5),
                shuffle=True,
                random_state=self.config.get('metrics', {}).get('random_state', 42)
            )

            scores = cross_val_score(
                model, self.X_train, self.y_train,
                cv=cv, scoring='f1', n_jobs=-1
            )

            return scores.mean()

        study = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.TPESampler(
                seed=self.config.get('metrics', {}).get('random_state', 42)
            )
        )
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

        best_params = study.best_params
        logger.info(f"   ✅ Best F1-Score CV: {study.best_value:.4f}")

        model = xgb.XGBClassifier(
            **best_params,
            random_state=self.config.get('metrics', {}).get('random_state', 42),
            n_jobs=-1,
            eval_metric='logloss'
        )
        model.fit(self.X_train, self.y_train)

        return model

    def _optimize_catboost(self, n_trials: int = 50) -> CatBoostClassifier:
        """Optimize CatBoost hyperparameters using Optuna."""
        logger.info("\n🔬 Optimizing CatBoost...")

        param_space = self.config.get('models', {}).get('catboost', {}).get('param_space', {})

        def objective(trial):
            params = {
                'iterations': trial.suggest_int(
                    'iterations',
                    *param_space.get('iterations', [100, 500])
                ),
                'depth': trial.suggest_int(
                    'depth',
                    *param_space.get('depth', [4, 10])
                ),
                'learning_rate': trial.suggest_float(
                    'learning_rate',
                    *param_space.get('learning_rate', [0.01, 0.3]),
                    log=True
                ),
                'l2_leaf_reg': trial.suggest_float(
                    'l2_leaf_reg',
                    *param_space.get('l2_leaf_reg', [1e-8, 10.0]),
                    log=True
                ),
                'border_count': trial.suggest_int(
                    'border_count',
                    *param_space.get('border_count', [32, 255])
                ),
            }

            model = CatBoostClassifier(
                **params,
                random_state=self.config.get('metrics', {}).get('random_state', 42),
                verbose=0
            )

            cv = StratifiedKFold(
                n_splits=self.config.get('metrics', {}).get('cv_folds', 5),
                shuffle=True,
                random_state=self.config.get('metrics', {}).get('random_state', 42)
            )

            scores = cross_val_score(
                model, self.X_train, self.y_train,
                cv=cv, scoring='f1', n_jobs=-1
            )

            return scores.mean()

        study = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.TPESampler(
                seed=self.config.get('metrics', {}).get('random_state', 42)
            )
        )
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

        best_params = study.best_params
        logger.info(f"   ✅ Best F1-Score CV: {study.best_value:.4f}")

        model = CatBoostClassifier(
            **best_params,
            random_state=self.config.get('metrics', {}).get('random_state', 42),
            verbose=0
        )
        model.fit(self.X_train, self.y_train)

        return model

    def train_models(self) -> None:
        """Train all configured models."""
        logger.info("="*80)
        logger.info("🎯 TRAINING MODELS")
        logger.info("="*80)

        algorithms = self.config.get('models', {}).get('algorithms', ['xgboost'])

        if 'xgboost' in algorithms:
            n_trials = self.config.get('models', {}).get('xgboost', {}).get('n_trials', 50)
            self.models['xgboost'] = self._optimize_xgboost(n_trials)

        if 'catboost' in algorithms:
            try:
                n_trials = self.config.get('models', {}).get('catboost', {}).get('n_trials', 50)
                self.models['catboost'] = self._optimize_catboost(n_trials)
            except AttributeError as e:
                if '__sklearn_tags__' in str(e):
                    logger.warning(
                        "⚠️ CatBoost sklearn compatibility issue - skipping"
                    )
                else:
                    raise

        logger.info(f"\n✅ Models trained: {', '.join(self.models.keys())}")

    def evaluate_models(self) -> None:
        """Evaluate models on test data with threshold optimization."""
        if self.X_test is None:
            logger.warning("⚠️ No test data available for evaluation")
            return

        logger.info("="*80)
        logger.info("📊 EVALUATING ON TEST DATA")
        logger.info("="*80)

        threshold_optimizer = ThresholdOptimizer(
            config=self.config,
            y_test=self.y_test,
            df_test=self.df_test
        )

        for name, model in self.models.items():
            logger.info(f"\n{'='*80}")
            logger.info(f"📊 {name.upper()}")
            logger.info(f"{'='*80}")

            # Get probabilities
            y_prob = model.predict_proba(self.X_test)[:, 1]

            # Optimize thresholds
            best_thresholds, _ = threshold_optimizer.optimize_thresholds(y_prob)
            self.best_thresholds[name] = best_thresholds

            # Create predictions
            t_low = best_thresholds['threshold_low']
            t_high = best_thresholds['threshold_high']

            y_pred = np.zeros(len(y_prob), dtype=int)
            mask_validation = y_prob >= t_high
            y_pred[mask_validation] = 1

            # Compute metrics on automated cases
            mask_rejet = y_prob <= t_low
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

            logger.info(f"\n📊 Metrics (on automated cases):")
            logger.info(f"   Accuracy  : {acc:.4f}")
            logger.info(f"   Precision : {prec:.4f}")
            logger.info(f"   Recall    : {rec:.4f}")
            logger.info(f"   F1-Score  : {f1:.4f}")
            logger.info(f"   ROC-AUC   : {auc:.4f}")

            # Save results
            self.results[name] = {
                'threshold_low': t_low,
                'threshold_high': t_high,
                'accuracy': acc,
                'precision': prec,
                'recall': rec,
                'f1': f1,
                'auc': auc,
                'auto_count': best_thresholds['auto'],
                'gain_net': best_thresholds['gain_net'],
                'y_pred': y_pred,
                'y_prob': y_prob
            }

    def save_models(self, output_dir: str) -> None:
        """
        Save trained models and preprocessor.

        Args:
            output_dir: Directory to save models
        """
        logger.info("="*80)
        logger.info("💾 SAVING MODELS")
        logger.info("="*80)

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Determine best model
        if self.results:
            best_model_name = max(
                self.results.items(),
                key=lambda x: x[1]['gain_net']
            )[0]

            logger.info(f"\n🏆 Best model (by net gain): {best_model_name.upper()}")
            logger.info(
                f"   Net Gain: {self.results[best_model_name]['gain_net']:,.0f} DH"
            )
        else:
            best_model_name = list(self.models.keys())[0]

        # Save each model
        for name, model in self.models.items():
            model_path = output_path / f'{name}_model.pkl'
            joblib.dump(model, model_path)
            logger.info(f"✅ {name.upper()} saved: {model_path}")

        # Save best model with generic name
        best_model_path = output_path / 'best_model.pkl'
        joblib.dump(self.models[best_model_name], best_model_path)
        logger.info(f"✅ Best model saved: {best_model_path}")

        # Save model info
        info_path = output_path / 'model_info.txt'
        with open(info_path, 'w', encoding='utf-8') as f:
            f.write(f"Best model: {best_model_name}\n")
            f.write(f"Trained on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            if self.results:
                f.write(f"Net Gain: {self.results[best_model_name]['gain_net']:,.0f} DH\n")
                f.write(f"Threshold Low: {self.results[best_model_name]['threshold_low']:.4f}\n")
                f.write(f"Threshold High: {self.results[best_model_name]['threshold_high']:.4f}\n")
                f.write(f"F1-Score: {self.results[best_model_name]['f1']:.4f}\n")
                f.write(f"ROC-AUC: {self.results[best_model_name]['auc']:.4f}\n")

        # Save preprocessor
        preprocessor_path = output_path / 'preprocessor.pkl'
        joblib.dump(self.preprocessor, preprocessor_path)
        logger.info(f"✅ Preprocessor saved: {preprocessor_path}")

        # Save thresholds
        if self.best_thresholds:
            thresholds_path = output_path / 'thresholds.pkl'
            joblib.dump(self.best_thresholds, thresholds_path)
            logger.info(f"✅ Thresholds saved: {thresholds_path}")

        logger.info(f"\n📂 Models available in: {output_path}")

    def generate_report(self, output_dir: str) -> None:
        """
        Generate training report.

        Args:
            output_dir: Directory to save report
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        report_path = output_path / f'training_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt'

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("TRAINING REPORT - PRODUCTION MODEL\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*80 + "\n\n")

            # Data info
            f.write("DATA:\n")
            f.write(f"Training samples: {len(self.df_train)}\n")
            if self.df_test is not None:
                f.write(f"Test samples: {len(self.df_test)}\n")
            f.write("\n")

            # Feature info
            info = self.preprocessor.get_feature_info()
            f.write("FEATURES:\n")
            f.write(f"Total features: {info['n_features']}\n")
            f.write(f"Families with robust stats: {info['family_stats_count']}\n")
            f.write(f"Categories with robust stats: {info['category_stats_count']}\n")
            f.write(f"Min samples for stats: {info['min_samples_for_stats']}\n\n")

            # Model results
            for name, results in self.results.items():
                f.write("="*80 + "\n")
                f.write(f"{name.upper()} RESULTS\n")
                f.write("="*80 + "\n\n")

                f.write(f"Thresholds:\n")
                f.write(f"  Low (Reject)     : {results['threshold_low']:.4f}\n")
                f.write(f"  High (Validate)  : {results['threshold_high']:.4f}\n\n")

                f.write(f"Metrics:\n")
                f.write(f"  Accuracy  : {results['accuracy']:.4f}\n")
                f.write(f"  Precision : {results['precision']:.4f}\n")
                f.write(f"  Recall    : {results['recall']:.4f}\n")
                f.write(f"  F1-Score  : {results['f1']:.4f}\n")
                f.write(f"  ROC-AUC   : {results['auc']:.4f}\n\n")

                f.write(f"Performance:\n")
                f.write(f"  Net Gain      : {results['gain_net']:,.0f} DH\n")
                auto_rate = results['auto_count'] / len(self.y_test) if self.y_test is not None else 0
                f.write(f"  Automation    : {auto_rate:.1%}\n\n")

            # Best model
            if self.results:
                best_name = max(self.results.items(), key=lambda x: x[1]['gain_net'])[0]
                f.write("="*80 + "\n")
                f.write(f"🏆 BEST MODEL: {best_name.upper()}\n")
                f.write("="*80 + "\n")
                f.write(f"Net Gain: {self.results[best_name]['gain_net']:,.0f} DH\n")

        logger.info(f"✅ Report saved: {report_path}")

    def run(
        self,
        train_file: str,
        test_file: Optional[str] = None,
        output_dir: Optional[str] = None
    ) -> None:
        """
        Run complete training pipeline.

        Args:
            train_file: Path to training data
            test_file: Path to test data (optional)
            output_dir: Output directory (optional)
        """
        # Load data
        self.load_data(train_file, test_file)

        # Prepare data
        self.prepare_data()

        # Train models
        self.train_models()

        # Evaluate if test data available
        if test_file:
            self.evaluate_models()

        # Save models
        if output_dir is None:
            output_dir = self.config.get('paths', {}).get('models_dir', 'production/models')

        self.save_models(output_dir)

        # Generate report
        self.generate_report(output_dir)

        logger.info("="*80)
        logger.info("✅ TRAINING COMPLETED")
        logger.info("="*80)
