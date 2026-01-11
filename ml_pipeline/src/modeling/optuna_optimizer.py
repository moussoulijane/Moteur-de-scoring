"""
Optimisation d'hyperparamètres avec Optuna
Support pour XGBoost, LightGBM et CatBoost
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import f1_score, roc_auc_score, precision_recall_curve, auc, make_scorer
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
import warnings
warnings.filterwarnings('ignore')


class OptunaOptimizer:
    """
    Optimiseur Optuna pour gradient boosting models
    """

    def __init__(
        self,
        n_trials=100,
        cv_folds=5,
        random_state=42,
        n_jobs=-1,
        optimization_metric='f1'
    ):
        self.n_trials = n_trials
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.optimization_metric = optimization_metric

        self.best_params = {}
        self.best_score = 0.0
        self.best_model = None
        self.study = None
        self.cv_results = {}

    def _get_xgboost_params(self, trial, class_imbalance_ratio):
        """Espace de recherche pour XGBoost"""
        return {
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000, step=100),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 20),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'gamma': trial.suggest_float('gamma', 0.0, 5.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            'scale_pos_weight': class_imbalance_ratio,  # Automatique
            'random_state': self.random_state,
            'n_jobs': self.n_jobs,
            'eval_metric': 'logloss'
        }

    def _get_lightgbm_params(self, trial, class_imbalance_ratio):
        """Espace de recherche pour LightGBM"""
        return {
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000, step=100),
            'num_leaves': trial.suggest_int('num_leaves', 20, 150),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            'scale_pos_weight': class_imbalance_ratio,
            'random_state': self.random_state,
            'n_jobs': self.n_jobs,
            'verbose': -1
        }

    def _get_catboost_params(self, trial, class_imbalance_ratio):
        """Espace de recherche pour CatBoost"""
        return {
            'depth': trial.suggest_int('depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'iterations': trial.suggest_int('iterations', 100, 1000, step=100),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-8, 10.0, log=True),
            'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
            'random_strength': trial.suggest_float('random_strength', 0.0, 10.0),
            'scale_pos_weight': class_imbalance_ratio,
            'random_state': self.random_state,
            'verbose': False,
            'thread_count': -1
        }

    def _create_model(self, model_type, params):
        """Crée le modèle avec les paramètres"""
        if model_type == 'xgboost':
            import xgboost as xgb
            return xgb.XGBClassifier(**params)
        elif model_type == 'lightgbm':
            import lightgbm as lgb
            return lgb.LGBMClassifier(**params)
        elif model_type == 'catboost':
            from catboost import CatBoostClassifier
            return CatBoostClassifier(**params)
        else:
            raise ValueError(f"Model type {model_type} not supported")

    def _objective(self, trial, X, y, model_type):
        """Fonction objectif pour Optuna"""

        # Calcul du ratio de déséquilibre
        class_counts = np.bincount(y)
        class_imbalance_ratio = class_counts[0] / class_counts[1]

        # Suggérer les hyperparamètres selon le modèle
        if model_type == 'xgboost':
            params = self._get_xgboost_params(trial, class_imbalance_ratio)
        elif model_type == 'lightgbm':
            params = self._get_lightgbm_params(trial, class_imbalance_ratio)
        elif model_type == 'catboost':
            params = self._get_catboost_params(trial, class_imbalance_ratio)

        # Créer le modèle
        model = self._create_model(model_type, params)

        # Validation croisée stratifiée
        skf = StratifiedKFold(
            n_splits=self.cv_folds,
            shuffle=True,
            random_state=self.random_state
        )

        # Scorer selon métrique
        if self.optimization_metric == 'f1':
            scorer = make_scorer(f1_score)
        elif self.optimization_metric == 'roc_auc':
            scorer = make_scorer(roc_auc_score, needs_proba=True)
        elif self.optimization_metric == 'pr_auc':
            def pr_auc_scorer(y_true, y_pred):
                precision, recall, _ = precision_recall_curve(y_true, y_pred)
                return auc(recall, precision)
            scorer = make_scorer(pr_auc_scorer, needs_proba=True)
        else:
            scorer = 'f1'

        # Cross-validation
        try:
            scores = cross_val_score(
                model, X, y,
                cv=skf,
                scoring=scorer,
                n_jobs=1  # Pour éviter conflits avec Optuna multiprocessing
            )
            mean_score = scores.mean()
            std_score = scores.std()

            # Pruning si performance médiocre
            trial.report(mean_score, 0)
            if trial.should_prune():
                raise optuna.TrialPruned()

            return mean_score

        except Exception as e:
            print(f"❌ Trial failed: {e}")
            return 0.0

    def optimize(self, X, y, model_type='xgboost'):
        """
        Optimise les hyperparamètres avec Optuna

        Args:
            X: Features
            y: Target
            model_type: 'xgboost', 'lightgbm', ou 'catboost'
        """
        print(f"\n🔬 OPTIMISATION {model_type.upper()}")
        print("=" * 60)
        print(f"Configuration:")
        print(f"  - Trials: {self.n_trials}")
        print(f"  - CV Folds: {self.cv_folds}")
        print(f"  - Métrique: {self.optimization_metric}")
        print(f"  - Samples: {len(X)}")
        print(f"  - Features: {X.shape[1]}")
        print(f"  - Class balance: {np.bincount(y)}")

        # Créer l'étude Optuna
        sampler = TPESampler(seed=self.random_state)
        pruner = MedianPruner(n_warmup_steps=5)

        self.study = optuna.create_study(
            direction='maximize',
            sampler=sampler,
            pruner=pruner
        )

        # Optimisation
        print(f"\n🔄 Démarrage de l'optimisation...")
        self.study.optimize(
            lambda trial: self._objective(trial, X, y, model_type),
            n_trials=self.n_trials,
            show_progress_bar=True,
            n_jobs=1  # Sequential pour éviter les conflits
        )

        # Meilleurs résultats
        self.best_params = self.study.best_params
        self.best_score = self.study.best_value

        print(f"\n✅ Optimisation terminée!")
        print(f"  - Meilleur score: {self.best_score:.4f}")
        print(f"  - Meilleurs paramètres:")
        for param, value in self.best_params.items():
            print(f"      {param:20s}: {value}")

        # Entraîner le meilleur modèle sur toutes les données
        print(f"\n🏋️  Entraînement du modèle final...")

        # Recréer les paramètres complets
        class_counts = np.bincount(y)
        class_imbalance_ratio = class_counts[0] / class_counts[1]

        if model_type == 'xgboost':
            final_params = {**self.best_params}
            final_params.update({
                'scale_pos_weight': class_imbalance_ratio,
                'random_state': self.random_state,
                'n_jobs': self.n_jobs,
                'eval_metric': 'logloss'
            })
        elif model_type == 'lightgbm':
            final_params = {**self.best_params}
            final_params.update({
                'scale_pos_weight': class_imbalance_ratio,
                'random_state': self.random_state,
                'n_jobs': self.n_jobs,
                'verbose': -1
            })
        elif model_type == 'catboost':
            final_params = {**self.best_params}
            final_params.update({
                'scale_pos_weight': class_imbalance_ratio,
                'random_state': self.random_state,
                'verbose': False,
                'thread_count': -1
            })

        self.best_model = self._create_model(model_type, final_params)
        self.best_model.fit(X, y)

        print(f"✅ Modèle final entraîné!")

        return self.best_model, self.best_params, self.best_score

    def get_optimization_history(self):
        """Retourne l'historique d'optimisation"""
        if self.study is None:
            return None

        history_df = self.study.trials_dataframe()
        return history_df

    def get_feature_importance(self):
        """Retourne l'importance des features du meilleur modèle"""
        if self.best_model is None:
            return None

        try:
            if hasattr(self.best_model, 'feature_importances_'):
                return self.best_model.feature_importances_
        except:
            return None
