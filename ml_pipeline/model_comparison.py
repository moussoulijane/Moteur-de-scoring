"""
COMPARAISON DE MODÈLES - XGBoost vs Random Forest vs CatBoost
Entraînement sur 2024, Test sur 2025
Avec calcul corrigé des pertes basé sur les montants réels
"""
import sys
sys.path.append('src')

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from catboost import CatBoostClassifier
import optuna
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Configuration
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (16, 10)

PRIX_UNITAIRE_DH = 169


class ProductionPreprocessor:
    """Preprocessing production avec gestion stricte de l'ordre des colonnes"""

    def __init__(self):
        self.scaler = RobustScaler()
        self.family_medians = {}
        self.categorical_encodings = {}
        self.feature_names_fitted = None

    def fit(self, df):
        """Fit sur données 2024"""
        print("\n🔧 Configuration du preprocessing...")

        X = df.copy()

        # Calculer médianes par famille
        print("📊 Calcul médianes par famille (base 2024)...")
        self.family_medians = X.groupby('Famille Produit')['Montant demandé'].median().to_dict()
        print(f"   ✅ {len(self.family_medians)} familles")

        # Encoder catégorielles
        print("🔢 Encodage catégorielles...")
        categorical_cols = ['Marché', 'Segment', 'Famille Produit', 'Catégorie', 'Sous-catégorie']

        for col in categorical_cols:
            if col in X.columns:
                self.categorical_encodings[col] = X[col].value_counts().to_dict()
                X[f'{col}_freq'] = X[col].map(self.categorical_encodings[col]).fillna(0)

        # Features
        X = self._create_features(X, fit_mode=True)

        # Colonnes numériques
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [c for c in numeric_cols if c != 'Fondee']

        # CRITICAL: Sauvegarder l'ordre des colonnes (trié alphabétiquement)
        self.feature_names_fitted = sorted(numeric_cols)
        print(f"📋 Features finales: {len(self.feature_names_fitted)}")

        # Fit scaler avec colonnes dans le bon ordre
        X_ordered = X[self.feature_names_fitted]
        self.scaler.fit(X_ordered)

        print(f"✅ Preprocessing configuré: {len(self.feature_names_fitted)} features")

        return self

    def transform(self, df):
        """Transform sur données 2024 ou 2025"""
        X = df.copy()

        # Encoder catégorielles avec encodages de 2024
        categorical_cols = ['Marché', 'Segment', 'Famille Produit', 'Catégorie', 'Sous-catégorie']

        for col in categorical_cols:
            if col in X.columns and col in self.categorical_encodings:
                X[f'{col}_freq'] = X[col].map(self.categorical_encodings[col]).fillna(0)

        # Features
        X = self._create_features(X, fit_mode=False)

        # Colonnes numériques
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [c for c in numeric_cols if c != 'Fondee']

        # CRITICAL: Utiliser EXACTEMENT les mêmes colonnes dans le même ordre
        # Gérer les colonnes manquantes (ajouter avec valeur 0)
        for col in self.feature_names_fitted:
            if col not in X.columns:
                X[col] = 0

        # Garder seulement les colonnes utilisées lors du fit, dans le bon ordre
        X = X[self.feature_names_fitted]

        # Scaler
        X[self.feature_names_fitted] = self.scaler.transform(X[self.feature_names_fitted])

        return X

    def _create_features(self, X, fit_mode=True):
        """Création des features engineered"""
        df = X.copy()

        # 1. Ratio couverture PNB
        if 'PNB analytique (vision commerciale) cumulé' in df.columns and 'Montant demandé' in df.columns:
            df['ratio_pnb_montant'] = (
                df['PNB analytique (vision commerciale) cumulé'] /
                (df['Montant demandé'] + 1)
            )

        # 2. Écart à la médiane de la famille (calculé sur 2024, appliqué partout)
        if 'Famille Produit' in df.columns and 'Montant demandé' in df.columns:
            df['ecart_mediane_famille'] = df.apply(
                lambda row: (
                    row['Montant demandé'] -
                    self.family_medians.get(row['Famille Produit'], row['Montant demandé'])
                ) / (self.family_medians.get(row['Famille Produit'], 1) + 1),
                axis=1
            )

        # 3. Log transformations
        if 'Montant demandé' in df.columns:
            df['log_montant'] = np.log1p(np.abs(df['Montant demandé']))

        if 'PNB analytique (vision commerciale) cumulé' in df.columns:
            df['log_pnb'] = np.log1p(np.abs(df['PNB analytique (vision commerciale) cumulé']))

        if 'anciennete_annees' in df.columns:
            df['log_anciennete'] = np.log1p(np.abs(df['anciennete_annees']))

        # 4. Features d'interaction
        if 'Montant demandé' in df.columns and 'anciennete_annees' in df.columns:
            df['montant_x_anciennete'] = df['Montant demandé'] * df['anciennete_annees']

        if 'PNB analytique (vision commerciale) cumulé' in df.columns and 'anciennete_annees' in df.columns:
            df['pnb_x_anciennete'] = df['PNB analytique (vision commerciale) cumulé'] * df['anciennete_annees']

        # Sélectionner colonnes numériques
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        # Garder colonnes nécessaires
        keep_cols = [col for col in numeric_cols
                     if col != 'Fondee']

        df_result = df[keep_cols]

        # CRITICAL: Nettoyer les inf et NaN
        df_result = self._clean_numeric_data(df_result)

        return df_result

    def _clean_numeric_data(self, df):
        """Nettoie les NaN et inf dans les colonnes numériques"""
        df_clean = df.copy()

        for col in df_clean.columns:
            if df_clean[col].dtype in [np.float64, np.float32, np.int64, np.int32]:
                # Remplacer inf et -inf par NaN
                df_clean[col] = df_clean[col].replace([np.inf, -np.inf], np.nan)

                # Remplacer NaN par la médiane ou 0
                median_val = df_clean[col].median()
                if pd.isna(median_val):
                    median_val = 0.0
                df_clean[col] = df_clean[col].fillna(median_val)

        return df_clean

    def fit_transform(self, df):
        """Fit puis transform"""
        self.fit(df)
        return self.transform(df)


class ModelComparison:
    """Comparaison de 3 modèles avec calcul corrigé des pertes"""

    def __init__(self):
        self.models = {}
        self.results = {}
        self.preprocessor = ProductionPreprocessor()

    def load_data(self):
        """Charger données 2024 et 2025"""
        print("\n" + "="*80)
        print("📂 CHARGEMENT DES DONNÉES")
        print("="*80)

        self.df_2024 = pd.read_excel('data/raw/reclamations_2024.xlsx')
        self.df_2025 = pd.read_excel('data/raw/reclamations_2025.xlsx')

        # Nettoyer les montants dès le chargement
        if 'Montant demandé' in self.df_2024.columns:
            self.df_2024['Montant demandé'] = pd.to_numeric(self.df_2024['Montant demandé'], errors='coerce').fillna(0)
            self.df_2024['Montant demandé'] = self.df_2024['Montant demandé'].replace([np.inf, -np.inf], 0).clip(lower=0)

        if 'Montant demandé' in self.df_2025.columns:
            self.df_2025['Montant demandé'] = pd.to_numeric(self.df_2025['Montant demandé'], errors='coerce').fillna(0)
            self.df_2025['Montant demandé'] = self.df_2025['Montant demandé'].replace([np.inf, -np.inf], 0).clip(lower=0)

        print(f"✅ 2024: {len(self.df_2024)} réclamations")
        print(f"✅ 2025: {len(self.df_2025)} réclamations")

    def prepare_data(self):
        """Preprocessing"""
        print("\n" + "="*80)
        print("🔧 PREPROCESSING")
        print("="*80)

        # Fit sur 2024
        X_train = self.preprocessor.fit_transform(self.df_2024)
        y_train = self.df_2024['Fondee'].values

        # Transform sur 2025
        X_test = self.preprocessor.transform(self.df_2025)
        y_test = self.df_2025['Fondee'].values

        print(f"\n📊 Shape 2024: {X_train.shape}")
        print(f"📊 Shape 2025: {X_test.shape}")

        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    def optimize_xgboost(self):
        """Optimiser XGBoost avec Optuna"""
        print("\n🔬 Optimisation XGBoost...")

        # Calculer scale_pos_weight pour gérer le déséquilibre
        scale_pos_weight = (self.y_train == 0).sum() / (self.y_train == 1).sum()
        print(f"   Scale pos weight: {scale_pos_weight:.2f}")

        def objective(trial):
            params = {
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'gamma': trial.suggest_float('gamma', 0, 5),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
                'scale_pos_weight': scale_pos_weight,
            }

            model = xgb.XGBClassifier(**params, random_state=42, n_jobs=-1)
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            scores = cross_val_score(model, self.X_train, self.y_train, cv=cv, scoring='f1', n_jobs=-1)

            return scores.mean()

        study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(objective, n_trials=50, show_progress_bar=False)

        best_params = study.best_params
        print(f"   ✅ Best F1-Score CV: {study.best_value:.4f}")

        model = xgb.XGBClassifier(**best_params, random_state=42, n_jobs=-1)
        model.fit(self.X_train, self.y_train)

        return model

    def optimize_random_forest(self):
        """Optimiser Random Forest avec Optuna"""
        print("\n🔬 Optimisation Random Forest...")

        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                'max_depth': trial.suggest_int('max_depth', 5, 30),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
            }

            model = RandomForestClassifier(**params, random_state=42, n_jobs=-1)
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            scores = cross_val_score(model, self.X_train, self.y_train, cv=cv, scoring='f1', n_jobs=-1)

            return scores.mean()

        study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(objective, n_trials=50, show_progress_bar=False)

        best_params = study.best_params
        print(f"   ✅ Best F1-Score CV: {study.best_value:.4f}")

        model = RandomForestClassifier(**best_params, random_state=42, n_jobs=-1)
        model.fit(self.X_train, self.y_train)

        return model

    def optimize_catboost(self):
        """Optimiser CatBoost avec Optuna"""
        print("\n🔬 Optimisation CatBoost...")

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
        """Entraîner les 3 modèles"""
        print("\n" + "="*80)
        print("🎯 ENTRAÎNEMENT DES MODÈLES")
        print("="*80)

        self.models['XGBoost'] = self.optimize_xgboost()
        self.models['RandomForest'] = self.optimize_random_forest()
        self.models['CatBoost'] = self.optimize_catboost()

        print("\n✅ Tous les modèles entraînés")

    def optimize_threshold_dual(self, y_prob, name):
        """Optimiser 2 seuils pour créer 3 zones: Rejet Auto / Audit Manuel / Validation Auto"""
        print(f"\n🎯 Optimisation des seuils de décision (2 seuils)...")

        montants_2025 = self.df_2025['Montant demandé'].values
        best_result = None
        best_gain_net = -float('inf')
        threshold_results = []

        # Tester différentes combinaisons de seuils
        for t_low in np.arange(0.10, 0.50, 0.02):
            for t_high in np.arange(0.50, 0.95, 0.02):
                if t_high <= t_low:
                    continue

                # 3 zones: <= t_low (Rejet), > t_low ET < t_high (Audit), >= t_high (Validation)
                mask_rejet = y_prob <= t_low  # Prédire Non Fondée (0)
                mask_audit = (y_prob > t_low) & (y_prob < t_high)
                mask_validation = y_prob >= t_high  # Prédire Fondée (1)

                # Créer les prédictions
                y_pred = np.zeros(len(y_prob), dtype=int)
                y_pred[mask_validation] = 1  # Validation = Fondée

                # Pour l'audit, on ne fait rien (manuel), donc on ne compte pas dans l'automatisation
                # Seuls les cas en rejet et validation sont automatisés

                # Calcul des erreurs uniquement sur les cas automatisés
                mask_auto = mask_rejet | mask_validation

                if mask_auto.sum() == 0:
                    continue

                # Sur les cas automatisés
                y_pred_auto = y_pred[mask_auto]
                y_true_auto = self.y_test[mask_auto]
                montants_auto = montants_2025[mask_auto]

                fp_mask = (y_true_auto == 0) & (y_pred_auto == 1)
                fn_mask = (y_true_auto == 1) & (y_pred_auto == 0)

                # Calcul financier - avec gestion des valeurs aberrantes
                # Nettoyer les montants (supprimer inf, NaN, valeurs négatives)
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

                # Calcul des précisions par zone
                if mask_rejet.sum() > 0:
                    prec_rejet = (self.y_test[mask_rejet] == 0).mean()
                else:
                    prec_rejet = 0

                if mask_validation.sum() > 0:
                    prec_validation = (self.y_test[mask_validation] == 1).mean()
                else:
                    prec_validation = 0

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

                # Critère: Maximiser gain NET avec contraintes de précision
                if prec_rejet >= 0.95 and prec_validation >= 0.93:
                    if gain_net > best_gain_net:
                        best_gain_net = gain_net
                        best_result = threshold_results[-1].copy()

        # Fallback si aucune solution ne satisfait les contraintes
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
        """Évaluer sur 2025 avec optimisation des seuils (2 seuils)"""
        print("\n" + "="*80)
        print("📊 ÉVALUATION SUR 2025")
        print("="*80)

        for name, model in self.models.items():
            print(f"\n{'='*80}")
            print(f"📊 {name}")
            print(f"{'='*80}")

            # Probabilités
            y_prob = model.predict_proba(self.X_test)[:, 1]

            # OPTIMISATION DES 2 SEUILS
            best_thresholds, threshold_results = self.optimize_threshold_dual(y_prob, name)

            # Créer les prédictions avec les 2 seuils
            t_low = best_thresholds['threshold_low']
            t_high = best_thresholds['threshold_high']

            y_pred = np.zeros(len(y_prob), dtype=int)
            mask_rejet = y_prob <= t_low
            mask_audit = (y_prob > t_low) & (y_prob < t_high)
            mask_validation = y_prob >= t_high
            y_pred[mask_validation] = 1

            # Métriques classiques (seulement sur les cas automatisés)
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
            print(f"   Seuil BAS  : {t_low:.2f}")
            print(f"   Seuil HAUT : {t_high:.2f}")
            print(f"   Accuracy   : {acc:.4f}")
            print(f"   Precision  : {prec:.4f}")
            print(f"   Recall     : {rec:.4f}")
            print(f"   F1-Score   : {f1:.4f}")
            print(f"   ROC-AUC    : {auc:.4f}")

            print(f"\n📊 Distribution des cas:")
            print(f"   Rejets auto      : {mask_rejet.sum()} ({100*mask_rejet.sum()/len(y_prob):.1f}%)")
            print(f"   Audits manuels   : {mask_audit.sum()} ({100*mask_audit.sum()/len(y_prob):.1f}%)")
            print(f"   Validations auto : {mask_validation.sum()} ({100*mask_validation.sum()/len(y_prob):.1f}%)")

            # Utiliser les résultats déjà calculés dans optimize_threshold_dual
            gain_net = best_thresholds['gain_net']
            auto = best_thresholds['auto']
            perte_fp = 0  # Recalculer pour affichage
            perte_fn = 0

            if mask_auto.sum() > 0:
                montants_2025 = self.df_2025['Montant demandé'].values
                montants_auto = montants_2025[mask_auto]

                # Nettoyer les montants (gestion inf, NaN, outliers)
                montants_auto_clean = np.nan_to_num(montants_auto, nan=0.0, posinf=0.0, neginf=0.0)
                if len(montants_auto_clean) > 0 and montants_auto_clean.max() > 0:
                    montants_auto_clean = np.clip(montants_auto_clean, 0, np.percentile(montants_auto_clean, 99))
                else:
                    montants_auto_clean = montants_auto_clean.clip(0)

                fp_mask = (y_true_auto == 0) & (y_pred_auto == 1)
                fn_mask = (y_true_auto == 1) & (y_pred_auto == 0)
                perte_fp = montants_auto_clean[fp_mask].sum()
                perte_fn = 2 * montants_auto_clean[fn_mask].sum()
                gain_brut = auto * PRIX_UNITAIRE_DH
            else:
                gain_brut = 0
                fp_mask = np.zeros(0, dtype=bool)
                fn_mask = np.zeros(0, dtype=bool)

            print(f"\n💰 Impact financier (CORRIGÉ):")
            print(f"   Automatisés : {auto}/{len(self.y_test)} ({100*auto/len(self.y_test):.1f}%)")
            print(f"   Gain brut   : {gain_brut:,.0f} DH")
            print(f"   Perte FP    : {perte_fp:,.0f} DH ({fp_mask.sum()} cas)")
            print(f"   Perte FN    : {perte_fn:,.0f} DH ({fn_mask.sum()} cas)")
            print(f"   Gain NET    : {gain_net:,.0f} DH")

            # Sauvegarder résultats
            self.results[name] = {
                'threshold_low': t_low,
                'threshold_high': t_high,
                'accuracy': acc,
                'precision': prec,
                'recall': rec,
                'f1': f1,
                'auc': auc,
                'auto': auto,
                'taux_auto': 100*auto/len(self.y_test),
                'n_rejet': mask_rejet.sum(),
                'n_audit': mask_audit.sum(),
                'n_validation': mask_validation.sum(),
                'prec_rejet': best_thresholds['prec_rejet'],
                'prec_validation': best_thresholds['prec_validation'],
                'gain_brut': gain_brut,
                'perte_fp': perte_fp,
                'perte_fn': perte_fn,
                'gain_net': gain_net,
                'fp_count': fp_mask.sum(),
                'fn_count': fn_mask.sum(),
                'y_pred': y_pred,
                'y_prob': y_prob,
                'threshold_results': threshold_results
            }

    def generate_comparison_chart(self):
        """Générer graphiques de comparaison"""
        print("\n" + "="*80)
        print("📊 GÉNÉRATION DES VISUALISATIONS")
        print("="*80)

        fig, axes = plt.subplots(3, 3, figsize=(18, 14))
        fig.suptitle('COMPARAISON DES MODÈLES - 2025 (avec seuils optimisés)', fontsize=16, fontweight='bold', y=0.995)

        models_list = list(self.results.keys())
        colors = ['#3498db', '#2ecc71', '#9b59b6']

        # 1. Accuracy
        ax = axes[0, 0]
        vals = [self.results[m]['accuracy'] for m in models_list]
        bars = ax.bar(models_list, vals, color=colors, alpha=0.7)
        ax.set_ylabel('Accuracy', fontweight='bold')
        ax.set_ylim([0.99, 1.0])
        ax.set_title('Accuracy', fontweight='bold')
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, val + 0.0001,
                   f'{val:.4f}', ha='center', fontweight='bold')

        # 2. Precision
        ax = axes[0, 1]
        vals = [self.results[m]['precision'] for m in models_list]
        bars = ax.bar(models_list, vals, color=colors, alpha=0.7)
        ax.set_ylabel('Precision', fontweight='bold')
        ax.set_ylim([0.99, 1.0])
        ax.set_title('Precision', fontweight='bold')
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, val + 0.0001,
                   f'{val:.4f}', ha='center', fontweight='bold')

        # 3. Recall
        ax = axes[0, 2]
        vals = [self.results[m]['recall'] for m in models_list]
        bars = ax.bar(models_list, vals, color=colors, alpha=0.7)
        ax.set_ylabel('Recall', fontweight='bold')
        ax.set_ylim([0.99, 1.0])
        ax.set_title('Recall', fontweight='bold')
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, val + 0.0001,
                   f'{val:.4f}', ha='center', fontweight='bold')

        # 4. F1-Score
        ax = axes[1, 0]
        vals = [self.results[m]['f1'] for m in models_list]
        bars = ax.bar(models_list, vals, color=colors, alpha=0.7)
        ax.set_ylabel('F1-Score', fontweight='bold')
        ax.set_ylim([0.99, 1.0])
        ax.set_title('F1-Score', fontweight='bold')
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, val + 0.0001,
                   f'{val:.4f}', ha='center', fontweight='bold')

        # 5. ROC-AUC
        ax = axes[1, 1]
        vals = [self.results[m]['auc'] for m in models_list]
        bars = ax.bar(models_list, vals, color=colors, alpha=0.7)
        ax.set_ylabel('ROC-AUC', fontweight='bold')
        ax.set_ylim([0.99, 1.0])
        ax.set_title('ROC-AUC', fontweight='bold')
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, val + 0.0001,
                   f'{val:.4f}', ha='center', fontweight='bold')

        # 6. Taux automatisation
        ax = axes[1, 2]
        vals = [self.results[m]['taux_auto'] for m in models_list]
        bars = ax.bar(models_list, vals, color=colors, alpha=0.7)
        ax.set_ylabel('Taux (%)', fontweight='bold')
        ax.set_title('Taux automatisation', fontweight='bold')
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, val + 0.5,
                   f'{val:.1f}%', ha='center', fontweight='bold')

        # 7. Gain NET
        ax = axes[2, 0]
        vals = [self.results[m]['gain_net'] for m in models_list]
        bars = ax.bar(models_list, vals, color=colors, alpha=0.7)
        ax.set_ylabel('Gain NET (DH)', fontweight='bold')
        ax.set_title('Gain NET (CORRIGÉ)', fontweight='bold')
        ax.ticklabel_format(style='plain', axis='y')
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, val + 10000,
                   f'{val:,.0f}', ha='center', fontweight='bold', fontsize=9)

        # 8. Perte FP (montants réels)
        ax = axes[2, 1]
        vals = [self.results[m]['perte_fp'] for m in models_list]
        bars = ax.bar(models_list, vals, color='#e74c3c', alpha=0.7)
        ax.set_ylabel('Perte FP (DH)', fontweight='bold')
        ax.set_title('Perte FP - Montants réels (CORRIGÉ)', fontweight='bold')
        ax.ticklabel_format(style='plain', axis='y')
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, val + 500,
                   f'{val:,.0f}', ha='center', fontweight='bold', fontsize=9)

        # 9. Perte FN (montants réels)
        ax = axes[2, 2]
        vals = [self.results[m]['perte_fn'] for m in models_list]
        bars = ax.bar(models_list, vals, color='#e67e22', alpha=0.7)
        ax.set_ylabel('Perte FN (DH)', fontweight='bold')
        ax.set_title('Perte FN - Montants réels x2 (CORRIGÉ)', fontweight='bold')
        ax.ticklabel_format(style='plain', axis='y')
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, val + 1000,
                   f'{val:,.0f}', ha='center', fontweight='bold', fontsize=9)

        plt.tight_layout()

        # Sauvegarder
        output_dir = Path('outputs/production/figures')
        output_dir.mkdir(parents=True, exist_ok=True)

        output_path = output_dir / 'model_comparison.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"   ✅ model_comparison.png")

        plt.close()

        # Graphique d'optimisation des seuils
        self.generate_threshold_optimization_chart(output_dir)

    def generate_threshold_optimization_chart(self, output_dir):
        """Graphique montrant les 3 zones de décision pour chaque modèle"""
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle('OPTIMISATION DES SEUILS (3 zones: Rejet / Audit / Validation)', fontsize=14, fontweight='bold')

        colors = {'XGBoost': '#3498db', 'RandomForest': '#2ecc71', 'CatBoost': '#9b59b6'}

        # Graphique 1: Distribution des 3 zones par modèle
        ax = axes[0]
        models_list = list(self.results.keys())
        x = np.arange(len(models_list))
        width = 0.25

        rejet_counts = [self.results[m]['n_rejet'] for m in models_list]
        audit_counts = [self.results[m]['n_audit'] for m in models_list]
        validation_counts = [self.results[m]['n_validation'] for m in models_list]

        ax.bar(x - width, rejet_counts, width, label='Rejet Auto', color='#e74c3c', alpha=0.7)
        ax.bar(x, audit_counts, width, label='Audit Manuel', color='#f39c12', alpha=0.7)
        ax.bar(x + width, validation_counts, width, label='Validation Auto', color='#2ecc71', alpha=0.7)

        ax.set_ylabel('Nombre de cas', fontweight='bold')
        ax.set_title('Distribution des décisions par modèle', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(models_list)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        # Ajouter les pourcentages
        for i, model in enumerate(models_list):
            total = 8000
            r_pct = 100 * rejet_counts[i] / total
            a_pct = 100 * audit_counts[i] / total
            v_pct = 100 * validation_counts[i] / total

            ax.text(i - width, rejet_counts[i] + 100, f'{r_pct:.0f}%', ha='center', fontsize=8)
            ax.text(i, audit_counts[i] + 100, f'{a_pct:.0f}%', ha='center', fontsize=8)
            ax.text(i + width, validation_counts[i] + 100, f'{v_pct:.0f}%', ha='center', fontsize=8)

        # Graphique 2: Précisions par zone
        ax = axes[1]
        x = np.arange(len(models_list))
        width = 0.35

        prec_rejet = [self.results[m]['prec_rejet'] * 100 for m in models_list]
        prec_validation = [self.results[m]['prec_validation'] * 100 for m in models_list]

        ax.bar(x - width/2, prec_rejet, width, label='Précision Rejet', color='#e74c3c', alpha=0.7)
        ax.bar(x + width/2, prec_validation, width, label='Précision Validation', color='#2ecc71', alpha=0.7)

        # Lignes d'objectif
        ax.axhline(y=95, color='red', linestyle='--', alpha=0.5, label='Objectif Rejet: 95%')
        ax.axhline(y=93, color='green', linestyle='--', alpha=0.5, label='Objectif Validation: 93%')

        ax.set_ylabel('Précision (%)', fontweight='bold')
        ax.set_title('Précisions par zone de décision', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(models_list)
        ax.set_ylim([85, 100])
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')

        # Ajouter les valeurs
        for i in range(len(models_list)):
            ax.text(i - width/2, prec_rejet[i] + 0.5, f'{prec_rejet[i]:.1f}%', ha='center', fontsize=8, fontweight='bold')
            ax.text(i + width/2, prec_validation[i] + 0.5, f'{prec_validation[i]:.1f}%', ha='center', fontsize=8, fontweight='bold')

        plt.tight_layout()

        output_path = output_dir / 'threshold_optimization.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"   ✅ threshold_optimization.png")

        plt.close()

    def generate_report(self):
        """Générer rapport texte"""
        output_dir = Path('outputs/production')
        output_dir.mkdir(parents=True, exist_ok=True)

        report_path = output_dir / 'rapport_comparison.txt'

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("RAPPORT COMPARAISON DE MODÈLES\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*80 + "\n\n")

            f.write("MODÈLES COMPARÉS:\n")
            f.write("- XGBoost (optimisé Optuna)\n")
            f.write("- Random Forest (optimisé Optuna)\n")
            f.write("- CatBoost (optimisé Optuna)\n\n")

            f.write("CALCUL DES PERTES (CORRIGÉ):\n")
            f.write("- Perte FP = Somme des montants demandés des faux positifs\n")
            f.write("- Perte FN = 2 × Somme des montants demandés des faux négatifs\n")
            f.write("- Gain NET = (Automatisés × 169 DH) - Perte FP - Perte FN\n\n")

            f.write("OPTIMISATION DES SEUILS (2 seuils):\n")
            f.write("- 3 zones de décision: Rejet Auto / Audit Manuel / Validation Auto\n")
            f.write("- Seuil BAS: en dessous → Rejet automatique (Non Fondée)\n")
            f.write("- Seuil HAUT: au dessus → Validation automatique (Fondée)\n")
            f.write("- Entre les 2: Audit manuel requis\n")
            f.write("- Critère d'optimisation: Maximisation du Gain NET\n")
            f.write("- Contraintes: Précision Rejet ≥95%, Précision Validation ≥93%\n\n")

            f.write("="*80 + "\n")
            f.write("RÉSULTATS SUR 2025\n")
            f.write("="*80 + "\n\n")

            for name in self.results.keys():
                r = self.results[name]
                f.write(f"{name}:\n")
                f.write(f"  Seuils:\n")
                f.write(f"    Seuil BAS (Rejet)      : {r['threshold_low']:.2f}\n")
                f.write(f"    Seuil HAUT (Validation): {r['threshold_high']:.2f}\n\n")

                f.write(f"  Distribution:\n")
                f.write(f"    Rejets auto      : {r['n_rejet']} ({100*r['n_rejet']/8000:.1f}%)\n")
                f.write(f"    Audits manuels   : {r['n_audit']} ({100*r['n_audit']/8000:.1f}%)\n")
                f.write(f"    Validations auto : {r['n_validation']} ({100*r['n_validation']/8000:.1f}%)\n\n")

                f.write(f"  Précisions par zone:\n")
                f.write(f"    Précision Rejet     : {r['prec_rejet']:.1%}\n")
                f.write(f"    Précision Validation: {r['prec_validation']:.1%}\n\n")

                f.write(f"  Métriques (sur cas automatisés):\n")
                f.write(f"    Accuracy  : {r['accuracy']:.4f}\n")
                f.write(f"    Precision : {r['precision']:.4f}\n")
                f.write(f"    Recall    : {r['recall']:.4f}\n")
                f.write(f"    F1-Score  : {r['f1']:.4f}\n")
                f.write(f"    ROC-AUC   : {r['auc']:.4f}\n")
                f.write(f"\n")
                f.write(f"  Automatisation:\n")
                f.write(f"    Taux      : {r['taux_auto']:.1f}%\n")
                f.write(f"    Nombre    : {r['auto']}/8000\n")
                f.write(f"\n")
                f.write(f"  Financier (CORRIGÉ):\n")
                f.write(f"    Gain brut : {r['gain_brut']:,.0f} DH\n")
                f.write(f"    Perte FP  : {r['perte_fp']:,.0f} DH ({r['fp_count']} cas)\n")
                f.write(f"    Perte FN  : {r['perte_fn']:,.0f} DH ({r['fn_count']} cas)\n")
                f.write(f"    Gain NET  : {r['gain_net']:,.0f} DH\n")
                f.write("\n" + "-"*80 + "\n\n")

            # Meilleur modèle
            best_model = max(self.results.keys(), key=lambda k: self.results[k]['gain_net'])
            f.write("="*80 + "\n")
            f.write("MEILLEUR MODÈLE (Gain NET):\n")
            f.write("="*80 + "\n")
            f.write(f"{best_model}\n")
            f.write(f"  Seuil BAS       : {self.results[best_model]['threshold_low']:.2f}\n")
            f.write(f"  Seuil HAUT      : {self.results[best_model]['threshold_high']:.2f}\n")
            f.write(f"  Gain NET        : {self.results[best_model]['gain_net']:,.0f} DH\n")
            f.write(f"  Taux Auto       : {self.results[best_model]['taux_auto']:.1f}%\n")
            f.write(f"  Précision Rejet : {self.results[best_model]['prec_rejet']:.1%}\n")
            f.write(f"  Précision Valid.: {self.results[best_model]['prec_validation']:.1%}\n")
            f.write(f"  F1-Score        : {self.results[best_model]['f1']:.4f}\n\n")

        print(f"   ✅ rapport_comparison.txt")

    def run(self):
        """Exécution complète"""
        self.load_data()
        self.prepare_data()
        self.train_models()
        self.evaluate_models()
        self.generate_comparison_chart()
        self.generate_report()

        # Analyse par famille pour le meilleur modèle
        self.analyze_by_family()

        print("\n" + "="*80)
        print("✅ COMPARAISON TERMINÉE")
        print("="*80)
        print(f"\n📂 Résultats: outputs/production/")

        # Afficher le meilleur
        best_model = max(self.results.keys(), key=lambda k: self.results[k]['gain_net'])
        print(f"\n🏆 MEILLEUR MODÈLE: {best_model}")
        print(f"   Gain NET: {self.results[best_model]['gain_net']:,.0f} DH")
        print(f"   F1-Score: {self.results[best_model]['f1']:.4f}")

    def analyze_by_family(self):
        """Analyse détaillée par famille de produit pour le meilleur modèle"""
        from analyze_by_family import FamilyAnalyzer

        print("\n" + "="*80)
        print("📊 ANALYSE PAR FAMILLE DE PRODUIT")
        print("="*80)

        # Trouver le meilleur modèle
        best_model = max(self.results.keys(), key=lambda k: self.results[k]['gain_net'])
        print(f"\nAnalyse basée sur: {best_model}")

        # Extraire les prédictions du meilleur modèle
        y_true = self.y_test
        y_pred = self.results[best_model]['y_pred']
        y_prob = self.results[best_model]['y_prob']

        # Créer l'analyseur
        analyzer = FamilyAnalyzer()

        # Exécuter l'analyse
        analyzer.run(y_true, y_pred, y_prob)


if __name__ == '__main__':
    comparison = ModelComparison()
    comparison.run()
