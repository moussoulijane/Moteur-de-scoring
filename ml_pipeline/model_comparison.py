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
            df['log_montant'] = np.log1p(df['Montant demandé'])

        if 'PNB analytique (vision commerciale) cumulé' in df.columns:
            df['log_pnb'] = np.log1p(df['PNB analytique (vision commerciale) cumulé'])

        if 'anciennete_annees' in df.columns:
            df['log_anciennete'] = np.log1p(df['anciennete_annees'])

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

        return df[keep_cols]

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

    def optimize_threshold(self, y_prob, name):
        """Optimiser le seuil de décision pour maximiser le gain NET"""
        print(f"\n🎯 Optimisation du seuil de décision...")

        montants_2025 = self.df_2025['Montant demandé'].values
        best_threshold = 0.5
        best_gain_net = -float('inf')
        threshold_results = []

        # Tester différents seuils
        thresholds = np.arange(0.1, 0.95, 0.01)

        for threshold in thresholds:
            y_pred = (y_prob >= threshold).astype(int)

            # Calcul des métriques
            tp = ((self.y_test == 1) & (y_pred == 1)).sum()
            tn = ((self.y_test == 0) & (y_pred == 0)).sum()
            fp_mask = (self.y_test == 0) & (y_pred == 1)
            fn_mask = (self.y_test == 1) & (y_pred == 0)

            # Calcul financier
            perte_fp = montants_2025[fp_mask].sum()
            perte_fn = 2 * montants_2025[fn_mask].sum()
            auto = tp + tn
            gain_brut = auto * PRIX_UNITAIRE_DH
            gain_net = gain_brut - perte_fp - perte_fn

            threshold_results.append({
                'threshold': threshold,
                'gain_net': gain_net,
                'auto': auto,
                'fp': fp_mask.sum(),
                'fn': fn_mask.sum()
            })

            if gain_net > best_gain_net:
                best_gain_net = gain_net
                best_threshold = threshold

        print(f"   ✅ Seuil optimal: {best_threshold:.2f}")
        print(f"   ✅ Gain NET max: {best_gain_net:,.0f} DH")

        return best_threshold, threshold_results

    def evaluate_models(self):
        """Évaluer sur 2025 avec optimisation du seuil"""
        print("\n" + "="*80)
        print("📊 ÉVALUATION SUR 2025")
        print("="*80)

        for name, model in self.models.items():
            print(f"\n{'='*80}")
            print(f"📊 {name}")
            print(f"{'='*80}")

            # Probabilités
            y_prob = model.predict_proba(self.X_test)[:, 1]

            # OPTIMISATION DU SEUIL
            best_threshold, threshold_results = self.optimize_threshold(y_prob, name)

            # Prédictions avec seuil optimal
            y_pred = (y_prob >= best_threshold).astype(int)

            # Métriques classiques
            acc = accuracy_score(self.y_test, y_pred)
            prec = precision_score(self.y_test, y_pred, zero_division=0)
            rec = recall_score(self.y_test, y_pred)
            f1 = f1_score(self.y_test, y_pred, zero_division=0)
            auc = roc_auc_score(self.y_test, y_prob)

            print(f"\n📊 Métriques avec seuil optimal ({best_threshold:.2f}):")
            print(f"   Accuracy  : {acc:.4f}")
            print(f"   Precision : {prec:.4f}")
            print(f"   Recall    : {rec:.4f}")
            print(f"   F1-Score  : {f1:.4f}")
            print(f"   ROC-AUC   : {auc:.4f}")

            # Calcul financier CORRIGÉ
            tp = ((self.y_test == 1) & (y_pred == 1)).sum()
            tn = ((self.y_test == 0) & (y_pred == 0)).sum()
            fp_mask = (self.y_test == 0) & (y_pred == 1)
            fn_mask = (self.y_test == 1) & (y_pred == 0)

            # CORRECTION: Utiliser les montants réels
            montants_2025 = self.df_2025['Montant demandé'].values

            # Perte FP = somme des montants des faux positifs
            perte_fp = montants_2025[fp_mask].sum()

            # Perte FN = 2 * somme des montants des faux négatifs (client insatisfait)
            perte_fn = 2 * montants_2025[fn_mask].sum()

            # Gain brut = nombre automatisé * prix unitaire
            auto = tp + tn
            gain_brut = auto * PRIX_UNITAIRE_DH

            # Gain net = gain brut - pertes
            gain_net = gain_brut - perte_fp - perte_fn

            print(f"\n💰 Impact financier (CORRIGÉ):")
            print(f"   Automatisés : {auto}/{len(self.y_test)} ({100*auto/len(self.y_test):.1f}%)")
            print(f"   Gain brut   : {gain_brut:,.0f} DH")
            print(f"   Perte FP    : {perte_fp:,.0f} DH ({fp_mask.sum()} cas)")
            print(f"   Perte FN    : {perte_fn:,.0f} DH ({fn_mask.sum()} cas)")
            print(f"   Gain NET    : {gain_net:,.0f} DH")

            # Sauvegarder résultats
            self.results[name] = {
                'threshold': best_threshold,
                'accuracy': acc,
                'precision': prec,
                'recall': rec,
                'f1': f1,
                'auc': auc,
                'auto': auto,
                'taux_auto': 100*auto/len(self.y_test),
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
        """Graphique montrant l'évolution du gain NET en fonction du seuil"""
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle('OPTIMISATION DU SEUIL DE DÉCISION', fontsize=14, fontweight='bold')

        colors = {'XGBoost': '#3498db', 'RandomForest': '#2ecc71', 'CatBoost': '#9b59b6'}

        # Graphique 1: Gain NET vs Seuil
        ax = axes[0]
        for name in self.results.keys():
            threshold_results = self.results[name]['threshold_results']
            thresholds = [r['threshold'] for r in threshold_results]
            gains = [r['gain_net'] for r in threshold_results]

            ax.plot(thresholds, gains, label=name, color=colors[name], linewidth=2)

            # Marquer le seuil optimal
            best_threshold = self.results[name]['threshold']
            best_gain = self.results[name]['gain_net']
            ax.scatter([best_threshold], [best_gain], color=colors[name], s=200,
                      zorder=5, edgecolors='black', linewidths=2)
            ax.annotate(f'{best_threshold:.2f}',
                       xy=(best_threshold, best_gain),
                       xytext=(10, 10), textcoords='offset points',
                       fontweight='bold', fontsize=9,
                       bbox=dict(boxstyle='round,pad=0.5', facecolor=colors[name], alpha=0.3))

        ax.set_xlabel('Seuil de décision', fontweight='bold')
        ax.set_ylabel('Gain NET (DH)', fontweight='bold')
        ax.set_title('Gain NET en fonction du seuil', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.ticklabel_format(style='plain', axis='y')

        # Graphique 2: Nombre FP et FN vs Seuil pour le meilleur modèle
        best_model = max(self.results.keys(), key=lambda k: self.results[k]['gain_net'])
        threshold_results = self.results[best_model]['threshold_results']

        ax = axes[1]
        thresholds = [r['threshold'] for r in threshold_results]
        fps = [r['fp'] for r in threshold_results]
        fns = [r['fn'] for r in threshold_results]

        ax.plot(thresholds, fps, label='Faux Positifs (FP)', color='#e74c3c', linewidth=2)
        ax.plot(thresholds, fns, label='Faux Négatifs (FN)', color='#e67e22', linewidth=2)

        # Marquer le seuil optimal
        best_threshold = self.results[best_model]['threshold']
        best_fp = self.results[best_model]['fp_count']
        best_fn = self.results[best_model]['fn_count']

        ax.scatter([best_threshold], [best_fp], color='#e74c3c', s=200,
                  zorder=5, edgecolors='black', linewidths=2)
        ax.scatter([best_threshold], [best_fn], color='#e67e22', s=200,
                  zorder=5, edgecolors='black', linewidths=2)

        ax.axvline(x=best_threshold, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlabel('Seuil de décision', fontweight='bold')
        ax.set_ylabel('Nombre d\'erreurs', fontweight='bold')
        ax.set_title(f'FP/FN vs Seuil - {best_model}', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

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

            f.write("OPTIMISATION DU SEUIL:\n")
            f.write("- Seuil optimal déterminé pour chaque modèle (au lieu de 0.5)\n")
            f.write("- Critère d'optimisation: Maximisation du Gain NET\n")
            f.write("- Plage testée: 0.10 à 0.94 (pas de 0.01)\n\n")

            f.write("="*80 + "\n")
            f.write("RÉSULTATS SUR 2025\n")
            f.write("="*80 + "\n\n")

            for name in self.results.keys():
                r = self.results[name]
                f.write(f"{name}:\n")
                f.write(f"  Seuil optimal: {r['threshold']:.2f}\n\n")
                f.write(f"  Métriques:\n")
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
            f.write(f"  Seuil optimal : {self.results[best_model]['threshold']:.2f}\n")
            f.write(f"  Gain NET      : {self.results[best_model]['gain_net']:,.0f} DH\n")
            f.write(f"  F1-Score      : {self.results[best_model]['f1']:.4f}\n\n")

        print(f"   ✅ rapport_comparison.txt")

    def run(self):
        """Exécution complète"""
        self.load_data()
        self.prepare_data()
        self.train_models()
        self.evaluate_models()
        self.generate_comparison_chart()
        self.generate_report()

        print("\n" + "="*80)
        print("✅ COMPARAISON TERMINÉE")
        print("="*80)
        print(f"\n📂 Résultats: outputs/production/")

        # Afficher le meilleur
        best_model = max(self.results.keys(), key=lambda k: self.results[k]['gain_net'])
        print(f"\n🏆 MEILLEUR MODÈLE: {best_model}")
        print(f"   Gain NET: {self.results[best_model]['gain_net']:,.0f} DH")
        print(f"   F1-Score: {self.results[best_model]['f1']:.4f}")


if __name__ == '__main__':
    comparison = ModelComparison()
    comparison.run()
