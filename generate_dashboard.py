"""
PIPELINE ML PRODUCTION - CLASSIFICATION RÉCLAMATIONS BANCAIRES
Version Production avec Optimisation Optuna + Règle Métier
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
import xgboost as xgb
import lightgbm as lgb
import optuna
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Configuration
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (14, 8)

PRIX_UNITAIRE_DH = 169


class ProductionPreprocessor:
    """Preprocessing production avec gestion stricte de l'ordre des colonnes"""

    def __init__(self):
        self.scaler = RobustScaler()
        self.family_medians = {}
        self.categorical_encodings = {}
        self.feature_names_fitted = None  # IMPORTANT: ordre des features après fit

    def fit(self, df):
        """Fit sur données 2024"""
        print("\n🔧 Configuration du preprocessing...")

        X = df.copy()

        # Calculer médianes par famille
        print("📊 Calcul médianes par famille (base 2024)...")
        self.family_medians = X.groupby('Famille Produit')['Montant demandé'].median().to_dict()
        print(f"   ✅ {len(self.family_medians)} familles")

        # Features engineering
        X = self._create_features(X, fit_mode=True)

        # Encoder catégorielles
        print("🔢 Encodage catégorielles...")
        cat_cols = ['Marché', 'Segment', 'Famille Produit', 'Catégorie', 'Sous-catégorie']
        for col in cat_cols:
            if col in X.columns:
                unique_vals = X[col].unique()
                self.categorical_encodings[col] = {val: idx for idx, val in enumerate(unique_vals)}

        X = self._encode_categorical(X)

        # Sélectionner colonnes numériques
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [c for c in numeric_cols if c != 'Fondee']

        # CRITICAL: Sauvegarder l'ordre des colonnes
        self.feature_names_fitted = sorted(numeric_cols)  # Trier pour garantir ordre

        print(f"📋 Features finales: {len(self.feature_names_fitted)}")

        # Fit scaler avec colonnes dans le bon ordre
        X_ordered = X[self.feature_names_fitted]
        self.scaler.fit(X_ordered)

        print(f"✅ Preprocessing configuré: {len(self.feature_names_fitted)} features")
        return self

    def transform(self, df):
        """Transform avec ordre garanti des colonnes"""
        X = df.copy()

        # Features engineering
        X = self._create_features(X, fit_mode=False)

        # Encoder catégorielles
        X = self._encode_categorical(X)

        # Sélectionner colonnes numériques
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

    def fit_transform(self, df):
        """Fit et transform"""
        self.fit(df)
        return self.transform(df)

    def _create_features(self, X, fit_mode=True):
        """Créer features métier"""
        df = X.copy()

        # 1. Ratio couverture PNB
        df['ratio_pnb_montant'] = (
            df['PNB analytique (vision commerciale) cumulé'] /
            (df['Montant demandé'] + 1)
        )

        # 2. Écart à la médiane de la famille
        df['ecart_mediane_famille'] = df.apply(
            lambda row: (
                row['Montant demandé'] -
                self.family_medians.get(row['Famille Produit'], row['Montant demandé'])
            ) / (self.family_medians.get(row['Famille Produit'], 1) + 1),
            axis=1
        )

        # 3. Log transformations
        df['log_montant'] = np.log1p(df['Montant demandé'])
        df['log_pnb'] = np.log1p(df['PNB analytique (vision commerciale) cumulé'])
        df['log_anciennete'] = np.log1p(df['anciennete_annees'])

        # 4. Features d'interaction
        df['montant_x_anciennete'] = df['Montant demandé'] * df['anciennete_annees']
        df['pnb_x_anciennete'] = df['PNB analytique (vision commerciale) cumulé'] * df['anciennete_annees']

        # Nettoyer NaN/inf
        df = df.replace([np.inf, -np.inf], np.nan)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col != 'Fondee':
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val if not pd.isna(median_val) else 0)

        return df

    def _encode_categorical(self, X):
        """Encoder catégorielles"""
        df = X.copy()

        for col, mapping in self.categorical_encodings.items():
            if col in df.columns:
                df[col] = df[col].map(mapping).fillna(-1)

        return df


class ProductionPipeline:
    """Pipeline production optimisé avec Optuna + règle métier"""

    def __init__(self, output_dir='outputs/production'):
        self.output_dir = output_dir
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        Path(f"{output_dir}/figures").mkdir(parents=True, exist_ok=True)
        Path(f"{output_dir}/models").mkdir(parents=True, exist_ok=True)

        self.preprocessor = ProductionPreprocessor()
        self.model = None
        self.best_params = None
        self.df_2024 = None
        self.df_2025 = None

    def load_data(self, path_2024, path_2025):
        """Charger données"""
        print("\n" + "="*80)
        print("📂 CHARGEMENT DES DONNÉES")
        print("="*80)

        self.df_2024 = pd.read_excel(path_2024)
        self.df_2025 = pd.read_excel(path_2025)

        print(f"✅ 2024: {len(self.df_2024)} réclamations")
        print(f"✅ 2025: {len(self.df_2025)} réclamations")

        # Afficher colonnes
        print(f"\n📋 Colonnes 2024: {len(self.df_2024.columns)}")
        print(f"📋 Colonnes 2025: {len(self.df_2025.columns)}")

    def optimize_model(self, X_train, y_train):
        """Optimisation Optuna pour modèle statistiquement fort"""
        print("\n" + "="*80)
        print("🔬 OPTIMISATION HYPERPARAMÈTRES (Optuna)")
        print("="*80)

        def objective(trial):
            """Fonction objectif pour Optuna"""
            params = {
                'objective': 'binary:logistic',
                'eval_metric': 'logloss',
                'random_state': 42,
                'n_jobs': -1,

                # Hyperparamètres à optimiser
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

            model = xgb.XGBClassifier(**params)

            # Cross-validation stratifiée
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='f1', n_jobs=-1)

            return scores.mean()

        # Optimisation
        print("🔄 Recherche des meilleurs hyperparamètres...")
        print("   (100 trials, 5-fold CV, optimisation F1-Score)")

        study = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=optuna.pruners.MedianPruner()
        )

        study.optimize(objective, n_trials=100, show_progress_bar=False, n_jobs=1)

        print(f"\n✅ Optimisation terminée!")
        print(f"   Meilleur F1-Score CV: {study.best_value:.4f}")
        print(f"\n📊 Meilleurs hyperparamètres:")
        for key, value in study.best_params.items():
            print(f"   {key:20s}: {value}")

        self.best_params = study.best_params
        return study.best_params

    def train_model(self):
        """Entraîner modèle optimisé"""
        print("\n" + "="*80)
        print("🎯 ENTRAÎNEMENT MODÈLE")
        print("="*80)

        # Preprocessing
        X_train = self.preprocessor.fit_transform(self.df_2024)
        y_train = self.df_2024['Fondee'].values

        print(f"\n📊 Shape: {X_train.shape}")

        # Optimisation Optuna
        best_params = self.optimize_model(X_train, y_train)

        # Entraîner modèle final avec meilleurs paramètres
        print("\n🏋️  Entraînement modèle final...")
        self.model = xgb.XGBClassifier(
            **best_params,
            objective='binary:logistic',
            eval_metric='logloss',
            random_state=42,
            n_jobs=-1
        )

        self.model.fit(X_train, y_train)

        # Métriques 2024
        y_pred_2024 = self.model.predict(X_train)
        y_prob_2024 = self.model.predict_proba(X_train)[:, 1]

        self.metrics_2024 = {
            'accuracy': accuracy_score(y_train, y_pred_2024),
            'precision': precision_score(y_train, y_pred_2024),
            'recall': recall_score(y_train, y_pred_2024),
            'f1': f1_score(y_train, y_pred_2024),
            'roc_auc': roc_auc_score(y_train, y_prob_2024)
        }

        print("\n📊 Métriques 2024 (entraînement):")
        for metric, value in self.metrics_2024.items():
            print(f"   {metric:12s}: {value:.4f}")

        # Validation statistique
        print("\n📊 Validation statistique (5-fold CV):")
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        cv_scores = {
            'accuracy': cross_val_score(self.model, X_train, y_train, cv=cv, scoring='accuracy'),
            'precision': cross_val_score(self.model, X_train, y_train, cv=cv, scoring='precision'),
            'recall': cross_val_score(self.model, X_train, y_train, cv=cv, scoring='recall'),
            'f1': cross_val_score(self.model, X_train, y_train, cv=cv, scoring='f1'),
            'roc_auc': cross_val_score(self.model, X_train, y_train, cv=cv, scoring='roc_auc')
        }

        for metric, scores in cv_scores.items():
            print(f"   {metric:12s}: {scores.mean():.4f} (+/- {scores.std()*2:.4f})")

        self.cv_scores = cv_scores

        # Sauvegarder
        joblib.dump(self.model, f'{self.output_dir}/models/model_production.pkl')
        joblib.dump(self.preprocessor, f'{self.output_dir}/models/preprocessor_production.pkl')
        joblib.dump(best_params, f'{self.output_dir}/models/best_params.pkl')
        print(f"\n✅ Modèle sauvegardé: {self.output_dir}/models/")

    def evaluate_2025(self):
        """Évaluer sur 2025"""
        print("\n" + "="*80)
        print("📊 ÉVALUATION SUR 2025")
        print("="*80)

        # Transform 2025
        X_test = self.preprocessor.transform(self.df_2025)
        y_test = self.df_2025['Fondee'].values

        print(f"📊 Shape 2025: {X_test.shape}")

        # Prédictions
        y_pred_2025 = self.model.predict(X_test)
        y_prob_2025 = self.model.predict_proba(X_test)[:, 1]

        # Métriques
        self.metrics_2025 = {
            'accuracy': accuracy_score(y_test, y_pred_2025),
            'precision': precision_score(y_test, y_pred_2025),
            'recall': recall_score(y_test, y_pred_2025),
            'f1': f1_score(y_test, y_pred_2025),
            'roc_auc': roc_auc_score(y_test, y_prob_2025)
        }

        print("\n📊 Métriques 2025:")
        for metric, value in self.metrics_2025.items():
            print(f"   {metric:12s}: {value:.4f}")

        # Comparaison
        print("\n📉 Dégradation 2024 → 2025:")
        for metric in self.metrics_2024.keys():
            degradation = ((self.metrics_2025[metric] - self.metrics_2024[metric]) /
                          self.metrics_2024[metric]) * 100
            print(f"   {metric:12s}: {degradation:+.2f}%")

        self.y_pred_2025 = y_pred_2025
        self.y_prob_2025 = y_prob_2025

    def apply_business_rule(self):
        """RÈGLE MÉTIER : 1 réclamation auto par client"""
        print("\n" + "="*80)
        print("🔒 APPLICATION RÈGLE MÉTIER : 1 RÉCLAMATION AUTO PAR CLIENT")
        print("="*80)

        df_scenario = self.df_2025.copy()
        df_scenario['y_pred'] = self.y_pred_2025
        df_scenario['y_prob'] = self.y_prob_2025
        df_scenario['y_true'] = self.df_2025['Fondee'].values

        # Convertir date
        df_scenario['Date de Qualification'] = pd.to_datetime(
            df_scenario['Date de Qualification'],
            errors='coerce'
        )

        # Identifier colonne client
        client_col = None
        for col in ['idtfcl', 'numero_compte', 'N compte', 'ID Client']:
            if col in df_scenario.columns:
                client_col = col
                break

        if client_col is None:
            print("⚠️  Colonne client non trouvée, création index")
            df_scenario['client_id'] = df_scenario.index
            client_col = 'client_id'

        print(f"📋 Colonne client: {client_col}")

        # Trier par client puis date
        df_scenario = df_scenario.sort_values([client_col, 'Date de Qualification'])

        # Marquer première réclamation
        df_scenario['is_first_reclamation'] = ~df_scenario.duplicated(subset=[client_col], keep='first')

        # Stats
        total_clients = df_scenario[client_col].nunique()
        total_reclamations = len(df_scenario)
        first_reclamations = df_scenario['is_first_reclamation'].sum()
        multi_reclamations = total_reclamations - first_reclamations

        print(f"\n📊 Statistiques:")
        print(f"   Clients uniques: {total_clients}")
        print(f"   Total réclamations: {total_reclamations}")
        print(f"   Premières: {first_reclamations}")
        print(f"   Multiples: {multi_reclamations} ({100*multi_reclamations/total_reclamations:.1f}%)")

        # Appliquer règle
        df_scenario['can_automate'] = df_scenario['is_first_reclamation']
        df_scenario['y_pred_with_rule'] = np.where(
            df_scenario['can_automate'],
            df_scenario['y_pred'],
            0
        )

        # Impact
        auto_without = df_scenario['y_pred'].sum()
        auto_with = df_scenario['y_pred_with_rule'].sum()
        blocked = auto_without - auto_with

        print(f"\n🚦 Impact règle:")
        print(f"   SANS règle: {auto_without}")
        print(f"   AVEC règle: {auto_with}")
        print(f"   Bloquées: {blocked} ({100*blocked/auto_without:.1f}%)")

        self.df_scenario = df_scenario

    def calculate_financial_impact(self):
        """Impact financier"""
        print("\n" + "="*80)
        print("💰 IMPACT FINANCIER")
        print("="*80)

        df = self.df_scenario

        # SANS règle
        tp_no = ((df['y_true'] == 1) & (df['y_pred'] == 1)).sum()
        tn_no = ((df['y_true'] == 0) & (df['y_pred'] == 0)).sum()
        fp_no = ((df['y_true'] == 0) & (df['y_pred'] == 1)).sum()
        fn_no = ((df['y_true'] == 1) & (df['y_pred'] == 0)).sum()

        auto_no = tp_no + tn_no
        gain_no = auto_no * PRIX_UNITAIRE_DH - fp_no * PRIX_UNITAIRE_DH - fn_no * 2 * PRIX_UNITAIRE_DH

        # AVEC règle
        tp_with = ((df['y_true'] == 1) & (df['y_pred_with_rule'] == 1)).sum()
        tn_with = ((df['y_true'] == 0) & (df['y_pred_with_rule'] == 0)).sum()
        fp_with = ((df['y_true'] == 0) & (df['y_pred_with_rule'] == 1)).sum()
        fn_with = ((df['y_true'] == 1) & (df['y_pred_with_rule'] == 0)).sum()

        auto_with = tp_with + tn_with
        gain_with = auto_with * PRIX_UNITAIRE_DH - fp_with * PRIX_UNITAIRE_DH - fn_with * 2 * PRIX_UNITAIRE_DH

        print(f"\n📊 SANS règle:")
        print(f"   Auto: {auto_no}/{len(df)} ({100*auto_no/len(df):.1f}%)")
        print(f"   Gain net: {gain_no:,.0f} DH")
        print(f"   FP: {fp_no}, FN: {fn_no}")

        print(f"\n📊 AVEC règle:")
        print(f"   Auto: {auto_with}/{len(df)} ({100*auto_with/len(df):.1f}%)")
        print(f"   Gain net: {gain_with:,.0f} DH")
        print(f"   FP: {fp_with}, FN: {fn_with}")

        print(f"\n💡 Différence: {gain_with - gain_no:+,.0f} DH")

        self.impact = {
            'sans_regle': {'auto': auto_no, 'taux_auto': 100*auto_no/len(df),
                          'gain_net': gain_no, 'fp': fp_no, 'fn': fn_no},
            'avec_regle': {'auto': auto_with, 'taux_auto': 100*auto_with/len(df),
                          'gain_net': gain_with, 'fp': fp_with, 'fn': fn_with}
        }

    def generate_visualizations(self):
        """Générer visualisations"""
        print("\n" + "="*80)
        print("📊 VISUALISATIONS")
        print("="*80)

        self._plot_comparison_2024_2025()
        self._plot_business_rule_impact()
        self._plot_financial_impact()

        print("✅ Visualisations générées")

    def _plot_comparison_2024_2025(self):
        """Comparaison 2024 vs 2025"""
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle('📊 Performance 2024 vs 2025 (Modèle Optimisé)',
                     fontsize=16, fontweight='bold')

        # Métriques
        ax = axes[0]
        metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        values_2024 = [self.metrics_2024[m] for m in metrics]
        values_2025 = [self.metrics_2025[m] for m in metrics]

        x = np.arange(len(metrics))
        width = 0.35

        ax.bar(x - width/2, values_2024, width, label='2024', color='#3498db', alpha=0.8)
        ax.bar(x + width/2, values_2025, width, label='2025', color='#e74c3c', alpha=0.8)

        ax.set_ylabel('Score', fontweight='bold')
        ax.set_title('Métriques', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([m.upper() for m in metrics], rotation=45)
        ax.legend()
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3, axis='y')

        # Dégradation
        ax = axes[1]
        degradations = [((self.metrics_2025[m] - self.metrics_2024[m]) /
                        self.metrics_2024[m]) * 100 for m in metrics]
        colors = ['#2ecc71' if d >= 0 else '#e74c3c' for d in degradations]

        bars = ax.barh(metrics, degradations, color=colors, alpha=0.7)
        ax.set_xlabel('Variation (%)', fontweight='bold')
        ax.set_title('Dégradation', fontweight='bold')
        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
        ax.grid(True, alpha=0.3, axis='x')

        for bar, val in zip(bars, degradations):
            ax.text(val + (0.5 if val > 0 else -0.5), bar.get_y() + bar.get_height()/2,
                   f'{val:+.1f}%', va='center', fontweight='bold')

        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/figures/comparison_2024_2025.png', dpi=300, bbox_inches='tight')
        print(f"   ✅ comparison_2024_2025.png")
        plt.close()

    def _plot_business_rule_impact(self):
        """Impact règle métier"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('🔒 Impact Règle Métier : 1 Réclamation/Client',
                     fontsize=16, fontweight='bold')

        df = self.df_scenario

        # Distribution réclamations
        ax = axes[0, 0]
        client_col = [c for c in df.columns if any(x in c.lower() for x in ['idtfcl', 'client', 'compte'])][0]
        recl_per_client = df[client_col].value_counts()
        dist = recl_per_client.value_counts().sort_index().head(10)

        ax.bar(dist.index, dist.values, color='#3498db', alpha=0.7)
        ax.set_xlabel('Réclamations par client', fontweight='bold')
        ax.set_ylabel('Nombre de clients', fontweight='bold')
        ax.set_title('Distribution', fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

        # Taux auto
        ax = axes[0, 1]
        categories = ['SANS règle', 'AVEC règle']
        taux = [self.impact['sans_regle']['taux_auto'], self.impact['avec_regle']['taux_auto']]
        colors_bar = ['#e74c3c', '#2ecc71']

        bars = ax.bar(categories, taux, color=colors_bar, alpha=0.7)
        ax.set_ylabel('Taux Auto (%)', fontweight='bold')
        ax.set_title('Automatisation', fontweight='bold')
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.3, axis='y')

        for bar, val in zip(bars, taux):
            ax.text(bar.get_x() + bar.get_width()/2, val + 2,
                   f'{val:.1f}%', ha='center', fontweight='bold')

        # Nombre auto
        ax = axes[1, 0]
        nb_auto = [self.impact['sans_regle']['auto'], self.impact['avec_regle']['auto']]

        bars = ax.bar(categories, nb_auto, color=colors_bar, alpha=0.7)
        ax.set_ylabel('Nombre', fontweight='bold')
        ax.set_title('Réclamations Automatisées', fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

        for bar, val in zip(bars, nb_auto):
            ax.text(bar.get_x() + bar.get_width()/2, val + 5,
                   f'{int(val)}', ha='center', fontweight='bold')

        # Pie chart
        ax = axes[1, 1]
        first = df['is_first_reclamation'].sum()
        multi = len(df) - first

        sizes = [first, multi]
        labels = [f'1ère réclamation\n({first})', f'Multiples\n({multi})']
        colors_pie = ['#2ecc71', '#e74c3c']

        wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors_pie,
                                          autopct='%1.1f%%', startangle=90, explode=(0.05, 0.05))
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')

        ax.set_title('Répartition', fontweight='bold')

        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/figures/business_rule_impact.png', dpi=300, bbox_inches='tight')
        print(f"   ✅ business_rule_impact.png")
        plt.close()

    def _plot_financial_impact(self):
        """Impact financier"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('💰 Impact Financier', fontsize=16, fontweight='bold')

        # Gain net
        ax = axes[0, 0]
        categories = ['SANS règle', 'AVEC règle']
        gains = [self.impact['sans_regle']['gain_net'], self.impact['avec_regle']['gain_net']]
        colors_bar = ['#2ecc71' if g > 0 else '#e74c3c' for g in gains]

        bars = ax.bar(categories, gains, color=colors_bar, alpha=0.7)
        ax.set_ylabel('Gain Net (DH)', fontweight='bold')
        ax.set_title('Gain Net', fontweight='bold')
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        ax.grid(True, alpha=0.3, axis='y')

        for bar, val in zip(bars, gains):
            ax.text(bar.get_x() + bar.get_width()/2, val + (1000 if val > 0 else -1000),
                   f'{val:,.0f} DH', ha='center', fontweight='bold')

        # FP/FN
        ax = axes[0, 1]
        x = np.arange(2)
        width = 0.35

        fp_vals = [self.impact['sans_regle']['fp'], self.impact['avec_regle']['fp']]
        fn_vals = [self.impact['sans_regle']['fn'], self.impact['avec_regle']['fn']]

        ax.bar(x - width/2, fp_vals, width, label='FP', color='#e74c3c', alpha=0.7)
        ax.bar(x + width/2, fn_vals, width, label='FN', color='#e67e22', alpha=0.7)

        ax.set_ylabel('Nombre', fontweight='bold')
        ax.set_title('Erreurs', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(categories)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        # Décomposition SANS
        ax = axes[1, 0]
        auto = self.impact['sans_regle']['auto']
        fp = self.impact['sans_regle']['fp']
        fn = self.impact['sans_regle']['fn']

        gain_brut = auto * PRIX_UNITAIRE_DH
        cout_fp = fp * PRIX_UNITAIRE_DH
        cout_fn = fn * 2 * PRIX_UNITAIRE_DH
        gain_net = gain_brut - cout_fp - cout_fn

        comps = ['Gain brut', 'Coût FP', 'Coût FN', 'Gain NET']
        vals = [gain_brut, -cout_fp, -cout_fn, gain_net]
        colors_comp = ['#2ecc71', '#e74c3c', '#e67e22',
                      '#2ecc71' if gain_net > 0 else '#e74c3c']

        bars = ax.bar(comps, vals, color=colors_comp, alpha=0.7)
        ax.set_ylabel('Montant (DH)', fontweight='bold')
        ax.set_title('SANS Règle', fontweight='bold')
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        ax.grid(True, alpha=0.3, axis='y')

        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2,
                   val + (500 if val > 0 else -500),
                   f'{val:,.0f}', ha='center', fontsize=9, fontweight='bold')

        # Décomposition AVEC
        ax = axes[1, 1]
        auto = self.impact['avec_regle']['auto']
        fp = self.impact['avec_regle']['fp']
        fn = self.impact['avec_regle']['fn']

        gain_brut = auto * PRIX_UNITAIRE_DH
        cout_fp = fp * PRIX_UNITAIRE_DH
        cout_fn = fn * 2 * PRIX_UNITAIRE_DH
        gain_net = gain_brut - cout_fp - cout_fn

        vals = [gain_brut, -cout_fp, -cout_fn, gain_net]
        colors_comp = ['#2ecc71', '#e74c3c', '#e67e22',
                      '#2ecc71' if gain_net > 0 else '#e74c3c']

        bars = ax.bar(comps, vals, color=colors_comp, alpha=0.7)
        ax.set_ylabel('Montant (DH)', fontweight='bold')
        ax.set_title('AVEC Règle', fontweight='bold')
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        ax.grid(True, alpha=0.3, axis='y')

        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2,
                   val + (500 if val > 0 else -500),
                   f'{val:,.0f}', ha='center', fontsize=9, fontweight='bold')

        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/figures/financial_impact.png', dpi=300, bbox_inches='tight')
        print(f"   ✅ financial_impact.png")
        plt.close()

    def generate_report(self):
        """Rapport"""
        lines = []
        lines.append("="*80)
        lines.append("RAPPORT PRODUCTION - MODÈLE OPTIMISÉ OPTUNA")
        lines.append("="*80)
        lines.append(f"\nDate: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        lines.append("\n" + "="*80)
        lines.append("1. OPTIMISATION MODÈLE")
        lines.append("="*80)
        lines.append("\nHyperparamètres optimisés (100 trials Optuna):")
        for key, value in self.best_params.items():
            lines.append(f"  {key:20s}: {value}")

        lines.append("\n" + "="*80)
        lines.append("2. PERFORMANCE")
        lines.append("="*80)
        lines.append("\n2024 (entraînement):")
        for metric, value in self.metrics_2024.items():
            lines.append(f"  {metric:12s}: {value:.4f}")

        lines.append("\nValidation croisée (5-fold):")
        for metric, scores in self.cv_scores.items():
            lines.append(f"  {metric:12s}: {scores.mean():.4f} (+/- {scores.std()*2:.4f})")

        lines.append("\n2025 (test):")
        for metric, value in self.metrics_2025.items():
            lines.append(f"  {metric:12s}: {value:.4f}")

        lines.append("\nDégradation:")
        for metric in self.metrics_2024.keys():
            deg = ((self.metrics_2025[metric] - self.metrics_2024[metric]) /
                   self.metrics_2024[metric]) * 100
            lines.append(f"  {metric:12s}: {deg:+.2f}%")

        lines.append("\n" + "="*80)
        lines.append("3. IMPACT FINANCIER")
        lines.append("="*80)
        lines.append(f"\nSANS règle: {self.impact['sans_regle']['gain_net']:,.0f} DH")
        lines.append(f"AVEC règle: {self.impact['avec_regle']['gain_net']:,.0f} DH")

        with open(f'{self.output_dir}/rapport_production.txt', 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))

        print(f"   ✅ rapport_production.txt")

    def run(self):
        """Pipeline complet"""
        print("\n" + "="*80)
        print("🚀 PIPELINE PRODUCTION - MODÈLE OPTIMISÉ")
        print("="*80)

        self.load_data('data/raw/reclamations_2024.xlsx', 'data/raw/reclamations_2025.xlsx')
        self.train_model()
        self.evaluate_2025()
        self.apply_business_rule()
        self.calculate_financial_impact()
        self.generate_visualizations()
        self.generate_report()

        print("\n" + "="*80)
        print("✅ TERMINÉ")
        print("="*80)
        print(f"\n📂 Résultats: {self.output_dir}/")


def main():
    pipeline = ProductionPipeline(output_dir='outputs/production')
    pipeline.run()


if __name__ == '__main__':
    main()
