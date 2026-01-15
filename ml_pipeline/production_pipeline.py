"""
PIPELINE ML PRODUCTION - CLASSIFICATION RÉCLAMATIONS BANCAIRES
Version Production Simplifiée avec Règle Métier : 1 réclamation auto par client
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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Configuration
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (14, 8)

PRIX_UNITAIRE_DH = 169  # Coût traitement manuel


class ProductionPreprocessor:
    """Preprocessing simplifié pour production"""

    def __init__(self):
        self.scaler = RobustScaler()
        self.family_medians = {}  # Médianes par famille (calculées sur 2024)
        self.features_to_use = [
            'Marché', 'Segment', 'Famille Produit', 'Catégorie', 'Sous-catégorie',
            'Montant demandé', 'PNB analytique (vision commerciale) cumulé',
            'anciennete_annees'
        ]
        self.categorical_encodings = {}

    def fit(self, df):
        """Fit sur données 2024"""
        print("\n🔧 Configuration du preprocessing...")

        X = df.copy()

        # Calculer médianes par famille (sur 2024 uniquement)
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
                # Encodage simple : mapping vers entiers
                unique_vals = X[col].unique()
                self.categorical_encodings[col] = {val: idx for idx, val in enumerate(unique_vals)}

        # Appliquer encodage
        X = self._encode_categorical(X)

        # Sélectionner colonnes numériques
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [c for c in numeric_cols if c != 'Fondee']

        # Fit scaler
        print("📏 Fit scaler...")
        self.scaler.fit(X[numeric_cols])

        print(f"✅ Preprocessing configuré: {len(numeric_cols)} features")
        return self

    def transform(self, df):
        """Transform (peut être appliqué sur 2024 ou 2025)"""
        X = df.copy()

        # Features engineering (utilise médianes de 2024)
        X = self._create_features(X, fit_mode=False)

        # Encoder catégorielles (utilise mapping de 2024)
        X = self._encode_categorical(X)

        # Sélectionner colonnes numériques
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [c for c in numeric_cols if c != 'Fondee']

        # Scaler
        X[numeric_cols] = self.scaler.transform(X[numeric_cols])

        # Supprimer colonnes non numériques
        X = X[numeric_cols]

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
        # Utilise toujours les médianes calculées sur 2024
        df['ecart_mediane_famille'] = df.apply(
            lambda row: (
                row['Montant demandé'] -
                self.family_medians.get(row['Famille Produit'], row['Montant demandé'])
            ) / (self.family_medians.get(row['Famille Produit'], 1) + 1),
            axis=1
        )

        # 3. Log montant (pour normaliser)
        df['log_montant'] = np.log1p(df['Montant demandé'])

        # 4. Log PNB
        df['log_pnb'] = np.log1p(df['PNB analytique (vision commerciale) cumulé'])

        # Nettoyer NaN/inf
        df = df.replace([np.inf, -np.inf], np.nan)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col != 'Fondee':
                df[col] = df[col].fillna(df[col].median())

        return df

    def _encode_categorical(self, X):
        """Encoder catégorielles avec mapping appris"""
        df = X.copy()

        for col, mapping in self.categorical_encodings.items():
            if col in df.columns:
                # Utiliser -1 pour valeurs inconnues
                df[col] = df[col].map(mapping).fillna(-1)

        return df


class ProductionPipeline:
    """Pipeline production avec règle métier : 1 réclamation auto par client"""

    def __init__(self, output_dir='outputs/production'):
        self.output_dir = output_dir
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        Path(f"{output_dir}/figures").mkdir(parents=True, exist_ok=True)
        Path(f"{output_dir}/models").mkdir(parents=True, exist_ok=True)

        self.preprocessor = ProductionPreprocessor()
        self.model = None
        self.df_2024 = None
        self.df_2025 = None

    def load_data(self, path_2024, path_2025):
        """Charger données 2024 et 2025"""
        print("\n" + "="*80)
        print("📂 CHARGEMENT DES DONNÉES")
        print("="*80)

        self.df_2024 = pd.read_excel(path_2024)
        self.df_2025 = pd.read_excel(path_2025)

        print(f"✅ 2024: {len(self.df_2024)} réclamations")
        print(f"✅ 2025: {len(self.df_2025)} réclamations")

        # Vérifier colonnes requises
        required_cols = [
            'Marché', 'Segment', 'Famille Produit', 'Catégorie', 'Sous-catégorie',
            'Montant demandé', 'PNB analytique (vision commerciale) cumulé',
            'anciennete_annees', 'Fondee', 'Date de Qualification'
        ]

        for col in required_cols:
            if col not in self.df_2024.columns:
                print(f"⚠️  Colonne manquante dans 2024: {col}")
            if col not in self.df_2025.columns:
                print(f"⚠️  Colonne manquante dans 2025: {col}")

    def train_model(self):
        """Entraîner modèle sur 2024"""
        print("\n" + "="*80)
        print("🎯 ENTRAÎNEMENT MODÈLE (2024)")
        print("="*80)

        # Preprocessing
        X_train = self.preprocessor.fit_transform(self.df_2024)
        y_train = self.df_2024['Fondee'].values

        print(f"\n📊 Shape: {X_train.shape}")
        print(f"   Features: {X_train.shape[1]}")
        print(f"   Samples: {X_train.shape[0]}")

        # Cross-validation
        print("\n🔄 Cross-validation (5-fold)...")
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        self.model = xgb.XGBClassifier(
            max_depth=6,
            learning_rate=0.1,
            n_estimators=100,
            random_state=42,
            eval_metric='logloss'
        )

        cv_scores = cross_val_score(self.model, X_train, y_train, cv=cv, scoring='f1')
        print(f"   F1-Score CV: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

        # Entraîner sur toutes les données
        print("\n🏋️  Entraînement sur données complètes...")
        self.model.fit(X_train, y_train)

        # Métriques 2024
        y_pred_2024 = self.model.predict(X_train)

        metrics_2024 = {
            'accuracy': accuracy_score(y_train, y_pred_2024),
            'precision': precision_score(y_train, y_pred_2024),
            'recall': recall_score(y_train, y_pred_2024),
            'f1': f1_score(y_train, y_pred_2024)
        }

        print("\n📊 Métriques 2024 (entraînement):")
        for metric, value in metrics_2024.items():
            print(f"   {metric:12s}: {value:.4f}")

        self.metrics_2024 = metrics_2024

        # Sauvegarder
        joblib.dump(self.model, f'{self.output_dir}/models/model_production.pkl')
        joblib.dump(self.preprocessor, f'{self.output_dir}/models/preprocessor_production.pkl')
        print(f"\n✅ Modèle sauvegardé: {self.output_dir}/models/")

    def evaluate_2025(self):
        """Évaluer sur 2025"""
        print("\n" + "="*80)
        print("📊 ÉVALUATION SUR 2025")
        print("="*80)

        # Transform 2025
        X_test = self.preprocessor.transform(self.df_2025)
        y_test = self.df_2025['Fondee'].values

        # Prédictions
        y_pred_2025 = self.model.predict(X_test)
        y_prob_2025 = self.model.predict_proba(X_test)[:, 1]

        # Métriques
        metrics_2025 = {
            'accuracy': accuracy_score(y_test, y_pred_2025),
            'precision': precision_score(y_test, y_pred_2025),
            'recall': recall_score(y_test, y_pred_2025),
            'f1': f1_score(y_test, y_pred_2025)
        }

        print("\n📊 Métriques 2025:")
        for metric, value in metrics_2025.items():
            print(f"   {metric:12s}: {value:.4f}")

        # Comparaison
        print("\n📉 Dégradation 2024 → 2025:")
        for metric in metrics_2024.keys():
            degradation = ((metrics_2025[metric] - self.metrics_2024[metric]) /
                          self.metrics_2024[metric]) * 100
            print(f"   {metric:12s}: {degradation:+.2f}%")

        self.metrics_2025 = metrics_2025
        self.y_pred_2025 = y_pred_2025
        self.y_prob_2025 = y_prob_2025

    def apply_business_rule(self):
        """
        RÈGLE MÉTIER : 1 seule réclamation automatisée par client

        Logique :
        1. Trier réclamations 2025 par date de qualification
        2. Pour chaque client, identifier sa première réclamation
        3. Seule la première réclamation peut être automatisée
        4. Les suivantes doivent être traitées manuellement
        """
        print("\n" + "="*80)
        print("🔒 APPLICATION RÈGLE MÉTIER : 1 RÉCLAMATION AUTO PAR CLIENT")
        print("="*80)

        # Préparer données
        df_scenario = self.df_2025.copy()
        df_scenario['y_pred'] = self.y_pred_2025
        df_scenario['y_prob'] = self.y_prob_2025
        df_scenario['y_true'] = self.df_2025['Fondee'].values

        # Convertir date
        df_scenario['Date de Qualification'] = pd.to_datetime(
            df_scenario['Date de Qualification'],
            errors='coerce'
        )

        # Identifier colonne client (plusieurs possibilités)
        client_col = None
        for col in ['idtfcl', 'N compte', 'numero_compte', 'ID Client']:
            if col in df_scenario.columns:
                client_col = col
                break

        if client_col is None:
            print("⚠️  Colonne client non trouvée, utilisation de l'index")
            df_scenario['client_id'] = df_scenario.index
            client_col = 'client_id'

        print(f"📋 Colonne client utilisée: {client_col}")

        # Trier par client puis par date
        df_scenario = df_scenario.sort_values([client_col, 'Date de Qualification'])

        # Marquer première réclamation par client
        df_scenario['is_first_reclamation'] = ~df_scenario.duplicated(subset=[client_col], keep='first')

        # Statistiques
        total_clients = df_scenario[client_col].nunique()
        total_reclamations = len(df_scenario)
        first_reclamations = df_scenario['is_first_reclamation'].sum()
        multi_reclamations = total_reclamations - first_reclamations

        print(f"\n📊 Statistiques clients:")
        print(f"   Total clients: {total_clients}")
        print(f"   Total réclamations: {total_reclamations}")
        print(f"   Premières réclamations: {first_reclamations}")
        print(f"   Réclamations multiples: {multi_reclamations}")
        print(f"   Taux réclamations multiples: {100*multi_reclamations/total_reclamations:.1f}%")

        # Appliquer règle : seules les premières réclamations peuvent être automatisées
        df_scenario['can_automate'] = df_scenario['is_first_reclamation']

        # Prédiction finale avec règle métier
        df_scenario['y_pred_with_rule'] = np.where(
            df_scenario['can_automate'],
            df_scenario['y_pred'],  # Utiliser prédiction modèle
            0  # Forcer traitement manuel pour réclamations suivantes
        )

        # Impact de la règle
        automated_without_rule = df_scenario['y_pred'].sum()
        automated_with_rule = df_scenario['y_pred_with_rule'].sum()
        blocked_by_rule = automated_without_rule - automated_with_rule

        print(f"\n🚦 Impact règle métier:")
        print(f"   Automatisées SANS règle: {automated_without_rule}")
        print(f"   Automatisées AVEC règle: {automated_with_rule}")
        print(f"   Bloquées par règle: {blocked_by_rule}")
        print(f"   Réduction: {100*blocked_by_rule/automated_without_rule:.1f}%")

        self.df_scenario = df_scenario

    def calculate_financial_impact(self):
        """Calculer impact financier avec règle métier"""
        print("\n" + "="*80)
        print("💰 CALCUL IMPACT FINANCIER")
        print("="*80)

        df = self.df_scenario

        # Scénario SANS règle métier
        tp_no_rule = ((df['y_true'] == 1) & (df['y_pred'] == 1)).sum()
        tn_no_rule = ((df['y_true'] == 0) & (df['y_pred'] == 0)).sum()
        fp_no_rule = ((df['y_true'] == 0) & (df['y_pred'] == 1)).sum()
        fn_no_rule = ((df['y_true'] == 1) & (df['y_pred'] == 0)).sum()

        auto_no_rule = tp_no_rule + tn_no_rule
        gain_no_rule = auto_no_rule * PRIX_UNITAIRE_DH
        cout_fp_no_rule = fp_no_rule * PRIX_UNITAIRE_DH
        cout_fn_no_rule = fn_no_rule * 2 * PRIX_UNITAIRE_DH
        gain_net_no_rule = gain_no_rule - cout_fp_no_rule - cout_fn_no_rule

        # Scénario AVEC règle métier
        tp_with_rule = ((df['y_true'] == 1) & (df['y_pred_with_rule'] == 1)).sum()
        tn_with_rule = ((df['y_true'] == 0) & (df['y_pred_with_rule'] == 0)).sum()
        fp_with_rule = ((df['y_true'] == 0) & (df['y_pred_with_rule'] == 1)).sum()
        fn_with_rule = ((df['y_true'] == 1) & (df['y_pred_with_rule'] == 0)).sum()

        auto_with_rule = tp_with_rule + tn_with_rule
        gain_with_rule = auto_with_rule * PRIX_UNITAIRE_DH
        cout_fp_with_rule = fp_with_rule * PRIX_UNITAIRE_DH
        cout_fn_with_rule = fn_with_rule * 2 * PRIX_UNITAIRE_DH
        gain_net_with_rule = gain_with_rule - cout_fp_with_rule - cout_fn_with_rule

        # Affichage
        print("\n📊 SANS règle métier:")
        print(f"   Automatisées: {auto_no_rule} / {len(df)} ({100*auto_no_rule/len(df):.1f}%)")
        print(f"   Gain brut: {gain_no_rule:,.0f} DH")
        print(f"   Coût FP: {cout_fp_no_rule:,.0f} DH")
        print(f"   Coût FN: {cout_fn_no_rule:,.0f} DH")
        print(f"   GAIN NET: {gain_net_no_rule:,.0f} DH")

        print("\n📊 AVEC règle métier (1 réclamation/client):")
        print(f"   Automatisées: {auto_with_rule} / {len(df)} ({100*auto_with_rule/len(df):.1f}%)")
        print(f"   Gain brut: {gain_with_rule:,.0f} DH")
        print(f"   Coût FP: {cout_fp_with_rule:,.0f} DH")
        print(f"   Coût FN: {cout_fn_with_rule:,.0f} DH")
        print(f"   GAIN NET: {gain_net_with_rule:,.0f} DH")

        print(f"\n💡 Impact règle métier:")
        diff_gain_net = gain_net_with_rule - gain_net_no_rule
        print(f"   Différence gain net: {diff_gain_net:+,.0f} DH")

        if diff_gain_net > 0:
            print(f"   ✅ La règle métier AMÉLIORE le gain net")
        else:
            print(f"   ⚠️  La règle métier RÉDUIT le gain net (mais sécurise la relation client)")

        self.impact = {
            'sans_regle': {
                'auto': auto_no_rule,
                'taux_auto': 100*auto_no_rule/len(df),
                'gain_net': gain_net_no_rule,
                'fp': fp_no_rule,
                'fn': fn_no_rule
            },
            'avec_regle': {
                'auto': auto_with_rule,
                'taux_auto': 100*auto_with_rule/len(df),
                'gain_net': gain_net_with_rule,
                'fp': fp_with_rule,
                'fn': fn_with_rule
            }
        }

    def generate_visualizations(self):
        """Générer visualisations"""
        print("\n" + "="*80)
        print("📊 GÉNÉRATION VISUALISATIONS")
        print("="*80)

        # 1. Comparaison 2024 vs 2025
        self._plot_comparison_2024_2025()

        # 2. Impact règle métier
        self._plot_business_rule_impact()

        # 3. Impact financier
        self._plot_financial_impact()

        print("\n✅ Visualisations générées")

    def _plot_comparison_2024_2025(self):
        """Comparaison performance 2024 vs 2025"""
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle('📊 Comparaison Performance 2024 vs 2025',
                     fontsize=16, fontweight='bold')

        # Graphique 1: Métriques
        ax = axes[0]
        metrics = ['accuracy', 'precision', 'recall', 'f1']
        values_2024 = [self.metrics_2024[m] for m in metrics]
        values_2025 = [self.metrics_2025[m] for m in metrics]

        x = np.arange(len(metrics))
        width = 0.35

        ax.bar(x - width/2, values_2024, width, label='2024', color='#3498db', alpha=0.8)
        ax.bar(x + width/2, values_2025, width, label='2025', color='#e74c3c', alpha=0.8)

        ax.set_ylabel('Score', fontweight='bold')
        ax.set_title('Métriques de Performance', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([m.capitalize() for m in metrics])
        ax.legend()
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3, axis='y')

        # Ajouter valeurs
        for i, (v24, v25) in enumerate(zip(values_2024, values_2025)):
            ax.text(i - width/2, v24 + 0.02, f'{v24:.3f}',
                   ha='center', fontsize=9, fontweight='bold')
            ax.text(i + width/2, v25 + 0.02, f'{v25:.3f}',
                   ha='center', fontsize=9, fontweight='bold')

        # Graphique 2: Dégradation
        ax = axes[1]
        degradations = [
            ((self.metrics_2025[m] - self.metrics_2024[m]) / self.metrics_2024[m]) * 100
            for m in metrics
        ]

        colors = ['#2ecc71' if d >= 0 else '#e74c3c' for d in degradations]
        bars = ax.barh(metrics, degradations, color=colors, alpha=0.7)

        ax.set_xlabel('Variation (%)', fontweight='bold')
        ax.set_title('Dégradation 2024 → 2025', fontweight='bold')
        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
        ax.grid(True, alpha=0.3, axis='x')

        # Ajouter valeurs
        for i, (bar, val) in enumerate(zip(bars, degradations)):
            ax.text(val + (1 if val > 0 else -1), i, f'{val:+.1f}%',
                   va='center', fontweight='bold', fontsize=10)

        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/figures/comparison_2024_2025.png',
                   dpi=300, bbox_inches='tight')
        print(f"   ✅ comparison_2024_2025.png")
        plt.close()

    def _plot_business_rule_impact(self):
        """Impact de la règle métier"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('🔒 Impact Règle Métier : 1 Réclamation Automatisée par Client',
                     fontsize=16, fontweight='bold')

        df = self.df_scenario

        # Graphique 1: Distribution réclamations par client
        ax = axes[0, 0]
        client_col = [c for c in df.columns if 'idtfcl' in c.lower() or
                     'client' in c.lower() or 'compte' in c.lower()][0]
        reclamations_per_client = df[client_col].value_counts()

        distribution = reclamations_per_client.value_counts().sort_index()
        ax.bar(distribution.index, distribution.values, color='#3498db', alpha=0.7)
        ax.set_xlabel('Nombre de réclamations par client', fontweight='bold')
        ax.set_ylabel('Nombre de clients', fontweight='bold')
        ax.set_title('Distribution Réclamations par Client', fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

        # Graphique 2: Taux automatisation
        ax = axes[0, 1]
        categories = ['SANS règle', 'AVEC règle']
        taux = [
            self.impact['sans_regle']['taux_auto'],
            self.impact['avec_regle']['taux_auto']
        ]
        colors_bar = ['#e74c3c', '#2ecc71']

        bars = ax.bar(categories, taux, color=colors_bar, alpha=0.7)
        ax.set_ylabel('Taux Automatisation (%)', fontweight='bold')
        ax.set_title('Taux d\'Automatisation', fontweight='bold')
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.3, axis='y')

        for bar, val in zip(bars, taux):
            ax.text(bar.get_x() + bar.get_width()/2, val + 2,
                   f'{val:.1f}%', ha='center', fontweight='bold', fontsize=12)

        # Graphique 3: Nombre automatisées
        ax = axes[1, 0]
        nb_auto = [
            self.impact['sans_regle']['auto'],
            self.impact['avec_regle']['auto']
        ]

        bars = ax.bar(categories, nb_auto, color=colors_bar, alpha=0.7)
        ax.set_ylabel('Nombre de Réclamations', fontweight='bold')
        ax.set_title('Réclamations Automatisées', fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

        for bar, val in zip(bars, nb_auto):
            ax.text(bar.get_x() + bar.get_width()/2, val + 5,
                   f'{int(val)}', ha='center', fontweight='bold', fontsize=12)

        # Graphique 4: Premières vs Multiples
        ax = axes[1, 1]
        first_count = df['is_first_reclamation'].sum()
        multi_count = len(df) - first_count

        sizes = [first_count, multi_count]
        labels = [f'1ère réclamation\n({first_count})',
                 f'Réclamations multiples\n({multi_count})']
        colors_pie = ['#2ecc71', '#e74c3c']
        explode = (0.05, 0.05)

        wedges, texts, autotexts = ax.pie(sizes, explode=explode, labels=labels,
                                          colors=colors_pie, autopct='%1.1f%%',
                                          shadow=True, startangle=90)
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
            autotext.set_fontsize(12)

        ax.set_title('Répartition Réclamations', fontweight='bold')

        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/figures/business_rule_impact.png',
                   dpi=300, bbox_inches='tight')
        print(f"   ✅ business_rule_impact.png")
        plt.close()

    def _plot_financial_impact(self):
        """Impact financier"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('💰 Impact Financier - Avec vs Sans Règle Métier',
                     fontsize=16, fontweight='bold')

        # Graphique 1: Gain net
        ax = axes[0, 0]
        categories = ['SANS règle', 'AVEC règle']
        gains = [
            self.impact['sans_regle']['gain_net'],
            self.impact['avec_regle']['gain_net']
        ]
        colors_bar = ['#e74c3c' if g < 0 else '#2ecc71' for g in gains]

        bars = ax.bar(categories, gains, color=colors_bar, alpha=0.7)
        ax.set_ylabel('Gain Net (DH)', fontweight='bold')
        ax.set_title('Gain Net Total', fontweight='bold')
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        ax.grid(True, alpha=0.3, axis='y')

        for bar, val in zip(bars, gains):
            ax.text(bar.get_x() + bar.get_width()/2, val + (1000 if val > 0 else -1000),
                   f'{val:,.0f} DH', ha='center', fontweight='bold', fontsize=11)

        # Graphique 2: FP et FN
        ax = axes[0, 1]
        x = np.arange(2)
        width = 0.35

        fp_values = [self.impact['sans_regle']['fp'], self.impact['avec_regle']['fp']]
        fn_values = [self.impact['sans_regle']['fn'], self.impact['avec_regle']['fn']]

        ax.bar(x - width/2, fp_values, width, label='Faux Positifs',
              color='#e74c3c', alpha=0.7)
        ax.bar(x + width/2, fn_values, width, label='Faux Négatifs',
              color='#e67e22', alpha=0.7)

        ax.set_ylabel('Nombre d\'Erreurs', fontweight='bold')
        ax.set_title('Erreurs : FP vs FN', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(categories)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        # Graphique 3: Décomposition gain (SANS règle)
        ax = axes[1, 0]
        auto = self.impact['sans_regle']['auto']
        fp = self.impact['sans_regle']['fp']
        fn = self.impact['sans_regle']['fn']

        gain_brut = auto * PRIX_UNITAIRE_DH
        cout_fp = fp * PRIX_UNITAIRE_DH
        cout_fn = fn * 2 * PRIX_UNITAIRE_DH
        gain_net = gain_brut - cout_fp - cout_fn

        components = ['Gain brut', 'Coût FP', 'Coût FN', 'Gain NET']
        values = [gain_brut, -cout_fp, -cout_fn, gain_net]
        colors_comp = ['#2ecc71', '#e74c3c', '#e67e22',
                      '#2ecc71' if gain_net > 0 else '#e74c3c']

        bars = ax.bar(components, values, color=colors_comp, alpha=0.7)
        ax.set_ylabel('Montant (DH)', fontweight='bold')
        ax.set_title('Décomposition Financière SANS Règle', fontweight='bold')
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        ax.grid(True, alpha=0.3, axis='y')

        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2,
                   val + (500 if val > 0 else -500),
                   f'{val:,.0f}', ha='center', fontsize=9, fontweight='bold')

        # Graphique 4: Décomposition gain (AVEC règle)
        ax = axes[1, 1]
        auto = self.impact['avec_regle']['auto']
        fp = self.impact['avec_regle']['fp']
        fn = self.impact['avec_regle']['fn']

        gain_brut = auto * PRIX_UNITAIRE_DH
        cout_fp = fp * PRIX_UNITAIRE_DH
        cout_fn = fn * 2 * PRIX_UNITAIRE_DH
        gain_net = gain_brut - cout_fp - cout_fn

        values = [gain_brut, -cout_fp, -cout_fn, gain_net]
        colors_comp = ['#2ecc71', '#e74c3c', '#e67e22',
                      '#2ecc71' if gain_net > 0 else '#e74c3c']

        bars = ax.bar(components, values, color=colors_comp, alpha=0.7)
        ax.set_ylabel('Montant (DH)', fontweight='bold')
        ax.set_title('Décomposition Financière AVEC Règle', fontweight='bold')
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        ax.grid(True, alpha=0.3, axis='y')

        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2,
                   val + (500 if val > 0 else -500),
                   f'{val:,.0f}', ha='center', fontsize=9, fontweight='bold')

        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/figures/financial_impact.png',
                   dpi=300, bbox_inches='tight')
        print(f"   ✅ financial_impact.png")
        plt.close()

    def generate_report(self):
        """Générer rapport texte"""
        print("\n📄 Génération rapport...")

        lines = []
        lines.append("="*80)
        lines.append("RAPPORT PRODUCTION - CLASSIFICATION RÉCLAMATIONS BANCAIRES")
        lines.append("="*80)
        lines.append(f"\nDate: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"Prix unitaire traitement: {PRIX_UNITAIRE_DH} DH")

        lines.append("\n" + "="*80)
        lines.append("1. DONNÉES")
        lines.append("="*80)
        lines.append(f"2024 (entraînement): {len(self.df_2024)} réclamations")
        lines.append(f"2025 (test): {len(self.df_2025)} réclamations")

        lines.append("\n" + "="*80)
        lines.append("2. PERFORMANCE MODÈLE")
        lines.append("="*80)
        lines.append("\n2024 (entraînement):")
        for metric, value in self.metrics_2024.items():
            lines.append(f"  {metric:12s}: {value:.4f}")

        lines.append("\n2025 (test):")
        for metric, value in self.metrics_2025.items():
            lines.append(f"  {metric:12s}: {value:.4f}")

        lines.append("\nDégradation 2024 → 2025:")
        for metric in self.metrics_2024.keys():
            degradation = ((self.metrics_2025[metric] - self.metrics_2024[metric]) /
                          self.metrics_2024[metric]) * 100
            lines.append(f"  {metric:12s}: {degradation:+.2f}%")

        lines.append("\n" + "="*80)
        lines.append("3. RÈGLE MÉTIER : 1 RÉCLAMATION AUTO PAR CLIENT")
        lines.append("="*80)

        df = self.df_scenario
        client_col = [c for c in df.columns if 'idtfcl' in c.lower() or
                     'client' in c.lower() or 'compte' in c.lower()][0]

        lines.append(f"\nClients uniques: {df[client_col].nunique()}")
        lines.append(f"Total réclamations: {len(df)}")
        lines.append(f"Premières réclamations: {df['is_first_reclamation'].sum()}")
        lines.append(f"Réclamations multiples: {(~df['is_first_reclamation']).sum()}")

        lines.append("\n" + "="*80)
        lines.append("4. IMPACT FINANCIER")
        lines.append("="*80)

        lines.append("\nSANS règle métier:")
        lines.append(f"  Automatisées: {self.impact['sans_regle']['auto']} "
                    f"({self.impact['sans_regle']['taux_auto']:.1f}%)")
        lines.append(f"  Gain net: {self.impact['sans_regle']['gain_net']:,.0f} DH")
        lines.append(f"  FP: {self.impact['sans_regle']['fp']}")
        lines.append(f"  FN: {self.impact['sans_regle']['fn']}")

        lines.append("\nAVEC règle métier (1 réclamation/client):")
        lines.append(f"  Automatisées: {self.impact['avec_regle']['auto']} "
                    f"({self.impact['avec_regle']['taux_auto']:.1f}%)")
        lines.append(f"  Gain net: {self.impact['avec_regle']['gain_net']:,.0f} DH")
        lines.append(f"  FP: {self.impact['avec_regle']['fp']}")
        lines.append(f"  FN: {self.impact['avec_regle']['fn']}")

        diff = self.impact['avec_regle']['gain_net'] - self.impact['sans_regle']['gain_net']
        lines.append(f"\nImpact règle métier: {diff:+,.0f} DH")

        # Sauvegarder
        report_path = f'{self.output_dir}/rapport_production.txt'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))

        print(f"   ✅ {report_path}")

    def run(self):
        """Exécuter pipeline complet"""
        print("\n" + "="*80)
        print("🚀 PIPELINE ML PRODUCTION")
        print("="*80)

        # 1. Charger données
        self.load_data(
            path_2024='data/raw/reclamations_2024.xlsx',
            path_2025='data/raw/reclamations_2025.xlsx'
        )

        # 2. Entraîner modèle
        self.train_model()

        # 3. Évaluer 2025
        self.evaluate_2025()

        # 4. Appliquer règle métier
        self.apply_business_rule()

        # 5. Impact financier
        self.calculate_financial_impact()

        # 6. Visualisations
        self.generate_visualizations()

        # 7. Rapport
        self.generate_report()

        print("\n" + "="*80)
        print("✅ PIPELINE TERMINÉ")
        print("="*80)
        print(f"\n📂 Résultats dans: {self.output_dir}/")
        print("\n📊 Fichiers générés:")
        print("   📈 Figures:")
        print("      • comparison_2024_2025.png")
        print("      • business_rule_impact.png")
        print("      • financial_impact.png")
        print("   💾 Modèles:")
        print("      • model_production.pkl")
        print("      • preprocessor_production.pkl")
        print("   📄 Rapport:")
        print("      • rapport_production.txt")


def main():
    """Point d'entrée"""
    pipeline = ProductionPipeline(output_dir='outputs/production')
    pipeline.run()


if __name__ == '__main__':
    main()
