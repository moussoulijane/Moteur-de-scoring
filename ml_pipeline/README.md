# 🏦 Pipeline ML Production - Classification des Réclamations Bancaires

## 📋 Vue d'Ensemble

Pipeline complet de Machine Learning pour la classification des réclamations bancaires (Fondée / Non Fondée) avec validation temporelle et détection de drift.

### ✨ Caractéristiques Principales

- ✅ **Preprocessing Robuste** : Feature engineering avancé avec 15+ features créés
- ✅ **Sélection de Features** : Multi-critères (variance, corrélation, importance)
- ✅ **Optimisation Optuna** : XGBoost/LightGBM/CatBoost avec 50+ trials
- ✅ **Calibration des Probabilités** : Méthodes isotonic/sigmoid
- ✅ **Validation Temporelle** : Test sur données 2025 (futures)
- ✅ **Analyse de Drift** : Tests statistiques KS et Chi²
- ✅ **Rapports Complets** : Métriques, visualisations, recommandations



## 📁 Structure du Projet

```
ml_pipeline/
├── data/
│   ├── raw/                    # Données brutes 2024 et 2025
│   └── processed/              # Données transformées
│
├── src/
│   ├── preprocessing/
│   │   └── preprocessor.py     # Feature engineering + encodage
│   ├── feature_selection/
│   │   └── selector.py         # Sélection multi-critères
│   ├── modeling/
│   │   └── optuna_optimizer.py # Optimisation hyperparamètres
│   ├── evaluation/
│   │   ├── calibrator.py       # Calibration probabilités
│   │   ├── metrics.py          # Calcul métriques
│   │   └── drift_analyzer.py   # Détection drift
│   └── utils/
│       └── data_generator.py   # Génération données synthétiques
│
├── outputs/
│   ├── models/                 # Modèles sauvegardés
│   ├── preprocessors/          # Transformers sauvegardés
│   └── reports/                # Rapports et visualisations
│
├── main_pipeline.py            # Pipeline principal
├── requirements.txt            # Dépendances
└── README.md                   # Ce fichier
```

## 🚀 Installation

### Prérequis
- Python 3.8+
- 8 GB RAM minimum

### Installation des Dépendances

```bash
pip install -r requirements.txt
```

**Dépendances principales:**
- pandas, numpy, scipy
- scikit-learn
- xgboost, lightgbm, catboost
- optuna
- matplotlib, seaborn
- openpyxl (pour Excel)
- shap (optionnel, pour explainability)

## 🎬 Utilisation



### 1. Exécution du Pipeline Complet

```bash
python main_pipeline.py
```

**Durée estimée:** 10-15 minutes (avec 50 trials Optuna)

### 2. Configuration Personnalisée

Éditez le bloc `config` dans `main_pipeline.py` :

```python
config = {
    'data_path_2024': 'data/raw/reclamations_2024.xlsx',
    'data_path_2025': 'data/raw/reclamations_2025.xlsx',
    'optuna_trials': 100,        # Nombre de trials (50-200)
    'cv_folds': 5,               # Nombre de folds CV
    'model_type': 'xgboost',     # xgboost, lightgbm, catboost
    'calibration_method': 'isotonic',  # isotonic ou sigmoid
    'random_state': 42
}
```

## 📊 Sorties Générées

### Modèles et Artefacts

```
outputs/
├── models/
│   ├── model_xgboost_20260111.pkl          # Modèle entraîné
│   ├── best_hyperparameters.json           # Hyperparamètres optimaux
│   └── metadata_20260111.json              # Métadonnées complètes
│
├── preprocessors/
│   ├── preprocessor.pkl                    # Pipeline preprocessing
│   └── feature_selector.pkl                # Sélecteur de features
│
└── reports/
    ├── RAPPORT_FINAL.txt                   # 📄 RAPPORT COMPLET
    ├── feature_importance.csv              # Importance des features
    ├── optuna_history.csv                  # Historique optimisation
    ├── metrics_2024.json                   # Métriques 2024
    ├── metrics_2025.json                   # Métriques 2025
    ├── drift_report_numerical.csv          # Rapport drift numériques
    ├── drift_report_categorical.csv        # Rapport drift catégorielles
    └── figures/
        ├── confusion_matrix_2024.png       # Confusion 2024
        ├── confusion_matrix_2025.png       # Confusion 2025
        ├── roc_curve_2024.png              # ROC 2024
        ├── roc_curve_2025.png              # ROC 2025
        ├── pr_curve_2024.png               # Precision-Recall 2024
        ├── pr_curve_2025.png               # Precision-Recall 2025
        ├── calibration_curve.png           # Calibration
        └── prob_distribution_comparison.png # Comparaison prédictions
```

## 🔧 Modules Détaillés

### 1. Preprocessing (`src/preprocessing/preprocessor.py`)

**Features Engineering:**
- Ratios : `ratio_pnb_montant`, `ratio_montant_famille`
- Temporels : `mois`, `trimestre`, `jour_semaine`, `est_weekend`
- Agrégations : `ratio_produits_anciennete`, `taux_reclamations_annuel`
- Flags : `is_high_value`, `is_frequent_claimer`, `is_senior`
- Interactions : `montant_x_anciennete`, `pnb_x_segment`
- Log-transform : `log_montant`, `log_pnb`, `log_anciennete`

**Encodage:**
- Target Encoding avec smoothing (évite overfitting)
- Traitement des outliers (IQR clipping)
- Standardisation robuste (RobustScaler)

### 2. Sélection de Features (`src/feature_selection/selector.py`)

**Critères d'élimination:**
1. ❌ Features avec >50% de valeurs manquantes
2. ❌ Features à variance quasi-nulle (< 0.01)
3. ❌ Features corrélées >0.95
4. ❌ Features à faible importance (consensus de 2+ méthodes)

**Méthodes d'importance:**
- Permutation Importance
- Native Feature Importance (Random Forest)
- SHAP values (optionnel)

### 3. Optimisation Optuna (`src/modeling/optuna_optimizer.py`)

**Hyperparamètres optimisés:**
- `max_depth` : [3, 10]
- `learning_rate` : [0.01, 0.3] (log scale)
- `n_estimators` : [100, 1000]
- `subsample` : [0.6, 1.0]
- `colsample_bytree` : [0.6, 1.0]
- `reg_alpha` (L1) : [1e-8, 10] (log scale)
- `reg_lambda` (L2) : [1e-8, 10] (log scale)
- `scale_pos_weight` : calculé automatiquement

**Stratégie:**
- TPESampler (Tree-structured Parzen Estimator)
- MedianPruner (arrêt précoce des mauvais trials)
- Validation croisée StratifiedKFold 5-fold
- Métrique d'optimisation : F1-Score

### 4. Calibration (`src/evaluation/calibrator.py`)

**Méthodes:**
- Isotonic Regression (non-paramétrique)
- Sigmoid (paramétrique)

**Métriques de calibration:**
- Expected Calibration Error (ECE)
- Brier Score

### 5. Analyse de Drift (`src/evaluation/drift_analyzer.py`)

**Tests statistiques:**
- **Kolmogorov-Smirnov** : features numériques
- **Chi²** : features catégorielles

**Seuil de signification:** p < 0.05

## 📈 Top Features Importantes

1. **Categorie_encoded** (1.0000) - Type de réclamation
2. **Famille_Produit_encoded** (0.5391) - Famille produit
3. **log_montant** (0.2295) - Log du montant demandé
4. **montant_x_anciennete** (0.2098) - Interaction
5. **ratio_montant_famille** (0.1894) - Ratio montant/médiane famille

## 🔍 Analyse des Résultats

### Pourquoi la dégradation sur 2025 ?

**Causes identifiées:**

1. **Drift temporel intentionnel** dans les données générées:
   - Taux de réclamations fondées : 53.5% → 49.1% (-8.1%)
   - Montant moyen : +15.1%
   - PNB moyen : +19.4%

2. **Distribution changeante des classes:**
   - Le modèle a appris sur une distribution 2024
   - La distribution 2025 est significativement différente

3. **Concept drift:**
   - Les critères de fondement peuvent avoir évolué
   - Les comportements clients changent

### Solutions Recommandées

✅ **Solution 1: Réentraînement**
- Réentraîner sur données combinées 2024 + 2025
- Validation croisée temporelle

✅ **Solution 2: Apprentissage Continu**
- Réentraînement mensuel/trimestriel
- Monitoring du drift en production
- Alertes automatiques

✅ **Solution 3: Modèles Adaptatifs**
- Online learning
- Ensemble avec poids temporels

## 🚨 Monitoring en Production

**KPIs à suivre:**
- Accuracy, F1-Score hebdomadaire
- Distribution des probabilités prédites
- Tests de drift mensuels
- Temps de réponse

**Seuils d'alerte:**
- Dégradation > 5% : ⚠️ Warning
- Dégradation > 10% : 🚨 Critical
- Drift détecté (p < 0.05) : 🔔 Investigation

## 🎓 Méthodologie

### Points Forts

✅ Validation temporelle (train 2024, test 2025)
✅ Feature engineering robuste (15+ features)
✅ Optimisation bayésienne (Optuna)
✅ Calibration des probabilités
✅ Détection de drift automatique
✅ Métriques complètes
✅ Reproductibilité (random_state fixé)

### Limites

⚠️ Pas de validation sur données réelles (synthétiques)
⚠️ Drift intentionnel très prononcé (démonstration)
⚠️ Pas d'ensemble de modèles
⚠️ Pas de SHAP pour explainability détaillée

## 📝 Licence

MIT License

## 👥 Auteur

Pipeline développé pour démonstration de best practices ML en production.

---

**Version:** 1.0.0
**Date:** Janvier 2026
**Statut:** ✅ Complet et Fonctionnel
