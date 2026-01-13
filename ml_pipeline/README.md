# 📖 Guide d'Utilisation - Pipeline ML Production

## 🎯 Pipeline de Classification des Réclamations Bancaires

Le pipeline est **100% adapté aux vraies colonnes** de votre base de données de production et inclut le **nettoyage automatique des montants**. Voici comment l'utiliser avec vos fichiers Excel réels.

## ✨ Nouvelles Fonctionnalités

### 🧹 Nettoyage Automatique des Montants

Le pipeline nettoie automatiquement les colonnes de montants dans différents formats:
- `"500,00 mad"` → `500.00`
- `"1 234,56 DH"` → `1234.56`
- `"1.234,56"` (format européen) → `1234.56`
- `"1,234.56"` (format US) → `1234.56`
- `"N/A"`, `""`, `null` → `NaN`

**Colonnes nettoyées automatiquement:**
- Montant demandé
- Montant
- Montant de réponse
- PNB analytique (vision commerciale) cumulé

---

## 📂 Étape 1: Préparer Vos Données

### Colonnes Requises pour 2024

Votre fichier `reclamations_2024.xlsx` doit contenir **au minimum** ces colonnes :

| Colonne | Type | Obligatoire | Description |
|---------|------|-------------|-------------|
| **Fondee** | int (0/1) | ✅ OUI | Variable cible (0=Non Fondée, 1=Fondée) |
| **Montant demandé** | float | ✅ OUI | Montant de la réclamation |
| **PNB analytique (vision commerciale) cumulé** | float | ✅ OUI | PNB du client |
| **anciennete_annees** | float | ✅ OUI | Ancienneté client en années |
| **Famille Produit** | string | ✅ OUI | Famille produit (Monétique, Crédit, etc.) |
| **Catégorie** | string | ✅ OUI | Catégorie de réclamation |
| Segment | string | ⭐ Recommandé | Segment client |
| Canal de Réception | string | ⭐ Recommandé | Canal de réception |
| Banque Privé | string (OUI/NON) | ⭐ Recommandé | Flag banque privée |
| Date de Qualification | date | ⭐ Recommandé | Date de qualification |
| Délai Estimé (j) | int | ⭐ Recommandé | Délai estimé |
| Montant de réponse | float | ⭐ Recommandé | Montant de réponse |

**Colonnes additionnelles supportées** (toutes celles de votre schéma sont supportées !) :
- Région, Réseau, Groupe, Statut, PP/PM, Marché
- Code Agence / CA Principal, Libellé Agence / CA Principal
- Priorité Client, Financière ou non, Wafacash
- Recevable, Motif d'irrecevabilité
- Source, BAS (spécifiques à 2024)
- Etc.

### Colonnes Requises pour 2025

Les mêmes colonnes que 2024, **PLUS** :
- Demandeur (spécifique 2025)
- Code GAB, Code anomalie GAB (spécifique Monétique)
- Motif de rejet UT, Date Rejet UT, etc.

---

## 🚀 Étape 2: Placer Vos Fichiers

```bash
# 1. Aller dans le dossier du pipeline
cd /home/user/Moteur-de-scoring/ml_pipeline

# 2. Supprimer les données synthétiques (optionnel)
rm data/raw/reclamations_*.xlsx

# 3. Copier VOS fichiers
cp /chemin/vers/vos/donnees/reclamations_2024.xlsx data/raw/
cp /chemin/vers/vos/donnees/reclamations_2025.xlsx data/raw/
```

**OU** simplement :

```bash
# Copier directement vos fichiers dans le bon dossier
cp ma_base_2024.xlsx ml_pipeline/data/raw/reclamations_2024.xlsx
cp ma_base_2025.xlsx ml_pipeline/data/raw/reclamations_2025.xlsx
```

---

## ⚙️ Étape 3: Configurer le Pipeline (Optionnel)

Ouvrez `main_pipeline.py` et ajustez la configuration si nécessaire:

```python
config = {
    'data_path_2024': 'data/raw/reclamations_2024.xlsx',  # ✅ Vos données
    'data_path_2025': 'data/raw/reclamations_2025.xlsx',  # ✅ Vos données
    'target_col': 'Fondee',                               # Variable cible
    'optuna_trials': 100,                                 # 100-200 pour production
    'cv_folds': 5,                                        # Cross-validation folds
    'model_type': 'xgboost',                              # xgboost, lightgbm, catboost
    'calibration_method': 'isotonic',                     # isotonic ou sigmoid
    'random_state': 42,
    'output_dir': 'outputs'
}
```

**Paramètres Clés:**

- `optuna_trials` :
  - 30-50 pour un test rapide (~5 min)
  - 100-150 pour production (~15 min)
  - 200+ pour optimisation maximale (~30 min)

- `model_type` :
  - `'xgboost'` : Excellent équilibre performance/vitesse
  - `'lightgbm'` : Plus rapide, bon pour gros volumes
  - `'catboost'` : Meilleur avec features catégorielles

---

## 🎬 Étape 4: Lancer le Pipeline

```bash
# Aller dans le dossier
cd /home/user/Moteur-de-scoring/ml_pipeline

# Lancer le pipeline complet
python main_pipeline.py
```

**Durée Estimée:**
- Avec 30 trials: ~5-8 minutes
- Avec 100 trials: ~15-20 minutes
- Avec 200 trials: ~30-40 minutes

---

## 📊 Étape 5: Consulter les Résultats

Tous les résultats sont dans le dossier `outputs/`:

### 📄 Rapports Principaux

```
outputs/reports/
├── RAPPORT_FINAL_REAL_COLS.txt              # 📄 RAPPORT COMPLET
├── family_analysis_2025.txt       # 📄 Analyse par famille produit
├── family_metrics_2025.csv        # 📊 Métriques par famille (CSV)
├── metrics_2024.json              # Métriques 2024
├── metrics_2025.json              # Métriques 2025
├── feature_importance.csv         # Importance des features
└── optuna_history.csv             # Historique optimisation
```

### 📈 Visualisations

```
outputs/reports/figures/
├── family_analysis_2025.png       # ⭐ ANALYSE PAR FAMILLE (NOUVEAU!)
├── confusion_matrix_2024.png      # Confusion 2024
├── confusion_matrix_2025.png      # Confusion 2025
├── roc_curve_2024.png             # ROC 2024
├── roc_curve_2025.png             # ROC 2025
├── pr_curve_2024.png              # Precision-Recall 2024
├── pr_curve_2025.png              # Precision-Recall 2025
├── calibration_curve.png          # Calibration
└── prob_distribution_comparison.png # Comparaison prédictions
```

### 💾 Modèles et Artefacts

```
outputs/models/
├── model_xgboost_YYYYMMDD_HHMMSS.pkl  # Modèle entraîné
├── best_hyperparameters.json           # Hyperparamètres optimaux
└── metadata_YYYYMMDD_HHMMSS.json       # Métadonnées complètes

outputs/preprocessors/
├── preprocessor.pkl                    # Preprocessing pipeline
└── feature_selector.pkl                # Sélecteur de features
```

---

## 🎨 Nouvelle Visualisation par Famille Produit

Le pipeline génère maintenant une **analyse complète par famille produit** avec :

1. **Barplot des métriques** (Accuracy, Precision, Recall, F1) par famille
2. **Volume de réclamations** par famille
3. **Taux de fondement** (Réel vs Prédit) par famille
4. **Montant moyen** par famille
5. **Confusion Matrix** - Meilleure famille
6. **Confusion Matrix** - Moins bonne famille
7. **Heatmap des métriques** toutes familles

📊 Exemple de ce que vous obtiendrez :

```
Famille                        N     Acc    Prec     Rec      F1   Fond%
--------------------------------------------------------------------------------
Monétique                   5832   82.4%   85.2%   79.8%   82.4%   68.3%
Crédit                      5123   79.1%   81.5%   76.2%   78.8%   52.1%
Frais bancaires             4891   74.5%   77.3%   71.2%   74.1%   38.7%
Epargne                     4654   72.8%   75.6%   69.3%   72.3%   41.2%
```

---

## 🔍 Vérifications Automatiques

Le pipeline vérifie automatiquement :

✅ **Colonnes manquantes** : Avertissement si colonnes importantes absentes
✅ **Types de données** : Conversion automatique si nécessaire
✅ **Valeurs manquantes** : Gestion intelligente (pas de fillna(0) brutal)
✅ **Outliers** : Détection et clipping automatique (IQR method)
✅ **Drift temporel** : Tests statistiques KS et Chi²
✅ **Déséquilibre des classes** : `scale_pos_weight` automatique
✅ **Calibration** : Vérification ECE et Brier Score

---

## 📋 Interpréter les Résultats

### Métriques Attendues (Production)

| Métrique | Bon | Acceptable | À Améliorer |
|----------|-----|------------|-------------|
| **Accuracy** | > 75% | 70-75% | < 70% |
| **F1-Score** | > 75% | 70-75% | < 70% |
| **ROC-AUC** | > 0.80 | 0.75-0.80 | < 0.75 |
| **Dégradation 2024→2025** | < 5% | 5-10% | > 10% |

### Analyse par Famille

Pour chaque famille produit, vous obtiendrez :

- **Performance spécifique** : Accuracy, Precision, Recall, F1
- **Taux de fondement** : Comparaison Réel vs Prédit
- **Volume de réclamations** : Nombre de cas par famille
- **Montant moyen** : Montant demandé moyen
- **Confusion matrix** : Détails des erreurs

**Utilité :**
- 🎯 Identifier les familles les plus/moins bien prédites
- 🔧 Ajuster les modèles par famille si nécessaire
- 📊 Comprendre les différences de performance
- 💡 Orienter les actions métier

---

## ⚠️ Cas Particuliers

### Colonnes Manquantes

Si certaines colonnes sont absentes, le pipeline :
- **Continue quand même** (features optionnelles)
- **Affiche un warning** pour les colonnes importantes
- **Adapte le preprocessing** automatiquement

### Données 2025 Non Disponibles

Si vous n'avez que 2024 :

```python
# Utiliser une partie de 2024 comme test
from sklearn.model_selection import train_test_split

df_2024 = pd.read_excel('data/raw/reclamations_2024.xlsx')
df_train, df_test = train_test_split(
    df_2024,
    test_size=0.2,
    stratify=df_2024['Fondee'],
    random_state=42
)

# Sauvegarder
df_train.to_excel('data/raw/reclamations_2024.xlsx', index=False)
df_test.to_excel('data/raw/reclamations_2025.xlsx', index=False)
```

### Noms de Colonnes Différents

Si vos colonnes ont des noms légèrement différents :

```python
# Renommer avant de sauvegarder
df.rename(columns={
    'montant': 'Montant demandé',
    'pnb': 'PNB analytique (vision commerciale) cumulé',
    'anciennete': 'anciennete_annees'
}, inplace=True)
```

---

## 🚀 Workflow Complet

```mermaid
graph LR
    A[Vos Données Excel] --> B[Copier dans data/raw/]
    B --> C[Lancer main_pipeline.py]
    C --> D[Preprocessing Automatique]
    D --> E[Sélection Features]
    E --> F[Optimisation Optuna]
    F --> G[Évaluation 2024/2025]
    G --> H[Analyse par Famille]
    H --> I[Analyse de Drift]
    I --> J[Rapport Complet]
    J --> K[Visualisations + Modèle Sauvegardé]
```

---

## 📞 Support

Si vous rencontrez des problèmes :

1. Vérifier que les colonnes obligatoires sont présentes
2. Vérifier que `Fondee` contient bien 0 et 1
3. Vérifier que les montants sont numériques
4. Vérifier que les dates sont au format date

---

## 🎉 Prêt !

Votre pipeline est maintenant **production-ready** avec les vraies colonnes de votre base de données !

```bash
# C'est aussi simple que :
cp mes_donnees_2024.xlsx ml_pipeline/data/raw/reclamations_2024.xlsx
cp mes_donnees_2025.xlsx ml_pipeline/data/raw/reclamations_2025.xlsx
cd ml_pipeline
python main_pipeline.py
```

**Et voilà ! 🚀**

---

**Version:** 2.0.0 - Avec Analyse par Famille Produit
**Date:** Janvier 2026
**Statut:** ✅ Production-Ready avec Vraies Colonnes
