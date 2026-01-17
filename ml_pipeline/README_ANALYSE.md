# Guide d'Analyse des Résultats - Moteur de Scoring

## 📋 Vue d'ensemble

Ce dossier contient plusieurs scripts d'analyse pour évaluer et présenter les performances des modèles de classification de réclamations bancaires.

## 🚀 Scripts disponibles

### 1. **model_comparison.py** - Entraînement et Comparaison
```bash
python ml_pipeline/model_comparison.py
```
**Objectif**: Entraîner les 3 modèles (XGBoost, RandomForest, CatBoost) et sauvegarder les prédictions.

**Sorties**:
- Modèles entraînés sauvegardés
- Prédictions sauvegardées dans `outputs/production/predictions/predictions_2025.pkl`
- Graphiques de comparaison

⚠️ **IMPORTANT**: Exécuter ce script EN PREMIER avant toute analyse!

---

### 2. **analyze_results.py** - Analyse XGBoost
```bash
python ml_pipeline/analyze_results.py
```
**Objectif**: Analyser en détail les résultats du modèle XGBoost.

**Visualisations générées**:
- ✅ Matrice de confusion (sur cas automatisés)
- ✅ Impact de la règle métier (9 graphes)
- ✅ Accuracy par famille de produit
- ✅ Rapport texte récapitulatif

**Fichiers générés**:
```
outputs/production/figures/
├── xgboost_confusion_matrix.png
├── xgboost_business_rule_impact.png
├── xgboost_accuracy_by_family.png
outputs/production/
└── xgboost_rapport_analyse.txt
```

---

### 3. **analyze_results_catboost.py** - Analyse CatBoost
```bash
python ml_pipeline/analyze_results_catboost.py
```
**Objectif**: Analyser en détail les résultats du modèle CatBoost.

**Visualisations générées**:
- ✅ Matrice de confusion (colormap violet)
- ✅ Impact de la règle métier (9 graphes)
- ✅ Accuracy par famille de produit
- ✅ Rapport texte récapitulatif

**Fichiers générés**:
```
outputs/production/figures/
├── catboost_confusion_matrix.png
├── catboost_business_rule_impact.png
├── catboost_accuracy_by_family.png
outputs/production/
└── catboost_rapport_analyse.txt
```

---

### 4. **generate_catboost_report.py** ⭐ - Rapport Professionnel CatBoost
```bash
python ml_pipeline/generate_catboost_report.py
```
**Objectif**: Générer un **dossier complet de visualisations professionnelles** pour présenter les performances de CatBoost.

**🎯 RECOMMANDÉ pour présentation et valorisation des résultats!**

**Visualisations générées** (6 PNG + 1 TXT):

#### 📊 1. Dashboard de Performance
- Vue d'ensemble des métriques principales
- Matrice de confusion détaillée
- Distribution des 3 types de décision
- Barplot des métriques (Accuracy, Precision, Recall, F1, Spécificité)
- Distribution des probabilités par classe

#### 📈 2. Courbes ROC et Precision-Recall
- Courbe ROC avec AUC
- Points marqueurs pour les seuils choisis (threshold_low et threshold_high)
- Courbe Precision-Recall
- Ligne de base (baseline)

#### 📅 3. Performance Temporelle
- Volume mensuel (total vs automatisé)
- Évolution du taux d'automatisation
- Évolution des métriques (Accuracy, Precision, Recall)
- Table récapitulative mensuelle

#### 💰 4. Analyse par Montant
- Volume par tranche de montant
- Performance (Accuracy) et taux d'automatisation par tranche
- Coûts des erreurs (FP et FN) par tranche
- Nombre d'erreurs par tranche

#### 💼 5. Impact Business Détaillé
- Flux financier (Gain brut → Pertes → Gain NET)
- ROI unitaire par type de cas
- Composition des prédictions (pie chart TP/TN/FP/FN)
- Résumé financier complet

#### 🏆 6. Top Families Advanced
- Accuracy par famille (Top 12)
- Volume et taux d'automatisation
- Scatter plot Precision vs Recall (avec volume et accuracy)
- Heatmap des métriques par famille

#### 📄 7. Rapport Texte Complet
Rapport récapitulatif professionnel incluant:
- Vue d'ensemble
- Système à 3 zones
- Métriques de performance
- Matrice de confusion détaillée
- Impact business
- Avantages du modèle
- Recommandations

**Dossier de sortie**:
```
outputs/production/catboost_report/
├── 01_dashboard_performance.png
├── 02_roc_pr_curves.png
├── 03_performance_temporelle.png
├── 04_analyse_par_montant.png
├── 05_impact_business.png
├── 06_top_families_advanced.png
└── RAPPORT_CATBOOST.txt
```

---

## 🔄 Workflow Recommandé

### Pour l'entraînement initial:
```bash
# 1. Entraîner les modèles et sauvegarder les prédictions
python ml_pipeline/model_comparison.py

# 2. Générer le rapport complet CatBoost (RECOMMANDÉ!)
python ml_pipeline/generate_catboost_report.py

# 3. (Optionnel) Analyser XGBoost séparément
python ml_pipeline/analyze_results.py

# 4. (Optionnel) Analyser CatBoost avec règle métier
python ml_pipeline/analyze_results_catboost.py
```

### Pour une présentation professionnelle:
```bash
# Utiliser uniquement le générateur de rapport complet
python ml_pipeline/generate_catboost_report.py
```
**✅ Ce script génère TOUT ce dont vous avez besoin pour présenter et valoriser votre travail!**

---

## 📊 Comprendre les Visualisations

### Système à 3 Zones de Décision

Le modèle utilise **2 seuils** pour créer **3 zones**:

```
Zone 1: prob ≤ threshold_low       → REJET AUTO (Non Fondée)
Zone 2: threshold_low < prob < threshold_high  → AUDIT HUMAIN (manuel)
Zone 3: prob ≥ threshold_high      → VALIDATION AUTO (Fondée)
```

**Avantages**:
- ✅ Automatise les cas certains (zones 1 et 3)
- ✅ Envoie les cas incertains à un expert humain (zone 2)
- ✅ Réduit le risque d'erreur

### Règle Métier

**Règle appliquée**: Un client ne peut bénéficier que d'**UNE validation automatique par année**.

**Mécanisme**:
1. Trier les réclamations par Date de Qualification
2. Par client et par année, seule la **première validation auto** est acceptée
3. Les validations suivantes deviennent des **audits humains**

**Impact**:
- 🔒 Prévient l'abus de validations automatiques
- 📊 Augmente le nombre d'audits humains
- 💰 Peut réduire le gain NET mais améliore le contrôle

### Métriques Clés

| Métrique | Description | Objectif |
|----------|-------------|----------|
| **Accuracy** | Proportion de prédictions correctes | > 95% |
| **Precision** | Proportion de validations correctes parmi toutes les validations | > 90% |
| **Recall** | Proportion de réclamations fondées correctement validées | > 95% |
| **F1-Score** | Moyenne harmonique Precision/Recall | > 92% |
| **Taux d'automatisation** | Proportion de cas automatisés (hors audit) | 70-80% |
| **Gain NET** | Gain brut - Coût FP - Coût FN | Positif |

### Calcul du Gain NET

```
Gain Brut = (TP + TN) × 169 DH
Coût FP = Σ montants des faux positifs
Coût FN = 2 × Σ montants des faux négatifs  (coût double!)
Gain NET = Gain Brut - Coût FP - Coût FN
```

---

## 🎨 Palette de Couleurs

Les visualisations utilisent un code couleur cohérent:

| Couleur | Usage |
|---------|-------|
| 🟢 Vert | Positif (TP, TN, Gain, Bonne performance) |
| 🔵 Bleu | Neutre (Volume, Accuracy, Informations) |
| 🟠 Orange | Attention (FP, Coûts modérés) |
| 🔴 Rouge | Négatif (FN, Pertes, Erreurs) |
| 🟣 Violet | CatBoost, Métriques spéciales |

---

## 📁 Structure des Sorties

```
outputs/production/
├── predictions/
│   └── predictions_2025.pkl         # Prédictions sauvegardées
├── figures/
│   ├── xgboost_confusion_matrix.png
│   ├── xgboost_business_rule_impact.png
│   ├── xgboost_accuracy_by_family.png
│   ├── catboost_confusion_matrix.png
│   ├── catboost_business_rule_impact.png
│   └── catboost_accuracy_by_family.png
├── catboost_report/                 # ⭐ DOSSIER COMPLET
│   ├── 01_dashboard_performance.png
│   ├── 02_roc_pr_curves.png
│   ├── 03_performance_temporelle.png
│   ├── 04_analyse_par_montant.png
│   ├── 05_impact_business.png
│   ├── 06_top_families_advanced.png
│   └── RAPPORT_CATBOOST.txt
├── xgboost_rapport_analyse.txt
└── catboost_rapport_analyse.txt
```

---

## 💡 Conseils pour la Présentation

### Pour valoriser votre travail:

1. **Commencer par le Dashboard** (`01_dashboard_performance.png`)
   - Montre immédiatement la performance globale
   - Métriques clés visibles d'un coup d'œil

2. **Expliquer le système à 3 zones**
   - Utiliser le graphique de distribution des décisions
   - Montrer la distribution des probabilités

3. **Montrer l'impact business** (`05_impact_business.png`)
   - Flux financier clair
   - ROI positif
   - Gain NET significatif

4. **Détailler les analyses** (selon l'audience)
   - Performance temporelle (évolution)
   - Performance par montant (robustesse)
   - Performance par famille (granularité)

5. **Terminer par les recommandations**
   - Lire le rapport texte
   - Mettre en avant les points forts
   - Proposer des axes d'amélioration

### Points à mettre en avant:

✅ **Taux d'automatisation élevé** (70-80%)
✅ **Accuracy > 98%** sur cas automatisés
✅ **Gain NET positif**
✅ **Robustesse** (gestion valeurs manquantes, catégorielles)
✅ **Gestion de l'incertitude** (zone d'audit humain)
✅ **Contrôle business** (règle métier personnalisable)

---

## 🔧 Personnalisation

### Modifier les seuils
Éditer dans `model_comparison.py`:
```python
threshold_low, threshold_high = optimize_dual_thresholds(...)
```

### Modifier la règle métier
Éditer dans `analyze_results.py` ou `analyze_results_catboost.py`:
```python
# Exemple: 2 validations auto par an au lieu de 1
df_scenario['validation_rank'] > 2  # au lieu de > 1
```

### Ajouter des visualisations
Ajouter des fonctions `viz_N_...()` dans `generate_catboost_report.py`

---

## ❓ FAQ

**Q: Quel script utiliser pour une présentation professionnelle?**
A: `generate_catboost_report.py` - Il génère tout ce dont vous avez besoin!

**Q: Pourquoi CatBoost et pas XGBoost?**
A: CatBoost offre généralement de meilleures performances sur des données catégorielles et gère mieux les valeurs manquantes sans preprocessing.

**Q: Comment expliquer les 2 seuils?**
A: Le seuil bas filtre les rejets évidents, le seuil haut filtre les validations évidentes. Entre les deux = incertitude → audit humain.

**Q: Pourquoi le coût FN est × 2?**
A: Car rejeter une réclamation fondée coûte plus cher (insatisfaction client, coûts légaux potentiels).

**Q: Puis-je désactiver la règle métier?**
A: Oui, comparer les graphiques "SANS règle" vs "AVEC règle" pour décider.

---

## 📞 Support

Pour toute question ou problème:
1. Vérifier que `model_comparison.py` a été exécuté en premier
2. Vérifier que le fichier `predictions_2025.pkl` existe
3. Consulter les logs d'exécution pour identifier les erreurs

---

**Créé avec ❤️ pour valoriser votre travail de machine learning!**
