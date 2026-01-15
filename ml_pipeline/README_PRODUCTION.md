# 🚀 Pipeline ML Production - Classification Réclamations Bancaires

## 🎯 Vue d'ensemble

Pipeline **simplifié et production-ready** avec **règle métier critique** : **1 seule réclamation automatisée par client**.

---

## ✨ Nouveautés Clés

### 🔒 Règle Métier Implémentée

**Principe** : Chaque client ne peut avoir qu'**UNE SEULE réclamation automatisée**.

**Logique** :
1. Les réclamations sont **triées par date de qualification**
2. Pour chaque client, seule sa **première réclamation** peut être automatisée
3. Les réclamations suivantes du même client → **traitement manuel obligatoire**

**Justification** :
- Éviter l'abus du système automatisé
- Sécuriser la relation client
- Réduire les faux négatifs critiques

### 📊 Features Simplifiées

Le pipeline utilise **uniquement les colonnes métier** :

**Colonnes directes** :
- Marché
- Segment
- Famille Produit
- Catégorie
- Sous-catégorie
- Montant demandé
- PNB analytique (vision commerciale) cumulé
- anciennete_annees

**Features calculées** :
1. **Ratio couverture PNB** = PNB / Montant demandé
2. **Écart à la médiane de famille** = (Montant - Médiane famille) / Médiane famille
   - ⚠️ Médiane calculée sur **2024 uniquement** et appliquée sur 2025
3. Log transformations (montant, PNB)

---

## 🚀 Utilisation

### Prérequis

```bash
cd /home/user/Moteur-de-scoring/ml_pipeline

# Vérifier que les données sont présentes
ls data/raw/reclamations_2024.xlsx
ls data/raw/reclamations_2025.xlsx
```

### Exécution

```bash
python production_pipeline.py
```

**Durée estimée** : ~5 minutes

---

## 📂 Structure des Résultats

```
outputs/production/
├── figures/
│   ├── comparison_2024_2025.png          # Comparaison performance
│   ├── business_rule_impact.png          # Impact règle métier
│   └── financial_impact.png              # Impact financier
│
├── models/
│   ├── model_production.pkl              # Modèle XGBoost entraîné
│   └── preprocessor_production.pkl       # Preprocessor
│
└── rapport_production.txt                # Rapport complet
```

---

## 📊 Visualisations Générées

### 1. `comparison_2024_2025.png` - Comparaison Performance

**2 graphiques** :

1. **Métriques 2024 vs 2025** (barres)
   - Accuracy, Precision, Recall, F1-Score
   - Comparaison visuelle directe

2. **Dégradation en %** (barres horizontales)
   - Variation de chaque métrique
   - Vert = amélioration, Rouge = dégradation

**Utilité** : Vérifier que le modèle reste performant sur 2025

---

### 2. `business_rule_impact.png` - Impact Règle Métier

**4 graphiques** :

1. **Distribution réclamations par client**
   - Combien de clients ont 1, 2, 3+ réclamations

2. **Taux automatisation : SANS vs AVEC règle**
   - Réduction du taux d'automatisation
   - Impact visible de la règle

3. **Nombre automatisées : SANS vs AVEC règle**
   - Volume absolu de réclamations automatisées

4. **Répartition : 1ère réclamation vs Multiples** (camembert)
   - Visualiser le % de réclamations multiples

**Utilité** : Comprendre l'impact de la règle métier sur l'automatisation

---

### 3. `financial_impact.png` - Impact Financier

**4 graphiques** :

1. **Gain net total : SANS vs AVEC règle**
   - Comparaison directe du gain net

2. **Erreurs FP et FN : SANS vs AVEC règle**
   - Impact de la règle sur les erreurs

3. **Décomposition financière SANS règle**
   - Gain brut, Coût FP, Coût FN, Gain NET

4. **Décomposition financière AVEC règle**
   - Même décomposition avec la règle appliquée

**Utilité** : Justifier financièrement la règle métier

---

## 💰 Calculs Financiers

### Prix Unitaire

```python
PRIX_UNITAIRE_DH = 169  # Coût traitement manuel d'une réclamation
```

### Formules

**Gain brut** :
```
Gain brut = (TP + TN) × 169 DH
```
Où : TP + TN = réclamations automatisées correctement

**Coûts** :
```
Coût FP = Nombre FP × 169 DH
Coût FN = Nombre FN × 2 × 169 DH
```
⚠️ FN coûtent 2× car client mécontent + re-traitement

**Gain net** :
```
Gain net = Gain brut - Coût FP - Coût FN
```

---

## 📋 Rapport Produit

Le fichier `rapport_production.txt` contient :

### 1. Données
- Nombre réclamations 2024 (entraînement)
- Nombre réclamations 2025 (test)

### 2. Performance Modèle
- Métriques 2024 : accuracy, precision, recall, F1
- Métriques 2025 : accuracy, precision, recall, F1
- Dégradation en % pour chaque métrique

### 3. Règle Métier
- Nombre clients uniques
- Total réclamations
- Premières réclamations
- Réclamations multiples

### 4. Impact Financier
- **SANS règle** : automatisées, gain net, FP, FN
- **AVEC règle** : automatisées, gain net, FP, FN
- Différence gain net

---

## 🔍 Exemple de Résultat

```
================================================================================
4. IMPACT FINANCIER
================================================================================

SANS règle métier:
  Automatisées: 1,856 (74.2%)
  Gain net: 245,628 DH
  FP: 128
  FN: 116

AVEC règle métier (1 réclamation/client):
  Automatisées: 1,623 (64.9%)
  Gain net: 218,455 DH
  FP: 95
  FN: 82

Impact règle métier: -27,173 DH
```

### Interprétation

- ✅ **Gain net positif** dans les deux cas
- ⚠️ **Règle réduit gain de 27k DH** mais :
  - Réduit FP de 33 (-26%)
  - Réduit FN de 34 (-29%) ← **Critique pour satisfaction client**
- 💡 **Trade-off acceptable** : sacrifier 27k DH pour éviter 34 clients mécontents

---

## 🔧 Personnalisation

### Changer le Prix Unitaire

Dans `production_pipeline.py`, ligne 13 :

```python
PRIX_UNITAIRE_DH = 169  # Modifier ici
```

### Modifier le Modèle

Dans la méthode `train_model()`, ligne ~160 :

```python
self.model = xgb.XGBClassifier(
    max_depth=6,           # Modifier hyperparamètres
    learning_rate=0.1,
    n_estimators=100,
    random_state=42
)
```

### Désactiver la Règle Métier (TEST UNIQUEMENT)

Dans `apply_business_rule()`, commenter la ligne :

```python
# Commenter cette ligne pour désactiver la règle
# df_scenario['can_automate'] = df_scenario['is_first_reclamation']

# Remplacer par :
df_scenario['can_automate'] = True  # Tous automatisables
```

⚠️ **Ne jamais désactiver en production !**

---

## 📊 Colonnes Requises

Votre fichier Excel **doit contenir** :

### Obligatoires

| Colonne | Type | Description |
|---------|------|-------------|
| **Fondee** | int (0/1) | Variable cible |
| **Marché** | string | Marché |
| **Segment** | string | Segment client |
| **Famille Produit** | string | Famille produit |
| **Catégorie** | string | Catégorie réclamation |
| **Sous-catégorie** | string | Sous-catégorie |
| **Montant demandé** | float | Montant |
| **PNB analytique (vision commerciale) cumulé** | float | PNB client |
| **anciennete_annees** | float | Ancienneté |
| **Date de Qualification** | date | Date qualification |

### Pour Règle Métier

Au moins **UNE** de ces colonnes pour identifier le client :
- `idtfcl`
- `N compte`
- `numero_compte`
- `ID Client`

---

## 🆘 Résolution de Problèmes

### Erreur : Colonne manquante

```
⚠️  Colonne manquante dans 2024: Marché
```

**Solution** : Vérifiez que votre fichier Excel contient bien toutes les colonnes requises.

### Erreur : Colonne client non trouvée

```
⚠️  Colonne client non trouvée, utilisation de l'index
```

**Impact** : La règle métier ne fonctionnera pas correctement.

**Solution** : Assurez-vous d'avoir une colonne `idtfcl`, `N compte`, ou `numero_compte`.

### Pas de réclamations multiples

```
Réclamations multiples: 0
```

**Vérification** : Normal si chaque client n'a vraiment qu'une seule réclamation dans vos données.

**Test** : La règle métier ne change rien dans ce cas (SANS règle = AVEC règle).

---

## 🎯 Checklist de Mise en Production

- [ ] Données 2024 et 2025 présentes
- [ ] Toutes les colonnes requises présentes
- [ ] Colonne client identifiable
- [ ] Prix unitaire correct (169 DH)
- [ ] Pipeline exécuté sans erreur
- [ ] Rapport généré et analysé
- [ ] Gain net positif confirmé
- [ ] Règle métier validée avec métier
- [ ] Visualisations consultées
- [ ] Modèle et preprocessor sauvegardés

---

## 📈 Métriques de Succès

✅ **Modèle acceptable si** :
- Accuracy 2025 ≥ 75%
- Dégradation 2024→2025 < 10%
- Gain net > 0 DH
- FN minimisés (< 10% des réclamations)

✅ **Règle métier validée si** :
- Gain net reste positif
- FN réduits (même si gain net baisse légèrement)
- % automatisation reste > 50%

---

## 💡 Améliorations Futures

1. **Optimiser hyperparamètres** avec Optuna
2. **Ajouter features** :
   - Historique réclamations client
   - Taux réclamations par produit
   - Saisonnalité
3. **Ajuster seuil de décision** pour minimiser FN
4. **A/B test** règle métier sur échantillon avant déploiement total

---

## 📞 Support

Pour toute question :
1. Vérifier `rapport_production.txt`
2. Consulter les visualisations
3. Vérifier les logs d'exécution

---

Bon déploiement en production ! 🚀
