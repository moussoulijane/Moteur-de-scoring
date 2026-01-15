# 📊 Guide d'Utilisation - Script de Visualisation 2025

## 🎯 Vue d'ensemble

Le script `visualize_results_2025.py` génère des visualisations avancées pour analyser les résultats du modèle sur les données 2025, incluant :

1. **🏆 Analyse par Famille Produit** - Identifier les familles avec les meilleurs succès
2. **⚠️ Analyse des Faux Positifs** - Comprendre les erreurs en termes de montants
3. **💰 Quantification Financière** - Calculer les pertes et gains (basé sur 169 DH/réclamation)

---

## 🚀 Utilisation

### Option 1: Après avoir exécuté le pipeline complet

Si vous avez déjà exécuté `main_pipeline.py`, les prédictions sont déjà sauvegardées. Lancez simplement :

```bash
cd /home/user/Moteur-de-scoring/ml_pipeline
python visualize_results_2025.py
```

### Option 2: Avec vos propres données

Si vous voulez uniquement visualiser vos données 2025 sans modèle :

```python
python visualize_results_2025.py
```

Le script détectera automatiquement si les prédictions existent ou non et adaptera les visualisations.

---

## 📈 Visualisations Générées

### 1. **family_success_2025.png** - Analyse par Famille

**4 graphiques :**
- 🥇 **Top 8 Familles par F1-Score** : Les familles où le modèle performe le mieux
- 📊 **Volume par Famille** : Les familles avec le plus de réclamations
- 📈 **Taux Fondées Réel vs Prédit** : Comparaison de la précision
- 💵 **Montant Moyen par Famille** : Impact financier par famille

**Utilité :**
- Identifier les familles où le modèle est le plus fiable
- Prioriser les efforts d'amélioration sur les familles critiques
- Comprendre la distribution des réclamations

### 2. **false_positives_analysis_2025.png** - Faux Positifs

**4 graphiques :**
- 📊 **Distribution par Tranche de Montant** : Où sont concentrés les FP ?
- 📦 **Boxplot des Montants** : Statistiques descriptives
- 🏢 **Top Familles avec FP** : Quelles familles génèrent le plus d'erreurs ?
- 💰 **Impact Financier par Famille** : Montant total des FP

**Définition - Faux Positif (FP) :**
> Réclamation **prédite comme Fondée** mais **réellement Non Fondée**
> - Impact : Client reçoit une réponse favorable alors qu'il ne devrait pas
> - Coût : 169 DH de traitement "gaspillé"

**Utilité :**
- Quantifier le risque financier des faux positifs
- Identifier les familles nécessitant plus d'attention manuelle
- Optimiser les seuils de décision

### 3. **financial_impact_2025.png** - Impact Financier

**4 graphiques :**
- 📊 **Matrice de Confusion** : Vue d'ensemble des prédictions
- 💵 **Bilan Financier** : Gains vs Coûts (barres)
- 🎯 **Taux d'Automatisation** : Proportion automatisée correctement
- 📋 **Métriques Clés** : Résumé textuel

**Calculs Financiers :**

```
Prix unitaire de traitement : 169 DH

📈 GAINS :
- Réclamations automatisées correctement (TP + TN) × 169 DH

❌ PERTES :
- Faux Positifs (FP) × 169 DH
  (Traitement inutile)

- Faux Négatifs (FN) × 2 × 169 DH
  (Client mécontent + re-traitement = double coût)

✅ GAIN NET = Gains - Pertes
📈 ROI = (Gain Net / Coûts) × 100%
```

**Définitions :**
- **TP (True Positive)** : Fondée prédite Fondée ✅
- **TN (True Negative)** : Non Fondée prédite Non Fondée ✅
- **FP (False Positive)** : Non Fondée prédite Fondée ❌
- **FN (False Negative)** : Fondée prédite Non Fondée ❌ (PIRE ERREUR - client mécontent)

**Utilité :**
- Calculer le ROI du modèle
- Justifier l'investissement en ML
- Identifier les opportunités d'optimisation

---

## 📂 Fichiers Générés

### Visualisations (PNG)

```
outputs/reports/figures/
├── family_success_2025.png              # Succès par famille
├── false_positives_analysis_2025.png    # Analyse FP
└── financial_impact_2025.png            # Impact financier
```

### Données (CSV/JSON)

```
outputs/reports/
├── family_metrics_2025.csv              # Métriques détaillées par famille
├── false_positives_analysis_2025.json   # Stats sur les FP
└── financial_impact_2025.json           # Calculs financiers détaillés
```

---

## 🔧 Personnalisation

### Changer le Prix Unitaire

Éditez la variable dans `visualize_results_2025.py` :

```python
PRIX_UNITAIRE_DH = 169  # Modifier selon vos coûts réels
```

### Modifier les Tranches de Montant

Dans la méthode `analyze_false_positives()` :

```python
df_fp['Tranche_Montant'] = pd.cut(
    df_fp['Montant'],
    bins=[0, 100, 500, 1000, 5000, 10000, np.inf],  # Modifier ici
    labels=['0-100 DH', '100-500 DH', '500-1k DH', '1k-5k DH', '5k-10k DH', '>10k DH']
)
```

### Changer les Couleurs

```python
COLORS = {
    'success': '#2ecc71',   # Vert
    'error': '#e74c3c',     # Rouge
    'warning': '#f39c12',   # Orange
    'info': '#3498db',      # Bleu
    'neutral': '#95a5a6'    # Gris
}
```

---

## 📊 Exemple d'Interprétation

### Résultat Typique

```
💰 QUANTIFICATION FINANCIÈRE
================================

🎯 Performance:
   • Réclamations traitées: 2,500
   • Automatisées correctement: 2,200
   • Taux d'automatisation: 88.0%

❌ Erreurs:
   • Faux Positifs (FP): 150
   • Faux Négatifs (FN): 150

💰 Impact Financier:
   • Prix unitaire: 169 DH
   • Gain brut: 371,800 DH
   • Coût FP: 25,350 DH
   • Coût FN: 50,700 DH

✅ GAIN NET: 295,750 DH
📈 ROI: 388.8%
```

### Interprétation

**Points Positifs :**
- ✅ 88% d'automatisation = Très bon
- ✅ ROI de 388% = Excellent retour sur investissement
- ✅ Gain net ~296k DH sur 2,500 réclamations

**Points d'Attention :**
- ⚠️ 150 FN = Clients mécontents potentiels (priorité #1 à réduire)
- ⚠️ 150 FP = Argent gaspillé mais moins critique
- 💡 Focus : Réduire les FN en priorité

**Actions Recommandées :**
1. Baisser le seuil de décision pour réduire les FN (quitte à augmenter un peu les FP)
2. Ajouter une revue manuelle pour les cas à haute probabilité de FN
3. Améliorer les features pour les familles avec beaucoup de FN

---

## 🆘 Résolution de Problèmes

### Erreur : "Fichier non trouvé"

```bash
❌ Fichier non trouvé: data/raw/reclamations_2025.xlsx
```

**Solution :**
```bash
# Option 1: Exécuter d'abord le pipeline
python main_pipeline.py

# Option 2: Copier vos données
cp /chemin/vers/vos/donnees/reclamations_2025.xlsx data/raw/
```

### Pas de prédictions disponibles

```
⚠️  Pas de prédictions - analyse sur vraies valeurs uniquement
```

**Solution :**
- Normal si vous n'avez pas encore exécuté `main_pipeline.py`
- Le script génèrera quand même des analyses descriptives
- Pour avoir l'analyse complète, exécutez d'abord le pipeline

### Colonnes manquantes

```
❌ Colonne 'Famille Produit' non trouvée
```

**Solution :**
- Vérifiez que vos données contiennent bien les colonnes requises
- Voir `README.md` pour la liste des colonnes nécessaires

---

## 📧 Support

Pour toute question sur les visualisations :
1. Vérifiez que le pipeline principal fonctionne : `python main_pipeline.py`
2. Consultez les logs générés dans `outputs/reports/`
3. Vérifiez les fichiers JSON pour les détails numériques

---

## 🎓 Concepts Clés

### Faux Négatifs (FN) - PIRE ERREUR

**Définition :** Réclamation **fondée** prédite comme **non fondée**

**Impact :**
- Client légitime reçoit un refus
- Client très mécontent → escalade
- Réputation de la banque affectée
- Coût estimé à 2× le traitement normal

**Priorité :** RÉDUIRE EN PRIORITÉ

### Faux Positifs (FP) - Erreur Moins Grave

**Définition :** Réclamation **non fondée** prédite comme **fondée**

**Impact :**
- Client reçoit une réponse favorable
- Perte financière directe
- Coût = traitement inutile (169 DH)

**Priorité :** Acceptable si permet de réduire les FN

### Gain Net

**Formule :**
```
Gain Net = (TP + TN) × 169 - FP × 169 - FN × 2 × 169
```

**Objectif :** Maximiser le gain net, pas seulement l'accuracy !

---

## ✅ Checklist d'Utilisation

- [ ] Pipeline principal exécuté (`main_pipeline.py`)
- [ ] Données 2025 présentes dans `data/raw/`
- [ ] Script de visualisation lancé
- [ ] 3 PNG générés dans `outputs/reports/figures/`
- [ ] CSV/JSON consultés pour détails numériques
- [ ] Interprétation des résultats effectuée
- [ ] Actions d'amélioration identifiées

---

Bon travail ! 🚀
