# Comparaison ML Pipeline V1 vs V2

## 🎯 Vue d'ensemble

Deux versions du pipeline ML pour la classification des réclamations:

| Aspect | V1 (`ml_pipeline`) | V2 (`ml_pipeline_v2`) ⭐ |
|--------|-------------------|------------------------|
| **Objectif** | Exploration maximale | Production-ready |
| **Features** | Toutes colonnes disponibles | Uniquement colonnes temps réel |
| **Robustesse** | Stats recalculées | Stats figées (2024) |
| **Complexité** | ~50-80 features | ~20-30 features |
| **Utilisation** | Recherche, analyse | Production, inférence |

## 📊 Différences détaillées

### 1. Features utilisées

#### V1 - Approche exploratoire
```python
Colonnes utilisées:
✓ Montant demandé
✓ PNB analytique (vision commerciale) cumulé  ← Peut ne pas être disponible en temps réel
✓ Famille Produit
✓ Catégorie
✓ Sous-catégorie
✓ Segment
✓ Marché
✓ anciennete_annees
✓ Toutes autres colonnes numériques
✗ Colonnes "Unnamed"  ← Problématique
```

#### V2 - Approche production ⭐
```python
Colonnes utilisées (uniquement disponibles en production):
✓ Montant demandé
✓ Délai estimé
✓ Famille Produit
✓ Catégorie
✓ Sous-catégorie
✓ Segment
✓ Marché
✓ anciennete_annees

Features calculées:
✓ Taux de fondée par famille (calculé sur 2024, robuste)
✓ Taux de fondée par catégorie (calculé sur 2024)
✓ Taux de fondée par sous-catégorie (calculé sur 2024)
✓ Écart à la médiane famille
✓ Ratios, logs, interactions
```

### 2. Robustesse statistique

#### V1
- Fréquences calculées sur toutes les données
- Pas de seuil minimum de cas
- Statistiques non figées

#### V2 ⭐
```python
RÈGLE: Minimum 30 cas pour calculer un taux de fondée

Exemple:
  Famille "Crédit Auto": 150 cas en 2024
    → Taux fondée = 45% ✓ (utilisé)

  Famille "Assurance Vie": 12 cas en 2024
    → Taux fondée = 80% ✗ (non utilisé, trop peu de cas)
    → Utilise taux global = 42% (fallback)

Avantage: Évite le surapprentissage sur catégories rares
```

### 3. Gestion des statistiques

#### V1
```python
# À chaque inférence, recalcule les fréquences
for col in categorical_cols:
    X[f'{col}_freq'] = X[col].map(
        X[col].value_counts().to_dict()  # ← Recalculé
    )
```

**Problème:** Les statistiques changent entre train et test !

#### V2 ⭐
```python
# Fit (sur 2024):
self.family_stats = {
    'taux': family_grouped['taux_fondee'].to_dict(),  # ← Sauvegardé
    'count': family_grouped['count'].to_dict(),
    'taux_global': 0.42
}

# Transform (sur 2025 ou production):
df['taux_fondee_famille'] = df['Famille Produit'].map(
    self.family_stats['taux']  # ← Réutilisé (figé)
).fillna(self.family_stats['taux_global'])
```

**Avantage:** Statistiques figées, pas de data leakage

### 4. Features calculées

#### V1
```python
Features:
- ratio_pnb_montant  ← PNB peut ne pas être disponible
- ecart_mediane_famille
- log_montant
- log_pnb  ← PNB peut ne pas être disponible
- log_anciennete
- montant_x_anciennete
- pnb_x_anciennete  ← PNB peut ne pas être disponible
```

#### V2 ⭐
```python
Features (toutes disponibles en production):
- taux_fondee_famille ⭐ (nouveau)
- taux_fondee_categorie ⭐ (nouveau)
- taux_fondee_souscategorie ⭐ (nouveau)
- taux_fondee_segment ⭐ (nouveau)
- count_famille ⭐ (nouveau - robustesse)
- ecart_mediane_famille
- ratio_montant_delai
- log_montant
- log_delai
- log_anciennete
- montant_x_anciennete
- delai_x_anciennete
- montant_x_delai
- montant_x_taux_famille ⭐ (interaction)
```

### 5. Nombre de features

| Métrique | V1 | V2 |
|----------|----|----|
| Features de base | ~8-10 | 8 |
| Features calculées | ~40-70 | ~15-25 |
| **Total** | **~50-80** | **~20-30** |
| Complexité | Élevée | Modérée |

**V2 = Plus simple, plus robuste, plus interprétable**

## 🚀 Quand utiliser quelle version ?

### Utilisez V1 si:
- ✅ Vous faites de la **recherche exploratoire**
- ✅ Vous voulez **tester toutes les features** possibles
- ✅ Vous avez **accès à toutes les colonnes** en production
- ✅ Vous voulez comparer **3 modèles** (XGBoost, RF, CatBoost)
- ✅ Vous êtes en phase d'**analyse** et d'**expérimentation**

### Utilisez V2 si: ⭐ (RECOMMANDÉ POUR PRODUCTION)
- ✅ Vous déployez en **production**
- ✅ Vous avez des **contraintes temps réel**
- ✅ Vous voulez un modèle **simple et robuste**
- ✅ Vous voulez éviter le **data leakage**
- ✅ Vous avez besoin de **statistiques figées**
- ✅ Vous voulez un modèle **interprétable**

## 📈 Performances comparées

### Métriques attendues (à vérifier après entraînement)

| Métrique | V1 | V2 | Commentaire |
|----------|----|----|-------------|
| F1-Score | ~0.995 | ~0.993-0.995 | V2 légèrement moins, mais plus robuste |
| Accuracy | ~0.996 | ~0.994-0.996 | Très similaire |
| ROC-AUC | ~0.999 | ~0.998-0.999 | Excellente dans les deux cas |
| Taux automatisation | 85-90% | 85-90% | Similaire |
| Gain NET | Élevé | Similaire ou meilleur | Dépend des seuils |
| **Production-ready** | ❌ | ✅ | **V2 gagnant** |
| **Robustesse** | ⚠️ | ✅ | **V2 gagnant** |
| **Simplicité** | ❌ | ✅ | **V2 gagnant** |

## 🔧 Migration V1 → V2

Si vous utilisez actuellement V1 et voulez migrer vers V2:

### Étape 1: Vérifier les colonnes disponibles
```python
# Vérifiez que vous avez ces colonnes en production:
required_columns = [
    'Montant demandé',
    'Délai estimé',
    'Famille Produit',
    'Catégorie',
    'Sous-catégorie',
    'Segment',
    'Marché',
    'anciennete_annees'
]
```

### Étape 2: Entraîner le modèle V2
```bash
cd ml_pipeline_v2
python model_comparison_v2.py
```

### Étape 3: Comparer les performances
```bash
# Comparer rapport V1 et V2
diff outputs/production/rapport_comparison.txt \
     outputs/production_v2/rapport_v2.txt
```

### Étape 4: Tester l'inférence
```bash
# Tester sur données 2025
python ml_pipeline_v2/inference_v2.py \
  --input_file data/raw/reclamations_2025.xlsx \
  --output_file test_v2.xlsx
```

### Étape 5: Valider et déployer
- ✅ Vérifier que les performances sont acceptables
- ✅ Tester sur quelques cas réels
- ✅ Déployer le modèle V2 en production
- ✅ Monitorer les performances

## 💡 Recommandations

### Pour la recherche et l'analyse
```
Utilisez V1 (ml_pipeline)
→ Exploration maximale des features
→ Comparaison de modèles
→ Analyse d'interprétabilité
```

### Pour la production
```
Utilisez V2 (ml_pipeline_v2) ⭐
→ Features disponibles en temps réel
→ Statistiques robustes et figées
→ Modèle simple et performant
→ Pas de data leakage
```

### Workflow idéal
```
1. Exploration avec V1
   ├─ Identifier les features importantes
   ├─ Comprendre les patterns
   └─ Tester différents modèles

2. Production avec V2
   ├─ Features production-ready uniquement
   ├─ Statistiques robustes (≥30 cas)
   ├─ Modèle CatBoost optimisé
   └─ Inférence temps réel
```

## 📚 Documentation

### V1
- `ml_pipeline/README_ANALYSE.md` - Guide d'analyse
- `ml_pipeline/model_comparison.py` - Comparaison de modèles
- `ml_pipeline/analyze_results.py` - Analyse XGBoost
- `ml_pipeline/analyze_results_catboost.py` - Analyse CatBoost
- `ml_pipeline/model_interpretability.py` - Interprétabilité

### V2 ⭐
- `ml_pipeline_v2/README_V2.md` - Documentation complète
- `ml_pipeline_v2/preprocessor_v2.py` - Preprocessing robuste
- `ml_pipeline_v2/model_comparison_v2.py` - Entraînement
- `ml_pipeline_v2/inference_v2.py` - Inférence production

## 🎓 Résumé

| Critère | V1 | V2 |
|---------|----|----|
| **Complexité** | 🔴 Élevée | 🟢 Modérée |
| **Robustesse** | 🟡 Moyenne | 🟢 Élevée |
| **Production** | 🔴 Non | 🟢 Oui |
| **Performance** | 🟢 Excellente | 🟢 Excellente |
| **Interprétabilité** | 🟡 Moyenne | 🟢 Élevée |
| **Maintenance** | 🔴 Difficile | 🟢 Facile |

**Conclusion: V2 est recommandé pour la production** ⭐
