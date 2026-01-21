# ML Pipeline V2 - Features Production-Ready

## 🎯 Objectif

Version améliorée du pipeline ML qui utilise **uniquement des features disponibles en temps réel** pour la production.

## 🔑 Différences clés avec V1

### V1 (ml_pipeline)
- Utilisait toutes les colonnes disponibles
- Certaines features (comme PNB cumulé) peuvent ne pas être disponibles en production
- Encodage des fréquences recalculé à chaque fois

### V2 (ml_pipeline_v2) ✅
- **Features production-ready uniquement**
- **Statistiques calculées sur 2024 et réutilisées** (taux de fondée par famille, catégorie, etc.)
- **Robustesse statistique** : seulement les catégories avec ≥30 cas
- **Pas de colonnes "Unnamed"**
- **Optimisé pour l'inférence temps réel**
- **Inclut PNB cumulé** (du dernier semestre)

## 📊 Features utilisées

### Colonnes de base (requises)
- `Montant demandé` ✅
- `Famille Produit` ✅
- `Délai estimé` ✅
- `Catégorie` ✅
- `Sous-catégorie` ✅
- `Segment` ✅
- `Marché` ✅
- `anciennete_annees` ✅
- `PNB analytique (vision commerciale) cumulé` ✅ (dernier semestre)

### Features calculées (automatiques)

#### 1. Taux de fondée (statistiquement robustes)
Calculés sur 2024 avec minimum 30 cas:
- `taux_fondee_famille` - Taux de réclamations fondées par famille
- `taux_fondee_categorie` - Taux par catégorie
- `taux_fondee_souscategorie` - Taux par sous-catégorie
- `taux_fondee_segment` - Taux par segment
- `count_famille` - Nombre de cas dans la famille (pour évaluer la robustesse)

#### 2. Écarts et ratios
- `ecart_mediane_famille` - Écart du montant à la médiane de la famille
- `ecart_pnb_mediane_famille` - Écart du PNB à la médiane de la famille
- `ratio_montant_delai` - Montant / Délai
- `ratio_montant_pnb` - Montant / PNB

#### 3. Transformations log
- `log_montant` - Log(1 + Montant demandé)
- `log_delai` - Log(1 + Délai estimé)
- `log_anciennete` - Log(1 + anciennete_annees)
- `log_pnb` - Log(1 + PNB cumulé)

#### 4. Interactions
- `montant_x_anciennete` - Montant × Ancienneté
- `delai_x_anciennete` - Délai × Ancienneté
- `montant_x_delai` - Montant × Délai
- `pnb_x_anciennete` - PNB × Ancienneté
- `montant_x_taux_famille` - Montant × Taux de fondée famille
- `pnb_x_taux_famille` - PNB × Taux de fondée famille

#### 5. Fréquences catégorielles
- `Marché_freq`
- `Segment_freq`
- `Famille Produit_freq`
- `Catégorie_freq`
- `Sous-catégorie_freq`

## 🚀 Workflow

### 1. Entraînement

```bash
cd ml_pipeline_v2
python model_comparison_v2.py
```

**Ce que fait ce script:**
- Charge données 2024 et 2025
- Calcule les **taux de fondée** sur 2024 (statistiquement renforcés)
- Entraîne CatBoost avec Optuna (50 trials)
- Optimise les **2 seuils** pour 3 zones de décision
- Évalue sur 2025
- Sauvegarde:
  - `outputs/production_v2/models/catboost_model_v2.pkl`
  - `outputs/production_v2/models/preprocessor_v2.pkl`
  - `outputs/production_v2/predictions/predictions_2025_v2.pkl`
  - `outputs/production_v2/rapport_v2.txt`

### 2. Inférence sur nouvelles données

```bash
python ml_pipeline_v2/inference_v2.py --input_file path/to/new_data.xlsx
```

**Avec règle métier (1 validation auto par client par an):**

```bash
python ml_pipeline_v2/inference_v2.py --input_file path/to/new_data.xlsx --apply_rule
```

**Avec fichier de sortie personnalisé:**

```bash
python ml_pipeline_v2/inference_v2.py \
  --input_file path/to/new_data.xlsx \
  --output_file path/to/results.xlsx \
  --apply_rule
```

### 3. Analyse des profils de réclamations

Avant ou après l'inférence, analysez les profils pour mieux comprendre vos données:

**Analyse sans prédictions (exploration initiale):**

```bash
python ml_pipeline_v2/analyze_claims_profile.py --input_file path/to/data.xlsx
```

**Analyse avec prédictions (après inférence):**

```bash
python ml_pipeline_v2/analyze_claims_profile.py \
  --input_file path/to/predictions.xlsx \
  --with_predictions
```

**Ce que fait ce script:**
- 📊 **Distributions**: Montant, délai, ancienneté, PNB, ratios
- 🏢 **Analyse par famille**: Montant moyen, volume, PNB moyen, délai moyen (Top 15)
- 🔗 **Corrélations**: Montant vs ancienneté, montant vs PNB, délai vs montant, PNB vs ancienneté
- 🎯 **Profils par décision** (si prédictions): Distribution par famille, montants moyens, ancienneté

**Graphiques générés:**
- `01_distributions.png` - 6 graphiques de distribution
- `02_analyse_famille.png` - 4 analyses par famille
- `03_correlations.png` - 4 scatter plots avec corrélations
- `04_profils_decisions.png` - 4 analyses par décision (si `--with_predictions`)
- `rapport_profils_*.txt` - Rapport texte récapitulatif

**Cas d'usage:**
- ✅ Comprendre les profils de réclamations avant de prédire
- ✅ Identifier les familles à fort montant/PNB
- ✅ Analyser les corrélations entre variables
- ✅ Interpréter les prédictions du modèle
- ✅ Détecter des patterns ou anomalies


## 📋 Système de décision (3 zones)

Le modèle utilise **2 seuils** optimisés:

| Zone | Condition | Décision | Code |
|------|-----------|----------|------|
| **Zone 1** | `prob ≤ seuil_bas` | **Rejet Auto** | 0 |
| **Zone 2** | `seuil_bas < prob < seuil_haut` | **Audit Humain** | -1 |
| **Zone 3** | `prob ≥ seuil_haut` | **Validation Auto** | 1 |

**Critères d'optimisation:**
- Maximiser le gain NET
- Contraintes:
  - Précision Rejet ≥ 95%
  - Précision Validation ≥ 93%

## 💰 Calcul financier

- **Gain brut** = (Rejet Auto + Validation Auto) × 169 DH
- **Perte FP** = Somme des montants des faux positifs
- **Perte FN** = 2 × Somme des montants des faux négatifs
- **Gain NET** = Gain brut - Perte FP - Perte FN

## 📊 Robustesse statistique

Les **taux de fondée** sont calculés uniquement pour les catégories ayant **≥30 cas** dans les données 2024.

**Pourquoi 30 cas minimum ?**
- Assure une stabilité statistique
- Évite le surapprentissage sur des catégories rares
- Les nouvelles catégories utilisent le **taux global** comme fallback

**Exemple:**
```
Famille Produit "Crédit Auto":
  - 2024: 150 cas, 45% fondées → Taux = 0.45 (utilisé)

Famille Produit "Assurance Vie":
  - 2024: 12 cas, 80% fondées → Trop peu de cas
  - Utilise taux global: 0.42 (fallback)
```

## 🔧 Règle métier

**Règle:** Un client ne peut avoir qu'**une seule validation automatique par année**.

Quand `--apply_rule` est activé:
1. Les réclamations sont triées par `Date de Qualification`
2. Pour chaque client/année, seule la **première validation auto** est gardée
3. Les suivantes sont converties en **Audit Humain**

## 📁 Structure des fichiers

```
ml_pipeline_v2/
├── preprocessor_v2.py           # Preprocessing production-ready
├── model_comparison_v2.py       # Entraînement et évaluation
├── inference_v2.py              # Script d'inférence
├── analyze_claims_profile.py   # Analyse exploratoire des profils
└── README_V2.md                 # Ce fichier

outputs/production_v2/
├── models/
│   ├── catboost_model_v2.pkl    # Modèle entraîné
│   └── preprocessor_v2.pkl      # Preprocessor avec stats 2024
├── predictions/
│   └── predictions_2025_v2.pkl  # Prédictions et seuils optimaux
└── rapport_v2.txt               # Rapport de performance

outputs/profile_analysis/
├── 01_distributions.png         # Distributions des variables
├── 02_analyse_famille.png       # Métriques par famille
├── 03_correlations.png          # Corrélations entre variables
├── 04_profils_decisions.png     # Profils par décision (optionnel)
└── rapport_profils_*.txt        # Rapport récapitulatif
```

## ✅ Avantages de la V2

1. **Production-ready** : Toutes les features sont disponibles en temps réel
2. **Robustesse** : Statistiques calculées sur 2024, pas de data leakage
3. **Stabilité** : Seuil minimum de 30 cas pour les statistiques
4. **Simplicité** : Pas de colonnes "Unnamed" ou de features obscures
5. **Transparence** : Features explicites et interprétables
6. **Performance** : Optimisation avec Optuna + 2 seuils
7. **Flexibilité** : Règle métier optionnelle

## 🎓 Utilisation recommandée

1. **Entraînement initial:** Lancez `model_comparison_v2.py` pour créer le modèle
2. **Validation:** Vérifiez les performances dans `rapport_v2.txt`
3. **Test inférence:** Testez avec quelques lignes de 2025
4. **Production:** Utilisez `inference_v2.py` sur de nouvelles données
5. **Monitoring:** Recalculez périodiquement les statistiques sur nouvelles données historiques

## 🔄 Mise à jour du modèle

Pour mettre à jour le modèle avec de nouvelles données historiques:

1. Remplacez `data/raw/reclamations_2024.xlsx` avec données les plus récentes
2. Relancez `python ml_pipeline_v2/model_comparison_v2.py`
3. Les nouveaux taux de fondée seront recalculés
4. Le modèle sera ré-entraîné avec les nouvelles statistiques

## 📞 Support

Pour toute question ou amélioration, consultez:
- Le code source avec commentaires détaillés
- Les rapports générés dans `outputs/production_v2/`
- Les logs de console lors de l'exécution
