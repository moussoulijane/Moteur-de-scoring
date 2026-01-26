# 📊 Guide de Génération de la Présentation Finale

Ce guide explique comment générer tous les graphiques nécessaires pour votre présentation finale.

## 📁 Structure de la Présentation

### **PARTIE 1: État des Lieux**
- Répartition des réclamations par marché (Nombre et Montant)
- Regroupement: "Particulier" + "Professionnel" = "Particulier & Professionnel"
- Évolution 2023-2024-2025

### **PARTIE 2: Architecture du Modèle**
- Les 3 piliers (Type Réclamation, Risque, Signalétique)
- Couche Analytique (IA) - Optimisation des poids
- Couche Décisionnelle (Modèle + 2 règles métier)
- 3 Décisions finales (Rejet Auto, Audit Humain, Validation Auto)

### **PARTIE 3: Résultats et Monitoring**
- Métriques de performance (2023 et 2025)
- Gain NET par année
- Impact des règles métier:
  - Règle #1: Maximum 1 validation par client par an
  - Règle #2: Montant validé ≤ PNB année dernière

---

## 🚀 Étapes de Génération

### **Étape 1: Génération Partie 1 et 2 (État des lieux + Architecture)**

Ces graphiques ne nécessitent PAS de données scorées.

```bash
python ml_pipeline_v2/generate_presentation_final.py \
  --data_2023 data/reclamations_2023.xlsx \
  --data_2024 data/reclamations_2024.xlsx \
  --data_2025 data/reclamations_2025.xlsx
```

**Fichiers générés:**
- `outputs/presentation_final/P1_etat_lieux_marche.png` - État des lieux par marché
- `outputs/presentation_final/P2_architecture_modele.png` - Architecture claire du modèle

---

### **Étape 2: Scorer les données (si pas déjà fait)**

Si vos fichiers n'ont pas encore les colonnes `Decision_Modele` et `Raison_Audit`, scorez-les:

```bash
# Scorer 2023
python ml_pipeline_v2/inference_v2.py \
  --input data/reclamations_2023.xlsx \
  --output outputs/predictions_2023_avec_regles.xlsx

# Scorer 2025
python ml_pipeline_v2/inference_v2.py \
  --input data/reclamations_2025.xlsx \
  --output outputs/predictions_2025_avec_regles.xlsx
```

---

### **Étape 3: Génération Partie 3 (Résultats + Monitoring)**

Avec les données scorées qui contiennent:
- `Decision_Modele`
- `Raison_Audit`
- `Fondée`

```bash
python ml_pipeline_v2/generate_monitoring_regles.py \
  --data_2023 outputs/predictions_2023_avec_regles.xlsx \
  --data_2025 outputs/predictions_2025_avec_regles.xlsx
```

**Fichier généré:**
- `outputs/presentation_final/P3_resultats_monitoring.png` - Résultats et monitoring

---

## 📊 Contenu Détaillé des Graphiques

### **P1: État des Lieux - Répartition par Marché**

4 graphiques:
1. **Répartition en NOMBRE** - Barres groupées par année (2023, 2024, 2025)
2. **Répartition en MONTANT** - Barres groupées par année (en Millions DH)
3. **Évolution NOMBRE** - Courbes montrant l'évolution temporelle
4. **Évolution MONTANT** - Courbes montrant l'évolution temporelle

Les marchés "Particulier" et "Professionnel" sont automatiquement regroupés.

---

### **P2: Architecture du Modèle**

Schéma vertical clair montrant:

1. **Les 3 Piliers** (niveau haut)
   - Pilier 1: Type Réclamation (Famille, Catégorie, Sous-catégorie)
   - Pilier 2: Risque (Montant, Délai, Ratio/PNB)
   - Pilier 3: Signalétique (PNB, Ancienneté, Segment/Marché)

2. **Couche Analytique** (niveau moyen)
   - "Optimisation automatique des poids de chaque pilier"
   - Pas de mention des noms de modèles (XGBoost, CatBoost)

3. **Couche Décisionnelle** (niveau bas)
   - Score du Modèle + Règles Métier
   - Règle #1: Maximum 1 validation/client/an
   - Règle #2: Montant validé ≤ PNB année dernière

4. **3 Décisions Finales**
   - ❌ Rejet Auto
   - 🔍 Audit Humain
   - ✅ Validation Auto

---

### **P3: Résultats et Monitoring**

Graphique 3×3 comprenant:

**Ligne 1 - Vue d'ensemble:**
- Métriques de performance (Précision, Rappel, F1-Score) pour 2023 et 2025
- Gain NET par année (Gain Brut vs Gain NET)
- Taux d'automatisation par année

**Ligne 2 - Impact règles 2023:**
- Nombre de cas convertis par chaque règle
- Montants concernés par les règles
- Récapitulatif détaillé

**Ligne 3 - Impact règles 2025:**
- Nombre de cas convertis par chaque règle
- Montants concernés par les règles
- Récapitulatif détaillé

---

## 📐 Calcul du Gain NET

Le gain est calculé selon la formule de `model_comparison_v2.py`:

```
Gain Brut = Nombre dossiers automatisés × 169 DH

Perte FP (Faux Positifs) = Somme des montants accordés à tort

Perte FN (Faux Négatifs) = 2 × Somme des montants refusés à tort

GAIN NET = Gain Brut - Perte FP - Perte FN
```

---

## ⚠️ Prérequis

### Pour Partie 1 et 2:
- Fichiers Excel bruts avec colonnes:
  - `Marché`
  - `Montant demandé`

### Pour Partie 3:
- Fichiers Excel scorés avec colonnes:
  - `Decision_Modele`
  - `Raison_Audit`
  - `Fondée`
  - `Montant demandé`

---

## 🎨 Personnalisation

Les graphiques sont générés en haute résolution (300 DPI) et sont prêts pour insertion dans PowerPoint.

Si vous souhaitez modifier:
- **Couleurs**: Modifiez les codes couleurs dans les scripts
- **Taille**: Changez `figsize=(18, 12)` dans les scripts
- **Titres**: Modifiez les `fig.suptitle()` dans les scripts

---

## 📝 Notes Importantes

1. **Regroupement automatique**: Les marchés "Particulier" et "Professionnel" sont automatiquement fusionnés en "Particulier & Professionnel"

2. **Architecture simplifiée**: La Partie 2 ne mentionne PAS les noms des modèles (XGBoost/CatBoost), seulement "Couche Analytique (Intelligence Artificielle)"

3. **Règles métier**: Les 2 règles sont clairement identifiées et leur impact est quantifié

4. **Gain NET**: Utilise la vraie formule avec Perte FP et FN (pas le simple coût évité)

---

## ❓ Dépannage

### "Colonne 'Marché' manquante"
→ Vérifiez l'orthographe exacte de la colonne dans vos fichiers Excel

### "Pas de décisions - Graphique ignoré"
→ Scorez d'abord vos données avec `inference_v2.py`

### "Colonne 'Raison_Audit' manquante"
→ Utilisez `inference_v2.py` qui ajoute automatiquement cette colonne

---

## 📂 Organisation des Fichiers

```
outputs/
└── presentation_final/
    ├── P1_etat_lieux_marche.png          # Partie 1
    ├── P2_architecture_modele.png         # Partie 2
    └── P3_resultats_monitoring.png        # Partie 3
```

---

## ✅ Checklist Finale

- [ ] Partie 1 générée (État des lieux)
- [ ] Partie 2 générée (Architecture)
- [ ] Données 2023 et 2025 scorées
- [ ] Partie 3 générée (Résultats + Monitoring)
- [ ] Tous les graphiques en 300 DPI
- [ ] Vérification visuelle des 3 graphiques

---

**Bonne présentation! 🎉**
