# 📊 Guide de Génération de la Présentation

Ce guide vous explique comment générer votre présentation complète pour l'opérationnalisation du modèle de scoring.

## 🎯 Vue d'ensemble

Vous disposez de **2 scripts complémentaires** :

1. **`generate_presentation_visuals.py`** - Génère les 6 graphiques PNG
2. **`generate_powerpoint.py`** - Génère la présentation PowerPoint avec contenu textuel

## 📋 Prérequis

Assurez-vous d'avoir :
- ✅ Fichier Excel 2024 avec colonne `Fondée`
- ✅ Fichier Excel 2025 avec inférence déjà effectuée (colonnes `Decision_Modele`, `Probabilite_Fondee`)
- ✅ (Optionnel) Fichier Excel 2023 avec colonne `Fondée`

## 🚀 Étapes de Génération

### Étape 1: Générer les Graphiques

```bash
python ml_pipeline_v2/generate_presentation_visuals.py \
  --data_2024 data/raw/reclamations_2024.xlsx \
  --data_2025 predictions_2025_v2.xlsx \
  --data_2023 data/raw/reclamations_2023.xlsx
```

**Résultat** : 6 fichiers PNG dans `outputs/presentation/` :
- `01_evolution_volume_montant.png`
- `02_fondee_vs_non_fondee.png`
- `03_repartition_famille.png`
- `04_repartition_marche.png`
- `05_architecture_modele.png`
- `06_resultats_2025_gain.png`

### Étape 2: Générer la Présentation PowerPoint

```bash
python ml_pipeline_v2/generate_powerpoint.py --output presentation_scoring.pptx
```

**Résultat** : Fichier `outputs/presentation/presentation_scoring.pptx` avec :
- ✅ 13 slides structurées
- ✅ Tout le contenu textuel
- ✅ Placeholders pour les graphiques

### Étape 3: Insérer les Graphiques

Ouvrez `presentation_scoring.pptx` dans PowerPoint et :

1. **Slide 3** - Supprimez le placeholder gris et insérez `01_evolution_volume_montant.png`
2. **Slide 4** - Supprimez le placeholder gris et insérez `02_fondee_vs_non_fondee.png`
3. **Slide 5** - Supprimez le placeholder gris et insérez `03_repartition_famille.png`
4. **Slide 6** - Supprimez le placeholder gris et insérez `04_repartition_marche.png`
5. **Slide 7** - Supprimez le placeholder gris et insérez `05_architecture_modele.png`
6. **Slide 9** - Supprimez le placeholder gris et insérez `06_resultats_2025_gain.png`

**Astuce** : Pour chaque image, utilisez "Insérer > Image" et ajustez la taille pour remplir l'espace disponible.

## 📊 Structure de la Présentation

### Slides 1-2: Introduction
- Page de titre
- Agenda

### Slides 3-6: Section I - État des Lieux
- **Slide 3**: Évolution volume et montant 2023-2025
- **Slide 4**: Analyse fondée vs non fondée
- **Slide 5**: Répartition par famille de produit
- **Slide 6**: Répartition par marché

### Slides 7-8: Section II - Présentation du Modèle
- **Slide 7**: Schéma d'architecture (diagramme visuel)
- **Slide 8**: Détail textuel de l'architecture
  - 3 Piliers (Type réclamation, Risque, Signalétique)
  - Couche analytique (IA avec optimisation)
  - Couche décisionnelle (Modèle + 2 règles métier)

### Slides 9-11: Section III - Résultats & Gains
- **Slide 9**: Résultats 2025 et calcul du gain
- **Slide 10**: Bénéfices quantifiables et qualitatifs
- **Slide 11**: Recommandations et prochaines étapes

### Slides 12-13: Conclusion
- **Slide 12**: Messages clés et call-to-action
- **Slide 13**: Questions

## 🎨 Personnalisation

### Couleurs utilisées
- **Titre** : Bleu foncé #2C3E50
- **Accent** : Vert #2ECC71
- **Sections** : Codes couleur par pilier
  - Bleu #3498DB (Type réclamation)
  - Rouge #E74C3C (Risque)
  - Vert #2ECC71 (Signalétique)
  - Violet #9B59B6 (IA)
  - Orange #F39C12 (Décisionnel)

### Polices
- Titres : 32-44pt, gras
- Contenu : 14-20pt
- Notes : 12pt, italique

## 💡 Conseils de Présentation

### Pour chaque section, insistez sur :

**État des lieux** :
- Tendances claires (hausse/baisse)
- Taux de fondée stable/variable
- Familles et marchés principaux

**Architecture du modèle** :
- **3 Piliers** = Vision holistique
- **IA** = Optimisation automatique (pas de biais humain)
- **Règles métier** = Garde-fous business

**Résultats** :
- Taux d'automatisation élevé
- Gain NET calculé précisément
- ROI positif

## ⚠️ Points d'attention

1. **Fichier 2025** : Doit OBLIGATOIREMENT contenir les résultats d'inférence
   ```bash
   # Si pas encore fait, exécutez d'abord :
   python ml_pipeline_v2/inference_v2.py \
     --input_file data/raw/reclamations_2025.xlsx \
     --output_file predictions_2025_v2.xlsx \
     --apply_rule
   ```

2. **Colonne Fondée** : Doit être présente dans les fichiers 2023/2024 pour les analyses fondée vs non fondée

3. **Colonnes requises** :
   - Montant demandé
   - Famille Produit
   - Marché
   - Date de Qualification
   - PNB analytique (vision commerciale) cumulé

## 📝 Checklist Finale

Avant la présentation :
- [ ] Tous les graphiques sont insérés
- [ ] Les images sont bien dimensionnées
- [ ] Les chiffres sont cohérents entre slides
- [ ] Les recommandations sont adaptées à votre contexte
- [ ] Le call-to-action est clair (GO/NO-GO)
- [ ] Durée : 20-25 minutes prévu

## 🔧 Dépannage

**Problème** : Erreur "Colonnes manquantes" lors de la génération des graphiques
**Solution** : Vérifiez que vos fichiers Excel contiennent toutes les colonnes requises

**Problème** : Graphiques vides ou incomplets
**Solution** : Vérifiez que les données 2023-2025 sont bien formatées (dates, montants en numérique)

**Problème** : Présentation PowerPoint ne s'ouvre pas
**Solution** : Installez python-pptx : `pip install python-pptx`

## 📞 Support

Pour toute question sur :
- La structure de la présentation → Consultez ce guide
- Les graphiques générés → Vérifiez `outputs/presentation/rapport_presentation_*.txt`
- Les résultats du modèle → Consultez `outputs/production_v2/`

---

**Bonne présentation ! 🎉**
