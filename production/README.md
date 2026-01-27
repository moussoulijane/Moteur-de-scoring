# Moteur de Scoring - Production System

Système de classification automatique des réclamations bancaires avec ML.

## 🎯 Fonctionnalités

- **Entraînement optimisé** : XGBoost et CatBoost avec optimisation Optuna
- **Préprocessing robuste** : Statistiques figées, features disponibles en temps réel
- **Versioning des modèles** : Gestion des versions avec métadonnées
- **Mode Batch** : Traitement de fichiers Excel
- **Mode Temps Réel** : API REST pour inférences instantanées
- **Règles métier** : Application automatique des règles d'affaires
- **Dockerisé** : Déploiement facile avec Docker

## 📁 Structure du Projet

```
production/
├── config/                  # Configuration
│   ├── config.yaml         # Configuration principale
│   └── config_loader.py    # Chargeur de config
├── src/                    # Code source
│   ├── api/               # API REST
│   ├── inference/         # Modules d'inférence
│   ├── models/            # Gestion des modèles
│   ├── preprocessing/     # Préprocessing
│   ├── training/          # Entraînement
│   └── utils/             # Utilitaires
├── models/                 # Modèles sauvegardés
├── tests/                  # Tests unitaires
├── logs/                   # Logs
├── data/                   # Données
├── train_model.py          # Script d'entraînement
├── batch_inference.py      # Script batch
├── requirements.txt        # Dépendances Python
├── Dockerfile             # Image Docker
├── docker-compose.yml     # Orchestration Docker
└── README.md              # Documentation

## 🚀 Installation

### Option 1: Installation locale

```bash
# Clone le projet
cd production

# Créer un environnement virtuel
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate  # Windows

# Installer les dépendances
pip install -r requirements.txt
```

### Option 2: Docker

```bash
# Build l'image
docker-compose build

# Lancer l'API
docker-compose up api
```

## 📊 Entraînement du Modèle

### 1. Préparer les données

Les données doivent être au format Excel avec les colonnes requises :
- `Montant demandé`
- `Famille Produit`
- `Fondee` (pour l'entraînement uniquement)

Colonnes optionnelles (recommandées) :
- `Délai estimé`
- `Catégorie`, `Sous-catégorie`
- `Segment`, `Marché`
- `anciennete_annees`
- `PNB analytique (vision commerciale) cumulé`

### 2. Lancer l'entraînement

```bash
python train_model.py \
    --train ../data/raw/reclamations_2024.xlsx \
    --test ../data/raw/reclamations_2025.xlsx \
    --output models/
```

Le système va :
1. Charger et préparer les données
2. Optimiser les hyperparamètres (Optuna)
3. Entraîner XGBoost et CatBoost
4. Optimiser les seuils de décision
5. Évaluer sur les données de test
6. Sauvegarder les modèles et métadonnées

### 3. Résultats

Les modèles entraînés seront sauvegardés dans `models/` :
- `best_model.pkl` : Meilleur modèle
- `xgboost_model.pkl` : Modèle XGBoost
- `catboost_model.pkl` : Modèle CatBoost
- `preprocessor.pkl` : Préprocesseur
- `thresholds.pkl` : Seuils optimisés
- `model_info.txt` : Informations du modèle

## 🔮 Inférence

### Mode 1: Batch (Fichiers Excel)

Pour traiter un fichier Excel complet :

```bash
python batch_inference.py \
    --input data/nouvelles_reclamations.xlsx \
    --output results/predictions.xlsx \
    --model best \
    --apply-rules
```

**Arguments** :
- `--input` : Fichier Excel d'entrée
- `--output` : Fichier Excel de sortie (optionnel, auto-généré si omis)
- `--model` : Modèle à utiliser (`best`, `xgboost`, `catboost`)
- `--apply-rules` : Appliquer les règles métier
- `--version` : Version du modèle (optionnel)

**Sortie** :
Le fichier Excel contiendra les colonnes originales plus :
- `Probabilite_Fondee` : Probabilité prédite [0-1]
- `Decision_Modele` : Rejet Auto / Audit Humain / Validation Auto
- `Decision_Code` : -1 (Rejet) / 0 (Audit) / 1 (Validation)
- `Raison_Audit` : Raison de l'audit (si règles appliquées)

### Mode 2: Temps Réel (API REST)

#### Démarrer l'API

```bash
# Locale
python src/api/app.py

# Ou avec Docker
docker-compose up api
```

L'API sera accessible sur `http://localhost:5000`

#### Endpoints

**1. Health Check**

```bash
curl http://localhost:5000/health
```

**2. Prédiction Unique**

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Montant demandé": 5000,
    "Famille Produit": "Cartes",
    "Délai estimé": 30,
    "Catégorie": "Débit non autorisé",
    "Segment": "Particuliers",
    "anciennete_annees": 5,
    "PNB analytique (vision commerciale) cumulé": 15000
  }'
```

**Réponse** :
```json
{
  "prediction": {
    "Probabilite_Fondee": 0.75,
    "Decision_Modele": "Validation Auto",
    "Decision_Code": 1
  },
  "timestamp": "2024-01-15T10:30:00"
}
```

**3. Prédiction Batch (petits lots)**

```bash
curl -X POST http://localhost:5000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{
    "complaints": [
      {"Montant demandé": 5000, "Famille Produit": "Cartes", ...},
      {"Montant demandé": 3000, "Famille Produit": "Comptes", ...}
    ]
  }'
```

**Réponse** :
```json
{
  "predictions": [...],
  "summary": {
    "total": 2,
    "Rejet Auto": 0,
    "Audit Humain": 0,
    "Validation Auto": 2
  },
  "timestamp": "2024-01-15T10:30:00"
}
```

**4. Informations du Modèle**

```bash
curl http://localhost:5000/model/info
```

## ⚙️ Configuration

La configuration se trouve dans `config/config.yaml`.

Sections principales :
- `data` : Fichiers de données
- `preprocessing` : Paramètres de préprocessing
- `models` : Configuration des modèles
- `thresholds` : Seuils de décision
- `business_rules` : Règles métier
- `api` : Configuration de l'API

Exemple de modification :

```yaml
business_rules:
  max_validations_per_client_per_year: 2  # Au lieu de 1
  check_amount_vs_pnb: true

api:
  host: "0.0.0.0"
  port: 8080  # Au lieu de 5000
```

## 🧪 Tests

Lancer les tests :

```bash
pytest tests/ -v
```

Avec couverture :

```bash
pytest tests/ --cov=src --cov-report=html
```

## 🐳 Déploiement Docker

### API seulement

```bash
docker-compose up -d api
```

### Avec entraînement

```bash
# Build
docker-compose build

# Entraîner le modèle
docker-compose --profile training up training

# Lancer l'API
docker-compose up -d api
```

## 📈 Système de Décision (3 Zones)

Le système utilise 2 seuils pour créer 3 zones :

```
┌─────────────────────────────────────────────────┐
│  Probabilité ≤ seuil_bas                        │
│  → REJET AUTOMATIQUE (code: -1)                 │
├─────────────────────────────────────────────────┤
│  seuil_bas < Probabilité < seuil_haut           │
│  → AUDIT HUMAIN (code: 0)                       │
├─────────────────────────────────────────────────┤
│  Probabilité ≥ seuil_haut                       │
│  → VALIDATION AUTOMATIQUE (code: 1)             │
└─────────────────────────────────────────────────┘
```

Les seuils sont optimisés automatiquement lors de l'entraînement pour maximiser :
- Le gain financier net
- Le taux d'automatisation
- La précision des décisions

## 📋 Règles Métier

**Règle #1** : Maximum 1 validation automatique par client par année
- La première validation de l'année est autorisée
- Les suivantes sont envoyées en audit humain

**Règle #2** : Montant validé ≤ PNB de l'année dernière
- Si montant > PNB cumulé → audit humain
- Protège contre les validations de montants anormalement élevés

Ces règles s'appliquent automatiquement avec `--apply-rules` (batch) ou `?apply_rules=true` (API).

## 🔄 Versioning des Modèles

Les modèles sont versionnés automatiquement :

```python
from src.inference import ModelManager

manager = ModelManager('models/')

# Lister les versions
versions = manager.list_versions()

# Charger une version spécifique
model, prep, thresh = manager.load_model(
    model_name='best',
    version='v_20240115_103000'
)

# Info sur une version
info = manager.get_version_info('v_20240115_103000')
```

## 📊 Monitoring et Logs

Les logs sont sauvegardés dans `logs/app.log` avec rotation automatique.

Niveau de log configurable dans `config.yaml` :

```yaml
logging:
  level: "INFO"  # DEBUG, INFO, WARNING, ERROR, CRITICAL
```

## 🛠️ Développement

### Structure du Code

- **Modulaire** : Chaque composant est indépendant
- **Typé** : Utilisation de type hints
- **Documenté** : Docstrings complètes
- **Testé** : Tests unitaires et d'intégration

### Ajouter un Nouveau Modèle

1. Créer une méthode dans `src/training/trainer.py` :

```python
def _optimize_lightgbm(self, n_trials: int = 50):
    # Implémentation
    pass
```

2. Ajouter dans `train_models()` :

```python
if 'lightgbm' in algorithms:
    self.models['lightgbm'] = self._optimize_lightgbm(n_trials)
```

3. Mettre à jour la config :

```yaml
models:
  algorithms:
    - xgboost
    - catboost
    - lightgbm
```

## 🤝 Support

Pour toute question ou problème :
1. Vérifier les logs dans `logs/`
2. Consulter la configuration dans `config/config.yaml`
3. Lancer les tests : `pytest tests/ -v`

## 📝 License

Propriétaire - Tous droits réservés
