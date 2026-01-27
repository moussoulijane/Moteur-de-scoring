# 🚀 Guide de Démarrage Rapide

## Installation et Premier Entraînement (5 minutes)

### 1. Installation

```bash
cd production
pip install -r requirements.txt
```

### 2. Entraîner le Modèle

```bash
python train_model.py \
    --train ../data/raw/reclamations_2024.xlsx \
    --test ../data/raw/reclamations_2025.xlsx
```

⏱️ **Durée** : 5-10 minutes (selon la machine)

✅ **Résultat** : Modèles sauvegardés dans `models/`

### 3. Test Rapide - Mode Batch

```bash
python batch_inference.py \
    --input ../data/raw/reclamations_2025.xlsx \
    --output test_predictions.xlsx
```

✅ **Résultat** : `test_predictions.xlsx` avec prédictions

### 4. Test Rapide - Mode API

#### Démarrer l'API

```bash
python src/api/app.py
```

#### Tester avec curl (nouveau terminal)

```bash
# Health check
curl http://localhost:5000/health

# Prédiction simple
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Montant demandé": 5000,
    "Famille Produit": "Cartes",
    "Délai estimé": 30
  }'
```

✅ **Résultat** : Réponse JSON avec prédiction

---

## Utilisation Quotidienne

### Traiter un nouveau fichier Excel

```bash
python batch_inference.py \
    --input mon_fichier.xlsx \
    --apply-rules
```

### Intégration API

```python
import requests

response = requests.post(
    'http://localhost:5000/predict',
    json={
        'Montant demandé': 5000,
        'Famille Produit': 'Cartes',
        'Délai estimé': 30
    }
)

prediction = response.json()
print(prediction['prediction']['Decision_Modele'])
```

---

## Docker (Production)

### Build et Run

```bash
# Build
docker-compose build

# Démarrer l'API
docker-compose up -d api

# Vérifier
curl http://localhost:5000/health
```

### Arrêter

```bash
docker-compose down
```

---

## Troubleshooting

### Erreur : "Model not found"

```bash
# Vérifier que les modèles existent
ls models/

# Si vide, entraîner d'abord
python train_model.py --train ../data/raw/reclamations_2024.xlsx
```

### Erreur : "Column not found"

Vérifier que votre fichier Excel contient au minimum :
- `Montant demandé`
- `Famille Produit`

### Port 5000 déjà utilisé

Modifier le port dans `config/config.yaml` :

```yaml
api:
  port: 8080
```

---

## Configuration Rapide

Modifier `config/config.yaml` pour :
- Changer les algorithmes utilisés
- Ajuster les seuils de décision
- Configurer les règles métier
- Modifier le port de l'API

---

## Prochaines Étapes

1. **Production** : Utiliser Docker pour le déploiement
2. **Monitoring** : Consulter les logs dans `logs/app.log`
3. **Tests** : Lancer `pytest tests/` pour valider
4. **Documentation** : Lire le `README.md` complet

---

## Support

Consulter `README.md` pour documentation complète.
