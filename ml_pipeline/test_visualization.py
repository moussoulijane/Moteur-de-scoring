"""
Script de test rapide pour la visualisation
Génère des données synthétiques pour démonstration
"""
import sys
sys.path.append('src')

import pandas as pd
import numpy as np
import joblib
from pathlib import Path

print("🧪 Test du Script de Visualisation")
print("="*60)

# Vérifier si les vraies données existent
data_path = 'data/raw/reclamations_2025.xlsx'
predictions_path = 'outputs/models/predictions_2025.pkl'

if Path(data_path).exists():
    print(f"✅ Données 2025 trouvées: {data_path}")

    if Path(predictions_path).exists():
        print(f"✅ Prédictions trouvées: {predictions_path}")
        print("\n💡 Vous pouvez lancer directement:")
        print("   python visualize_results_2025.py")
    else:
        print(f"⚠️  Prédictions non trouvées: {predictions_path}")
        print("\n💡 Options:")
        print("   1. Exécuter d'abord: python main_pipeline.py")
        print("   2. Ou lancer quand même pour analyse descriptive:")
        print("      python visualize_results_2025.py")
else:
    print(f"❌ Données 2025 non trouvées: {data_path}")
    print("\n💡 Solutions:")
    print("   1. Copier vos données:")
    print("      cp /chemin/vers/reclamations_2025.xlsx data/raw/")
    print("   2. Ou générer des données synthétiques:")
    print("      python -c 'from utils.data_generator import *; gen = RealColumnDataGenerator(); gen.generate_and_save()'")

print("\n" + "="*60)
print("🎯 Pour tester avec données synthétiques:")
print("="*60)

# Créer données synthétiques minimales pour démo
print("\n📊 Création de données synthétiques de démonstration...")

np.random.seed(42)
n_samples = 500

# Générer données
familles = ['Monétique', 'Crédit', 'Épargne', 'Assurance', 'Transfert']
segments = ['Grand Public', 'Particuliers', 'Premium', 'VVIP']

df_demo = pd.DataFrame({
    'No Demande': range(1, n_samples + 1),
    'Famille Produit': np.random.choice(familles, n_samples),
    'Segment': np.random.choice(segments, n_samples),
    'Montant demandé': np.random.lognormal(6, 1.5, n_samples),  # ~500 DH médiane
    'PNB analytique (vision commerciale) cumulé': np.random.lognormal(9, 1, n_samples),
    'anciennete_annees': np.random.exponential(5, n_samples),
    'Fondee': np.random.randint(0, 2, n_samples)
})

# Générer prédictions simulées (avec quelques erreurs)
y_true = df_demo['Fondee'].values
# Modèle avec ~85% accuracy
y_pred = y_true.copy()
# Introduire 15% d'erreurs
error_idx = np.random.choice(n_samples, size=int(0.15 * n_samples), replace=False)
y_pred[error_idx] = 1 - y_pred[error_idx]

# Probabilités cohérentes
y_prob = np.where(y_pred == 1,
                  np.random.beta(8, 2, n_samples),  # Si prédit 1, proba haute
                  np.random.beta(2, 8, n_samples))  # Si prédit 0, proba basse

# Sauvegarder
Path('data/raw').mkdir(parents=True, exist_ok=True)
Path('outputs/models').mkdir(parents=True, exist_ok=True)

demo_data_path = 'data/raw/reclamations_2025_DEMO.xlsx'
demo_pred_path = 'outputs/models/predictions_2025_DEMO.pkl'

df_demo.to_excel(demo_data_path, index=False)
joblib.dump({
    'y_true': y_true,
    'y_pred': y_pred,
    'y_prob': y_prob
}, demo_pred_path)

print(f"✅ Données démo créées: {demo_data_path}")
print(f"✅ Prédictions démo créées: {demo_pred_path}")

print("\n🎨 Pour visualiser ces données de démo:")
print("="*60)
print(f"""
# Modifier temporairement visualize_results_2025.py ligne ~700:
# Changer:
#   data_path = 'data/raw/reclamations_2025.xlsx'
#   predictions_path = 'outputs/models/predictions_2025.pkl'
# En:
#   data_path = '{demo_data_path}'
#   predictions_path = '{demo_pred_path}'

# Puis lancer:
python visualize_results_2025.py
""")

print("✅ Test terminé!")
print("="*60)
