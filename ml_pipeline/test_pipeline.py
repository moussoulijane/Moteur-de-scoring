"""
Test rapide du pipeline avec les vraies colonnes
"""
import sys
sys.path.append('src')

import pandas as pd
from preprocessing.preprocessor import RobustPreprocessor
from evaluation.family_analysis import FamilyProductAnalyzer

# Charger les données
print("📂 Chargement des données...")
df_2024 = pd.read_excel('data/raw/reclamations_2024.xlsx')
df_2025 = pd.read_excel('data/raw/reclamations_2025.xlsx')

print(f"✅ 2024: {len(df_2024)} lignes, {len(df_2024.columns)} colonnes")
print(f"✅ 2025: {len(df_2025)} lignes, {len(df_2025.columns)} colonnes")

print(f"\n📊 Colonnes 2024:")
print(df_2024.columns.tolist())

print(f"\n📊 Colonnes 2025:")
print(df_2025.columns.tolist())

# Test preprocessing
print("\n⚙️  Test du preprocessing...")
preprocessor = RobustPreprocessor(target_col='Fondee')

print("\n🔧 Fit sur 2024...")
preprocessor.fit(df_2024)

print("\n🔄 Transform 2024...")
X_train = preprocessor.transform(df_2024)
y_train = df_2024['Fondee'].values

print(f"✅ X_train shape: {X_train.shape}")
print(f"✅ Features: {X_train.columns.tolist()[:10]}...")

print("\n🔄 Transform 2025...")
X_test = preprocessor.transform(df_2025)
y_test = df_2025['Fondee'].values

print(f"✅ X_test shape: {X_test.shape}")

# Test analyse par famille (simulation)
print("\n📊 Test analyse par famille...")
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=50, max_depth=8, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

analyzer = FamilyProductAnalyzer()
df_analysis = analyzer.analyze_by_family(df_2025, y_test, y_pred, y_prob)

print(f"\n✅ Analyse par famille terminée: {len(df_analysis)} familles")

# Créer visualisation
analyzer.plot_family_comparison(
    save_path='outputs/reports/figures/family_analysis_2025_test.png',
    year='2025'
)

print("\n✅ Test complet réussi!")
