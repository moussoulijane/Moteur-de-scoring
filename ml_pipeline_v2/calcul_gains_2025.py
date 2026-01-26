#!/usr/bin/env python3
"""
Calcul des gains financiers sur 2025 AVANT l'application des règles métier
"""
import sys
sys.path.append('src')

import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# Configuration
PRIX_UNITAIRE_DH = 169  # Prix unitaire par dossier traité

print("="*80)
print("💰 CALCUL DES GAINS FINANCIERS - 2025 (AVANT Règles Métier)")
print("="*80)

# 1. Charger les prédictions de model_comparison
predictions_path = Path('outputs/production_v2/predictions/predictions_2025_v2.pkl')
if not predictions_path.exists():
    print(f"\n❌ Fichier de prédictions introuvable: {predictions_path}")
    print("\nVeuillez d'abord exécuter:")
    print("   python ml_pipeline_v2/model_comparison_v2.py")
    sys.exit(1)

predictions_data = joblib.load(predictions_path)

# 2. Charger les données 2025
data_path = Path('data/raw/reclamations_2025.xlsx')
if not data_path.exists():
    print(f"❌ Fichier de données introuvable: {data_path}")
    sys.exit(1)

df_2025 = pd.read_excel(data_path)

# 3. Extraire les informations
best_model = predictions_data['best_model']
y_prob = predictions_data[best_model]['y_prob']
threshold_low = predictions_data[best_model]['threshold_low']
threshold_high = predictions_data[best_model]['threshold_high']
y_true = predictions_data['y_true']

print(f"\n🏆 Modèle utilisé: {best_model}")
print(f"   Seuil BAS:  {threshold_low:.4f}")
print(f"   Seuil HAUT: {threshold_high:.4f}")

# 4. Créer les décisions AVANT règles métier
y_pred = np.zeros(len(y_prob), dtype=int)
mask_rejet = y_prob <= threshold_low
mask_audit = (y_prob > threshold_low) & (y_prob < threshold_high)
mask_validation = y_prob >= threshold_high
y_pred[mask_validation] = 1

# 5. Identifier les cas automatisés (Rejet + Validation)
mask_auto = mask_rejet | mask_validation
n_auto = mask_auto.sum()
n_audit = mask_audit.sum()

print(f"\n📊 Répartition des décisions (AVANT règles métier):")
print(f"   • Rejet Auto:      {mask_rejet.sum():,} ({100*mask_rejet.sum()/len(y_prob):.1f}%)")
print(f"   • Audit Humain:    {n_audit:,} ({100*n_audit/len(y_prob):.1f}%)")
print(f"   • Validation Auto: {mask_validation.sum():,} ({100*mask_validation.sum()/len(y_prob):.1f}%)")
print(f"   • TOTAL AUTO:      {n_auto:,} ({100*n_auto/len(y_prob):.1f}%)")

# 6. Calcul du GAIN BRUT
gain_brut = n_auto * PRIX_UNITAIRE_DH

print(f"\n" + "="*80)
print("💵 CALCUL DU GAIN BRUT")
print("="*80)
print(f"   Nombre de cas automatisés: {n_auto:,}")
print(f"   Prix unitaire par dossier:  {PRIX_UNITAIRE_DH} DH")
print(f"   ─────────────────────────────────────")
print(f"   GAIN BRUT:                  {gain_brut:,.0f} DH")
print(f"                               {gain_brut/1e6:.2f} Millions DH")

# 7. Calcul des PERTES (uniquement sur cas automatisés)
if 'Montant demandé' not in df_2025.columns:
    print("\n⚠️  Colonne 'Montant demandé' manquante - Impossible de calculer les pertes")
    sys.exit(1)

# Filtrer pour ne garder que les cas automatisés
y_pred_auto = y_pred[mask_auto]
y_true_auto = y_true[mask_auto]
montants_auto = df_2025['Montant demandé'].values[mask_auto]

# Identifier les erreurs
fp_mask = (y_true_auto == 0) & (y_pred_auto == 1)  # Faux Positifs (validé à tort)
fn_mask = (y_true_auto == 1) & (y_pred_auto == 0)  # Faux Négatifs (rejeté à tort)

# Nettoyer les montants (enlever NaN, valeurs extrêmes)
montants_clean = np.nan_to_num(montants_auto, nan=0.0, posinf=0.0, neginf=0.0)
if len(montants_clean) > 0 and montants_clean.max() > 0:
    montants_clean = np.clip(montants_clean, 0, np.percentile(montants_clean[montants_clean > 0], 99))
else:
    montants_clean = montants_clean.clip(0)

# Calcul des pertes
perte_fp = montants_clean[fp_mask].sum()  # Montants versés à tort
perte_fn = 2 * montants_clean[fn_mask].sum()  # Coût double (insatisfaction client + montant)

print(f"\n" + "="*80)
print("📉 CALCUL DES PERTES")
print("="*80)

print(f"\n1️⃣  FAUX POSITIFS (Validations à tort):")
print(f"   Nombre de FP:               {fp_mask.sum():,}")
print(f"   Montants versés à tort:     {perte_fp:,.0f} DH")
print(f"                               {perte_fp/1e6:.2f} Millions DH")
print(f"   Impact: Argent payé alors que réclamation non fondée")

print(f"\n2️⃣  FAUX NÉGATIFS (Rejets à tort):")
print(f"   Nombre de FN:               {fn_mask.sum():,}")
print(f"   Montants non versés (FN):   {montants_clean[fn_mask].sum():,.0f} DH")
print(f"   Coût estimé (x2):           {perte_fn:,.0f} DH")
print(f"                               {perte_fn/1e6:.2f} Millions DH")
print(f"   Impact: Insatisfaction client + perte de confiance (coût doublé)")

perte_totale = perte_fp + perte_fn

print(f"\n   ─────────────────────────────────────")
print(f"   PERTE TOTALE:               {perte_totale:,.0f} DH")
print(f"                               {perte_totale/1e6:.2f} Millions DH")

# 8. Calcul du GAIN NET
gain_net = gain_brut - perte_totale

print(f"\n" + "="*80)
print("🎯 GAIN NET FINAL")
print("="*80)
print(f"   Gain BRUT:                  {gain_brut:,.0f} DH")
print(f"   - Perte FP (montants):      {perte_fp:,.0f} DH")
print(f"   - Perte FN (coût x2):       {perte_fn:,.0f} DH")
print(f"   ─────────────────────────────────────")
print(f"   GAIN NET:                   {gain_net:,.0f} DH")
print(f"                               {gain_net/1e6:.2f} Millions DH")

# 9. Résumé final
print(f"\n" + "="*80)
print("📊 RÉSUMÉ FINANCIER - 2025 (AVANT Règles Métier)")
print("="*80)

print(f"""
╔════════════════════════════════════════════════════╗
║            GAINS FINANCIERS 2025                   ║
╚════════════════════════════════════════════════════╝

📈 GAIN BRUT:
   {gain_brut:>18,} DH  =  {gain_brut/1e6:>8.2f} Millions DH

📉 PERTES:
   • Faux Positifs (FP):
     {perte_fp:>18,} DH  =  {perte_fp/1e6:>8.2f} Millions DH

   • Faux Négatifs (FN):
     {perte_fn:>18,} DH  =  {perte_fn/1e6:>8.2f} Millions DH

   • TOTAL Pertes:
     {perte_totale:>18,} DH  =  {perte_totale/1e6:>8.2f} Millions DH

💰 GAIN NET:
   {gain_net:>18,} DH  =  {gain_net/1e6:>8.2f} Millions DH

📊 INDICATEURS:
   • Taux automatisation:    {100*n_auto/len(y_prob):>6.1f}%
   • Nombre FP:              {fp_mask.sum():>6,}
   • Nombre FN:              {fn_mask.sum():>6,}
   • Ratio Gain/Perte:       {gain_brut/perte_totale if perte_totale > 0 else 0:>6.1f}x

""")

print("="*80)
print("✅ Calcul terminé")
print("="*80)

# 10. Sauvegarder les résultats dans un fichier
output_dir = Path('outputs/production_v2')
output_dir.mkdir(parents=True, exist_ok=True)

output_file = output_dir / 'gains_financiers_2025_avant_regles.txt'
with open(output_file, 'w', encoding='utf-8') as f:
    f.write("="*80 + "\n")
    f.write("💰 GAINS FINANCIERS - 2025 (AVANT Règles Métier)\n")
    f.write("="*80 + "\n\n")

    f.write(f"Modèle: {best_model}\n")
    f.write(f"Seuil BAS:  {threshold_low:.4f}\n")
    f.write(f"Seuil HAUT: {threshold_high:.4f}\n\n")

    f.write("RÉPARTITION DES DÉCISIONS:\n")
    f.write(f"  Rejet Auto:      {mask_rejet.sum():,} ({100*mask_rejet.sum()/len(y_prob):.1f}%)\n")
    f.write(f"  Audit Humain:    {n_audit:,} ({100*n_audit/len(y_prob):.1f}%)\n")
    f.write(f"  Validation Auto: {mask_validation.sum():,} ({100*mask_validation.sum()/len(y_prob):.1f}%)\n")
    f.write(f"  TOTAL AUTO:      {n_auto:,} ({100*n_auto/len(y_prob):.1f}%)\n\n")

    f.write("="*80 + "\n")
    f.write("CALCUL FINANCIER\n")
    f.write("="*80 + "\n\n")

    f.write(f"GAIN BRUT:           {gain_brut:>15,} DH  =  {gain_brut/1e6:>8.2f} Millions DH\n\n")

    f.write("PERTES:\n")
    f.write(f"  Faux Positifs:     {perte_fp:>15,} DH  =  {perte_fp/1e6:>8.2f} Millions DH\n")
    f.write(f"  Faux Négatifs:     {perte_fn:>15,} DH  =  {perte_fn/1e6:>8.2f} Millions DH\n")
    f.write(f"  TOTAL Pertes:      {perte_totale:>15,} DH  =  {perte_totale/1e6:>8.2f} Millions DH\n\n")

    f.write(f"GAIN NET:            {gain_net:>15,} DH  =  {gain_net/1e6:>8.2f} Millions DH\n\n")

    f.write("INDICATEURS:\n")
    f.write(f"  Taux automatisation:  {100*n_auto/len(y_prob):.1f}%\n")
    f.write(f"  Nombre FP:            {fp_mask.sum():,}\n")
    f.write(f"  Nombre FN:            {fn_mask.sum():,}\n")
    f.write(f"  Ratio Gain/Perte:     {gain_brut/perte_totale if perte_totale > 0 else 0:.1f}x\n")

print(f"\n📁 Résultats sauvegardés dans: {output_file}")
