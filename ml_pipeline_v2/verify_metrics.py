#!/usr/bin/env python3
"""
Script de vérification: Compare les métriques entre model_comparison et visualizer
"""
import sys
sys.path.append('src')

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

print("="*80)
print("VÉRIFICATION DE COHÉRENCE DES MÉTRIQUES")
print("="*80)

# 1. Charger les prédictions de model_comparison
predictions_path = Path('outputs/production_v2/predictions/predictions_2025_v2.pkl')
predictions_data = joblib.load(predictions_path)

best_model = predictions_data['best_model']
y_prob = predictions_data[best_model]['y_prob']
threshold_low = predictions_data[best_model]['threshold_low']
threshold_high = predictions_data[best_model]['threshold_high']
y_true = predictions_data['y_true']

print(f"\n📊 Modèle: {best_model}")
print(f"   Seuil BAS:  {threshold_low:.4f}")
print(f"   Seuil HAUT: {threshold_high:.4f}")

# 2. Recalculer les prédictions
y_pred = np.zeros(len(y_prob), dtype=int)
mask_rejet = y_prob <= threshold_low
mask_audit = (y_prob > threshold_low) & (y_prob < threshold_high)
mask_validation = y_prob >= threshold_high
y_pred[mask_validation] = 1

# 3. Calculer métriques UNIQUEMENT sur cas automatisés (même logique que model_comparison)
mask_auto = mask_rejet | mask_validation

print(f"\n📈 Distribution:")
print(f"   Rejet Auto:      {mask_rejet.sum():,} ({100*mask_rejet.sum()/len(y_prob):.1f}%)")
print(f"   Audit Humain:    {mask_audit.sum():,} ({100*mask_audit.sum()/len(y_prob):.1f}%)")
print(f"   Validation Auto: {mask_validation.sum():,} ({100*mask_validation.sum()/len(y_prob):.1f}%)")

if mask_auto.sum() > 0:
    y_pred_auto = y_pred[mask_auto]
    y_true_auto = y_true[mask_auto]

    acc = accuracy_score(y_true_auto, y_pred_auto)
    prec = precision_score(y_true_auto, y_pred_auto, zero_division=0)
    rec = recall_score(y_true_auto, y_pred_auto, zero_division=0)
    f1 = f1_score(y_true_auto, y_pred_auto, zero_division=0)

    print(f"\n✅ MÉTRIQUES SUR CAS AUTOMATISÉS (cohérent avec model_comparison):")
    print(f"   Accuracy:   {acc:.4f} ({100*acc:.2f}%)")
    print(f"   Precision:  {prec:.4f} ({100*prec:.2f}%)")
    print(f"   Recall:     {rec:.4f} ({100*rec:.2f}%)")
    print(f"   F1-Score:   {f1:.4f} ({100*f1:.2f}%)")

    # Matrice de confusion
    vp = ((y_true_auto == 1) & (y_pred_auto == 1)).sum()
    vn = ((y_true_auto == 0) & (y_pred_auto == 0)).sum()
    fp = ((y_true_auto == 0) & (y_pred_auto == 1)).sum()
    fn = ((y_true_auto == 1) & (y_pred_auto == 0)).sum()

    print(f"\n📊 Matrice de confusion (cas automatisés):")
    print(f"   VP: {vp:,}    VN: {vn:,}")
    print(f"   FP: {fp:,}    FN: {fn:,}")

# 4. Calculer aussi sur TOUS les cas pour comparaison
y_pred_all = y_pred  # Inclut les audits comme rejet (0)
acc_all = accuracy_score(y_true, y_pred_all)
prec_all = precision_score(y_true, y_pred_all, zero_division=0)
rec_all = recall_score(y_true, y_pred_all, zero_division=0)
f1_all = f1_score(y_true, y_pred_all, zero_division=0)

print(f"\n⚠️  MÉTRIQUES SUR TOUS LES CAS (pour comparaison):")
print(f"   Accuracy:   {acc_all:.4f} ({100*acc_all:.2f}%)")
print(f"   Precision:  {prec_all:.4f} ({100*prec_all:.2f}%)")
print(f"   Recall:     {rec_all:.4f} ({100*rec_all:.2f}%)")
print(f"   F1-Score:   {f1_all:.4f} ({100*f1_all:.2f}%)")

print("\n" + "="*80)
print("✅ CONCLUSION:")
print("Les métriques DOIVENT être calculées sur les CAS AUTOMATISÉS uniquement")
print("pour être cohérentes avec model_comparison_v2.py")
print("="*80)
