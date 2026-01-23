#!/usr/bin/env python3
"""
Visualisation graphique des gains financiers 2025 AVANT règles métier
"""
import sys
sys.path.append('src')

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from matplotlib.patches import FancyBboxPatch

# Configuration
sns.set_style('whitegrid')
PRIX_UNITAIRE_DH = 169

print("="*80)
print("💰 GÉNÉRATION GRAPHIQUE DES GAINS FINANCIERS - 2025")
print("="*80)

# 1. Charger les prédictions
predictions_path = Path('outputs/production_v2/predictions/predictions_2025_v2.pkl')
if not predictions_path.exists():
    print(f"\n❌ Fichier de prédictions introuvable: {predictions_path}")
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

print(f"\n🏆 Modèle: {best_model}")

# 4. Créer les décisions AVANT règles métier
y_pred = np.zeros(len(y_prob), dtype=int)
mask_rejet = y_prob <= threshold_low
mask_audit = (y_prob > threshold_low) & (y_prob < threshold_high)
mask_validation = y_prob >= threshold_high
y_pred[mask_validation] = 1

# 5. Cas automatisés
mask_auto = mask_rejet | mask_validation
n_auto = mask_auto.sum()

# 6. Calcul du GAIN BRUT
gain_brut = n_auto * PRIX_UNITAIRE_DH

# 7. Calcul des PERTES
y_pred_auto = y_pred[mask_auto]
y_true_auto = y_true[mask_auto]
montants_auto = df_2025['Montant demandé'].values[mask_auto]

fp_mask = (y_true_auto == 0) & (y_pred_auto == 1)
fn_mask = (y_true_auto == 1) & (y_pred_auto == 0)

montants_clean = np.nan_to_num(montants_auto, nan=0.0, posinf=0.0, neginf=0.0)
if len(montants_clean) > 0 and montants_clean.max() > 0:
    montants_clean = np.clip(montants_clean, 0, np.percentile(montants_clean[montants_clean > 0], 99))

perte_fp = montants_clean[fp_mask].sum()
perte_fn = 2 * montants_clean[fn_mask].sum()
perte_totale = perte_fp + perte_fn

# 8. GAIN NET
gain_net = gain_brut - perte_totale

print(f"\n📊 Résultats:")
print(f"   Gain BRUT:  {gain_brut:,.0f} DH ({gain_brut/1e6:.2f}M)")
print(f"   Pertes:     {perte_totale:,.0f} DH ({perte_totale/1e6:.2f}M)")
print(f"   Gain NET:   {gain_net:,.0f} DH ({gain_net/1e6:.2f}M)")

# ================================================================================
# CRÉATION DU GRAPHIQUE
# ================================================================================

fig = plt.figure(figsize=(18, 10))
fig.suptitle('GAINS FINANCIERS 2025 - AVANT Application des Règles Métier',
             fontsize=22, fontweight='bold', y=0.98)

# ================================================================================
# 1. Graphique en cascade (Waterfall)
# ================================================================================
ax1 = plt.subplot(2, 3, 1)

categories = ['Gain\nBrut', 'Perte\nFP', 'Perte\nFN', 'Gain\nNET']
values = [gain_brut/1e6, -(perte_fp/1e6), -(perte_fn/1e6), gain_net/1e6]
colors = ['#2ECC71', '#E74C3C', '#E67E22', '#3498DB']

bars = ax1.bar(categories, values, color=colors, alpha=0.85,
               edgecolor='black', linewidth=2, width=0.6)

ax1.set_ylabel('Millions DH', fontweight='bold', fontsize=14)
ax1.set_title('Cascade des Gains', fontweight='bold', fontsize=16)
ax1.grid(True, alpha=0.3, axis='y')
ax1.axhline(y=0, color='black', linestyle='-', linewidth=1.5)

for bar, val in zip(bars, values):
    height = bar.get_height()
    va = 'bottom' if val >= 0 else 'top'
    offset = 0.03 if val >= 0 else -0.03
    ax1.text(bar.get_x() + bar.get_width()/2., height + offset,
            f'{abs(val):.2f}M', ha='center', va=va,
            fontweight='bold', fontsize=12)

# ================================================================================
# 2. Gain NET (Box principal)
# ================================================================================
ax2 = plt.subplot(2, 3, 2)
ax2.axis('off')
ax2.set_xlim(0, 10)
ax2.set_ylim(0, 10)

rect = FancyBboxPatch((1, 2), 8, 6,
                      boxstyle="round,pad=0.4",
                      facecolor='#27AE60', edgecolor='black', linewidth=4)
ax2.add_patch(rect)

ax2.text(5, 7.5, 'GAIN NET', ha='center', va='center',
        fontsize=18, fontweight='bold', color='white')
ax2.text(5, 5.5, f'{gain_net/1e6:.2f}', ha='center', va='center',
        fontsize=42, fontweight='bold', color='white')
ax2.text(5, 4.2, 'Millions DH', ha='center', va='center',
        fontsize=16, style='italic', color='white')
ax2.text(5, 3, f'({gain_net:,.0f} DH)', ha='center', va='center',
        fontsize=11, color='white')

# ================================================================================
# 3. Décomposition (Pie chart)
# ================================================================================
ax3 = plt.subplot(2, 3, 3)

total = gain_brut
sizes = [gain_net, perte_totale]
labels = [f'Gain NET\n{gain_net/1e6:.2f}M DH', f'Pertes\n{perte_totale/1e6:.2f}M DH']
colors_pie = ['#2ECC71', '#E74C3C']
explode = (0.1, 0)

wedges, texts, autotexts = ax3.pie(sizes, explode=explode, labels=labels,
                                    autopct='%1.2f%%', colors=colors_pie,
                                    shadow=True, startangle=90,
                                    textprops={'fontsize': 11, 'weight': 'bold'})

for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontsize(12)

ax3.set_title(f'Répartition (Total: {gain_brut/1e6:.2f}M DH)',
             fontweight='bold', fontsize=14)

# ================================================================================
# 4. Détail des pertes
# ================================================================================
ax4 = plt.subplot(2, 3, 4)

pertes_cats = ['Faux\nPositifs', 'Faux\nNégatifs']
pertes_vals = [perte_fp/1e6, perte_fn/1e6]
pertes_colors = ['#E74C3C', '#E67E22']

bars = ax4.bar(pertes_cats, pertes_vals, color=pertes_colors,
              alpha=0.8, edgecolor='black', linewidth=2, width=0.5)

ax4.set_ylabel('Millions DH', fontweight='bold', fontsize=12)
ax4.set_title('Détail des Pertes', fontweight='bold', fontsize=14)
ax4.grid(True, alpha=0.3, axis='y')

for bar, val, count in zip(bars, pertes_vals, [fp_mask.sum(), fn_mask.sum()]):
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height + max(pertes_vals)*0.05,
            f'{val:.4f}M\n({count} cas)', ha='center', va='bottom',
            fontweight='bold', fontsize=10)

# ================================================================================
# 5. Indicateurs clés
# ================================================================================
ax5 = plt.subplot(2, 3, 5)
ax5.axis('off')

indicators_text = f"""
╔════════════════════════════════════╗
║       INDICATEURS CLÉS             ║
╚════════════════════════════════════╝

📊 VOLUME:
  • Cas automatisés:   {n_auto:,}
  • Taux automatisation: {100*n_auto/len(y_prob):.1f}%

💵 PRIX:
  • Prix unitaire:      {PRIX_UNITAIRE_DH} DH/cas

❌ ERREURS:
  • Faux Positifs (FP): {fp_mask.sum():,} cas
  • Faux Négatifs (FN): {fn_mask.sum():,} cas
  • Taux erreur:        {100*(fp_mask.sum()+fn_mask.sum())/n_auto:.2f}%

💰 RATIO:
  • Gain/Perte:         {gain_brut/perte_totale if perte_totale > 0 else 0:,.0f}x
"""

ax5.text(0.5, 0.5, indicators_text, transform=ax5.transAxes,
        fontsize=10, verticalalignment='center', horizontalalignment='center',
        family='monospace',
        bbox=dict(boxstyle='round,pad=0.8', facecolor='#E8F8F5',
                 edgecolor='#16A085', linewidth=3, alpha=0.9))

# ================================================================================
# 6. Résumé financier
# ================================================================================
ax6 = plt.subplot(2, 3, 6)
ax6.axis('off')

summary_text = f"""
╔════════════════════════════════════════════╗
║         RÉSUMÉ FINANCIER 2025              ║
╚════════════════════════════════════════════╝

💵 GAIN BRUT:
   {gain_brut:>18,} DH
   {gain_brut/1e6:>18.2f} Millions DH

📉 PERTES TOTALES:
   • FP: {perte_fp:>15,.0f} DH
   • FN: {perte_fn:>15,.0f} DH
   ────────────────────────────
   {perte_totale:>18,.0f} DH
   {perte_totale/1e6:>18.4f} Millions DH

💰 GAIN NET:
   {gain_net:>18,} DH
   {gain_net/1e6:>18.2f} Millions DH

🎯 EFFICACITÉ:
   {100*gain_net/gain_brut:.2f}% du gain brut conservé
"""

ax6.text(0.5, 0.5, summary_text, transform=ax6.transAxes,
        fontsize=9, verticalalignment='center', horizontalalignment='center',
        family='monospace',
        bbox=dict(boxstyle='round,pad=0.8', facecolor='#FEF9E7',
                 edgecolor='#F39C12', linewidth=3, alpha=0.9))

plt.tight_layout(rect=[0, 0, 1, 0.96])

# Sauvegarder
output_dir = Path('outputs/results_model_comparison')
output_dir.mkdir(parents=True, exist_ok=True)
output_path = output_dir / 'G5_gains_financiers_2025.png'

plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"\n✅ Graphique sauvegardé: {output_path}")
plt.close()

print("\n" + "="*80)
print("📊 GRAPHIQUE DES GAINS GÉNÉRÉ AVEC SUCCÈS")
print("="*80)
print(f"\n📂 Fichier: {output_path}")
print("\n📈 Résultats affichés:")
print(f"  • Cascade des gains (Brut → Pertes → Net)")
print(f"  • Gain NET principal: {gain_net/1e6:.2f}M DH")
print(f"  • Répartition Gain/Pertes")
print(f"  • Détail des pertes FP/FN")
print(f"  • Indicateurs clés")
print(f"  • Résumé financier complet")
print("="*80)
