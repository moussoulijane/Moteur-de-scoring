#!/usr/bin/env python3
"""
Génération du schéma de la couche décisionnelle
Représente le flux de décision avec les règles métier
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
from pathlib import Path

# Configuration
output_dir = Path('outputs/results_model_comparison')
output_dir.mkdir(parents=True, exist_ok=True)

fig, ax = plt.subplots(1, 1, figsize=(16, 12))
ax.set_xlim(0, 10)
ax.set_ylim(0, 14)
ax.axis('off')

# Titre principal
ax.text(5, 13.5, 'COUCHE DÉCISIONNELLE DU MODÈLE ML',
        ha='center', va='center', fontsize=20, fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#34495E',
                 edgecolor='black', linewidth=3, alpha=0.9),
        color='white')

# ================================================================================
# ÉTAPE 1: MODÈLE ML
# ================================================================================
y_start = 12

# Box Modèle ML
rect_model = FancyBboxPatch((1.5, y_start-0.6), 7, 1.2,
                            boxstyle="round,pad=0.1",
                            facecolor='#3498DB', edgecolor='black', linewidth=3)
ax.add_patch(rect_model)
ax.text(5, y_start, '📊 MODÈLE MACHINE LEARNING (XGBoost)',
        ha='center', va='center', fontsize=14, fontweight='bold', color='white')
ax.text(5, y_start-0.35, 'Prédiction: Probabilité que la réclamation soit FONDÉE',
        ha='center', va='center', fontsize=11, style='italic', color='white')

# Flèche vers seuils
arrow1 = FancyArrowPatch((5, y_start-0.7), (5, y_start-1.3),
                        arrowstyle='->', mutation_scale=30, linewidth=3,
                        color='black')
ax.add_patch(arrow1)

# ================================================================================
# ÉTAPE 2: APPLICATION DES SEUILS
# ================================================================================
y_seuils = 10

# Box Seuils
rect_seuils = FancyBboxPatch((1.5, y_seuils-0.5), 7, 1,
                            boxstyle="round,pad=0.1",
                            facecolor='#9B59B6', edgecolor='black', linewidth=3)
ax.add_patch(rect_seuils)
ax.text(5, y_seuils+0.2, '🎯 APPLICATION DES SEUILS OPTIMISÉS',
        ha='center', va='center', fontsize=13, fontweight='bold', color='white')
ax.text(5, y_seuils-0.2, 'Seuil BAS: 0.43  |  Seuil HAUT: 0.50',
        ha='center', va='center', fontsize=11, fontweight='bold', color='white')

# Flèches vers 3 zones
arrow_left = FancyArrowPatch((3.5, y_seuils-0.6), (1.5, y_seuils-1.5),
                            arrowstyle='->', mutation_scale=25, linewidth=2.5,
                            color='black')
ax.add_patch(arrow_left)

arrow_center = FancyArrowPatch((5, y_seuils-0.6), (5, y_seuils-1.5),
                              arrowstyle='->', mutation_scale=25, linewidth=2.5,
                              color='black')
ax.add_patch(arrow_center)

arrow_right = FancyArrowPatch((6.5, y_seuils-0.6), (8.5, y_seuils-1.5),
                             arrowstyle='->', mutation_scale=25, linewidth=2.5,
                             color='black')
ax.add_patch(arrow_right)

# ================================================================================
# ÉTAPE 3: TROIS ZONES DE DÉCISION
# ================================================================================
y_zones = 7.5

# Zone 1: Rejet Auto (gauche)
rect_rejet = FancyBboxPatch((0.2, y_zones-0.6), 2.5, 1.2,
                           boxstyle="round,pad=0.1",
                           facecolor='#E74C3C', edgecolor='black', linewidth=2.5)
ax.add_patch(rect_rejet)
ax.text(1.45, y_zones+0.3, '❌ REJET AUTO',
        ha='center', va='center', fontsize=12, fontweight='bold', color='white')
ax.text(1.45, y_zones-0.05, 'Probabilité ≤ 0.43',
        ha='center', va='center', fontsize=10, color='white')
ax.text(1.45, y_zones-0.35, '(Non fondée)',
        ha='center', va='center', fontsize=9, style='italic', color='white')

# Zone 2: Audit Humain (centre)
rect_audit = FancyBboxPatch((3.75, y_zones-0.6), 2.5, 1.2,
                           boxstyle="round,pad=0.1",
                           facecolor='#F39C12', edgecolor='black', linewidth=2.5)
ax.add_patch(rect_audit)
ax.text(5, y_zones+0.3, '👤 AUDIT HUMAIN',
        ha='center', va='center', fontsize=12, fontweight='bold', color='white')
ax.text(5, y_zones-0.05, '0.43 < Prob < 0.50',
        ha='center', va='center', fontsize=10, color='white')
ax.text(5, y_zones-0.35, '(Zone incertaine)',
        ha='center', va='center', fontsize=9, style='italic', color='white')

# Zone 3: Validation Auto (droite)
rect_validation = FancyBboxPatch((7.3, y_zones-0.6), 2.5, 1.2,
                                boxstyle="round,pad=0.1",
                                facecolor='#2ECC71', edgecolor='black', linewidth=2.5)
ax.add_patch(rect_validation)
ax.text(8.55, y_zones+0.3, '✅ VALIDATION AUTO',
        ha='center', va='center', fontsize=12, fontweight='bold', color='white')
ax.text(8.55, y_zones-0.05, 'Probabilité ≥ 0.50',
        ha='center', va='center', fontsize=10, color='white')
ax.text(8.55, y_zones-0.35, '(Fondée)',
        ha='center', va='center', fontsize=9, style='italic', color='white')

# Flèches depuis rejet et audit (décision finale directe)
arrow_rejet_final = FancyArrowPatch((1.45, y_zones-0.7), (1.45, 2.2),
                                   arrowstyle='->', mutation_scale=20, linewidth=2,
                                   color='#E74C3C', linestyle='--')
ax.add_patch(arrow_rejet_final)

arrow_audit_final = FancyArrowPatch((5, y_zones-0.7), (5, 2.2),
                                   arrowstyle='->', mutation_scale=20, linewidth=2,
                                   color='#F39C12', linestyle='--')
ax.add_patch(arrow_audit_final)

# Flèche depuis validation vers règles métier
arrow_validation_rules = FancyArrowPatch((8.55, y_zones-0.7), (8.55, y_zones-1.5),
                                        arrowstyle='->', mutation_scale=25, linewidth=2.5,
                                        color='black')
ax.add_patch(arrow_validation_rules)

# ================================================================================
# ÉTAPE 4: RÈGLES MÉTIER (seulement pour Validation Auto)
# ================================================================================
y_rules = 5.5

# Titre des règles
ax.text(8.55, y_rules+0.8, '🔧 RÈGLES MÉTIER',
        ha='center', va='center', fontsize=13, fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#34495E',
                 edgecolor='black', linewidth=2),
        color='white')

# Règle 1
rect_rule1 = FancyBboxPatch((7.1, y_rules-0.5), 2.9, 0.7,
                           boxstyle="round,pad=0.08",
                           facecolor='#5DADE2', edgecolor='black', linewidth=2)
ax.add_patch(rect_rule1)
ax.text(8.55, y_rules-0.05, '📋 Règle #1',
        ha='center', va='center', fontsize=11, fontweight='bold')
ax.text(8.55, y_rules-0.3, 'Max 1 validation/client/an',
        ha='center', va='center', fontsize=9)

# Flèche règle 1
arrow_rule1 = FancyArrowPatch((8.55, y_rules-0.6), (8.55, y_rules-1.2),
                             arrowstyle='->', mutation_scale=20, linewidth=2,
                             color='black')
ax.add_patch(arrow_rule1)

# Règle 2
rect_rule2 = FancyBboxPatch((7.1, y_rules-2), 2.9, 0.7,
                           boxstyle="round,pad=0.08",
                           facecolor='#5DADE2', edgecolor='black', linewidth=2)
ax.add_patch(rect_rule2)
ax.text(8.55, y_rules-1.55, '📋 Règle #2',
        ha='center', va='center', fontsize=11, fontweight='bold')
ax.text(8.55, y_rules-1.8, 'Montant > PNB du client',
        ha='center', va='center', fontsize=9)

# Flèche vers décision
arrow_rule2_decision = FancyArrowPatch((8.55, y_rules-2.1), (8.55, 2.2),
                                      arrowstyle='->', mutation_scale=20, linewidth=2,
                                      color='#2ECC71')
ax.add_patch(arrow_rule2_decision)

# ================================================================================
# ÉTAPE 5: DÉCISION FINALE
# ================================================================================
y_final = 1

# Box décision finale
rect_final = FancyBboxPatch((1, y_final-0.6), 8, 1.2,
                           boxstyle="round,pad=0.1",
                           facecolor='#16A085', edgecolor='black', linewidth=3)
ax.add_patch(rect_final)
ax.text(5, y_final+0.3, '🎯 DÉCISION FINALE',
        ha='center', va='center', fontsize=14, fontweight='bold', color='white')
ax.text(5, y_final-0.15, 'Rejet Auto  |  Audit Humain  |  Validation Auto (si règles OK)',
        ha='center', va='center', fontsize=11, color='white')

# ================================================================================
# LÉGENDE
# ================================================================================
y_legend = 0.3

# Légende des règles
legend_text = """
💡 LOGIQUE DES RÈGLES MÉTIER:
   • Les règles métier s'appliquent UNIQUEMENT aux cas "Validation Auto"
   • Si UNE des règles est déclenchée → Conversion en "Audit Humain"
   • Les cas "Rejet Auto" et "Audit Humain" ne passent PAS par les règles métier
"""
ax.text(5, y_legend-0.1, legend_text,
        ha='center', va='top', fontsize=9, family='monospace',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#F8F9F9',
                 edgecolor='#34495E', linewidth=2, alpha=0.9))

# Ajout des statistiques dans un coin
stats_text = """
📊 STATISTIQUES (Modèle XGBoost):
  • Accuracy: 99.86%
  • Précision: 99.95%
  • Rappel: 99.77%
  • Taux automatisation: 100%
"""
ax.text(9.7, 13, stats_text,
        ha='right', va='top', fontsize=8, family='monospace',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#E8F8F5',
                 edgecolor='#16A085', linewidth=2, alpha=0.9))

plt.tight_layout()
output_path = output_dir / 'schema_couche_decisionnelle.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"✅ Schéma sauvegardé: {output_path}")
plt.close()

print("\n" + "="*80)
print("📊 SCHÉMA DE LA COUCHE DÉCISIONNELLE GÉNÉRÉ")
print("="*80)
print(f"\n📂 Fichier: {output_path}")
print("\nLe schéma montre:")
print("  1. Modèle ML → Prédiction de probabilité")
print("  2. Application des seuils → 3 zones de décision")
print("  3. Règles métier → Appliquées aux validations automatiques")
print("  4. Décision finale")
print("="*80)
