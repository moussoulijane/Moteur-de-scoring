#!/usr/bin/env python3
"""
Génération du graphique des résultats 2023
Affiche le taux d'automatisation (pie chart) et l'accuracy
"""
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from matplotlib.patches import FancyBboxPatch

# Configuration
sns.set_style('whitegrid')
output_dir = Path('outputs/results_model_comparison')
output_dir.mkdir(parents=True, exist_ok=True)

# Données 2023
taux_auto_2023 = 88.0
accuracy_2023 = 92.0
taux_audit_2023 = 12.0

# Créer la figure
fig = plt.figure(figsize=(14, 7))
fig.suptitle('RÉSULTATS SYSTÈME 2023',
             fontsize=22, fontweight='bold', y=0.96)

# ================================================================================
# 1. Taux d'automatisation (Pie chart)
# ================================================================================
ax1 = plt.subplot(1, 2, 1)
sizes = [taux_auto_2023, taux_audit_2023]
labels = ['Automatisé', 'Audit Humain']
colors = ['#3498DB', '#E67E22']
explode = (0.1, 0)

wedges, texts, autotexts = ax1.pie(sizes, explode=explode, labels=labels,
                                    autopct='%1.1f%%', colors=colors,
                                    shadow=True, startangle=90,
                                    textprops={'fontsize': 14, 'weight': 'bold'})

for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontsize(16)

ax1.set_title('Taux d\'Automatisation', fontweight='bold', fontsize=18, pad=20)

# ================================================================================
# 2. Accuracy
# ================================================================================
ax2 = plt.subplot(1, 2, 2)
ax2.axis('off')
ax2.set_xlim(0, 10)
ax2.set_ylim(0, 10)

# Box principale pour l'accuracy
rect_acc = FancyBboxPatch((1.5, 2.5), 7, 5,
                          boxstyle="round,pad=0.4",
                          facecolor='#27AE60', edgecolor='black', linewidth=4)
ax2.add_patch(rect_acc)

# Texte ACCURACY
ax2.text(5, 6.5, 'ACCURACY', ha='center', va='center',
        fontsize=20, fontweight='bold', color='white')

# Valeur de l'accuracy
ax2.text(5, 4.5, f'{accuracy_2023:.0f}%', ha='center', va='center',
        fontsize=48, fontweight='bold', color='white')

# Sous-titre
ax2.text(5, 3.2, 'Précision globale du système', ha='center', va='center',
        fontsize=12, style='italic', color='white')

# Ajouter des statistiques en bas
ax2.text(5, 1.2, f'📊 {taux_auto_2023:.0f}% de cas automatisés  |  {accuracy_2023:.0f}% de précision',
        ha='center', va='center',
        fontsize=11, fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#F8F9F9',
                 edgecolor='#34495E', linewidth=2))

plt.tight_layout(rect=[0, 0, 1, 0.93])
output_path = output_dir / 'G4_resultats_2023.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"✅ Graphique sauvegardé: {output_path}")
plt.close()

print("\n" + "="*80)
print("📊 GRAPHIQUE DES RÉSULTATS 2023 GÉNÉRÉ")
print("="*80)
print(f"\n📂 Fichier: {output_path}")
print("\nRésultats 2023:")
print(f"  • Taux d'automatisation: {taux_auto_2023:.1f}%")
print(f"  • Accuracy:              {accuracy_2023:.1f}%")
print(f"  • Audit humain:          {taux_audit_2023:.1f}%")
print("="*80)
