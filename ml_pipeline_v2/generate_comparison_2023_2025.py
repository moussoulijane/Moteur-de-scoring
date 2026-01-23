#!/usr/bin/env python3
"""
Génération du graphique de comparaison 2023 vs 2025
Montre l'évolution du taux d'automatisation et de l'accuracy
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# Configuration
sns.set_style('whitegrid')
output_dir = Path('outputs/results_model_comparison')
output_dir.mkdir(parents=True, exist_ok=True)

# Données 2023 et 2025
data = {
    '2023': {
        'taux_auto': 88.0,
        'accuracy': 92.0,
        'taux_audit': 12.0,
        'label': 'Ancien Système (2023)'
    },
    '2025': {
        'taux_auto': 100.0,
        'accuracy': 99.86,
        'taux_audit': 0.0,
        'label': 'Nouveau Modèle ML (2025)'
    }
}

# Créer la figure
fig = plt.figure(figsize=(18, 8))
fig.suptitle('COMPARAISON 2023 vs 2025 - AMÉLIORATION DES PERFORMANCES',
             fontsize=20, fontweight='bold', y=0.98)

# ================================================================================
# GRAPHIQUES POUR 2023
# ================================================================================

# 1. Taux d'automatisation 2023 (Pie chart)
ax1 = plt.subplot(2, 4, 1)
sizes_2023 = [data['2023']['taux_auto'], data['2023']['taux_audit']]
labels_2023 = ['Automatisé', 'Audit\nHumain']
colors_2023 = ['#3498DB', '#E67E22']
explode_2023 = (0.1, 0)

wedges, texts, autotexts = ax1.pie(sizes_2023, explode=explode_2023, labels=labels_2023,
                                    autopct='%1.1f%%', colors=colors_2023,
                                    shadow=True, startangle=90,
                                    textprops={'fontsize': 11, 'weight': 'bold'})

for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontsize(12)

ax1.set_title('2023: Taux d\'Automatisation', fontweight='bold', fontsize=13)

# 2. Accuracy 2023 (Gauge style)
ax2 = plt.subplot(2, 4, 2)
ax2.axis('off')
ax2.set_xlim(0, 10)
ax2.set_ylim(0, 10)

# Box pour l'accuracy
from matplotlib.patches import FancyBboxPatch
rect_acc_2023 = FancyBboxPatch((1, 2), 8, 6,
                               boxstyle="round,pad=0.3",
                               facecolor='#3498DB', edgecolor='black', linewidth=3)
ax2.add_patch(rect_acc_2023)

ax2.text(5, 6.5, '2023', ha='center', va='center',
        fontsize=14, fontweight='bold', color='white')
ax2.text(5, 4.5, 'ACCURACY', ha='center', va='center',
        fontsize=13, fontweight='bold', color='white')
ax2.text(5, 3, f'{data["2023"]["accuracy"]:.1f}%', ha='center', va='center',
        fontsize=26, fontweight='bold', color='white')

# ================================================================================
# GRAPHIQUES POUR 2025
# ================================================================================

# 3. Taux d'automatisation 2025 (Pie chart)
ax3 = plt.subplot(2, 4, 3)
sizes_2025 = [data['2025']['taux_auto'], data['2025']['taux_audit']]
labels_2025 = ['Automatisé', 'Audit\nHumain']
colors_2025 = ['#2ECC71', '#E67E22']
explode_2025 = (0.1, 0)

wedges, texts, autotexts = ax3.pie(sizes_2025, explode=explode_2025, labels=labels_2025,
                                    autopct='%1.1f%%', colors=colors_2025,
                                    shadow=True, startangle=90,
                                    textprops={'fontsize': 11, 'weight': 'bold'})

for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontsize(12)

ax3.set_title('2025: Taux d\'Automatisation', fontweight='bold', fontsize=13)

# 4. Accuracy 2025 (Gauge style)
ax4 = plt.subplot(2, 4, 4)
ax4.axis('off')
ax4.set_xlim(0, 10)
ax4.set_ylim(0, 10)

# Box pour l'accuracy
rect_acc_2025 = FancyBboxPatch((1, 2), 8, 6,
                               boxstyle="round,pad=0.3",
                               facecolor='#2ECC71', edgecolor='black', linewidth=3)
ax4.add_patch(rect_acc_2025)

ax4.text(5, 6.5, '2025', ha='center', va='center',
        fontsize=14, fontweight='bold', color='white')
ax4.text(5, 4.5, 'ACCURACY', ha='center', va='center',
        fontsize=13, fontweight='bold', color='white')
ax4.text(5, 3, f'{data["2025"]["accuracy"]:.2f}%', ha='center', va='center',
        fontsize=26, fontweight='bold', color='white')

# ================================================================================
# GRAPHIQUES DE COMPARAISON
# ================================================================================

# 5. Comparaison des taux d'automatisation (Barres)
ax5 = plt.subplot(2, 4, 5)
years = ['2023', '2025']
taux_auto_values = [data['2023']['taux_auto'], data['2025']['taux_auto']]
colors_bars = ['#3498DB', '#2ECC71']

bars = ax5.bar(years, taux_auto_values, color=colors_bars,
              alpha=0.8, edgecolor='black', linewidth=2, width=0.5)

ax5.set_ylabel('Taux d\'Automatisation (%)', fontweight='bold', fontsize=12)
ax5.set_title('Évolution du Taux d\'Automatisation', fontweight='bold', fontsize=13)
ax5.set_ylim(0, 105)
ax5.grid(True, alpha=0.3, axis='y')

for bar, val in zip(bars, taux_auto_values):
    height = bar.get_height()
    ax5.text(bar.get_x() + bar.get_width()/2., height + 2,
            f'{val:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=12)

# Flèche d'amélioration
improvement_auto = data['2025']['taux_auto'] - data['2023']['taux_auto']
ax5.annotate('', xy=(1, data['2025']['taux_auto']), xytext=(0, data['2023']['taux_auto']),
            arrowprops=dict(arrowstyle='->', lw=2.5, color='green'))
ax5.text(0.5, (data['2023']['taux_auto'] + data['2025']['taux_auto'])/2,
        f'+{improvement_auto:.1f}%', ha='center', va='center',
        fontsize=11, fontweight='bold', color='green',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='green', linewidth=2))

# 6. Comparaison de l'accuracy (Barres)
ax6 = plt.subplot(2, 4, 6)
accuracy_values = [data['2023']['accuracy'], data['2025']['accuracy']]

bars = ax6.bar(years, accuracy_values, color=colors_bars,
              alpha=0.8, edgecolor='black', linewidth=2, width=0.5)

ax6.set_ylabel('Accuracy (%)', fontweight='bold', fontsize=12)
ax6.set_title('Évolution de l\'Accuracy', fontweight='bold', fontsize=13)
ax6.set_ylim(85, 101)
ax6.grid(True, alpha=0.3, axis='y')

for bar, val in zip(bars, accuracy_values):
    height = bar.get_height()
    ax6.text(bar.get_x() + bar.get_width()/2., height + 0.3,
            f'{val:.2f}%', ha='center', va='bottom', fontweight='bold', fontsize=12)

# Flèche d'amélioration
improvement_acc = data['2025']['accuracy'] - data['2023']['accuracy']
ax6.annotate('', xy=(1, data['2025']['accuracy']), xytext=(0, data['2023']['accuracy']),
            arrowprops=dict(arrowstyle='->', lw=2.5, color='green'))
ax6.text(0.5, (data['2023']['accuracy'] + data['2025']['accuracy'])/2,
        f'+{improvement_acc:.2f}%', ha='center', va='center',
        fontsize=11, fontweight='bold', color='green',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='green', linewidth=2))

# 7. Statistiques 2023
ax7 = plt.subplot(2, 4, 7)
ax7.axis('off')

stats_2023 = f"""
╔══════════════════════════════╗
║      SYSTÈME 2023            ║
╚══════════════════════════════╝

📊 PERFORMANCE:
  • Taux automatisation: {data['2023']['taux_auto']:.1f}%
  • Accuracy:            {data['2023']['accuracy']:.1f}%
  • Audit humain:        {data['2023']['taux_audit']:.1f}%

⚙️ MÉTHODE:
  Système basé sur des règles
  métier statiques
"""

ax7.text(0.5, 0.5, stats_2023, transform=ax7.transAxes,
        fontsize=10, verticalalignment='center', horizontalalignment='center',
        family='monospace',
        bbox=dict(boxstyle='round,pad=0.8', facecolor='#D6EAF8',
                 edgecolor='#3498DB', linewidth=3, alpha=0.9))

# 8. Statistiques 2025
ax8 = plt.subplot(2, 4, 8)
ax8.axis('off')

stats_2025 = f"""
╔══════════════════════════════╗
║      SYSTÈME 2025            ║
╚══════════════════════════════╝

📊 PERFORMANCE:
  • Taux automatisation: {data['2025']['taux_auto']:.1f}%
  • Accuracy:            {data['2025']['accuracy']:.2f}%
  • Audit humain:        {data['2025']['taux_audit']:.1f}%

🤖 MÉTHODE:
  Machine Learning (XGBoost)
  + Règles métier intelligentes

🎯 AMÉLIORATION:
  • Automatisation: +{improvement_auto:.1f}%
  • Accuracy:       +{improvement_acc:.2f}%
"""

ax8.text(0.5, 0.5, stats_2025, transform=ax8.transAxes,
        fontsize=10, verticalalignment='center', horizontalalignment='center',
        family='monospace',
        bbox=dict(boxstyle='round,pad=0.8', facecolor='#D5F4E6',
                 edgecolor='#2ECC71', linewidth=3, alpha=0.9))

plt.tight_layout(rect=[0, 0, 1, 0.95])
output_path = output_dir / 'G4_comparison_2023_vs_2025.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"✅ Graphique sauvegardé: {output_path}")
plt.close()

print("\n" + "="*80)
print("📊 GRAPHIQUE DE COMPARAISON 2023 vs 2025 GÉNÉRÉ")
print("="*80)
print(f"\n📂 Fichier: {output_path}")
print("\nComparaison:")
print(f"  • Taux d'automatisation: {data['2023']['taux_auto']:.1f}% → {data['2025']['taux_auto']:.1f}% (+{improvement_auto:.1f}%)")
print(f"  • Accuracy:              {data['2023']['accuracy']:.1f}% → {data['2025']['accuracy']:.2f}% (+{improvement_acc:.2f}%)")
print("="*80)
