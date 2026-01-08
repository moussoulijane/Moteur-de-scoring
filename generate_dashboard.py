#!/usr/bin/env python3
"""
Inférence Complète + Graphiques d'Impact
=========================================
- Inférence sur toute la base
- Matrice de confusion
- Impact FP/FN (nombre + montant)
- Gain total automatisation (169 MAD/réclamation)

Usage:
    python inference_impact.py --data fichier.xlsx --output resultats/
"""

import pandas as pd
import numpy as np
import argparse
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings('ignore')

# Configuration
COUT_TRAITEMENT_MANUEL = 169  # MAD par réclamation
SEUIL_REJET = 0.30
SEUIL_VALIDATION = 0.70

# Style des graphiques
plt.style.use('seaborn-v0_8-whitegrid')
COLORS = {
    'fp': '#e74c3c',      # Rouge - Faux Positifs
    'fn': '#f39c12',      # Orange - Faux Négatifs
    'tp': '#27ae60',      # Vert - Vrais Positifs
    'tn': '#3498db',      # Bleu - Vrais Négatifs
    'gain': '#2ecc71',    # Vert clair - Gains
    'perte': '#c0392b',   # Rouge foncé - Pertes
    'auto': '#9b59b6',    # Violet - Automatisé
}


def detect_columns(df):
    """Détecte les colonnes cible et montant."""
    # Cible
    target_col = None
    for col in df.columns:
        if 'fond' in col.lower():
            target_col = col
            break
    
    # Montant
    montant_col = None
    for col in df.columns:
        col_lower = col.lower()
        if 'montant' in col_lower:
            montant_col = col
            break
    
    return target_col, montant_col


def prepare_data(df, target_col):
    """Prépare X et y."""
    y = df[target_col].copy()
    if y.dtype == 'object':
        y = y.apply(lambda x: 1 if str(x).lower() in ['oui', 'yes', '1', 'fondée', 'fondee', 'true', 'o'] else 0)
    elif y.dtype == 'bool':
        y = y.astype(int)
    y = y.fillna(0).astype(int)
    
    X = df.drop(columns=[target_col]).copy()
    cols_to_drop = [c for c in X.columns if any(x in c.lower() for x in ['id', 'date', 'dt_', 'timestamp'])]
    X = X.drop(columns=cols_to_drop, errors='ignore')
    
    for col in X.columns:
        if pd.api.types.is_datetime64_any_dtype(X[col]) or pd.api.types.is_timedelta64_dtype(X[col]):
            X = X.drop(columns=[col])
        elif X[col].dtype == 'bool':
            X[col] = X[col].astype(int)
        elif X[col].dtype == 'object':
            X[col] = LabelEncoder().fit_transform(X[col].astype(str).fillna('NA'))
        elif pd.api.types.is_numeric_dtype(X[col]):
            X[col] = X[col].fillna(0)
    
    X = X.select_dtypes(include=[np.number]).fillna(0)
    return X, y


def train_model(X, y):
    """Entraîne le modèle sur un split puis prédit sur toute la base."""
    import xgboost as xgb
    
    # Split pour entraînement
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.15, stratify=y_train, random_state=42)
    
    # Entraînement
    model = xgb.XGBClassifier(
        max_depth=6, learning_rate=0.05, n_estimators=300,
        random_state=42, verbosity=0
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    
    return model


def plot_confusion_matrix(y_true, y_pred, output_path):
    """Graphique matrice de confusion."""
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Matrice
    matrix = np.array([[tn, fp], [fn, tp]])
    
    # Couleurs
    colors = np.array([[COLORS['tn'], COLORS['fp']], 
                       [COLORS['fn'], COLORS['tp']]])
    
    for i in range(2):
        for j in range(2):
            color = colors[i, j]
            ax.add_patch(plt.Rectangle((j, 1-i), 1, 1, fill=True, color=color, alpha=0.7))
            
            # Valeur
            value = matrix[i, j]
            pct = value / matrix.sum() * 100
            
            # Labels
            labels = [['Vrai Négatif\n(Bon Rejet)', 'Faux Positif\n(Fausse Validation)'],
                      ['Faux Négatif\n(Faux Rejet)', 'Vrai Positif\n(Bonne Validation)']]
            
            ax.text(j + 0.5, 1.5 - i, f'{value:,}', ha='center', va='center',
                    fontsize=28, fontweight='bold', color='white')
            ax.text(j + 0.5, 1.2 - i, f'({pct:.1f}%)', ha='center', va='center',
                    fontsize=14, color='white')
            ax.text(j + 0.5, 0.85 - i + 1, labels[i][j], ha='center', va='center',
                    fontsize=10, color='white', style='italic')
    
    # Axes
    ax.set_xlim(0, 2)
    ax.set_ylim(0, 2)
    ax.set_xticks([0.5, 1.5])
    ax.set_yticks([0.5, 1.5])
    ax.set_xticklabels(['Prédit: Non Fondée', 'Prédit: Fondée'], fontsize=12)
    ax.set_yticklabels(['Réel: Fondée', 'Réel: Non Fondée'], fontsize=12)
    ax.set_xlabel('Prédiction', fontsize=14, fontweight='bold')
    ax.set_ylabel('Réalité', fontsize=14, fontweight='bold')
    
    # Titre
    total = matrix.sum()
    accuracy = (tn + tp) / total * 100
    ax.set_title(f'Matrice de Confusion\nTotal: {total:,} réclamations | Accuracy: {accuracy:.1f}%',
                 fontsize=16, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✅ Matrice de confusion: {output_path}")
    
    return tn, fp, fn, tp


def plot_impact_fp_fn(fp_count, fn_count, fp_montant, fn_montant, output_path):
    """Graphique impact FP et FN (nombre + montant)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # === Graphique 1: Nombre ===
    ax1 = axes[0]
    categories = ['Faux Positifs\n(Fausses Validations)', 'Faux Négatifs\n(Faux Rejets)']
    values = [fp_count, fn_count]
    colors = [COLORS['fp'], COLORS['fn']]
    
    bars1 = ax1.bar(categories, values, color=colors, edgecolor='white', linewidth=2)
    
    # Annotations
    for bar, val in zip(bars1, values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(values)*0.02,
                 f'{val:,}', ha='center', va='bottom', fontsize=16, fontweight='bold')
    
    ax1.set_ylabel('Nombre de réclamations', fontsize=12, fontweight='bold')
    ax1.set_title('Impact en NOMBRE', fontsize=14, fontweight='bold')
    ax1.set_ylim(0, max(values) * 1.15)
    
    # Total
    total_erreurs = fp_count + fn_count
    ax1.text(0.5, 0.95, f'Total erreurs: {total_erreurs:,}', transform=ax1.transAxes,
             ha='center', fontsize=12, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # === Graphique 2: Montant ===
    ax2 = axes[1]
    montants = [fp_montant, fn_montant]
    
    bars2 = ax2.bar(categories, montants, color=colors, edgecolor='white', linewidth=2)
    
    # Annotations
    for bar, val in zip(bars2, montants):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(montants)*0.02,
                 f'{val:,.0f} MAD', ha='center', va='bottom', fontsize=14, fontweight='bold')
    
    ax2.set_ylabel('Montant (MAD)', fontsize=12, fontweight='bold')
    ax2.set_title('Impact en MONTANT', fontsize=14, fontweight='bold')
    ax2.set_ylim(0, max(montants) * 1.15 if max(montants) > 0 else 1)
    
    # Total
    total_montant = fp_montant + fn_montant
    ax2.text(0.5, 0.95, f'Total: {total_montant:,.0f} MAD', transform=ax2.transAxes,
             ha='center', fontsize=12, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Titre global
    fig.suptitle('IMPACT DES ERREURS DE PRÉDICTION', fontsize=16, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✅ Impact FP/FN: {output_path}")


def plot_impact_total(n_auto, n_manuel, gain_auto, fp_montant, fn_montant, output_path):
    """Graphique impact total avec gains et pertes."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # === Graphique 1: Automatisation ===
    ax1 = axes[0]
    
    categories = ['Automatisé', 'Manuel']
    values = [n_auto, n_manuel]
    colors_bar = [COLORS['auto'], '#95a5a6']
    
    bars = ax1.bar(categories, values, color=colors_bar, edgecolor='white', linewidth=2)
    
    total = n_auto + n_manuel
    for bar, val in zip(bars, values):
        pct = val / total * 100
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + total*0.02,
                 f'{val:,}\n({pct:.1f}%)', ha='center', va='bottom', fontsize=14, fontweight='bold')
    
    ax1.set_ylabel('Nombre de réclamations', fontsize=12, fontweight='bold')
    ax1.set_title('TAUX D\'AUTOMATISATION', fontsize=14, fontweight='bold')
    ax1.set_ylim(0, max(values) * 1.2)
    
    # === Graphique 2: Bilan Financier ===
    ax2 = axes[1]
    
    # Calculs
    perte_fp = fp_montant  # Montant perdu sur faux positifs
    perte_fn = fn_montant  # Montant d'insatisfaction (ou à rembourser plus tard)
    gain_net = gain_auto - perte_fp
    
    categories = ['Gain\nAutomatisation', 'Perte\nFaux Positifs', 'Perte\nFaux Négatifs', 'GAIN NET']
    values = [gain_auto, -perte_fp, -perte_fn, gain_net]
    colors_fin = [COLORS['gain'], COLORS['perte'], COLORS['fn'], 
                  COLORS['gain'] if gain_net > 0 else COLORS['perte']]
    
    bars2 = ax2.bar(categories, values, color=colors_fin, edgecolor='white', linewidth=2)
    
    for bar, val in zip(bars2, values):
        y_pos = bar.get_height() + (max(abs(v) for v in values) * 0.02 * (1 if val >= 0 else -1))
        va = 'bottom' if val >= 0 else 'top'
        ax2.text(bar.get_x() + bar.get_width()/2, y_pos,
                 f'{val:+,.0f} MAD', ha='center', va=va, fontsize=12, fontweight='bold')
    
    ax2.axhline(y=0, color='black', linewidth=1)
    ax2.set_ylabel('Montant (MAD)', fontsize=12, fontweight='bold')
    ax2.set_title('BILAN FINANCIER', fontsize=14, fontweight='bold')
    
    # Ajuster les limites
    max_val = max(abs(v) for v in values)
    ax2.set_ylim(-max_val * 1.3, max_val * 1.3)
    
    # Titre global
    fig.suptitle(f'IMPACT TOTAL - Gain par réclamation automatisée: {COUT_TRAITEMENT_MANUEL} MAD',
                 fontsize=16, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✅ Impact total: {output_path}")


def plot_summary_dashboard(metrics, n_auto, n_manuel, gain_auto, fp_count, fn_count, 
                           fp_montant, fn_montant, output_path):
    """Dashboard récapitulatif."""
    fig = plt.figure(figsize=(16, 10))
    
    # Grille
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # === KPIs en haut ===
    ax_kpi = fig.add_subplot(gs[0, :])
    ax_kpi.axis('off')
    
    kpis = [
        ('Accuracy', f"{metrics['accuracy']*100:.1f}%", COLORS['tp']),
        ('Précision', f"{metrics['precision']*100:.1f}%", COLORS['tn']),
        ('Recall', f"{metrics['recall']*100:.1f}%", COLORS['auto']),
        ('F1-Score', f"{metrics['f1']*100:.1f}%", COLORS['gain']),
        ('Taux Auto', f"{n_auto/(n_auto+n_manuel)*100:.1f}%", COLORS['auto']),
    ]
    
    for i, (label, value, color) in enumerate(kpis):
        x = 0.1 + i * 0.18
        ax_kpi.text(x, 0.7, value, fontsize=28, fontweight='bold', color=color,
                    ha='center', transform=ax_kpi.transAxes)
        ax_kpi.text(x, 0.3, label, fontsize=12, ha='center', transform=ax_kpi.transAxes)
    
    ax_kpi.set_title('MÉTRIQUES CLÉS', fontsize=16, fontweight='bold', pad=20)
    
    # === Matrice de confusion (petite) ===
    ax_cm = fig.add_subplot(gs[1, 0])
    cm = metrics['confusion_matrix']
    im = ax_cm.imshow([[cm[0,0], cm[0,1]], [cm[1,0], cm[1,1]]], cmap='Blues', alpha=0.7)
    
    for i in range(2):
        for j in range(2):
            ax_cm.text(j, i, f'{cm[i,j]:,}', ha='center', va='center', fontsize=14, fontweight='bold')
    
    ax_cm.set_xticks([0, 1])
    ax_cm.set_yticks([0, 1])
    ax_cm.set_xticklabels(['Non Fondée', 'Fondée'])
    ax_cm.set_yticklabels(['Non Fondée', 'Fondée'])
    ax_cm.set_xlabel('Prédit')
    ax_cm.set_ylabel('Réel')
    ax_cm.set_title('Matrice de Confusion', fontweight='bold')
    
    # === Erreurs en nombre ===
    ax_err = fig.add_subplot(gs[1, 1])
    bars = ax_err.bar(['FP', 'FN'], [fp_count, fn_count], color=[COLORS['fp'], COLORS['fn']])
    for bar, val in zip(bars, [fp_count, fn_count]):
        ax_err.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{val:,}',
                    ha='center', va='bottom', fontsize=12, fontweight='bold')
    ax_err.set_title('Erreurs (Nombre)', fontweight='bold')
    ax_err.set_ylabel('Nombre')
    
    # === Erreurs en montant ===
    ax_mont = fig.add_subplot(gs[1, 2])
    bars = ax_mont.bar(['FP', 'FN'], [fp_montant, fn_montant], color=[COLORS['fp'], COLORS['fn']])
    for bar, val in zip(bars, [fp_montant, fn_montant]):
        ax_mont.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{val:,.0f}',
                     ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax_mont.set_title('Erreurs (Montant MAD)', fontweight='bold')
    ax_mont.set_ylabel('MAD')
    
    # === Automatisation ===
    ax_auto = fig.add_subplot(gs[2, 0])
    ax_auto.pie([n_auto, n_manuel], labels=['Automatisé', 'Manuel'],
                colors=[COLORS['auto'], '#95a5a6'], autopct='%1.1f%%',
                explode=(0.05, 0), startangle=90)
    ax_auto.set_title('Taux Automatisation', fontweight='bold')
    
    # === Bilan financier ===
    ax_fin = fig.add_subplot(gs[2, 1:])
    gain_net = gain_auto - fp_montant
    
    categories = ['Gain Auto', 'Perte FP', 'GAIN NET']
    values = [gain_auto, -fp_montant, gain_net]
    colors_fin = [COLORS['gain'], COLORS['perte'], COLORS['gain'] if gain_net > 0 else COLORS['perte']]
    
    bars = ax_fin.barh(categories, values, color=colors_fin)
    ax_fin.axvline(x=0, color='black', linewidth=1)
    
    for bar, val in zip(bars, values):
        x_pos = val + (max(abs(v) for v in values) * 0.02 * (1 if val >= 0 else -1))
        ax_fin.text(x_pos, bar.get_y() + bar.get_height()/2, f'{val:+,.0f} MAD',
                    ha='left' if val >= 0 else 'right', va='center', fontsize=11, fontweight='bold')
    
    ax_fin.set_title('Bilan Financier', fontweight='bold')
    ax_fin.set_xlabel('MAD')
    
    # Titre global
    fig.suptitle('DASHBOARD RÉCAPITULATIF - INFÉRENCE SUR TOUTE LA BASE',
                 fontsize=18, fontweight='bold', y=0.98)
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✅ Dashboard: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', '-d', required=True, help='Fichier Excel')
    parser.add_argument('--output', '-o', default='resultats_inference', help='Dossier de sortie')
    parser.add_argument('--montant-col', '-m', default=None, help='Nom de la colonne montant (auto-détecté si non spécifié)')
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("INFÉRENCE COMPLÈTE + GRAPHIQUES D'IMPACT")
    print("=" * 60)
    
    # 1. Charger
    print(f"\n📂 Chargement: {args.data}")
    df = pd.read_excel(args.data)
    print(f"   {len(df)} lignes")
    
    # 2. Détecter colonnes
    target_col, montant_col = detect_columns(df)
    if args.montant_col:
        montant_col = args.montant_col
    
    print(f"   Cible: {target_col}")
    print(f"   Montant: {montant_col}")
    
    if montant_col is None:
        print("   ⚠️ Colonne montant non trouvée, utilisation de 0")
    
    # 3. Préparer
    X, y = prepare_data(df, target_col)
    print(f"   Features: {X.shape[1]}")
    
    # 4. Entraîner
    print("\n🚀 Entraînement XGBoost...")
    model = train_model(X, y)
    
    # 5. Inférence sur TOUTE la base
    print("\n🔮 Inférence sur toute la base...")
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]
    
    # 6. Calculs
    cm = confusion_matrix(y, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    # Montants
    if montant_col and montant_col in df.columns:
        montants = df[montant_col].fillna(0).values
    else:
        montants = np.zeros(len(df))
    
    # Masques
    mask_fp = (y_pred == 1) & (y.values == 0)
    mask_fn = (y_pred == 0) & (y.values == 1)
    
    fp_montant = montants[mask_fp].sum()
    fn_montant = montants[mask_fn].sum()
    
    # Automatisation
    mask_auto = (y_proba <= SEUIL_REJET) | (y_proba >= SEUIL_VALIDATION)
    n_auto = mask_auto.sum()
    n_manuel = len(df) - n_auto
    
    gain_auto = n_auto * COUT_TRAITEMENT_MANUEL
    
    # Métriques
    metrics = {
        'accuracy': accuracy_score(y, y_pred),
        'precision': precision_score(y, y_pred),
        'recall': recall_score(y, y_pred),
        'f1': f1_score(y, y_pred),
        'confusion_matrix': cm
    }
    
    # 7. Afficher résultats
    print(f"\n📊 RÉSULTATS:")
    print(f"   Accuracy: {metrics['accuracy']*100:.1f}%")
    print(f"   Précision: {metrics['precision']*100:.1f}%")
    print(f"   Recall: {metrics['recall']*100:.1f}%")
    print(f"   F1-Score: {metrics['f1']*100:.1f}%")
    print(f"\n   Matrice de confusion:")
    print(f"   TN={tn:,} | FP={fp:,}")
    print(f"   FN={fn:,} | TP={tp:,}")
    print(f"\n   Faux Positifs: {fp:,} ({fp_montant:,.0f} MAD)")
    print(f"   Faux Négatifs: {fn:,} ({fn_montant:,.0f} MAD)")
    print(f"\n   Automatisé: {n_auto:,} ({n_auto/len(df)*100:.1f}%)")
    print(f"   Manuel: {n_manuel:,} ({n_manuel/len(df)*100:.1f}%)")
    print(f"\n   Gain automatisation: {gain_auto:,.0f} MAD")
    print(f"   Gain net (après pertes FP): {gain_auto - fp_montant:,.0f} MAD")
    
    # 8. Générer graphiques
    print(f"\n📈 Génération des graphiques...")
    
    # Matrice de confusion
    plot_confusion_matrix(y, y_pred, output_dir / '1_matrice_confusion.png')
    
    # Impact FP/FN
    plot_impact_fp_fn(fp, fn, fp_montant, fn_montant, output_dir / '2_impact_fp_fn.png')
    
    # Impact total
    plot_impact_total(n_auto, n_manuel, gain_auto, fp_montant, fn_montant,
                      output_dir / '3_impact_total.png')
    
    # Dashboard
    plot_summary_dashboard(metrics, n_auto, n_manuel, gain_auto, fp, fn,
                           fp_montant, fn_montant, output_dir / '4_dashboard.png')
    
    # 9. Export Excel récapitulatif
    print(f"\n💾 Export Excel...")
    summary_data = {
        'Métrique': ['Accuracy', 'Précision', 'Recall', 'F1-Score', 
                     'Vrais Négatifs', 'Faux Positifs', 'Faux Négatifs', 'Vrais Positifs',
                     'Montant FP (MAD)', 'Montant FN (MAD)',
                     'N Automatisé', 'N Manuel', 'Taux Automatisation',
                     'Gain Automatisation (MAD)', 'Gain Net (MAD)'],
        'Valeur': [f"{metrics['accuracy']*100:.1f}%", f"{metrics['precision']*100:.1f}%",
                   f"{metrics['recall']*100:.1f}%", f"{metrics['f1']*100:.1f}%",
                   tn, fp, fn, tp,
                   f"{fp_montant:,.0f}", f"{fn_montant:,.0f}",
                   n_auto, n_manuel, f"{n_auto/len(df)*100:.1f}%",
                   f"{gain_auto:,.0f}", f"{gain_auto - fp_montant:,.0f}"]
    }
    
    df_summary = pd.DataFrame(summary_data)
    df_summary.to_excel(output_dir / 'resume_inference.xlsx', index=False)
    print(f"✅ Résumé Excel: {output_dir / 'resume_inference.xlsx'}")
    
    print("\n" + "=" * 60)
    print("✅ TERMINÉ")
    print("=" * 60)
    print(f"\n📁 Fichiers générés dans: {output_dir}")
    print("   • 1_matrice_confusion.png")
    print("   • 2_impact_fp_fn.png")
    print("   • 3_impact_total.png")
    print("   • 4_dashboard.png")
    print("   • resume_inference.xlsx")


if __name__ == "__main__":
    main()
