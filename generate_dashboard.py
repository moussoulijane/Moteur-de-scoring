#!/usr/bin/env python3
"""
=============================================================================
PROJET COMPLET - CLASSIFICATION DES RÉCLAMATIONS BANCAIRES
=============================================================================
Génère toutes les visualisations et analyses:
1. Matrice de confusion (test set)
2. Performance par famille
3. Impact total
4. Impact automatisation familles >95% accuracy
5. Impact automatisation familles >90% accuracy
6. Familles avec le plus de pertes (montant)

Usage:
    python run_project.py --data fichier.xlsx --output outputs/
=============================================================================
"""

import pandas as pd
import numpy as np
import argparse
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score, 
    recall_score, f1_score, roc_auc_score
)
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================
COUT_TRAITEMENT_MANUEL = 169  # MAD par réclamation
SEUIL_REJET = 0.30
SEUIL_VALIDATION = 0.70
SEUIL_ACCURACY_HIGH = 0.95  # 95%
SEUIL_ACCURACY_MEDIUM = 0.90  # 90%

# Couleurs
COLORS = {
    'fp': '#e74c3c',
    'fn': '#f39c12',
    'tp': '#27ae60',
    'tn': '#3498db',
    'gain': '#2ecc71',
    'perte': '#c0392b',
    'auto': '#9b59b6',
    'excellent': '#27ae60',
    'good': '#3498db',
    'warning': '#f39c12',
    'danger': '#e74c3c',
}

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 10


# =============================================================================
# FONCTIONS UTILITAIRES
# =============================================================================

def detect_columns(df):
    """Détecte les colonnes importantes."""
    target_col = None
    for col in df.columns:
        if 'fond' in col.lower():
            target_col = col
            break
    
    montant_col = None
    for col in df.columns:
        if 'montant' in col.lower():
            montant_col = col
            break
    
    famille_col = None
    for col in df.columns:
        col_lower = col.lower()
        if 'famille' in col_lower:
            famille_col = col
            break
    
    return target_col, montant_col, famille_col


def prepare_target(df, target_col):
    """Prépare la cible."""
    y = df[target_col].copy()
    if y.dtype == 'object':
        y = y.apply(lambda x: 1 if str(x).lower() in ['oui', 'yes', '1', 'fondée', 'fondee', 'true', 'o'] else 0)
    elif y.dtype == 'bool':
        y = y.astype(int)
    return y.fillna(0).astype(int)


def prepare_features(df, target_col):
    """Prépare les features."""
    X = df.drop(columns=[target_col]).copy()
    cols_to_drop = [c for c in X.columns if any(x in c.lower() for x in ['id', 'date', 'dt_', 'timestamp', 'num_'])]
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
    
    return X.select_dtypes(include=[np.number]).fillna(0)


def train_xgboost(X_train, y_train, X_val, y_val):
    """Entraîne XGBoost."""
    import xgboost as xgb
    
    model = xgb.XGBClassifier(
        max_depth=6,
        learning_rate=0.05,
        n_estimators=300,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=0
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    return model


# =============================================================================
# VISUALISATIONS
# =============================================================================

def plot_confusion_matrix(y_true, y_pred, output_path):
    """1. Matrice de confusion sur le test set."""
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    total = cm.sum()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Dessiner les cellules
    cell_colors = [[COLORS['tn'], COLORS['fp']], [COLORS['fn'], COLORS['tp']]]
    labels = [['Vrai Négatif\n(Bon Rejet)', 'Faux Positif\n(Fausse Validation)'],
              ['Faux Négatif\n(Faux Rejet)', 'Vrai Positif\n(Bonne Validation)']]
    values = [[tn, fp], [fn, tp]]
    
    for i in range(2):
        for j in range(2):
            rect = plt.Rectangle((j, 1-i), 1, 1, fill=True, 
                                   color=cell_colors[i][j], alpha=0.75)
            ax.add_patch(rect)
            
            val = values[i][j]
            pct = val / total * 100
            
            ax.text(j + 0.5, 1.55 - i, f'{val:,}', ha='center', va='center',
                    fontsize=30, fontweight='bold', color='white')
            ax.text(j + 0.5, 1.25 - i, f'({pct:.1f}%)', ha='center', va='center',
                    fontsize=14, color='white')
            ax.text(j + 0.5, 0.95 - i + 1, labels[i][j], ha='center', va='center',
                    fontsize=9, color='white', style='italic')
    
    ax.set_xlim(0, 2)
    ax.set_ylim(0, 2)
    ax.set_xticks([0.5, 1.5])
    ax.set_yticks([0.5, 1.5])
    ax.set_xticklabels(['Prédit: Non Fondée', 'Prédit: Fondée'], fontsize=11)
    ax.set_yticklabels(['Réel: Fondée', 'Réel: Non Fondée'], fontsize=11)
    ax.set_xlabel('PRÉDICTION', fontsize=12, fontweight='bold')
    ax.set_ylabel('RÉALITÉ', fontsize=12, fontweight='bold')
    
    accuracy = (tn + tp) / total * 100
    ax.set_title(f'MATRICE DE CONFUSION - ENSEMBLE DE TEST\n'
                 f'Total: {total:,} | Accuracy: {accuracy:.1f}%',
                 fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    return {'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp, 'accuracy': accuracy}


def plot_performance_par_famille(df_test, y_test, y_pred, famille_col, output_path):
    """2. Performance par famille avec barres horizontales."""
    results = []
    
    for famille in df_test[famille_col].unique():
        mask = df_test[famille_col] == famille
        if mask.sum() < 5:
            continue
        
        y_t = y_test[mask]
        y_p = y_pred[mask]
        
        acc = accuracy_score(y_t, y_p)
        prec = precision_score(y_t, y_p, zero_division=0)
        rec = recall_score(y_t, y_p, zero_division=0)
        f1 = f1_score(y_t, y_p, zero_division=0)
        
        results.append({
            'Famille': famille,
            'N': mask.sum(),
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'F1': f1
        })
    
    df_perf = pd.DataFrame(results).sort_values('Accuracy', ascending=True)
    
    # Graphique
    fig, ax = plt.subplots(figsize=(12, max(8, len(df_perf) * 0.4)))
    
    y_pos = np.arange(len(df_perf))
    
    # Couleurs selon accuracy
    colors = []
    for acc in df_perf['Accuracy']:
        if acc >= 0.95:
            colors.append(COLORS['excellent'])
        elif acc >= 0.90:
            colors.append(COLORS['good'])
        elif acc >= 0.80:
            colors.append(COLORS['warning'])
        else:
            colors.append(COLORS['danger'])
    
    bars = ax.barh(y_pos, df_perf['Accuracy'] * 100, color=colors, edgecolor='white', height=0.7)
    
    # Annotations
    for i, (idx, row) in enumerate(df_perf.iterrows()):
        ax.text(row['Accuracy'] * 100 + 1, i, f"{row['Accuracy']*100:.1f}%",
                va='center', fontsize=10, fontweight='bold')
        ax.text(2, i, f"N={row['N']}", va='center', fontsize=8, color='white')
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f[:35] for f in df_perf['Famille']], fontsize=9)
    ax.set_xlabel('Accuracy (%)', fontsize=11, fontweight='bold')
    ax.set_xlim(0, 110)
    
    # Lignes de référence
    ax.axvline(x=95, color='green', linestyle='--', linewidth=2, label='Objectif 95%')
    ax.axvline(x=90, color='orange', linestyle='--', linewidth=2, label='Seuil 90%')
    
    ax.legend(loc='lower right')
    ax.set_title('PERFORMANCE PAR FAMILLE PRODUIT', fontsize=14, fontweight='bold', pad=15)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    return df_perf


def plot_impact_total(n_total, n_auto, fp_count, fn_count, fp_montant, fn_montant, output_path):
    """3. Impact total."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    gain_auto = n_auto * COUT_TRAITEMENT_MANUEL
    n_manuel = n_total - n_auto
    
    # === 1. Automatisation (pie) ===
    ax1 = axes[0, 0]
    sizes = [n_auto, n_manuel]
    labels = [f'Automatisé\n{n_auto:,} ({n_auto/n_total*100:.1f}%)',
              f'Manuel\n{n_manuel:,} ({n_manuel/n_total*100:.1f}%)']
    colors_pie = [COLORS['auto'], '#bdc3c7']
    explode = (0.05, 0)
    
    ax1.pie(sizes, labels=labels, colors=colors_pie, explode=explode,
            autopct='', startangle=90, textprops={'fontsize': 10})
    ax1.set_title('TAUX D\'AUTOMATISATION', fontsize=12, fontweight='bold')
    
    # === 2. Erreurs en nombre ===
    ax2 = axes[0, 1]
    cats = ['Faux Positifs', 'Faux Négatifs']
    vals = [fp_count, fn_count]
    colors_bar = [COLORS['fp'], COLORS['fn']]
    
    bars = ax2.bar(cats, vals, color=colors_bar, edgecolor='white', linewidth=2)
    for bar, val in zip(bars, vals):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(vals)*0.02,
                 f'{val:,}', ha='center', va='bottom', fontsize=14, fontweight='bold')
    
    ax2.set_ylabel('Nombre', fontsize=11)
    ax2.set_title(f'ERREURS EN NOMBRE\nTotal: {fp_count + fn_count:,}', fontsize=12, fontweight='bold')
    
    # === 3. Erreurs en montant ===
    ax3 = axes[1, 0]
    montants = [fp_montant, fn_montant]
    
    bars2 = ax3.bar(cats, montants, color=colors_bar, edgecolor='white', linewidth=2)
    for bar, val in zip(bars2, montants):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(montants)*0.02,
                 f'{val:,.0f} MAD', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax3.set_ylabel('Montant (MAD)', fontsize=11)
    ax3.set_title(f'ERREURS EN MONTANT\nTotal: {fp_montant + fn_montant:,.0f} MAD', 
                  fontsize=12, fontweight='bold')
    
    # === 4. Bilan financier ===
    ax4 = axes[1, 1]
    
    gain_net = gain_auto - fp_montant
    cats_fin = ['Gain\nAutomatisation', 'Perte\nFaux Positifs', 'GAIN NET']
    vals_fin = [gain_auto, -fp_montant, gain_net]
    colors_fin = [COLORS['gain'], COLORS['perte'], 
                  COLORS['gain'] if gain_net > 0 else COLORS['perte']]
    
    bars3 = ax4.bar(cats_fin, vals_fin, color=colors_fin, edgecolor='white', linewidth=2)
    for bar, val in zip(bars3, vals_fin):
        y_pos = bar.get_height() + (max(abs(v) for v in vals_fin) * 0.03 * (1 if val >= 0 else -1))
        ax4.text(bar.get_x() + bar.get_width()/2, y_pos,
                 f'{val:+,.0f} MAD', ha='center', va='bottom' if val >= 0 else 'top',
                 fontsize=11, fontweight='bold')
    
    ax4.axhline(y=0, color='black', linewidth=1)
    ax4.set_ylabel('Montant (MAD)', fontsize=11)
    ax4.set_title('BILAN FINANCIER', fontsize=12, fontweight='bold')
    
    fig.suptitle(f'IMPACT TOTAL - {n_total:,} RÉCLAMATIONS\n'
                 f'(Gain: {COUT_TRAITEMENT_MANUEL} MAD par réclamation automatisée)',
                 fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    return {'gain_auto': gain_auto, 'gain_net': gain_net}


def plot_impact_by_threshold(df_perf, df_test, y_test, y_pred, y_proba, 
                              famille_col, montant_col, threshold, title, output_path):
    """4 & 5. Impact si on automatise les familles > seuil accuracy."""
    
    # Familles éligibles
    familles_ok = df_perf[df_perf['Accuracy'] >= threshold]['Famille'].tolist()
    
    if len(familles_ok) == 0:
        print(f"   ⚠️ Aucune famille avec accuracy >= {threshold*100:.0f}%")
        return None
    
    # Filtrer
    mask = df_test[famille_col].isin(familles_ok)
    df_filtered = df_test[mask]
    y_t = y_test[mask]
    y_p = y_pred[mask]
    y_prob = y_proba[mask]
    
    n_total = len(df_filtered)
    
    # Calculs
    mask_auto = (y_prob <= SEUIL_REJET) | (y_prob >= SEUIL_VALIDATION)
    n_auto = mask_auto.sum()
    n_manuel = n_total - n_auto
    
    mask_fp = (y_p == 1) & (y_t == 0)
    mask_fn = (y_p == 0) & (y_t == 1)
    
    fp_count = mask_fp.sum()
    fn_count = mask_fn.sum()
    
    if montant_col and montant_col in df_filtered.columns:
        montants = df_filtered[montant_col].fillna(0).values
        fp_montant = montants[mask_fp].sum()
        fn_montant = montants[mask_fn].sum()
    else:
        fp_montant, fn_montant = 0, 0
    
    gain_auto = n_auto * COUT_TRAITEMENT_MANUEL
    gain_net = gain_auto - fp_montant
    
    # Graphique
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 1. Familles concernées
    ax1 = axes[0]
    df_ok = df_perf[df_perf['Famille'].isin(familles_ok)].sort_values('Accuracy', ascending=True)
    y_pos = np.arange(len(df_ok))
    
    colors = [COLORS['excellent'] if acc >= 0.95 else COLORS['good'] 
              for acc in df_ok['Accuracy']]
    
    ax1.barh(y_pos, df_ok['Accuracy'] * 100, color=colors, edgecolor='white')
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels([f[:25] for f in df_ok['Famille']], fontsize=8)
    ax1.set_xlabel('Accuracy (%)')
    ax1.axvline(x=threshold * 100, color='red', linestyle='--', linewidth=2)
    ax1.set_title(f'{len(familles_ok)} FAMILLES ÉLIGIBLES', fontsize=11, fontweight='bold')
    
    # 2. Erreurs
    ax2 = axes[1]
    cats = ['Faux Positifs', 'Faux Négatifs']
    
    x = np.arange(2)
    width = 0.35
    
    bars1 = ax2.bar(x - width/2, [fp_count, fn_count], width, label='Nombre', color=[COLORS['fp'], COLORS['fn']])
    
    ax2.set_xticks(x)
    ax2.set_xticklabels(cats)
    ax2.set_ylabel('Nombre')
    ax2.set_title(f'ERREURS\nFP: {fp_count} | FN: {fn_count}', fontsize=11, fontweight='bold')
    
    for bar, val in zip(bars1, [fp_count, fn_count]):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                 f'{val}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # 3. Bilan
    ax3 = axes[2]
    cats_fin = ['Gain Auto', 'Perte FP', 'GAIN NET']
    vals_fin = [gain_auto, -fp_montant, gain_net]
    colors_fin = [COLORS['gain'], COLORS['perte'], 
                  COLORS['gain'] if gain_net > 0 else COLORS['perte']]
    
    bars3 = ax3.bar(cats_fin, vals_fin, color=colors_fin, edgecolor='white')
    ax3.axhline(y=0, color='black', linewidth=1)
    
    for bar, val in zip(bars3, vals_fin):
        y_p = bar.get_height() + (max(abs(v) for v in vals_fin) * 0.05 * (1 if val >= 0 else -1))
        ax3.text(bar.get_x() + bar.get_width()/2, y_p,
                 f'{val:+,.0f}', ha='center', va='bottom' if val >= 0 else 'top',
                 fontsize=10, fontweight='bold')
    
    ax3.set_ylabel('MAD')
    ax3.set_title(f'BILAN FINANCIER\n{n_auto:,} auto / {n_total:,} total', 
                  fontsize=11, fontweight='bold')
    
    fig.suptitle(f'{title}\n({n_total:,} réclamations)', fontsize=13, fontweight='bold', y=1.05)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    return {
        'familles': familles_ok,
        'n_total': n_total,
        'n_auto': n_auto,
        'fp': fp_count,
        'fn': fn_count,
        'fp_montant': fp_montant,
        'fn_montant': fn_montant,
        'gain_auto': gain_auto,
        'gain_net': gain_net
    }


def plot_familles_plus_pertes(df_test, y_test, y_pred, famille_col, montant_col, output_path):
    """6. Familles avec le plus de pertes en montant."""
    
    results = []
    
    for famille in df_test[famille_col].unique():
        mask = df_test[famille_col] == famille
        if mask.sum() < 5:
            continue
        
        y_t = y_test[mask]
        y_p = y_pred[mask]
        
        mask_fp = (y_p == 1) & (y_t == 0)
        mask_fn = (y_p == 0) & (y_t == 1)
        
        if montant_col and montant_col in df_test.columns:
            sub = df_test[mask]
            fp_montant = sub.loc[mask_fp, montant_col].fillna(0).sum()
            fn_montant = sub.loc[mask_fn, montant_col].fillna(0).sum()
        else:
            fp_montant, fn_montant = 0, 0
        
        total_perte = fp_montant + fn_montant
        
        results.append({
            'Famille': famille,
            'N': mask.sum(),
            'FP': mask_fp.sum(),
            'FN': mask_fn.sum(),
            'Montant_FP': fp_montant,
            'Montant_FN': fn_montant,
            'Total_Perte': total_perte,
            'Accuracy': accuracy_score(y_t, y_p)
        })
    
    df_pertes = pd.DataFrame(results).sort_values('Total_Perte', ascending=False)
    
    # Top 10 pertes
    df_top = df_pertes.head(10)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 1. Pertes par famille (barres empilées)
    ax1 = axes[0]
    y_pos = np.arange(len(df_top))
    
    bars_fp = ax1.barh(y_pos, df_top['Montant_FP'], color=COLORS['fp'], label='Pertes FP', edgecolor='white')
    bars_fn = ax1.barh(y_pos, df_top['Montant_FN'], left=df_top['Montant_FP'], 
                       color=COLORS['fn'], label='Pertes FN', edgecolor='white')
    
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels([f[:30] for f in df_top['Famille']], fontsize=9)
    ax1.set_xlabel('Montant (MAD)', fontsize=11)
    ax1.legend(loc='lower right')
    ax1.set_title('TOP 10 FAMILLES - PERTES EN MONTANT', fontsize=12, fontweight='bold')
    
    # Annotations
    for i, (idx, row) in enumerate(df_top.iterrows()):
        total = row['Total_Perte']
        ax1.text(total + max(df_top['Total_Perte'])*0.02, i, 
                 f'{total:,.0f} MAD', va='center', fontsize=9, fontweight='bold')
    
    # 2. Détail avec accuracy
    ax2 = axes[1]
    
    x = np.arange(len(df_top))
    width = 0.6
    
    colors = [COLORS['danger'] if acc < 0.90 else COLORS['warning'] if acc < 0.95 else COLORS['excellent']
              for acc in df_top['Accuracy']]
    
    bars = ax2.bar(x, df_top['Accuracy'] * 100, width, color=colors, edgecolor='white')
    
    ax2.axhline(y=95, color='green', linestyle='--', linewidth=2, label='Objectif 95%')
    ax2.axhline(y=90, color='orange', linestyle='--', linewidth=2, label='Seuil 90%')
    
    ax2.set_xticks(x)
    ax2.set_xticklabels([f[:15] for f in df_top['Famille']], rotation=45, ha='right', fontsize=8)
    ax2.set_ylabel('Accuracy (%)', fontsize=11)
    ax2.set_ylim(0, 105)
    ax2.legend(loc='lower right')
    ax2.set_title('ACCURACY DES FAMILLES À RISQUE', fontsize=12, fontweight='bold')
    
    for bar, (idx, row) in zip(bars, df_top.iterrows()):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                 f'{row["Accuracy"]*100:.0f}%', ha='center', va='bottom', fontsize=8)
    
    fig.suptitle('ANALYSE DES FAMILLES AVEC LE PLUS DE PERTES', fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    return df_pertes


# =============================================================================
# EXPORT EXCEL
# =============================================================================

def export_excel(output_dir, cm_results, df_perf, impact_total, impact_95, impact_90, df_pertes):
    """Exporte tous les résultats en Excel."""
    
    with pd.ExcelWriter(output_dir / 'rapport_complet.xlsx', engine='openpyxl') as writer:
        
        # 1. Résumé global
        summary = pd.DataFrame({
            'Métrique': [
                'Total Test', 'Accuracy', 'Vrais Négatifs', 'Faux Positifs', 
                'Faux Négatifs', 'Vrais Positifs', 'Gain Automatisation (MAD)', 'Gain Net (MAD)'
            ],
            'Valeur': [
                cm_results['tn'] + cm_results['fp'] + cm_results['fn'] + cm_results['tp'],
                f"{cm_results['accuracy']:.1f}%",
                cm_results['tn'], cm_results['fp'], cm_results['fn'], cm_results['tp'],
                f"{impact_total['gain_auto']:,.0f}",
                f"{impact_total['gain_net']:,.0f}"
            ]
        })
        summary.to_excel(writer, sheet_name='Resume_Global', index=False)
        
        # 2. Performance par famille
        df_perf_export = df_perf.copy()
        df_perf_export['Accuracy'] = df_perf_export['Accuracy'].apply(lambda x: f"{x*100:.1f}%")
        df_perf_export['Precision'] = df_perf_export['Precision'].apply(lambda x: f"{x*100:.1f}%")
        df_perf_export['Recall'] = df_perf_export['Recall'].apply(lambda x: f"{x*100:.1f}%")
        df_perf_export['F1'] = df_perf_export['F1'].apply(lambda x: f"{x*100:.1f}%")
        df_perf_export.to_excel(writer, sheet_name='Performance_Famille', index=False)
        
        # 3. Impact seuil 95%
        if impact_95:
            df_95 = pd.DataFrame({
                'Métrique': ['Familles éligibles', 'N Total', 'N Automatisé', 'FP', 'FN',
                            'Montant FP', 'Montant FN', 'Gain Auto', 'Gain Net'],
                'Valeur': [len(impact_95['familles']), impact_95['n_total'], impact_95['n_auto'],
                          impact_95['fp'], impact_95['fn'], f"{impact_95['fp_montant']:,.0f}",
                          f"{impact_95['fn_montant']:,.0f}", f"{impact_95['gain_auto']:,.0f}",
                          f"{impact_95['gain_net']:,.0f}"]
            })
            df_95.to_excel(writer, sheet_name='Impact_Seuil_95', index=False)
            
            # Liste des familles
            pd.DataFrame({'Familles_95%': impact_95['familles']}).to_excel(
                writer, sheet_name='Familles_95', index=False)
        
        # 4. Impact seuil 90%
        if impact_90:
            df_90 = pd.DataFrame({
                'Métrique': ['Familles éligibles', 'N Total', 'N Automatisé', 'FP', 'FN',
                            'Montant FP', 'Montant FN', 'Gain Auto', 'Gain Net'],
                'Valeur': [len(impact_90['familles']), impact_90['n_total'], impact_90['n_auto'],
                          impact_90['fp'], impact_90['fn'], f"{impact_90['fp_montant']:,.0f}",
                          f"{impact_90['fn_montant']:,.0f}", f"{impact_90['gain_auto']:,.0f}",
                          f"{impact_90['gain_net']:,.0f}"]
            })
            df_90.to_excel(writer, sheet_name='Impact_Seuil_90', index=False)
            
            pd.DataFrame({'Familles_90%': impact_90['familles']}).to_excel(
                writer, sheet_name='Familles_90', index=False)
        
        # 5. Familles avec pertes
        df_pertes_export = df_pertes.copy()
        df_pertes_export['Accuracy'] = df_pertes_export['Accuracy'].apply(lambda x: f"{x*100:.1f}%")
        df_pertes_export.to_excel(writer, sheet_name='Familles_Pertes', index=False)
    
    print(f"✅ Rapport Excel: {output_dir / 'rapport_complet.xlsx'}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Projet complet classification réclamations')
    parser.add_argument('--data', '-d', required=True, help='Fichier Excel')
    parser.add_argument('--output', '-o', default='outputs', help='Dossier de sortie')
    parser.add_argument('--montant-col', '-m', default=None, help='Colonne montant')
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    viz_dir = output_dir / 'visualizations'
    viz_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("PROJET COMPLET - CLASSIFICATION DES RÉCLAMATIONS BANCAIRES")
    print("=" * 70)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    
    # 1. Charger
    print(f"\n📂 Chargement: {args.data}")
    df = pd.read_excel(args.data)
    print(f"   {len(df):,} lignes")
    
    # 2. Détecter colonnes
    target_col, montant_col, famille_col = detect_columns(df)
    if args.montant_col:
        montant_col = args.montant_col
    
    print(f"   Cible: {target_col}")
    print(f"   Montant: {montant_col}")
    print(f"   Famille: {famille_col}")
    
    # 3. Préparer données
    print("\n🔧 Préparation des données...")
    y = prepare_target(df, target_col)
    X = prepare_features(df, target_col)
    print(f"   Features: {X.shape[1]}")
    
    # 4. Split
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, df.index, test_size=0.2, stratify=y, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.15, stratify=y_train, random_state=42
    )
    
    df_test = df.iloc[idx_test].reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)
    
    print(f"   Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}")
    
    # 5. Entraîner
    print("\n🚀 Entraînement XGBoost...")
    model = train_xgboost(X_train, y_train, X_val, y_val)
    
    # 6. Prédictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # 7. Calculs globaux
    mask_auto = (y_proba <= SEUIL_REJET) | (y_proba >= SEUIL_VALIDATION)
    n_auto = mask_auto.sum()
    
    mask_fp = (y_pred == 1) & (y_test.values == 0)
    mask_fn = (y_pred == 0) & (y_test.values == 1)
    
    if montant_col and montant_col in df_test.columns:
        montants = df_test[montant_col].fillna(0).values
        fp_montant = montants[mask_fp].sum()
        fn_montant = montants[mask_fn].sum()
    else:
        fp_montant, fn_montant = 0, 0
    
    # ==========================================================================
    # GÉNÉRATION DES VISUALISATIONS
    # ==========================================================================
    print("\n📊 Génération des visualisations...")
    
    # 1. Matrice de confusion
    print("   1. Matrice de confusion...")
    cm_results = plot_confusion_matrix(y_test, y_pred, viz_dir / '1_matrice_confusion.png')
    
    # 2. Performance par famille
    print("   2. Performance par famille...")
    df_perf = plot_performance_par_famille(df_test, y_test, y_pred, famille_col,
                                            viz_dir / '2_performance_famille.png')
    
    # 3. Impact total
    print("   3. Impact total...")
    impact_total = plot_impact_total(
        len(df_test), n_auto, mask_fp.sum(), mask_fn.sum(),
        fp_montant, fn_montant, viz_dir / '3_impact_total.png'
    )
    
    # 4. Impact seuil 95%
    print("   4. Impact familles >95% accuracy...")
    impact_95 = plot_impact_by_threshold(
        df_perf, df_test, y_test, y_pred, y_proba,
        famille_col, montant_col, SEUIL_ACCURACY_HIGH,
        'IMPACT - FAMILLES AVEC ACCURACY ≥ 95%',
        viz_dir / '4_impact_seuil_95.png'
    )
    
    # 5. Impact seuil 90%
    print("   5. Impact familles >90% accuracy...")
    impact_90 = plot_impact_by_threshold(
        df_perf, df_test, y_test, y_pred, y_proba,
        famille_col, montant_col, SEUIL_ACCURACY_MEDIUM,
        'IMPACT - FAMILLES AVEC ACCURACY ≥ 90%',
        viz_dir / '5_impact_seuil_90.png'
    )
    
    # 6. Familles avec plus de pertes
    print("   6. Familles avec le plus de pertes...")
    df_pertes = plot_familles_plus_pertes(
        df_test, y_test, y_pred, famille_col, montant_col,
        viz_dir / '6_familles_pertes.png'
    )
    
    # ==========================================================================
    # EXPORT EXCEL
    # ==========================================================================
    print("\n💾 Export Excel...")
    export_excel(output_dir, cm_results, df_perf, impact_total, impact_95, impact_90, df_pertes)
    
    # ==========================================================================
    # RÉSUMÉ
    # ==========================================================================
    print("\n" + "=" * 70)
    print("📋 RÉSUMÉ")
    print("=" * 70)
    print(f"\n   Accuracy globale: {cm_results['accuracy']:.1f}%")
    print(f"   Taux automatisation: {n_auto/len(df_test)*100:.1f}%")
    print(f"   Gain net: {impact_total['gain_net']:,.0f} MAD")
    
    if impact_95:
        print(f"\n   Familles ≥95%: {len(impact_95['familles'])} familles")
        print(f"   → Gain net: {impact_95['gain_net']:,.0f} MAD")
    
    if impact_90:
        print(f"\n   Familles ≥90%: {len(impact_90['familles'])} familles")
        print(f"   → Gain net: {impact_90['gain_net']:,.0f} MAD")
    
    print(f"\n   Top 3 familles à risque (pertes):")
    for i, (_, row) in enumerate(df_pertes.head(3).iterrows(), 1):
        print(f"   {i}. {row['Famille'][:30]} : {row['Total_Perte']:,.0f} MAD")
    
    print("\n" + "=" * 70)
    print("✅ PROJET TERMINÉ")
    print("=" * 70)
    print(f"\n📁 Fichiers générés dans: {output_dir}")
    print(f"   📊 visualizations/")
    print(f"      • 1_matrice_confusion.png")
    print(f"      • 2_performance_famille.png")
    print(f"      • 3_impact_total.png")
    print(f"      • 4_impact_seuil_95.png")
    print(f"      • 5_impact_seuil_90.png")
    print(f"      • 6_familles_pertes.png")
    print(f"   📄 rapport_complet.xlsx")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
=============================================================================
PROJET COMPLET - CLASSIFICATION DES RÉCLAMATIONS BANCAIRES
=============================================================================
Génère toutes les visualisations et analyses:
1. Matrice de confusion (test set)
2. Performance par famille
3. Impact total
4. Impact automatisation familles >95% accuracy
5. Impact automatisation familles >90% accuracy
6. Familles avec le plus de pertes (montant)

Usage:
    python run_project.py --data fichier.xlsx --output outputs/
=============================================================================
"""

import pandas as pd
import numpy as np
import argparse
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score, 
    recall_score, f1_score, roc_auc_score
)
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================
COUT_TRAITEMENT_MANUEL = 169  # MAD par réclamation
SEUIL_REJET = 0.30
SEUIL_VALIDATION = 0.70
SEUIL_ACCURACY_HIGH = 0.95  # 95%
SEUIL_ACCURACY_MEDIUM = 0.90  # 90%

# Couleurs
COLORS = {
    'fp': '#e74c3c',
    'fn': '#f39c12',
    'tp': '#27ae60',
    'tn': '#3498db',
    'gain': '#2ecc71',
    'perte': '#c0392b',
    'auto': '#9b59b6',
    'excellent': '#27ae60',
    'good': '#3498db',
    'warning': '#f39c12',
    'danger': '#e74c3c',
}

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 10


# =============================================================================
# FONCTIONS UTILITAIRES
# =============================================================================

def detect_columns(df):
    """Détecte les colonnes importantes."""
    target_col = None
    for col in df.columns:
        if 'fond' in col.lower():
            target_col = col
            break
    
    montant_col = None
    for col in df.columns:
        if 'montant' in col.lower():
            montant_col = col
            break
    
    famille_col = None
    for col in df.columns:
        col_lower = col.lower()
        if 'famille' in col_lower:
            famille_col = col
            break
    
    return target_col, montant_col, famille_col


def prepare_target(df, target_col):
    """Prépare la cible."""
    y = df[target_col].copy()
    if y.dtype == 'object':
        y = y.apply(lambda x: 1 if str(x).lower() in ['oui', 'yes', '1', 'fondée', 'fondee', 'true', 'o'] else 0)
    elif y.dtype == 'bool':
        y = y.astype(int)
    return y.fillna(0).astype(int)


def prepare_features(df, target_col):
    """Prépare les features."""
    X = df.drop(columns=[target_col]).copy()
    cols_to_drop = [c for c in X.columns if any(x in c.lower() for x in ['id', 'date', 'dt_', 'timestamp', 'num_'])]
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
    
    return X.select_dtypes(include=[np.number]).fillna(0)


def train_xgboost(X_train, y_train, X_val, y_val):
    """Entraîne XGBoost."""
    import xgboost as xgb
    
    model = xgb.XGBClassifier(
        max_depth=6,
        learning_rate=0.05,
        n_estimators=300,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=0
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    return model


# =============================================================================
# VISUALISATIONS
# =============================================================================

def plot_confusion_matrix(y_true, y_pred, output_path):
    """1. Matrice de confusion sur le test set."""
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    total = cm.sum()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Dessiner les cellules
    cell_colors = [[COLORS['tn'], COLORS['fp']], [COLORS['fn'], COLORS['tp']]]
    labels = [['Vrai Négatif\n(Bon Rejet)', 'Faux Positif\n(Fausse Validation)'],
              ['Faux Négatif\n(Faux Rejet)', 'Vrai Positif\n(Bonne Validation)']]
    values = [[tn, fp], [fn, tp]]
    
    for i in range(2):
        for j in range(2):
            rect = plt.Rectangle((j, 1-i), 1, 1, fill=True, 
                                   color=cell_colors[i][j], alpha=0.75)
            ax.add_patch(rect)
            
            val = values[i][j]
            pct = val / total * 100
            
            ax.text(j + 0.5, 1.55 - i, f'{val:,}', ha='center', va='center',
                    fontsize=30, fontweight='bold', color='white')
            ax.text(j + 0.5, 1.25 - i, f'({pct:.1f}%)', ha='center', va='center',
                    fontsize=14, color='white')
            ax.text(j + 0.5, 0.95 - i + 1, labels[i][j], ha='center', va='center',
                    fontsize=9, color='white', style='italic')
    
    ax.set_xlim(0, 2)
    ax.set_ylim(0, 2)
    ax.set_xticks([0.5, 1.5])
    ax.set_yticks([0.5, 1.5])
    ax.set_xticklabels(['Prédit: Non Fondée', 'Prédit: Fondée'], fontsize=11)
    ax.set_yticklabels(['Réel: Fondée', 'Réel: Non Fondée'], fontsize=11)
    ax.set_xlabel('PRÉDICTION', fontsize=12, fontweight='bold')
    ax.set_ylabel('RÉALITÉ', fontsize=12, fontweight='bold')
    
    accuracy = (tn + tp) / total * 100
    ax.set_title(f'MATRICE DE CONFUSION - ENSEMBLE DE TEST\n'
                 f'Total: {total:,} | Accuracy: {accuracy:.1f}%',
                 fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    return {'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp, 'accuracy': accuracy}


def plot_performance_par_famille(df_test, y_test, y_pred, famille_col, output_path):
    """2. Performance par famille avec barres horizontales."""
    results = []
    
    for famille in df_test[famille_col].unique():
        mask = df_test[famille_col] == famille
        if mask.sum() < 5:
            continue
        
        y_t = y_test[mask]
        y_p = y_pred[mask]
        
        acc = accuracy_score(y_t, y_p)
        prec = precision_score(y_t, y_p, zero_division=0)
        rec = recall_score(y_t, y_p, zero_division=0)
        f1 = f1_score(y_t, y_p, zero_division=0)
        
        results.append({
            'Famille': famille,
            'N': mask.sum(),
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'F1': f1
        })
    
    df_perf = pd.DataFrame(results).sort_values('Accuracy', ascending=True)
    
    # Graphique
    fig, ax = plt.subplots(figsize=(12, max(8, len(df_perf) * 0.4)))
    
    y_pos = np.arange(len(df_perf))
    
    # Couleurs selon accuracy
    colors = []
    for acc in df_perf['Accuracy']:
        if acc >= 0.95:
            colors.append(COLORS['excellent'])
        elif acc >= 0.90:
            colors.append(COLORS['good'])
        elif acc >= 0.80:
            colors.append(COLORS['warning'])
        else:
            colors.append(COLORS['danger'])
    
    bars = ax.barh(y_pos, df_perf['Accuracy'] * 100, color=colors, edgecolor='white', height=0.7)
    
    # Annotations
    for i, (idx, row) in enumerate(df_perf.iterrows()):
        ax.text(row['Accuracy'] * 100 + 1, i, f"{row['Accuracy']*100:.1f}%",
                va='center', fontsize=10, fontweight='bold')
        ax.text(2, i, f"N={row['N']}", va='center', fontsize=8, color='white')
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f[:35] for f in df_perf['Famille']], fontsize=9)
    ax.set_xlabel('Accuracy (%)', fontsize=11, fontweight='bold')
    ax.set_xlim(0, 110)
    
    # Lignes de référence
    ax.axvline(x=95, color='green', linestyle='--', linewidth=2, label='Objectif 95%')
    ax.axvline(x=90, color='orange', linestyle='--', linewidth=2, label='Seuil 90%')
    
    ax.legend(loc='lower right')
    ax.set_title('PERFORMANCE PAR FAMILLE PRODUIT', fontsize=14, fontweight='bold', pad=15)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    return df_perf


def plot_impact_total(n_total, n_auto, fp_count, fn_count, fp_montant, fn_montant, output_path):
    """3. Impact total."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    gain_auto = n_auto * COUT_TRAITEMENT_MANUEL
    n_manuel = n_total - n_auto
    
    # === 1. Automatisation (pie) ===
    ax1 = axes[0, 0]
    sizes = [n_auto, n_manuel]
    labels = [f'Automatisé\n{n_auto:,} ({n_auto/n_total*100:.1f}%)',
              f'Manuel\n{n_manuel:,} ({n_manuel/n_total*100:.1f}%)']
    colors_pie = [COLORS['auto'], '#bdc3c7']
    explode = (0.05, 0)
    
    ax1.pie(sizes, labels=labels, colors=colors_pie, explode=explode,
            autopct='', startangle=90, textprops={'fontsize': 10})
    ax1.set_title('TAUX D\'AUTOMATISATION', fontsize=12, fontweight='bold')
    
    # === 2. Erreurs en nombre ===
    ax2 = axes[0, 1]
    cats = ['Faux Positifs', 'Faux Négatifs']
    vals = [fp_count, fn_count]
    colors_bar = [COLORS['fp'], COLORS['fn']]
    
    bars = ax2.bar(cats, vals, color=colors_bar, edgecolor='white', linewidth=2)
    for bar, val in zip(bars, vals):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(vals)*0.02,
                 f'{val:,}', ha='center', va='bottom', fontsize=14, fontweight='bold')
    
    ax2.set_ylabel('Nombre', fontsize=11)
    ax2.set_title(f'ERREURS EN NOMBRE\nTotal: {fp_count + fn_count:,}', fontsize=12, fontweight='bold')
    
    # === 3. Erreurs en montant ===
    ax3 = axes[1, 0]
    montants = [fp_montant, fn_montant]
    
    bars2 = ax3.bar(cats, montants, color=colors_bar, edgecolor='white', linewidth=2)
    for bar, val in zip(bars2, montants):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(montants)*0.02,
                 f'{val:,.0f} MAD', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax3.set_ylabel('Montant (MAD)', fontsize=11)
    ax3.set_title(f'ERREURS EN MONTANT\nTotal: {fp_montant + fn_montant:,.0f} MAD', 
                  fontsize=12, fontweight='bold')
    
    # === 4. Bilan financier ===
    ax4 = axes[1, 1]
    
    gain_net = gain_auto - fp_montant
    cats_fin = ['Gain\nAutomatisation', 'Perte\nFaux Positifs', 'GAIN NET']
    vals_fin = [gain_auto, -fp_montant, gain_net]
    colors_fin = [COLORS['gain'], COLORS['perte'], 
                  COLORS['gain'] if gain_net > 0 else COLORS['perte']]
    
    bars3 = ax4.bar(cats_fin, vals_fin, color=colors_fin, edgecolor='white', linewidth=2)
    for bar, val in zip(bars3, vals_fin):
        y_pos = bar.get_height() + (max(abs(v) for v in vals_fin) * 0.03 * (1 if val >= 0 else -1))
        ax4.text(bar.get_x() + bar.get_width()/2, y_pos,
                 f'{val:+,.0f} MAD', ha='center', va='bottom' if val >= 0 else 'top',
                 fontsize=11, fontweight='bold')
    
    ax4.axhline(y=0, color='black', linewidth=1)
    ax4.set_ylabel('Montant (MAD)', fontsize=11)
    ax4.set_title('BILAN FINANCIER', fontsize=12, fontweight='bold')
    
    fig.suptitle(f'IMPACT TOTAL - {n_total:,} RÉCLAMATIONS\n'
                 f'(Gain: {COUT_TRAITEMENT_MANUEL} MAD par réclamation automatisée)',
                 fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    return {'gain_auto': gain_auto, 'gain_net': gain_net}


def plot_impact_by_threshold(df_perf, df_test, y_test, y_pred, y_proba, 
                              famille_col, montant_col, threshold, title, output_path):
    """4 & 5. Impact si on automatise les familles > seuil accuracy."""
    
    # Familles éligibles
    familles_ok = df_perf[df_perf['Accuracy'] >= threshold]['Famille'].tolist()
    
    if len(familles_ok) == 0:
        print(f"   ⚠️ Aucune famille avec accuracy >= {threshold*100:.0f}%")
        return None
    
    # Filtrer
    mask = df_test[famille_col].isin(familles_ok)
    df_filtered = df_test[mask]
    y_t = y_test[mask]
    y_p = y_pred[mask]
    y_prob = y_proba[mask]
    
    n_total = len(df_filtered)
    
    # Calculs
    mask_auto = (y_prob <= SEUIL_REJET) | (y_prob >= SEUIL_VALIDATION)
    n_auto = mask_auto.sum()
    n_manuel = n_total - n_auto
    
    mask_fp = (y_p == 1) & (y_t == 0)
    mask_fn = (y_p == 0) & (y_t == 1)
    
    fp_count = mask_fp.sum()
    fn_count = mask_fn.sum()
    
    if montant_col and montant_col in df_filtered.columns:
        montants = df_filtered[montant_col].fillna(0).values
        fp_montant = montants[mask_fp].sum()
        fn_montant = montants[mask_fn].sum()
    else:
        fp_montant, fn_montant = 0, 0
    
    gain_auto = n_auto * COUT_TRAITEMENT_MANUEL
    gain_net = gain_auto - fp_montant
    
    # Graphique
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 1. Familles concernées
    ax1 = axes[0]
    df_ok = df_perf[df_perf['Famille'].isin(familles_ok)].sort_values('Accuracy', ascending=True)
    y_pos = np.arange(len(df_ok))
    
    colors = [COLORS['excellent'] if acc >= 0.95 else COLORS['good'] 
              for acc in df_ok['Accuracy']]
    
    ax1.barh(y_pos, df_ok['Accuracy'] * 100, color=colors, edgecolor='white')
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels([f[:25] for f in df_ok['Famille']], fontsize=8)
    ax1.set_xlabel('Accuracy (%)')
    ax1.axvline(x=threshold * 100, color='red', linestyle='--', linewidth=2)
    ax1.set_title(f'{len(familles_ok)} FAMILLES ÉLIGIBLES', fontsize=11, fontweight='bold')
    
    # 2. Erreurs
    ax2 = axes[1]
    cats = ['Faux Positifs', 'Faux Négatifs']
    
    x = np.arange(2)
    width = 0.35
    
    bars1 = ax2.bar(x - width/2, [fp_count, fn_count], width, label='Nombre', color=[COLORS['fp'], COLORS['fn']])
    
    ax2.set_xticks(x)
    ax2.set_xticklabels(cats)
    ax2.set_ylabel('Nombre')
    ax2.set_title(f'ERREURS\nFP: {fp_count} | FN: {fn_count}', fontsize=11, fontweight='bold')
    
    for bar, val in zip(bars1, [fp_count, fn_count]):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                 f'{val}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # 3. Bilan
    ax3 = axes[2]
    cats_fin = ['Gain Auto', 'Perte FP', 'GAIN NET']
    vals_fin = [gain_auto, -fp_montant, gain_net]
    colors_fin = [COLORS['gain'], COLORS['perte'], 
                  COLORS['gain'] if gain_net > 0 else COLORS['perte']]
    
    bars3 = ax3.bar(cats_fin, vals_fin, color=colors_fin, edgecolor='white')
    ax3.axhline(y=0, color='black', linewidth=1)
    
    for bar, val in zip(bars3, vals_fin):
        y_p = bar.get_height() + (max(abs(v) for v in vals_fin) * 0.05 * (1 if val >= 0 else -1))
        ax3.text(bar.get_x() + bar.get_width()/2, y_p,
                 f'{val:+,.0f}', ha='center', va='bottom' if val >= 0 else 'top',
                 fontsize=10, fontweight='bold')
    
    ax3.set_ylabel('MAD')
    ax3.set_title(f'BILAN FINANCIER\n{n_auto:,} auto / {n_total:,} total', 
                  fontsize=11, fontweight='bold')
    
    fig.suptitle(f'{title}\n({n_total:,} réclamations)', fontsize=13, fontweight='bold', y=1.05)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    return {
        'familles': familles_ok,
        'n_total': n_total,
        'n_auto': n_auto,
        'fp': fp_count,
        'fn': fn_count,
        'fp_montant': fp_montant,
        'fn_montant': fn_montant,
        'gain_auto': gain_auto,
        'gain_net': gain_net
    }


def plot_familles_plus_pertes(df_test, y_test, y_pred, famille_col, montant_col, output_path):
    """6. Familles avec le plus de pertes en montant."""
    
    results = []
    
    for famille in df_test[famille_col].unique():
        mask = df_test[famille_col] == famille
        if mask.sum() < 5:
            continue
        
        y_t = y_test[mask]
        y_p = y_pred[mask]
        
        mask_fp = (y_p == 1) & (y_t == 0)
        mask_fn = (y_p == 0) & (y_t == 1)
        
        if montant_col and montant_col in df_test.columns:
            sub = df_test[mask]
            fp_montant = sub.loc[mask_fp, montant_col].fillna(0).sum()
            fn_montant = sub.loc[mask_fn, montant_col].fillna(0).sum()
        else:
            fp_montant, fn_montant = 0, 0
        
        total_perte = fp_montant + fn_montant
        
        results.append({
            'Famille': famille,
            'N': mask.sum(),
            'FP': mask_fp.sum(),
            'FN': mask_fn.sum(),
            'Montant_FP': fp_montant,
            'Montant_FN': fn_montant,
            'Total_Perte': total_perte,
            'Accuracy': accuracy_score(y_t, y_p)
        })
    
    df_pertes = pd.DataFrame(results).sort_values('Total_Perte', ascending=False)
    
    # Top 10 pertes
    df_top = df_pertes.head(10)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 1. Pertes par famille (barres empilées)
    ax1 = axes[0]
    y_pos = np.arange(len(df_top))
    
    bars_fp = ax1.barh(y_pos, df_top['Montant_FP'], color=COLORS['fp'], label='Pertes FP', edgecolor='white')
    bars_fn = ax1.barh(y_pos, df_top['Montant_FN'], left=df_top['Montant_FP'], 
                       color=COLORS['fn'], label='Pertes FN', edgecolor='white')
    
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels([f[:30] for f in df_top['Famille']], fontsize=9)
    ax1.set_xlabel('Montant (MAD)', fontsize=11)
    ax1.legend(loc='lower right')
    ax1.set_title('TOP 10 FAMILLES - PERTES EN MONTANT', fontsize=12, fontweight='bold')
    
    # Annotations
    for i, (idx, row) in enumerate(df_top.iterrows()):
        total = row['Total_Perte']
        ax1.text(total + max(df_top['Total_Perte'])*0.02, i, 
                 f'{total:,.0f} MAD', va='center', fontsize=9, fontweight='bold')
    
    # 2. Détail avec accuracy
    ax2 = axes[1]
    
    x = np.arange(len(df_top))
    width = 0.6
    
    colors = [COLORS['danger'] if acc < 0.90 else COLORS['warning'] if acc < 0.95 else COLORS['excellent']
              for acc in df_top['Accuracy']]
    
    bars = ax2.bar(x, df_top['Accuracy'] * 100, width, color=colors, edgecolor='white')
    
    ax2.axhline(y=95, color='green', linestyle='--', linewidth=2, label='Objectif 95%')
    ax2.axhline(y=90, color='orange', linestyle='--', linewidth=2, label='Seuil 90%')
    
    ax2.set_xticks(x)
    ax2.set_xticklabels([f[:15] for f in df_top['Famille']], rotation=45, ha='right', fontsize=8)
    ax2.set_ylabel('Accuracy (%)', fontsize=11)
    ax2.set_ylim(0, 105)
    ax2.legend(loc='lower right')
    ax2.set_title('ACCURACY DES FAMILLES À RISQUE', fontsize=12, fontweight='bold')
    
    for bar, (idx, row) in zip(bars, df_top.iterrows()):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                 f'{row["Accuracy"]*100:.0f}%', ha='center', va='bottom', fontsize=8)
    
    fig.suptitle('ANALYSE DES FAMILLES AVEC LE PLUS DE PERTES', fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    return df_pertes


# =============================================================================
# EXPORT EXCEL
# =============================================================================

def export_excel(output_dir, cm_results, df_perf, impact_total, impact_95, impact_90, df_pertes):
    """Exporte tous les résultats en Excel."""
    
    with pd.ExcelWriter(output_dir / 'rapport_complet.xlsx', engine='openpyxl') as writer:
        
        # 1. Résumé global
        summary = pd.DataFrame({
            'Métrique': [
                'Total Test', 'Accuracy', 'Vrais Négatifs', 'Faux Positifs', 
                'Faux Négatifs', 'Vrais Positifs', 'Gain Automatisation (MAD)', 'Gain Net (MAD)'
            ],
            'Valeur': [
                cm_results['tn'] + cm_results['fp'] + cm_results['fn'] + cm_results['tp'],
                f"{cm_results['accuracy']:.1f}%",
                cm_results['tn'], cm_results['fp'], cm_results['fn'], cm_results['tp'],
                f"{impact_total['gain_auto']:,.0f}",
                f"{impact_total['gain_net']:,.0f}"
            ]
        })
        summary.to_excel(writer, sheet_name='Resume_Global', index=False)
        
        # 2. Performance par famille
        df_perf_export = df_perf.copy()
        df_perf_export['Accuracy'] = df_perf_export['Accuracy'].apply(lambda x: f"{x*100:.1f}%")
        df_perf_export['Precision'] = df_perf_export['Precision'].apply(lambda x: f"{x*100:.1f}%")
        df_perf_export['Recall'] = df_perf_export['Recall'].apply(lambda x: f"{x*100:.1f}%")
        df_perf_export['F1'] = df_perf_export['F1'].apply(lambda x: f"{x*100:.1f}%")
        df_perf_export.to_excel(writer, sheet_name='Performance_Famille', index=False)
        
        # 3. Impact seuil 95%
        if impact_95:
            df_95 = pd.DataFrame({
                'Métrique': ['Familles éligibles', 'N Total', 'N Automatisé', 'FP', 'FN',
                            'Montant FP', 'Montant FN', 'Gain Auto', 'Gain Net'],
                'Valeur': [len(impact_95['familles']), impact_95['n_total'], impact_95['n_auto'],
                          impact_95['fp'], impact_95['fn'], f"{impact_95['fp_montant']:,.0f}",
                          f"{impact_95['fn_montant']:,.0f}", f"{impact_95['gain_auto']:,.0f}",
                          f"{impact_95['gain_net']:,.0f}"]
            })
            df_95.to_excel(writer, sheet_name='Impact_Seuil_95', index=False)
            
            # Liste des familles
            pd.DataFrame({'Familles_95%': impact_95['familles']}).to_excel(
                writer, sheet_name='Familles_95', index=False)
        
        # 4. Impact seuil 90%
        if impact_90:
            df_90 = pd.DataFrame({
                'Métrique': ['Familles éligibles', 'N Total', 'N Automatisé', 'FP', 'FN',
                            'Montant FP', 'Montant FN', 'Gain Auto', 'Gain Net'],
                'Valeur': [len(impact_90['familles']), impact_90['n_total'], impact_90['n_auto'],
                          impact_90['fp'], impact_90['fn'], f"{impact_90['fp_montant']:,.0f}",
                          f"{impact_90['fn_montant']:,.0f}", f"{impact_90['gain_auto']:,.0f}",
                          f"{impact_90['gain_net']:,.0f}"]
            })
            df_90.to_excel(writer, sheet_name='Impact_Seuil_90', index=False)
            
            pd.DataFrame({'Familles_90%': impact_90['familles']}).to_excel(
                writer, sheet_name='Familles_90', index=False)
        
        # 5. Familles avec pertes
        df_pertes_export = df_pertes.copy()
        df_pertes_export['Accuracy'] = df_pertes_export['Accuracy'].apply(lambda x: f"{x*100:.1f}%")
        df_pertes_export.to_excel(writer, sheet_name='Familles_Pertes', index=False)
    
    print(f"✅ Rapport Excel: {output_dir / 'rapport_complet.xlsx'}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Projet complet classification réclamations')
    parser.add_argument('--data', '-d', required=True, help='Fichier Excel')
    parser.add_argument('--output', '-o', default='outputs', help='Dossier de sortie')
    parser.add_argument('--montant-col', '-m', default=None, help='Colonne montant')
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    viz_dir = output_dir / 'visualizations'
    viz_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("PROJET COMPLET - CLASSIFICATION DES RÉCLAMATIONS BANCAIRES")
    print("=" * 70)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    
    # 1. Charger
    print(f"\n📂 Chargement: {args.data}")
    df = pd.read_excel(args.data)
    print(f"   {len(df):,} lignes")
    
    # 2. Détecter colonnes
    target_col, montant_col, famille_col = detect_columns(df)
    if args.montant_col:
        montant_col = args.montant_col
    
    print(f"   Cible: {target_col}")
    print(f"   Montant: {montant_col}")
    print(f"   Famille: {famille_col}")
    
    # 3. Préparer données
    print("\n🔧 Préparation des données...")
    y = prepare_target(df, target_col)
    X = prepare_features(df, target_col)
    print(f"   Features: {X.shape[1]}")
    
    # 4. Split
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, df.index, test_size=0.2, stratify=y, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.15, stratify=y_train, random_state=42
    )
    
    df_test = df.iloc[idx_test].reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)
    
    print(f"   Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}")
    
    # 5. Entraîner
    print("\n🚀 Entraînement XGBoost...")
    model = train_xgboost(X_train, y_train, X_val, y_val)
    
    # 6. Prédictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # 7. Calculs globaux
    mask_auto = (y_proba <= SEUIL_REJET) | (y_proba >= SEUIL_VALIDATION)
    n_auto = mask_auto.sum()
    
    mask_fp = (y_pred == 1) & (y_test.values == 0)
    mask_fn = (y_pred == 0) & (y_test.values == 1)
    
    if montant_col and montant_col in df_test.columns:
        montants = df_test[montant_col].fillna(0).values
        fp_montant = montants[mask_fp].sum()
        fn_montant = montants[mask_fn].sum()
    else:
        fp_montant, fn_montant = 0, 0
    
    # ==========================================================================
    # GÉNÉRATION DES VISUALISATIONS
    # ==========================================================================
    print("\n📊 Génération des visualisations...")
    
    # 1. Matrice de confusion
    print("   1. Matrice de confusion...")
    cm_results = plot_confusion_matrix(y_test, y_pred, viz_dir / '1_matrice_confusion.png')
    
    # 2. Performance par famille
    print("   2. Performance par famille...")
    df_perf = plot_performance_par_famille(df_test, y_test, y_pred, famille_col,
                                            viz_dir / '2_performance_famille.png')
    
    # 3. Impact total
    print("   3. Impact total...")
    impact_total = plot_impact_total(
        len(df_test), n_auto, mask_fp.sum(), mask_fn.sum(),
        fp_montant, fn_montant, viz_dir / '3_impact_total.png'
    )
    
    # 4. Impact seuil 95%
    print("   4. Impact familles >95% accuracy...")
    impact_95 = plot_impact_by_threshold(
        df_perf, df_test, y_test, y_pred, y_proba,
        famille_col, montant_col, SEUIL_ACCURACY_HIGH,
        'IMPACT - FAMILLES AVEC ACCURACY ≥ 95%',
        viz_dir / '4_impact_seuil_95.png'
    )
    
    # 5. Impact seuil 90%
    print("   5. Impact familles >90% accuracy...")
    impact_90 = plot_impact_by_threshold(
        df_perf, df_test, y_test, y_pred, y_proba,
        famille_col, montant_col, SEUIL_ACCURACY_MEDIUM,
        'IMPACT - FAMILLES AVEC ACCURACY ≥ 90%',
        viz_dir / '5_impact_seuil_90.png'
    )
    
    # 6. Familles avec plus de pertes
    print("   6. Familles avec le plus de pertes...")
    df_pertes = plot_familles_plus_pertes(
        df_test, y_test, y_pred, famille_col, montant_col,
        viz_dir / '6_familles_pertes.png'
    )
    
    # ==========================================================================
    # EXPORT EXCEL
    # ==========================================================================
    print("\n💾 Export Excel...")
    export_excel(output_dir, cm_results, df_perf, impact_total, impact_95, impact_90, df_pertes)
    
    # ==========================================================================
    # RÉSUMÉ
    # ==========================================================================
    print("\n" + "=" * 70)
    print("📋 RÉSUMÉ")
    print("=" * 70)
    print(f"\n   Accuracy globale: {cm_results['accuracy']:.1f}%")
    print(f"   Taux automatisation: {n_auto/len(df_test)*100:.1f}%")
    print(f"   Gain net: {impact_total['gain_net']:,.0f} MAD")
    
    if impact_95:
        print(f"\n   Familles ≥95%: {len(impact_95['familles'])} familles")
        print(f"   → Gain net: {impact_95['gain_net']:,.0f} MAD")
    
    if impact_90:
        print(f"\n   Familles ≥90%: {len(impact_90['familles'])} familles")
        print(f"   → Gain net: {impact_90['gain_net']:,.0f} MAD")
    
    print(f"\n   Top 3 familles à risque (pertes):")
    for i, (_, row) in enumerate(df_pertes.head(3).iterrows(), 1):
        print(f"   {i}. {row['Famille'][:30]} : {row['Total_Perte']:,.0f} MAD")
    
    print("\n" + "=" * 70)
    print("✅ PROJET TERMINÉ")
    print("=" * 70)
    print(f"\n📁 Fichiers générés dans: {output_dir}")
    print(f"   📊 visualizations/")
    print(f"      • 1_matrice_confusion.png")
    print(f"      • 2_performance_famille.png")
    print(f"      • 3_impact_total.png")
    print(f"      • 4_impact_seuil_95.png")
    print(f"      • 5_impact_seuil_90.png")
    print(f"      • 6_familles_pertes.png")
    print(f"   📄 rapport_complet.xlsx")


if __name__ == "__main__":
    main()
