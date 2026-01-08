#!/usr/bin/env python3
"""
Inférence + Impact - UNIQUEMENT 4 Familles Performantes
=========================================================
Filtre sur:
- Clôture de compte
- Ristourne / intérêt clientèle
- Opérations sur GAB
- Cartes Wafacash

Usage:
    python inference_4_familles.py --data fichier.xlsx --output resultats/
"""

import pandas as pd
import numpy as np
import argparse
import matplotlib.pyplot as plt
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

# Les 4 familles performantes (en vert)
FAMILLES_CIBLES = [
    'Clôture de compte',
    'Ristourne / intérêt clientèle',
    'Opérations. sur GAB',
    'Opérations sur GAB',  # Variante possible
    'Cartes Wafacash',
]

# Style
plt.style.use('seaborn-v0_8-whitegrid')
COLORS = {
    'fp': '#e74c3c',
    'fn': '#f39c12', 
    'tp': '#27ae60',
    'tn': '#3498db',
    'gain': '#2ecc71',
    'perte': '#c0392b',
    'auto': '#9b59b6',
}


def detect_columns(df):
    """Détecte colonnes cible, montant et famille."""
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
        if 'famille' in col_lower or 'family' in col_lower:
            famille_col = col
            break
        if 'produit' in col_lower and df[col].dtype == 'object':
            famille_col = col
    
    return target_col, montant_col, famille_col


def filter_familles(df, famille_col):
    """Filtre sur les 4 familles cibles."""
    mask = df[famille_col].isin(FAMILLES_CIBLES)
    
    # Si pas de match exact, essayer avec contains
    if mask.sum() == 0:
        for famille in FAMILLES_CIBLES:
            mask = mask | df[famille_col].str.contains(famille.split()[0], case=False, na=False)
    
    return df[mask].copy()


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
    """Entraîne XGBoost."""
    import xgboost as xgb
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.15, stratify=y_train, random_state=42)
    
    model = xgb.XGBClassifier(
        max_depth=6, learning_rate=0.05, n_estimators=300,
        random_state=42, verbosity=0
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    
    return model


def plot_confusion_matrix(y_true, y_pred, title, output_path):
    """Matrice de confusion."""
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    matrix = np.array([[tn, fp], [fn, tp]])
    colors = np.array([[COLORS['tn'], COLORS['fp']], [COLORS['fn'], COLORS['tp']]])
    
    for i in range(2):
        for j in range(2):
            ax.add_patch(plt.Rectangle((j, 1-i), 1, 1, fill=True, color=colors[i,j], alpha=0.7))
            value = matrix[i, j]
            pct = value / matrix.sum() * 100
            labels = [['Vrai Négatif', 'Faux Positif'], ['Faux Négatif', 'Vrai Positif']]
            
            ax.text(j + 0.5, 1.5 - i, f'{value:,}', ha='center', va='center',
                    fontsize=32, fontweight='bold', color='white')
            ax.text(j + 0.5, 1.2 - i, f'({pct:.1f}%)', ha='center', va='center',
                    fontsize=16, color='white')
            ax.text(j + 0.5, 0.9 - i + 1, labels[i][j], ha='center', va='center',
                    fontsize=11, color='white', style='italic')
    
    ax.set_xlim(0, 2)
    ax.set_ylim(0, 2)
    ax.set_xticks([0.5, 1.5])
    ax.set_yticks([0.5, 1.5])
    ax.set_xticklabels(['Prédit: Non Fondée', 'Prédit: Fondée'], fontsize=12)
    ax.set_yticklabels(['Réel: Fondée', 'Réel: Non Fondée'], fontsize=12)
    
    total = matrix.sum()
    accuracy = (tn + tp) / total * 100
    ax.set_title(f'{title}\nTotal: {total:,} | Accuracy: {accuracy:.1f}%',
                 fontsize=16, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✅ {output_path}")
    
    return tn, fp, fn, tp


def plot_impact_errors(fp_count, fn_count, fp_montant, fn_montant, output_path):
    """Impact FP/FN en nombre et montant."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Nombre
    ax1 = axes[0]
    cats = ['Faux Positifs\n(Fausses Validations)', 'Faux Négatifs\n(Faux Rejets)']
    vals = [fp_count, fn_count]
    colors = [COLORS['fp'], COLORS['fn']]
    
    bars = ax1.bar(cats, vals, color=colors, edgecolor='white', linewidth=2)
    for bar, val in zip(bars, vals):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(vals)*0.02,
                 f'{val:,}', ha='center', va='bottom', fontsize=18, fontweight='bold')
    
    ax1.set_ylabel('Nombre', fontsize=12, fontweight='bold')
    ax1.set_title('IMPACT EN NOMBRE', fontsize=14, fontweight='bold')
    ax1.set_ylim(0, max(vals) * 1.2 if max(vals) > 0 else 1)
    
    total = fp_count + fn_count
    ax1.text(0.5, 0.92, f'Total erreurs: {total:,}', transform=ax1.transAxes,
             ha='center', fontsize=13, fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='#ffeb3b', alpha=0.8))
    
    # Montant
    ax2 = axes[1]
    montants = [fp_montant, fn_montant]
    
    bars2 = ax2.bar(cats, montants, color=colors, edgecolor='white', linewidth=2)
    for bar, val in zip(bars2, montants):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(montants)*0.02,
                 f'{val:,.0f} MAD', ha='center', va='bottom', fontsize=16, fontweight='bold')
    
    ax2.set_ylabel('Montant (MAD)', fontsize=12, fontweight='bold')
    ax2.set_title('IMPACT EN MONTANT', fontsize=14, fontweight='bold')
    ax2.set_ylim(0, max(montants) * 1.2 if max(montants) > 0 else 1)
    
    total_montant = fp_montant + fn_montant
    ax2.text(0.5, 0.92, f'Total: {total_montant:,.0f} MAD', transform=ax2.transAxes,
             ha='center', fontsize=13, fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='#ffeb3b', alpha=0.8))
    
    fig.suptitle('IMPACT DES ERREURS - 4 FAMILLES PERFORMANTES', fontsize=16, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✅ {output_path}")


def plot_gain_total(n_auto, n_manuel, gain_auto, fp_montant, fn_montant, output_path):
    """Graphique gain total."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Automatisation
    ax1 = axes[0]
    cats = ['Automatisé', 'Manuel']
    vals = [n_auto, n_manuel]
    colors_bar = [COLORS['auto'], '#bdc3c7']
    
    bars = ax1.bar(cats, vals, color=colors_bar, edgecolor='white', linewidth=2)
    total = n_auto + n_manuel
    for bar, val in zip(bars, vals):
        pct = val / total * 100 if total > 0 else 0
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + total*0.02,
                 f'{val:,}\n({pct:.1f}%)', ha='center', va='bottom', fontsize=14, fontweight='bold')
    
    ax1.set_ylabel('Nombre de réclamations', fontsize=12, fontweight='bold')
    ax1.set_title('TAUX D\'AUTOMATISATION', fontsize=14, fontweight='bold')
    ax1.set_ylim(0, max(vals) * 1.25)
    
    # Bilan financier
    ax2 = axes[1]
    
    gain_net = gain_auto - fp_montant
    
    cats2 = ['Gain\nAutomatisation', 'Perte\nFaux Positifs', 'Perte\nFaux Négatifs', 'GAIN NET']
    vals2 = [gain_auto, -fp_montant, -fn_montant, gain_net]
    colors2 = [COLORS['gain'], COLORS['perte'], COLORS['fn'],
               COLORS['gain'] if gain_net > 0 else COLORS['perte']]
    
    bars2 = ax2.bar(cats2, vals2, color=colors2, edgecolor='white', linewidth=2)
    
    for bar, val in zip(bars2, vals2):
        y_pos = bar.get_height() + (max(abs(v) for v in vals2) * 0.03 * (1 if val >= 0 else -1))
        va = 'bottom' if val >= 0 else 'top'
        ax2.text(bar.get_x() + bar.get_width()/2, y_pos,
                 f'{val:+,.0f} MAD', ha='center', va=va, fontsize=11, fontweight='bold')
    
    ax2.axhline(y=0, color='black', linewidth=1.5)
    ax2.set_ylabel('Montant (MAD)', fontsize=12, fontweight='bold')
    ax2.set_title('BILAN FINANCIER', fontsize=14, fontweight='bold')
    
    max_val = max(abs(v) for v in vals2) if vals2 else 1
    ax2.set_ylim(-max_val * 1.3, max_val * 1.3)
    
    fig.suptitle(f'GAIN TOTAL - 4 FAMILLES PERFORMANTES\n(169 MAD économisés par réclamation automatisée)',
                 fontsize=16, fontweight='bold', y=1.05)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✅ {output_path}")


def plot_detail_par_famille(df_results, famille_col, output_path):
    """Détail par famille."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    familles = df_results[famille_col].unique()
    
    for idx, famille in enumerate(familles[:4]):
        ax = axes[idx // 2, idx % 2]
        sub = df_results[df_results[famille_col] == famille]
        
        n_total = len(sub)
        n_fp = sub['is_fp'].sum()
        n_fn = sub['is_fn'].sum()
        n_correct = n_total - n_fp - n_fn
        
        vals = [n_correct, n_fp, n_fn]
        labels = ['Correct', 'Faux Positif', 'Faux Négatif']
        colors = [COLORS['tp'], COLORS['fp'], COLORS['fn']]
        
        bars = ax.bar(labels, vals, color=colors)
        for bar, val in zip(bars, vals):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + n_total*0.02,
                        f'{val}', ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        accuracy = n_correct / n_total * 100 if n_total > 0 else 0
        ax.set_title(f'{famille[:30]}\nN={n_total} | Accuracy={accuracy:.1f}%', fontsize=11, fontweight='bold')
        ax.set_ylim(0, max(vals) * 1.2 if max(vals) > 0 else 1)
    
    fig.suptitle('DÉTAIL PAR FAMILLE', fontsize=16, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✅ {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', '-d', required=True)
    parser.add_argument('--output', '-o', default='resultats_4_familles')
    parser.add_argument('--montant-col', '-m', default=None)
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("INFÉRENCE - 4 FAMILLES PERFORMANTES")
    print("=" * 60)
    
    # 1. Charger
    print(f"\n📂 Chargement: {args.data}")
    df_full = pd.read_excel(args.data)
    print(f"   Base complète: {len(df_full)} lignes")
    
    # 2. Détecter colonnes
    target_col, montant_col, famille_col = detect_columns(df_full)
    if args.montant_col:
        montant_col = args.montant_col
    
    print(f"   Cible: {target_col}")
    print(f"   Montant: {montant_col}")
    print(f"   Famille: {famille_col}")
    
    # 3. Filtrer sur les 4 familles
    print(f"\n🎯 Filtrage sur les 4 familles performantes...")
    df = filter_familles(df_full, famille_col)
    print(f"   Familles trouvées: {df[famille_col].unique().tolist()}")
    print(f"   Lignes après filtre: {len(df)}")
    
    if len(df) == 0:
        print("❌ Aucune ligne trouvée pour ces familles!")
        print(f"   Familles disponibles: {df_full[famille_col].unique().tolist()}")
        return
    
    # 4. Préparer
    X, y = prepare_data(df, target_col)
    print(f"   Features: {X.shape[1]}")
    
    # 5. Entraîner sur base complète puis prédire sur filtrée
    print("\n🚀 Entraînement XGBoost sur base complète...")
    X_full, y_full = prepare_data(df_full, target_col)
    model = train_model(X_full, y_full)
    
    # 6. Inférence sur les 4 familles
    print("\n🔮 Inférence sur les 4 familles...")
    
    # Aligner les colonnes
    missing_cols = set(X_full.columns) - set(X.columns)
    for col in missing_cols:
        X[col] = 0
    X = X[X_full.columns]
    
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]
    
    # 7. Calculs
    cm = confusion_matrix(y, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    # Montants
    if montant_col and montant_col in df.columns:
        montants = df[montant_col].fillna(0).values
    else:
        montants = np.zeros(len(df))
    
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
        'precision': precision_score(y, y_pred, zero_division=0),
        'recall': recall_score(y, y_pred, zero_division=0),
        'f1': f1_score(y, y_pred, zero_division=0),
    }
    
    # 8. Afficher
    print(f"\n📊 RÉSULTATS - 4 FAMILLES:")
    print(f"   Total: {len(df):,} réclamations")
    print(f"   Accuracy: {metrics['accuracy']*100:.1f}%")
    print(f"   Précision: {metrics['precision']*100:.1f}%")
    print(f"   Recall: {metrics['recall']*100:.1f}%")
    print(f"   F1-Score: {metrics['f1']*100:.1f}%")
    print(f"\n   TN={tn:,} | FP={fp:,}")
    print(f"   FN={fn:,} | TP={tp:,}")
    print(f"\n   Faux Positifs: {fp:,} ({fp_montant:,.0f} MAD)")
    print(f"   Faux Négatifs: {fn:,} ({fn_montant:,.0f} MAD)")
    print(f"\n   Automatisé: {n_auto:,} ({n_auto/len(df)*100:.1f}%)")
    print(f"   Gain automatisation: {gain_auto:,.0f} MAD")
    print(f"   GAIN NET: {gain_auto - fp_montant:,.0f} MAD")
    
    # 9. Graphiques
    print(f"\n📈 Génération des graphiques...")
    
    plot_confusion_matrix(y, y_pred, 'MATRICE DE CONFUSION - 4 FAMILLES',
                          output_dir / '1_matrice_confusion.png')
    
    plot_impact_errors(fp, fn, fp_montant, fn_montant,
                       output_dir / '2_impact_erreurs.png')
    
    plot_gain_total(n_auto, n_manuel, gain_auto, fp_montant, fn_montant,
                    output_dir / '3_gain_total.png')
    
    # Détail par famille
    df_results = df.copy()
    df_results['prediction'] = y_pred
    df_results['proba'] = y_proba
    df_results['is_fp'] = mask_fp
    df_results['is_fn'] = mask_fn
    
    plot_detail_par_famille(df_results, famille_col, output_dir / '4_detail_familles.png')
    
    # 10. Excel
    print(f"\n💾 Export Excel...")
    
    # Résumé global
    summary = pd.DataFrame({
        'Métrique': ['Total réclamations', 'Accuracy', 'Précision', 'Recall', 'F1-Score',
                     'Vrais Négatifs', 'Faux Positifs', 'Faux Négatifs', 'Vrais Positifs',
                     'Montant FP (MAD)', 'Montant FN (MAD)',
                     'N Automatisé', 'N Manuel', 'Taux Automatisation',
                     'Gain Automatisation (MAD)', 'GAIN NET (MAD)'],
        'Valeur': [len(df), f"{metrics['accuracy']*100:.1f}%", f"{metrics['precision']*100:.1f}%",
                   f"{metrics['recall']*100:.1f}%", f"{metrics['f1']*100:.1f}%",
                   tn, fp, fn, tp,
                   f"{fp_montant:,.0f}", f"{fn_montant:,.0f}",
                   n_auto, n_manuel, f"{n_auto/len(df)*100:.1f}%",
                   f"{gain_auto:,.0f}", f"{gain_auto - fp_montant:,.0f}"]
    })
    
    # Détail par famille
    detail_famille = []
    for famille in df[famille_col].unique():
        sub = df_results[df_results[famille_col] == famille]
        n = len(sub)
        n_fp = sub['is_fp'].sum()
        n_fn = sub['is_fn'].sum()
        
        if montant_col and montant_col in sub.columns:
            montant_fp = sub.loc[sub['is_fp'], montant_col].sum()
            montant_fn = sub.loc[sub['is_fn'], montant_col].sum()
        else:
            montant_fp, montant_fn = 0, 0
        
        n_auto_f = ((sub['proba'] <= SEUIL_REJET) | (sub['proba'] >= SEUIL_VALIDATION)).sum()
        
        detail_famille.append({
            'Famille': famille,
            'N_Total': n,
            'N_Automatise': n_auto_f,
            'Taux_Auto_%': round(n_auto_f / n * 100, 1) if n > 0 else 0,
            'N_Faux_Positifs': n_fp,
            'N_Faux_Negatifs': n_fn,
            'Montant_FP': montant_fp,
            'Montant_FN': montant_fn,
            'Gain_Auto_MAD': n_auto_f * COUT_TRAITEMENT_MANUEL,
        })
    
    df_detail = pd.DataFrame(detail_famille)
    
    with pd.ExcelWriter(output_dir / 'resume_4_familles.xlsx', engine='openpyxl') as writer:
        summary.to_excel(writer, sheet_name='Resume_Global', index=False)
        df_detail.to_excel(writer, sheet_name='Detail_par_Famille', index=False)
    
    print(f"✅ {output_dir / 'resume_4_familles.xlsx'}")
    
    print("\n" + "=" * 60)
    print("✅ TERMINÉ")
    print("=" * 60)
    print(f"\n📁 Fichiers dans: {output_dir}")
    print("   • 1_matrice_confusion.png")
    print("   • 2_impact_erreurs.png")
    print("   • 3_gain_total.png")
    print("   • 4_detail_familles.png")
    print("   • resume_4_familles.xlsx")


if __name__ == "__main__":
    main()

