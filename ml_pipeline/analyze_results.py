"""
ANALYSE POST-ENTRAÎNEMENT - XGBoost uniquement avec 3 Zones de Décision et Règle Métier
Génère des analyses détaillées sans réentraîner le modèle
Usage: python analyze_results.py
"""
import sys
sys.path.append('src')

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

# Configuration
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (15, 10)

PRIX_UNITAIRE_DH = 169


def load_results():
    """Charger les résultats du modèle XGBoost avec les seuils"""
    print("\n" + "="*80)
    print("📂 CHARGEMENT DES RÉSULTATS - XGBoost")
    print("="*80)

    # Charger les données 2025
    df_2025 = pd.read_excel('data/raw/reclamations_2025.xlsx')
    print(f"✅ Données 2025: {len(df_2025)} réclamations")

    # Essayer de charger les vraies prédictions
    predictions_path = Path('outputs/production/predictions/predictions_2025.pkl')

    if predictions_path.exists():
        print(f"✅ Chargement des prédictions depuis: {predictions_path}")

        predictions_data = joblib.load(predictions_path)

        y_true = predictions_data['y_true']

        if 'XGBoost' in predictions_data:
            xgboost_results = {
                'y_prob': predictions_data['XGBoost']['y_prob'],
                'threshold_low': predictions_data['XGBoost']['threshold_low'],
                'threshold_high': predictions_data['XGBoost']['threshold_high']
            }
            print(f"   ✓ XGBoost (seuils: {xgboost_results['threshold_low']:.4f} / {xgboost_results['threshold_high']:.4f})")
        else:
            print("❌ XGBoost non trouvé dans les prédictions!")
            return None, None, None

        print(f"✅ Prédictions XGBoost chargées")

    else:
        print("⚠️  Fichier de prédictions non trouvé!")
        print("   Exécutez d'abord: python model_comparison.py")
        return None, None, None

    return df_2025, y_true, xgboost_results


def create_3zone_predictions(y_prob, threshold_low, threshold_high):
    """Créer les prédictions avec 3 zones de décision"""
    # Zone 1: Rejet auto (prob <= threshold_low) → prédiction = 0
    # Zone 2: Audit humain (threshold_low < prob < threshold_high) → prédiction = -1 (manuel)
    # Zone 3: Validation auto (prob >= threshold_high) → prédiction = 1

    y_pred = np.full(len(y_prob), -1, dtype=int)  # Par défaut: audit humain
    y_pred[y_prob <= threshold_low] = 0  # Rejet auto
    y_pred[y_prob >= threshold_high] = 1  # Validation auto

    return y_pred


def apply_business_rule(df_2025, y_true, y_prob, threshold_low, threshold_high):
    """Appliquer la règle métier: 1 validation auto par client par année"""
    print(f"\n🔒 Application de la règle métier - XGBoost")
    print("="*80)

    df_scenario = df_2025.copy()
    df_scenario['y_true'] = y_true
    df_scenario['y_prob'] = y_prob

    # Créer les prédictions avec 3 zones
    y_pred_original = create_3zone_predictions(y_prob, threshold_low, threshold_high)
    df_scenario['y_pred_original'] = y_pred_original

    # Convertir la date
    df_scenario['Date de Qualification'] = pd.to_datetime(
        df_scenario['Date de Qualification'],
        errors='coerce'
    )
    df_scenario['Annee'] = df_scenario['Date de Qualification'].dt.year

    # Identifier la colonne client
    client_col = None
    for col in ['idtfcl', 'numero_compte', 'N compte', 'ID Client']:
        if col in df_scenario.columns:
            client_col = col
            break

    if client_col is None:
        print("⚠️  Aucune colonne client trouvée, utilisation de l'index")
        client_col = 'index'
        df_scenario['index'] = df_scenario.index

    # Trier par client, année, puis date
    df_scenario = df_scenario.sort_values([client_col, 'Annee', 'Date de Qualification'])

    # RÈGLE: 1 validation auto par client par année
    # On ne bloque que les validations automatiques (y_pred == 1)
    df_scenario['is_validation_auto'] = (df_scenario['y_pred_original'] == 1)

    # Compter les validations auto par client/année
    df_scenario['validation_rank'] = df_scenario.groupby([client_col, 'Annee'])['is_validation_auto'].cumsum()

    # Appliquer la règle: seule la première validation auto est acceptée
    # Les validations suivantes deviennent des audits humains (y_pred = -1)
    df_scenario['y_pred_with_rule'] = df_scenario['y_pred_original'].copy()
    df_scenario.loc[
        (df_scenario['is_validation_auto']) & (df_scenario['validation_rank'] > 1),
        'y_pred_with_rule'
    ] = -1  # Bloquer → audit humain

    # Statistiques
    n_validations_original = (df_scenario['y_pred_original'] == 1).sum()
    n_validations_blocked = n_validations_original - (df_scenario['y_pred_with_rule'] == 1).sum()
    n_rejets = (df_scenario['y_pred_original'] == 0).sum()
    n_audits_original = (df_scenario['y_pred_original'] == -1).sum()
    n_audits_with_rule = (df_scenario['y_pred_with_rule'] == -1).sum()

    print(f"\n📊 Distribution SANS règle:")
    print(f"   Rejets auto      : {n_rejets} ({100*n_rejets/len(df_scenario):.1f}%)")
    print(f"   Audits humains   : {n_audits_original} ({100*n_audits_original/len(df_scenario):.1f}%)")
    print(f"   Validations auto : {n_validations_original} ({100*n_validations_original/len(df_scenario):.1f}%)")

    print(f"\n📊 Distribution AVEC règle:")
    print(f"   Rejets auto      : {n_rejets} ({100*n_rejets/len(df_scenario):.1f}%)")
    print(f"   Audits humains   : {n_audits_with_rule} ({100*n_audits_with_rule/len(df_scenario):.1f}%)")
    print(f"   Validations auto : {(df_scenario['y_pred_with_rule'] == 1).sum()} ({100*(df_scenario['y_pred_with_rule'] == 1).sum()/len(df_scenario):.1f}%)")

    print(f"\n📊 Impact de la règle:")
    print(f"   Validations bloquées : {n_validations_blocked}")
    print(f"   Nouveaux audits      : {n_audits_with_rule - n_audits_original}")

    return df_scenario


def calculate_financial_impact(df_scenario):
    """Calculer l'impact financier avec et sans règle (3 zones)"""
    print(f"\n💰 Calcul de l'impact financier - XGBoost")
    print("="*80)

    montants = df_scenario['Montant demandé'].values
    montants = pd.to_numeric(montants, errors='coerce').fillna(0)
    montants = np.clip(montants, 0, np.percentile(montants, 99))

    y_true = df_scenario['y_true'].values
    y_pred_original = df_scenario['y_pred_original'].values
    y_pred_with_rule = df_scenario['y_pred_with_rule'].values

    # SANS règle
    # Seuls les cas automatisés (rejet=0 ou validation=1) sont comptés
    mask_auto_sans = (y_pred_original != -1)
    y_true_auto_sans = y_true[mask_auto_sans]
    y_pred_auto_sans = y_pred_original[mask_auto_sans]
    montants_auto_sans = montants[mask_auto_sans]

    tp_sans = ((y_true_auto_sans == 1) & (y_pred_auto_sans == 1)).sum()
    tn_sans = ((y_true_auto_sans == 0) & (y_pred_auto_sans == 0)).sum()
    fp_mask_sans = (y_true_auto_sans == 0) & (y_pred_auto_sans == 1)
    fn_mask_sans = (y_true_auto_sans == 1) & (y_pred_auto_sans == 0)

    auto_sans = tp_sans + tn_sans
    gain_brut_sans = auto_sans * PRIX_UNITAIRE_DH
    perte_fp_sans = montants_auto_sans[fp_mask_sans].sum()
    perte_fn_sans = 2 * montants_auto_sans[fn_mask_sans].sum()
    gain_net_sans = gain_brut_sans - perte_fp_sans - perte_fn_sans

    # AVEC règle
    mask_auto_avec = (y_pred_with_rule != -1)
    y_true_auto_avec = y_true[mask_auto_avec]
    y_pred_auto_avec = y_pred_with_rule[mask_auto_avec]
    montants_auto_avec = montants[mask_auto_avec]

    tp_avec = ((y_true_auto_avec == 1) & (y_pred_auto_avec == 1)).sum()
    tn_avec = ((y_true_auto_avec == 0) & (y_pred_auto_avec == 0)).sum()
    fp_mask_avec = (y_true_auto_avec == 0) & (y_pred_auto_avec == 1)
    fn_mask_avec = (y_true_auto_avec == 1) & (y_pred_auto_avec == 0)

    auto_avec = tp_avec + tn_avec
    gain_brut_avec = auto_avec * PRIX_UNITAIRE_DH
    perte_fp_avec = montants_auto_avec[fp_mask_avec].sum()
    perte_fn_avec = 2 * montants_auto_avec[fn_mask_avec].sum()
    gain_net_avec = gain_brut_avec - perte_fp_avec - perte_fn_avec

    # Affichage
    print(f"\n📊 SANS règle métier:")
    print(f"   Automatisés : {mask_auto_sans.sum()}/{len(y_true)} ({100*mask_auto_sans.sum()/len(y_true):.1f}%)")
    print(f"   Corrects    : {auto_sans} ({100*auto_sans/mask_auto_sans.sum():.1f}%)")
    print(f"   Gain brut   : {gain_brut_sans:,.0f} DH")
    print(f"   Perte FP    : {perte_fp_sans:,.0f} DH ({fp_mask_sans.sum()} cas)")
    print(f"   Perte FN    : {perte_fn_sans:,.0f} DH ({fn_mask_sans.sum()} cas)")
    print(f"   Gain NET    : {gain_net_sans:,.0f} DH")

    print(f"\n📊 AVEC règle métier:")
    print(f"   Automatisés : {mask_auto_avec.sum()}/{len(y_true)} ({100*mask_auto_avec.sum()/len(y_true):.1f}%)")
    accuracy_avec = 100*auto_avec/mask_auto_avec.sum() if mask_auto_avec.sum() > 0 else 0
    print(f"   Corrects    : {auto_avec} ({accuracy_avec:.1f}%)")
    print(f"   Gain brut   : {gain_brut_avec:,.0f} DH")
    print(f"   Perte FP    : {perte_fp_avec:,.0f} DH ({fp_mask_avec.sum()} cas)")
    print(f"   Perte FN    : {perte_fn_avec:,.0f} DH ({fn_mask_avec.sum()} cas)")
    print(f"   Gain NET    : {gain_net_avec:,.0f} DH")

    difference = gain_net_avec - gain_net_sans
    print(f"\n💡 Différence: {difference:+,.0f} DH ({100*difference/gain_net_sans:+.2f}%)")

    return {
        'sans_regle': {
            'total': len(y_true),
            'auto': mask_auto_sans.sum(),
            'audit': (~mask_auto_sans).sum(),
            'taux_auto': 100*mask_auto_sans.sum()/len(y_true),
            'gain_brut': gain_brut_sans,
            'perte_fp': perte_fp_sans,
            'perte_fn': perte_fn_sans,
            'gain_net': gain_net_sans,
            'fp': fp_mask_sans.sum(),
            'fn': fn_mask_sans.sum(),
            'tp': tp_sans,
            'tn': tn_sans
        },
        'avec_regle': {
            'total': len(y_true),
            'auto': mask_auto_avec.sum(),
            'audit': (~mask_auto_avec).sum(),
            'taux_auto': 100*mask_auto_avec.sum()/len(y_true),
            'gain_brut': gain_brut_avec,
            'perte_fp': perte_fp_avec,
            'perte_fn': perte_fn_avec,
            'gain_net': gain_net_avec,
            'fp': fp_mask_avec.sum(),
            'fn': fn_mask_avec.sum(),
            'tp': tp_avec,
            'tn': tn_avec
        },
        'difference': difference
    }


def generate_confusion_matrix(df_scenario, y_true, y_prob, threshold_low, threshold_high):
    """Générer la matrice de confusion pour XGBoost"""
    print("\n" + "="*80)
    print("📊 GÉNÉRATION DE LA MATRICE DE CONFUSION")
    print("="*80)

    # Créer prédictions avec 3 zones
    y_pred = create_3zone_predictions(y_prob, threshold_low, threshold_high)

    # Matrice de confusion seulement sur cas automatisés
    mask_auto = (y_pred != -1)
    y_true_auto = y_true[mask_auto]
    y_pred_auto = y_pred[mask_auto]

    cm = confusion_matrix(y_true_auto, y_pred_auto)
    cm_percent = cm.astype('float') / cm.sum() * 100

    # Créer la figure
    fig, ax = plt.subplots(figsize=(8, 6))

    # Annotations
    annotations = []
    for i in range(2):
        row = []
        for j in range(2):
            row.append(f'{cm[i,j]}\n({cm_percent[i,j]:.1f}%)')
        annotations.append(row)

    # Heatmap
    sns.heatmap(cm, annot=annotations, fmt='', cmap='Blues', ax=ax,
                xticklabels=['Non Fondée', 'Fondée'],
                yticklabels=['Non Fondée', 'Fondée'],
                cbar_kws={'label': 'Nombre'},
                annot_kws={'size': 12, 'weight': 'bold'})

    ax.set_xlabel('Prédiction', fontweight='bold', fontsize=12)
    ax.set_ylabel('Réalité', fontweight='bold', fontsize=12)
    ax.set_title('MATRICE DE CONFUSION - XGBoost (sur cas automatisés)',
                 fontweight='bold', fontsize=14, pad=20)

    # Stats
    tn, fp, fn, tp = cm.ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    taux_auto = 100 * mask_auto.sum() / len(y_pred)

    stats_text = f'Accuracy: {accuracy:.2%}  |  Taux automatisation: {taux_auto:.1f}%  |  Erreurs: {fp+fn} (FP={fp}, FN={fn})'
    ax.text(0.5, -0.15, stats_text,
            ha='center', transform=ax.transAxes, fontsize=11,
            style='italic', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.tight_layout()

    output_dir = Path('outputs/production/figures')
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / 'xgboost_confusion_matrix.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Sauvegardé: xgboost_confusion_matrix.png")

    plt.close()


def generate_business_rule_visualizations(impact, df_scenario):
    """Générer les visualisations de la règle métier pour XGBoost"""
    print("\n" + "="*80)
    print("📊 GÉNÉRATION DES VISUALISATIONS RÈGLE MÉTIER")
    print("="*80)

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    fig.suptitle('IMPACT DE LA RÈGLE MÉTIER - XGBoost', fontsize=16, fontweight='bold', y=0.98)

    # 1. Taux d'automatisation
    ax1 = fig.add_subplot(gs[0, 0])
    categories = ['SANS règle', 'AVEC règle']
    taux = [impact['sans_regle']['taux_auto'], impact['avec_regle']['taux_auto']]
    colors = ['#3498db', '#e74c3c']

    bars = ax1.bar(categories, taux, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax1.set_ylabel('Taux automatisation (%)', fontweight='bold')
    ax1.set_title('Taux d\'Automatisation', fontweight='bold', fontsize=12)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_ylim([0, 100])

    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
               f'{height:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

    # 2. Gain NET
    ax2 = fig.add_subplot(gs[0, 1])
    gains = [impact['sans_regle']['gain_net'], impact['avec_regle']['gain_net']]
    colors_gain = ['#2ecc71', '#f39c12']

    bars = ax2.bar(categories, gains, color=colors_gain, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax2.set_ylabel('Gain NET (DH)', fontweight='bold')
    ax2.set_title('Gain NET', fontweight='bold', fontsize=12)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.ticklabel_format(style='plain', axis='y')

    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 10000,
               f'{height:,.0f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    # 3. Différence de gain
    ax3 = fig.add_subplot(gs[0, 2])
    difference = impact['difference']
    color_diff = '#27ae60' if difference >= 0 else '#e74c3c'

    bar = ax3.bar(['Impact Règle'], [difference], color=color_diff, alpha=0.8,
                  edgecolor='black', linewidth=1.5)
    ax3.set_ylabel('Différence Gain NET (DH)', fontweight='bold')
    ax3.set_title('Impact de la Règle Métier', fontweight='bold', fontsize=12)
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=2, alpha=0.5)
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.ticklabel_format(style='plain', axis='y')

    height = bar[0].get_height()
    label_y = height + 1000 if height > 0 else height - 1000
    ax3.text(bar[0].get_x() + bar[0].get_width()/2., label_y,
           f'{height:+,.0f} DH', ha='center', va='bottom' if height > 0 else 'top',
           fontsize=10, fontweight='bold')

    # 4. Distribution par zone - SANS règle
    ax4 = fig.add_subplot(gs[1, 0])

    n_rejet_sans = (df_scenario['y_pred_original'] == 0).sum()
    n_audit_sans = (df_scenario['y_pred_original'] == -1).sum()
    n_valid_sans = (df_scenario['y_pred_original'] == 1).sum()

    zones_sans = ['Rejet\nAuto', 'Audit\nHumain', 'Validation\nAuto']
    values_sans = [n_rejet_sans, n_audit_sans, n_valid_sans]
    colors_zones = ['#e74c3c', '#f39c12', '#2ecc71']

    bars = ax4.bar(zones_sans, values_sans, color=colors_zones, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax4.set_ylabel('Nombre de cas', fontweight='bold')
    ax4.set_title('Distribution par Zone - SANS Règle', fontweight='bold', fontsize=12)
    ax4.grid(True, alpha=0.3, axis='y')

    for bar, val in zip(bars, values_sans):
        height = bar.get_height()
        pct = 100 * val / len(df_scenario)
        ax4.text(bar.get_x() + bar.get_width()/2., height + 50,
               f'{int(height)}\n({pct:.1f}%)', ha='center', va='bottom', fontsize=9, fontweight='bold')

    # 5. Distribution par zone - AVEC règle
    ax5 = fig.add_subplot(gs[1, 1])

    n_rejet_avec = (df_scenario['y_pred_with_rule'] == 0).sum()
    n_audit_avec = (df_scenario['y_pred_with_rule'] == -1).sum()
    n_valid_avec = (df_scenario['y_pred_with_rule'] == 1).sum()

    zones_avec = ['Rejet\nAuto', 'Audit\nHumain', 'Validation\nAuto']
    values_avec = [n_rejet_avec, n_audit_avec, n_valid_avec]

    bars = ax5.bar(zones_avec, values_avec, color=colors_zones, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax5.set_ylabel('Nombre de cas', fontweight='bold')
    ax5.set_title('Distribution par Zone - AVEC Règle', fontweight='bold', fontsize=12)
    ax5.grid(True, alpha=0.3, axis='y')

    for bar, val in zip(bars, values_avec):
        height = bar.get_height()
        pct = 100 * val / len(df_scenario)
        ax5.text(bar.get_x() + bar.get_width()/2., height + 50,
               f'{int(height)}\n({pct:.1f}%)', ha='center', va='bottom', fontsize=9, fontweight='bold')

    # 6. Erreurs FP et FN
    ax6 = fig.add_subplot(gs[1, 2])

    err_categories = ['SANS règle', 'AVEC règle']
    fp_values = [impact['sans_regle']['fp'], impact['avec_regle']['fp']]
    fn_values = [impact['sans_regle']['fn'], impact['avec_regle']['fn']]

    x_pos = np.arange(len(err_categories))
    width = 0.35

    bars1 = ax6.bar(x_pos - width/2, fp_values, width, label='FP (Faux Positifs)',
                   color='#e67e22', alpha=0.8, edgecolor='black', linewidth=1)
    bars2 = ax6.bar(x_pos + width/2, fn_values, width, label='FN (Faux Négatifs)',
                   color='#9b59b6', alpha=0.8, edgecolor='black', linewidth=1)

    ax6.set_ylabel('Nombre d\'erreurs', fontweight='bold')
    ax6.set_title('Erreurs par Type', fontweight='bold', fontsize=12)
    ax6.set_xticks(x_pos)
    ax6.set_xticklabels(err_categories)
    ax6.legend()
    ax6.grid(True, alpha=0.3, axis='y')

    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax6.text(bar.get_x() + bar.get_width()/2., height + 1,
                   f'{int(height)}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    # 7. Comparaison des gains et pertes - SANS règle
    ax7 = fig.add_subplot(gs[2, 0])

    composants_sans = ['Gain\nBrut', 'Perte\nFP', 'Perte\nFN', 'Gain\nNET']
    values_comp_sans = [
        impact['sans_regle']['gain_brut'],
        -impact['sans_regle']['perte_fp'],
        -impact['sans_regle']['perte_fn'],
        impact['sans_regle']['gain_net']
    ]
    colors_comp = ['#27ae60', '#e74c3c', '#c0392b', '#2980b9']

    bars = ax7.bar(composants_sans, values_comp_sans, color=colors_comp, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax7.set_ylabel('Montant (DH)', fontweight='bold')
    ax7.set_title('Composition du Gain - SANS Règle', fontweight='bold', fontsize=12)
    ax7.axhline(y=0, color='black', linestyle='-', linewidth=1.5)
    ax7.grid(True, alpha=0.3, axis='y')
    ax7.ticklabel_format(style='plain', axis='y')

    for bar in bars:
        height = bar.get_height()
        label_y = height + 5000 if height > 0 else height - 5000
        ax7.text(bar.get_x() + bar.get_width()/2., label_y,
               f'{height:,.0f}', ha='center', va='bottom' if height > 0 else 'top',
               fontsize=8, fontweight='bold')

    # 8. Comparaison des gains et pertes - AVEC règle
    ax8 = fig.add_subplot(gs[2, 1])

    composants_avec = ['Gain\nBrut', 'Perte\nFP', 'Perte\nFN', 'Gain\nNET']
    values_comp_avec = [
        impact['avec_regle']['gain_brut'],
        -impact['avec_regle']['perte_fp'],
        -impact['avec_regle']['perte_fn'],
        impact['avec_regle']['gain_net']
    ]

    bars = ax8.bar(composants_avec, values_comp_avec, color=colors_comp, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax8.set_ylabel('Montant (DH)', fontweight='bold')
    ax8.set_title('Composition du Gain - AVEC Règle', fontweight='bold', fontsize=12)
    ax8.axhline(y=0, color='black', linestyle='-', linewidth=1.5)
    ax8.grid(True, alpha=0.3, axis='y')
    ax8.ticklabel_format(style='plain', axis='y')

    for bar in bars:
        height = bar.get_height()
        label_y = height + 5000 if height > 0 else height - 5000
        ax8.text(bar.get_x() + bar.get_width()/2., label_y,
               f'{height:,.0f}', ha='center', va='bottom' if height > 0 else 'top',
               fontsize=8, fontweight='bold')

    # 9. Récapitulatif des métriques
    ax9 = fig.add_subplot(gs[2, 2])
    ax9.axis('off')

    summary_text = f"""
    📊 RÉSUMÉ DES MÉTRIQUES - XGBoost

    SANS règle métier:
    • Automatisés: {impact['sans_regle']['auto']}/{impact['sans_regle']['total']} ({impact['sans_regle']['taux_auto']:.1f}%)
    • TP: {impact['sans_regle']['tp']} | TN: {impact['sans_regle']['tn']}
    • FP: {impact['sans_regle']['fp']} | FN: {impact['sans_regle']['fn']}
    • Gain NET: {impact['sans_regle']['gain_net']:,.0f} DH

    AVEC règle métier:
    • Automatisés: {impact['avec_regle']['auto']}/{impact['avec_regle']['total']} ({impact['avec_regle']['taux_auto']:.1f}%)
    • TP: {impact['avec_regle']['tp']} | TN: {impact['avec_regle']['tn']}
    • FP: {impact['avec_regle']['fp']} | FN: {impact['avec_regle']['fn']}
    • Gain NET: {impact['avec_regle']['gain_net']:,.0f} DH

    IMPACT:
    • Différence: {impact['difference']:+,.0f} DH
    • Variation: {100*impact['difference']/impact['sans_regle']['gain_net']:+.2f}%
    """

    ax9.text(0.1, 0.5, summary_text, transform=ax9.transAxes,
            fontsize=10, verticalalignment='center', family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

    plt.tight_layout()

    output_dir = Path('outputs/production/figures')
    output_path = output_dir / 'xgboost_business_rule_impact.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Sauvegardé: xgboost_business_rule_impact.png")

    plt.close()


def analyze_by_family(df_2025, y_true, y_prob, threshold_low, threshold_high):
    """Analyser l'accuracy par famille pour XGBoost"""
    print("\n" + "="*80)
    print("📊 ANALYSE PAR FAMILLE DE PRODUIT - XGBoost")
    print("="*80)

    y_pred = create_3zone_predictions(y_prob, threshold_low, threshold_high)

    families = df_2025['Famille Produit'].unique()
    results = []

    for family in families:
        mask_family = df_2025['Famille Produit'] == family

        if mask_family.sum() == 0:
            continue

        # Sur cette famille
        y_true_fam = y_true[mask_family]
        y_pred_fam = y_pred[mask_family]

        # Seulement sur les cas automatisés
        mask_auto_fam = (y_pred_fam != -1)

        if mask_auto_fam.sum() == 0:
            continue

        y_true_auto = y_true_fam[mask_auto_fam]
        y_pred_auto = y_pred_fam[mask_auto_fam]

        accuracy = accuracy_score(y_true_auto, y_pred_auto)

        results.append({
            'Famille': str(family)[:50],
            'N_Total': mask_family.sum(),
            'N_Auto': mask_auto_fam.sum(),
            'Taux_Auto': 100 * mask_auto_fam.sum() / mask_family.sum(),
            'Accuracy': accuracy
        })

    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values('N_Total', ascending=False)

    print(f"  ✓ {len(families)} familles analysées")
    print(f"\nTop 5 familles par volume:")
    print(df_results.head(5)[['Famille', 'N_Total', 'N_Auto', 'Accuracy']].to_string(index=False))

    # Générer la visualisation
    generate_family_accuracy_chart(df_results)

    return df_results


def generate_family_accuracy_chart(df_results):
    """Générer le graphique d'accuracy par famille pour XGBoost"""
    print("\n📊 Génération du graphique d'accuracy par famille...")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    fig.suptitle('ACCURACY PAR FAMILLE DE PRODUIT - XGBoost', fontsize=16, fontweight='bold')

    # Top 15 familles par volume
    df_top = df_results.head(15).copy()
    # Inverser l'ordre pour avoir le Top 1 en haut
    df_top = df_top.iloc[::-1].reset_index(drop=True)

    # Graphique 1: Accuracy
    colors = ['#27ae60' if acc >= 0.98 else '#f39c12' if acc >= 0.95 else '#e74c3c'
              for acc in df_top['Accuracy']]

    y_pos = range(len(df_top))
    bars = ax1.barh(y_pos, df_top['Accuracy'] * 100, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(df_top['Famille'], fontsize=8)
    ax1.set_xlabel('Accuracy (%)', fontweight='bold', fontsize=11)
    ax1.set_title('Accuracy par Famille (Top 15)', fontweight='bold', fontsize=13)
    ax1.grid(True, alpha=0.3, axis='x')
    ax1.set_xlim([85, 100])

    # Ajouter les valeurs
    for i, (_, row) in enumerate(df_top.iterrows()):
        acc_val = row['Accuracy'] * 100
        volume = row['N_Total']
        ax1.text(acc_val + 0.3, i, f'{acc_val:.1f}% (n={volume})',
                va='center', fontsize=8, fontweight='bold')

    # Graphique 2: Volume et taux d'automatisation
    ax1_2 = ax2.twinx()

    x_pos = range(len(df_top))

    bars1 = ax2.bar(x_pos, df_top['N_Total'], color='#3498db', alpha=0.6, label='Volume total', edgecolor='black', linewidth=1)
    line1 = ax1_2.plot(x_pos, df_top['Taux_Auto'], color='#e74c3c', marker='o', linewidth=2,
                       markersize=8, label='Taux automatisation', linestyle='--')

    ax2.set_xlabel('Famille de Produit', fontweight='bold', fontsize=11)
    ax2.set_ylabel('Volume', fontweight='bold', fontsize=11, color='#3498db')
    ax1_2.set_ylabel('Taux automatisation (%)', fontweight='bold', fontsize=11, color='#e74c3c')
    ax2.set_title('Volume et Taux d\'Automatisation (Top 15)', fontweight='bold', fontsize=13)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(df_top['Famille'], rotation=45, ha='right', fontsize=8)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.tick_params(axis='y', labelcolor='#3498db')
    ax1_2.tick_params(axis='y', labelcolor='#e74c3c')

    # Légende combinée
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax1_2.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    plt.tight_layout()

    output_dir = Path('outputs/production/figures')
    output_path = output_dir / 'xgboost_accuracy_by_family.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Sauvegardé: xgboost_accuracy_by_family.png")

    plt.close()


def save_summary_report(impact, df_family):
    """Sauvegarder un rapport récapitulatif pour XGBoost"""
    print("\n📝 Génération du rapport récapitulatif...")

    output_dir = Path('outputs/production')
    output_dir.mkdir(parents=True, exist_ok=True)

    report_path = output_dir / 'xgboost_rapport_analyse.txt'

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("RAPPORT D'ANALYSE - XGBoost avec 3 ZONES + RÈGLE MÉTIER\n")
        f.write("="*80 + "\n\n")

        f.write("SYSTÈME À 3 ZONES DE DÉCISION:\n")
        f.write("- Zone 1 (prob <= seuil_bas): REJET AUTO (Non Fondée)\n")
        f.write("- Zone 2 (seuil_bas < prob < seuil_haut): AUDIT HUMAIN (manuel)\n")
        f.write("- Zone 3 (prob >= seuil_haut): VALIDATION AUTO (Fondée)\n\n")

        f.write("RÈGLE MÉTIER APPLIQUÉE:\n")
        f.write("- Un client ne peut bénéficier que d'UNE validation automatique par année\n")
        f.write("- Les réclamations sont triées par Date de Qualification\n")
        f.write("- Seule la première validation auto est acceptée pour chaque client/année\n")
        f.write("- Les validations suivantes sont transformées en audits humains\n\n")

        f.write("="*80 + "\n")
        f.write("RÉSULTATS XGBoost\n")
        f.write("="*80 + "\n\n")

        f.write(f"SANS RÈGLE MÉTIER:\n")
        f.write(f"  Taux auto   : {impact['sans_regle']['taux_auto']:.1f}%\n")
        f.write(f"  Automatisés : {impact['sans_regle']['auto']}\n")
        f.write(f"  Audits      : {impact['sans_regle']['audit']}\n")
        f.write(f"  TP: {impact['sans_regle']['tp']} | TN: {impact['sans_regle']['tn']} | FP: {impact['sans_regle']['fp']} | FN: {impact['sans_regle']['fn']}\n")
        f.write(f"  Gain brut   : {impact['sans_regle']['gain_brut']:,.0f} DH\n")
        f.write(f"  Perte FP    : {impact['sans_regle']['perte_fp']:,.0f} DH\n")
        f.write(f"  Perte FN    : {impact['sans_regle']['perte_fn']:,.0f} DH\n")
        f.write(f"  Gain NET    : {impact['sans_regle']['gain_net']:,.0f} DH\n")

        f.write(f"\nAVEC RÈGLE MÉTIER:\n")
        f.write(f"  Taux auto   : {impact['avec_regle']['taux_auto']:.1f}%\n")
        f.write(f"  Automatisés : {impact['avec_regle']['auto']}\n")
        f.write(f"  Audits      : {impact['avec_regle']['audit']}\n")
        f.write(f"  TP: {impact['avec_regle']['tp']} | TN: {impact['avec_regle']['tn']} | FP: {impact['avec_regle']['fp']} | FN: {impact['avec_regle']['fn']}\n")
        f.write(f"  Gain brut   : {impact['avec_regle']['gain_brut']:,.0f} DH\n")
        f.write(f"  Perte FP    : {impact['avec_regle']['perte_fp']:,.0f} DH\n")
        f.write(f"  Perte FN    : {impact['avec_regle']['perte_fn']:,.0f} DH\n")
        f.write(f"  Gain NET    : {impact['avec_regle']['gain_net']:,.0f} DH\n")

        f.write(f"\nIMPACT DE LA RÈGLE:\n")
        f.write(f"  Différence  : {impact['difference']:+,.0f} DH\n")
        f.write(f"  Variation   : {100*impact['difference']/impact['sans_regle']['gain_net']:+.2f}%\n")

        f.write("\n" + "="*80 + "\n")
        f.write("ANALYSE PAR FAMILLE (Top 10)\n")
        f.write("="*80 + "\n\n")

        for _, row in df_family.head(10).iterrows():
            f.write(f"{row['Famille'][:50]:50s} | ")
            f.write(f"Volume: {row['N_Total']:4d} | ")
            f.write(f"Auto: {row['N_Auto']:4d} ({row['Taux_Auto']:5.1f}%) | ")
            f.write(f"Acc: {row['Accuracy']*100:5.2f}%\n")

    print(f"✅ Sauvegardé: xgboost_rapport_analyse.txt")


def main():
    """Fonction principale"""
    print("="*80)
    print("ANALYSE POST-ENTRAÎNEMENT - XGBoost avec 3 ZONES + RÈGLE MÉTIER")
    print("="*80)

    # Charger les résultats
    df_2025, y_true, xgboost_results = load_results()

    if df_2025 is None:
        return

    y_prob = xgboost_results['y_prob']
    threshold_low = xgboost_results['threshold_low']
    threshold_high = xgboost_results['threshold_high']

    # Générer la matrice de confusion
    generate_confusion_matrix(df_2025, y_true, y_prob, threshold_low, threshold_high)

    # Appliquer la règle métier
    df_scenario = apply_business_rule(df_2025, y_true, y_prob, threshold_low, threshold_high)

    # Calculer l'impact financier
    impact = calculate_financial_impact(df_scenario)

    # Générer les visualisations de comparaison
    generate_business_rule_visualizations(impact, df_scenario)

    # Analyser par famille
    df_family = analyze_by_family(df_2025, y_true, y_prob, threshold_low, threshold_high)

    # Sauvegarder le rapport
    save_summary_report(impact, df_family)

    print("\n" + "="*80)
    print("✅ ANALYSE TERMINÉE - XGBoost")
    print("="*80)
    print("\n📂 Fichiers générés:")
    print("   - outputs/production/figures/xgboost_confusion_matrix.png")
    print("   - outputs/production/figures/xgboost_business_rule_impact.png")
    print("   - outputs/production/figures/xgboost_accuracy_by_family.png")
    print("   - outputs/production/xgboost_rapport_analyse.txt")


if __name__ == '__main__':
    main()
