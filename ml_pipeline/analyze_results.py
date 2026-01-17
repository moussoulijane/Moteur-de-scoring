"""
ANALYSE POST-ENTRAÎNEMENT - Avec 3 Zones de Décision et Règle Métier
Génère des analyses détaillées sans réentraîner les modèles
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
from sklearn.metrics import confusion_matrix, accuracy_score
import warnings
warnings.filterwarnings('ignore')

# Configuration
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (18, 12)

PRIX_UNITAIRE_DH = 169


def load_results():
    """Charger les résultats des modèles avec les seuils"""
    print("\n" + "="*80)
    print("📂 CHARGEMENT DES RÉSULTATS")
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
        models_results = {}

        for model_name in ['XGBoost', 'RandomForest', 'CatBoost']:
            if model_name in predictions_data:
                models_results[model_name] = {
                    'y_prob': predictions_data[model_name]['y_prob'],
                    'threshold_low': predictions_data[model_name]['threshold_low'],
                    'threshold_high': predictions_data[model_name]['threshold_high']
                }
                print(f"   ✓ {model_name} (seuils: {models_results[model_name]['threshold_low']:.2f} / {models_results[model_name]['threshold_high']:.2f})")

        print(f"✅ Prédictions chargées pour {len(models_results)} modèles")

    else:
        print("⚠️  Fichier de prédictions non trouvé!")
        print("   Exécutez d'abord: python model_comparison.py")
        return None, None, None

    return df_2025, y_true, models_results


def create_3zone_predictions(y_prob, threshold_low, threshold_high):
    """Créer les prédictions avec 3 zones de décision"""
    # Zone 1: Rejet auto (prob <= threshold_low) → prédiction = 0
    # Zone 2: Audit humain (threshold_low < prob < threshold_high) → prédiction = -1 (manuel)
    # Zone 3: Validation auto (prob >= threshold_high) → prédiction = 1

    y_pred = np.full(len(y_prob), -1, dtype=int)  # Par défaut: audit humain
    y_pred[y_prob <= threshold_low] = 0  # Rejet auto
    y_pred[y_prob >= threshold_high] = 1  # Validation auto

    return y_pred


def apply_business_rule(df_2025, y_true, y_prob, threshold_low, threshold_high, model_name):
    """Appliquer la règle métier: 1 validation auto par client par année"""
    print(f"\n🔒 Application de la règle métier - {model_name}")
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


def calculate_financial_impact(df_scenario, model_name):
    """Calculer l'impact financier avec et sans règle (3 zones)"""
    print(f"\n💰 Calcul de l'impact financier - {model_name}")
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
    print(f"   Corrects    : {auto_avec} ({100*auto_avec/mask_auto_avec.sum():.1f}% si > 0 else 0)")
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
            'fn': fn_mask_sans.sum()
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
            'fn': fn_mask_avec.sum()
        },
        'difference': difference
    }


def generate_confusion_matrices(df_2025, y_true, models_results):
    """Générer les matrices de confusion pour tous les modèles"""
    print("\n" + "="*80)
    print("📊 GÉNÉRATION DES MATRICES DE CONFUSION")
    print("="*80)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('MATRICES DE CONFUSION PAR MODÈLE - 2025 (sur cas automatisés)',
                 fontsize=16, fontweight='bold', y=1.02)

    models_list = ['XGBoost', 'RandomForest', 'CatBoost']
    colors_map = ['Blues', 'Greens', 'Purples']

    for idx, (model_name, cmap) in enumerate(zip(models_list, colors_map)):
        ax = axes[idx]

        y_prob = models_results[model_name]['y_prob']
        threshold_low = models_results[model_name]['threshold_low']
        threshold_high = models_results[model_name]['threshold_high']

        # Créer prédictions avec 3 zones
        y_pred = create_3zone_predictions(y_prob, threshold_low, threshold_high)

        # Matrice de confusion seulement sur cas automatisés
        mask_auto = (y_pred != -1)
        y_true_auto = y_true[mask_auto]
        y_pred_auto = y_pred[mask_auto]

        cm = confusion_matrix(y_true_auto, y_pred_auto)
        cm_percent = cm.astype('float') / cm.sum() * 100

        # Annotations
        annotations = []
        for i in range(2):
            row = []
            for j in range(2):
                row.append(f'{cm[i,j]}\n({cm_percent[i,j]:.1f}%)')
            annotations.append(row)

        # Heatmap
        sns.heatmap(cm, annot=annotations, fmt='', cmap=cmap, ax=ax,
                    xticklabels=['Non Fondée', 'Fondée'],
                    yticklabels=['Non Fondée', 'Fondée'],
                    cbar_kws={'label': 'Nombre'},
                    annot_kws={'size': 11, 'weight': 'bold'})

        ax.set_xlabel('Prédiction', fontweight='bold', fontsize=11)
        ax.set_ylabel('Réalité', fontweight='bold', fontsize=11)
        ax.set_title(f'{model_name}', fontweight='bold', fontsize=13)

        # Stats
        tn, fp, fn, tp = cm.ravel()
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        taux_auto = 100 * mask_auto.sum() / len(y_pred)

        ax.text(0.5, -0.18, f'Accuracy: {accuracy:.2%}  |  Auto: {taux_auto:.1f}%  |  Erreurs: {fp+fn}',
                ha='center', transform=ax.transAxes, fontsize=10, style='italic')

    plt.tight_layout()

    output_dir = Path('outputs/production/figures')
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / 'confusion_matrices.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Sauvegardé: confusion_matrices.png")

    plt.close()


def generate_business_rule_visualizations(all_impacts, all_scenarios):
    """Générer les visualisations de la règle métier"""
    print("\n" + "="*80)
    print("📊 GÉNÉRATION DES VISUALISATIONS RÈGLE MÉTIER")
    print("="*80)

    models_list = list(all_impacts.keys())

    # Figure 1: Comparaisons avec/sans règle
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('IMPACT DE LA RÈGLE MÉTIER - Comparaison SANS vs AVEC',
                fontsize=16, fontweight='bold', y=0.995)

    x = np.arange(len(models_list))
    width = 0.35

    # 1. Taux d'automatisation
    ax = axes[0, 0]
    auto_sans = [all_impacts[m]['sans_regle']['taux_auto'] for m in models_list]
    auto_avec = [all_impacts[m]['avec_regle']['taux_auto'] for m in models_list]

    bars1 = ax.bar(x - width/2, auto_sans, width, label='SANS règle',
                   color='#3498db', alpha=0.8, edgecolor='black', linewidth=1)
    bars2 = ax.bar(x + width/2, auto_avec, width, label='AVEC règle',
                   color='#e74c3c', alpha=0.8, edgecolor='black', linewidth=1)

    ax.set_ylabel('Taux automatisation (%)', fontweight='bold')
    ax.set_title('Taux d\'Automatisation', fontweight='bold', fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(models_list)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                   f'{height:.1f}%', ha='center', va='bottom', fontsize=8, fontweight='bold')

    # 2. Gain NET
    ax = axes[0, 1]
    gains_sans = [all_impacts[m]['sans_regle']['gain_net'] for m in models_list]
    gains_avec = [all_impacts[m]['avec_regle']['gain_net'] for m in models_list]

    bars1 = ax.bar(x - width/2, gains_sans, width, label='SANS règle',
                   color='#2ecc71', alpha=0.8, edgecolor='black', linewidth=1)
    bars2 = ax.bar(x + width/2, gains_avec, width, label='AVEC règle',
                   color='#f39c12', alpha=0.8, edgecolor='black', linewidth=1)

    ax.set_ylabel('Gain NET (DH)', fontweight='bold')
    ax.set_title('Gain NET', fontweight='bold', fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(models_list)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.ticklabel_format(style='plain', axis='y')

    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 10000,
                   f'{height:,.0f}', ha='center', va='bottom', fontsize=7, fontweight='bold')

    # 3. Différence de gain
    ax = axes[0, 2]
    differences = [all_impacts[m]['difference'] for m in models_list]
    colors_diff = ['#27ae60' if d >= 0 else '#e74c3c' for d in differences]

    bars = ax.bar(models_list, differences, color=colors_diff, alpha=0.8,
                  edgecolor='black', linewidth=1)
    ax.set_ylabel('Différence Gain NET (DH)', fontweight='bold')
    ax.set_title('Impact de la Règle Métier', fontweight='bold', fontsize=13)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1.5, alpha=0.5)
    ax.grid(True, alpha=0.3, axis='y')
    ax.ticklabel_format(style='plain', axis='y')

    for bar in bars:
        height = bar.get_height()
        label_y = height + 1000 if height > 0 else height - 1000
        ax.text(bar.get_x() + bar.get_width()/2., label_y,
               f'{height:+,.0f}', ha='center', va='bottom' if height > 0 else 'top',
               fontsize=9, fontweight='bold')

    # 4. Nombre de cas par zone (SANS règle)
    ax = axes[1, 0]

    # Calculer les rejets, audits, validations pour chaque modèle SANS règle
    rejets_sans = []
    audits_sans = []
    validations_sans = []

    for model_name in models_list:
        df_sc = all_scenarios[model_name]
        rejets_sans.append((df_sc['y_pred_original'] == 0).sum())
        audits_sans.append((df_sc['y_pred_original'] == -1).sum())
        validations_sans.append((df_sc['y_pred_original'] == 1).sum())

    width_zone = 0.25
    ax.bar(x - width_zone, rejets_sans, width_zone, label='Rejet Auto', color='#e74c3c', alpha=0.8)
    ax.bar(x, audits_sans, width_zone, label='Audit Humain', color='#f39c12', alpha=0.8)
    ax.bar(x + width_zone, validations_sans, width_zone, label='Validation Auto', color='#2ecc71', alpha=0.8)

    ax.set_ylabel('Nombre de cas', fontweight='bold')
    ax.set_title('Distribution par Zone - SANS Règle', fontweight='bold', fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(models_list)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # 5. Nombre de cas par zone (AVEC règle)
    ax = axes[1, 1]

    rejets_avec = []
    audits_avec = []
    validations_avec = []

    for model_name in models_list:
        df_sc = all_scenarios[model_name]
        rejets_avec.append((df_sc['y_pred_with_rule'] == 0).sum())
        audits_avec.append((df_sc['y_pred_with_rule'] == -1).sum())
        validations_avec.append((df_sc['y_pred_with_rule'] == 1).sum())

    ax.bar(x - width_zone, rejets_avec, width_zone, label='Rejet Auto', color='#e74c3c', alpha=0.8)
    ax.bar(x, audits_avec, width_zone, label='Audit Humain', color='#f39c12', alpha=0.8)
    ax.bar(x + width_zone, validations_avec, width_zone, label='Validation Auto', color='#2ecc71', alpha=0.8)

    ax.set_ylabel('Nombre de cas', fontweight='bold')
    ax.set_title('Distribution par Zone - AVEC Règle', fontweight='bold', fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(models_list)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # 6. Erreurs FP et FN
    ax = axes[1, 2]

    err_sans = [all_impacts[m]['sans_regle']['fp'] + all_impacts[m]['sans_regle']['fn'] for m in models_list]
    err_avec = [all_impacts[m]['avec_regle']['fp'] + all_impacts[m]['avec_regle']['fn'] for m in models_list]

    bars1 = ax.bar(x - width/2, err_sans, width, label='SANS règle',
                   color='#9b59b6', alpha=0.8, edgecolor='black', linewidth=1)
    bars2 = ax.bar(x + width/2, err_avec, width, label='AVEC règle',
                   color='#8e44ad', alpha=0.8, edgecolor='black', linewidth=1)

    ax.set_ylabel('Nombre d\'erreurs', fontweight='bold')
    ax.set_title('Total Erreurs (FP + FN)', fontweight='bold', fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(models_list)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 2,
                   f'{int(height)}', ha='center', va='bottom', fontsize=8, fontweight='bold')

    plt.tight_layout()

    output_dir = Path('outputs/production/figures')
    output_path = output_dir / 'business_rule_impact.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Sauvegardé: business_rule_impact.png")

    plt.close()


def analyze_by_family(df_2025, y_true, models_results):
    """Analyser l'accuracy par famille pour tous les modèles"""
    print("\n" + "="*80)
    print("📊 ANALYSE PAR FAMILLE DE PRODUIT")
    print("="*80)

    models_list = list(models_results.keys())
    families = df_2025['Famille Produit'].unique()

    results_all = {}

    for model_name in models_list:
        print(f"\n{model_name}:")

        y_prob = models_results[model_name]['y_prob']
        threshold_low = models_results[model_name]['threshold_low']
        threshold_high = models_results[model_name]['threshold_high']

        y_pred = create_3zone_predictions(y_prob, threshold_low, threshold_high)

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
                'Famille': str(family)[:40],
                'N_Total': mask_family.sum(),
                'N_Auto': mask_auto_fam.sum(),
                'Taux_Auto': 100 * mask_auto_fam.sum() / mask_family.sum(),
                'Accuracy': accuracy
            })

        df_results = pd.DataFrame(results)
        df_results = df_results.sort_values('N_Total', ascending=False)
        results_all[model_name] = df_results

        print(f"  ✓ {len(families)} familles analysées")

    # Générer la visualisation
    generate_family_accuracy_chart(results_all)

    return results_all


def generate_family_accuracy_chart(results_all):
    """Générer le graphique d'accuracy par famille"""
    print("\n📊 Génération du graphique d'accuracy par famille...")

    models_list = list(results_all.keys())

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle('ACCURACY PAR FAMILLE DE PRODUIT (Top 10)', fontsize=16, fontweight='bold')

    for idx, model_name in enumerate(models_list):
        ax = axes[idx]
        df_res = results_all[model_name].head(10)

        colors = ['#27ae60' if acc >= 0.98 else '#f39c12' if acc >= 0.95 else '#e74c3c'
                  for acc in df_res['Accuracy']]

        bars = ax.barh(range(len(df_res)), df_res['Accuracy'] * 100, color=colors, alpha=0.8)
        ax.set_yticks(range(len(df_res)))
        ax.set_yticklabels(df_res['Famille'], fontsize=9)
        ax.set_xlabel('Accuracy (%)', fontweight='bold')
        ax.set_title(f'{model_name}', fontweight='bold', fontsize=13)
        ax.grid(True, alpha=0.3, axis='x')
        ax.set_xlim([90, 100])

        # Ajouter les valeurs
        for i, (_, row) in enumerate(df_res.iterrows()):
            acc_val = row['Accuracy'] * 100
            volume = row['N_Total']
            ax.text(acc_val + 0.3, i, f'{acc_val:.1f}% (n={volume})',
                    va='center', fontsize=8, fontweight='bold')

    plt.tight_layout()

    output_dir = Path('outputs/production/figures')
    output_path = output_dir / 'accuracy_by_family.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Sauvegardé: accuracy_by_family.png")

    plt.close()


def save_summary_report(all_impacts):
    """Sauvegarder un rapport récapitulatif"""
    print("\n📝 Génération du rapport récapitulatif...")

    output_dir = Path('outputs/production')
    output_dir.mkdir(parents=True, exist_ok=True)

    report_path = output_dir / 'rapport_regle_metier.txt'

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("RAPPORT D'ANALYSE - IMPACT DE LA RÈGLE MÉTIER\n")
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
        f.write("RÉSULTATS PAR MODÈLE\n")
        f.write("="*80 + "\n\n")

        for model_name in all_impacts.keys():
            impact = all_impacts[model_name]

            f.write(f"{model_name}:\n")
            f.write(f"\n  SANS RÈGLE MÉTIER:\n")
            f.write(f"    Taux auto   : {impact['sans_regle']['taux_auto']:.1f}%\n")
            f.write(f"    Automatisés : {impact['sans_regle']['auto']}\n")
            f.write(f"    Audits      : {impact['sans_regle']['audit']}\n")
            f.write(f"    Gain brut   : {impact['sans_regle']['gain_brut']:,.0f} DH\n")
            f.write(f"    Perte FP    : {impact['sans_regle']['perte_fp']:,.0f} DH\n")
            f.write(f"    Perte FN    : {impact['sans_regle']['perte_fn']:,.0f} DH\n")
            f.write(f"    Gain NET    : {impact['sans_regle']['gain_net']:,.0f} DH\n")
            f.write(f"    Erreurs     : FP={impact['sans_regle']['fp']}, FN={impact['sans_regle']['fn']}\n")

            f.write(f"\n  AVEC RÈGLE MÉTIER:\n")
            f.write(f"    Taux auto   : {impact['avec_regle']['taux_auto']:.1f}%\n")
            f.write(f"    Automatisés : {impact['avec_regle']['auto']}\n")
            f.write(f"    Audits      : {impact['avec_regle']['audit']}\n")
            f.write(f"    Gain brut   : {impact['avec_regle']['gain_brut']:,.0f} DH\n")
            f.write(f"    Perte FP    : {impact['avec_regle']['perte_fp']:,.0f} DH\n")
            f.write(f"    Perte FN    : {impact['avec_regle']['perte_fn']:,.0f} DH\n")
            f.write(f"    Gain NET    : {impact['avec_regle']['gain_net']:,.0f} DH\n")
            f.write(f"    Erreurs     : FP={impact['avec_regle']['fp']}, FN={impact['avec_regle']['fn']}\n")

            f.write(f"\n  IMPACT:\n")
            f.write(f"    Différence  : {impact['difference']:+,.0f} DH\n")
            f.write(f"    Variation   : {100*impact['difference']/impact['sans_regle']['gain_net']:+.2f}%\n")

            f.write("\n" + "-"*80 + "\n\n")

        # Meilleur modèle
        best_sans = max(all_impacts.keys(), key=lambda k: all_impacts[k]['sans_regle']['gain_net'])
        best_avec = max(all_impacts.keys(), key=lambda k: all_impacts[k]['avec_regle']['gain_net'])

        f.write("="*80 + "\n")
        f.write("MEILLEURS MODÈLES\n")
        f.write("="*80 + "\n")
        f.write(f"\nSANS règle: {best_sans} ({all_impacts[best_sans]['sans_regle']['gain_net']:,.0f} DH)\n")
        f.write(f"AVEC règle: {best_avec} ({all_impacts[best_avec]['avec_regle']['gain_net']:,.0f} DH)\n\n")

    print(f"✅ Sauvegardé: rapport_regle_metier.txt")


def main():
    """Fonction principale"""
    print("="*80)
    print("ANALYSE POST-ENTRAÎNEMENT - 3 ZONES + RÈGLE MÉTIER")
    print("="*80)

    # Charger les résultats
    df_2025, y_true, models_results = load_results()

    if df_2025 is None:
        return

    # Générer les matrices de confusion
    generate_confusion_matrices(df_2025, y_true, models_results)

    # Appliquer la règle métier et calculer l'impact pour chaque modèle
    all_impacts = {}
    all_scenarios = {}

    for model_name, results in models_results.items():
        y_prob = results['y_prob']
        threshold_low = results['threshold_low']
        threshold_high = results['threshold_high']

        # Appliquer la règle métier
        df_scenario = apply_business_rule(df_2025, y_true, y_prob, threshold_low, threshold_high, model_name)
        all_scenarios[model_name] = df_scenario

        # Calculer l'impact financier
        impact = calculate_financial_impact(df_scenario, model_name)
        all_impacts[model_name] = impact

    # Générer les visualisations de comparaison
    generate_business_rule_visualizations(all_impacts, all_scenarios)

    # Analyser par famille
    analyze_by_family(df_2025, y_true, models_results)

    # Sauvegarder le rapport
    save_summary_report(all_impacts)

    print("\n" + "="*80)
    print("✅ ANALYSE TERMINÉE")
    print("="*80)
    print("\n📂 Fichiers générés:")
    print("   - outputs/production/figures/confusion_matrices.png")
    print("   - outputs/production/figures/business_rule_impact.png")
    print("   - outputs/production/figures/accuracy_by_family.png")
    print("   - outputs/production/rapport_regle_metier.txt")


if __name__ == '__main__':
    main()
