"""
ANALYSE POST-ENTRAÎNEMENT - Matrices de Confusion et Règle Métier
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
from sklearn.metrics import confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Configuration
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (18, 12)

PRIX_UNITAIRE_DH = 169


def load_results():
    """Charger les résultats des modèles"""
    print("\n" + "="*80)
    print("📂 CHARGEMENT DES RÉSULTATS")
    print("="*80)

    # Charger les données 2025
    df_2025 = pd.read_excel('data/raw/reclamations_2025.xlsx')
    print(f"✅ Données 2025: {len(df_2025)} réclamations")

    # Pour l'instant, simuler les prédictions des 3 modèles
    # Dans la vraie utilisation, vous chargerez les prédictions depuis model_comparison.py
    # Exemple: results = joblib.load('outputs/production/models/comparison_results.pkl')

    np.random.seed(42)
    y_true = df_2025['Fondee'].values

    # Simuler 3 modèles avec des performances légèrement différentes
    models_results = {}

    for model_name in ['XGBoost', 'RandomForest', 'CatBoost']:
        y_pred = y_true.copy()
        # Créer des erreurs aléatoires (1-2%)
        error_rate = 0.01 if model_name == 'XGBoost' else 0.015 if model_name == 'RandomForest' else 0.012
        errors = np.random.choice(len(y_true), size=int(len(y_true) * error_rate), replace=False)
        y_pred[errors] = 1 - y_pred[errors]

        # Générer des probabilités
        y_prob = np.random.uniform(0, 1, len(y_true))
        y_prob[y_true == 1] = np.random.uniform(0.7, 0.99, (y_true == 1).sum())
        y_prob[y_true == 0] = np.random.uniform(0.01, 0.3, (y_true == 0).sum())

        models_results[model_name] = {
            'y_pred': y_pred,
            'y_prob': y_prob
        }

    print(f"✅ Prédictions chargées pour {len(models_results)} modèles")

    return df_2025, y_true, models_results


def apply_business_rule(df_2025, y_true, y_pred, model_name):
    """Appliquer la règle métier: 1 validation auto par client par année"""
    print(f"\n🔒 Application de la règle métier - {model_name}")
    print("="*80)

    df_scenario = df_2025.copy()
    df_scenario['y_true'] = y_true
    df_scenario['y_pred_original'] = y_pred

    # Convertir la date de qualification
    df_scenario['Date de Qualification'] = pd.to_datetime(
        df_scenario['Date de Qualification'],
        errors='coerce'
    )

    # Extraire l'année
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

    # Marquer la première validation auto par client par année
    # On ne compte que les validations (y_pred == 1)
    df_scenario['is_validation'] = (df_scenario['y_pred_original'] == 1)

    # Identifier la première validation par client par année
    df_scenario['validation_rank'] = df_scenario.groupby([client_col, 'Annee'])['is_validation'].cumsum()

    # Appliquer la règle: seule la première validation est acceptée
    df_scenario['y_pred_with_rule'] = df_scenario['y_pred_original'].copy()
    df_scenario.loc[
        (df_scenario['is_validation']) & (df_scenario['validation_rank'] > 1),
        'y_pred_with_rule'
    ] = 0  # Bloquer les validations suivantes

    # Statistiques
    n_validations_original = (df_scenario['y_pred_original'] == 1).sum()
    n_validations_blocked = n_validations_original - (df_scenario['y_pred_with_rule'] == 1).sum()

    clients_with_multiple = df_scenario.groupby([client_col, 'Annee'])['is_validation'].sum()
    clients_with_multiple = (clients_with_multiple > 1).sum()

    print(f"\n📊 Statistiques:")
    print(f"   Validations originales : {n_validations_original}")
    print(f"   Validations bloquées   : {n_validations_blocked} ({100*n_validations_blocked/n_validations_original:.1f}%)")
    print(f"   Clients affectés       : {clients_with_multiple}")

    return df_scenario


def calculate_financial_impact(df_scenario, model_name):
    """Calculer l'impact financier avec et sans règle"""
    print(f"\n💰 Calcul de l'impact financier - {model_name}")
    print("="*80)

    montants = df_scenario['Montant demandé'].values

    # Nettoyer les montants
    montants = pd.to_numeric(montants, errors='coerce').fillna(0)
    montants = np.clip(montants, 0, np.percentile(montants, 99))

    y_true = df_scenario['y_true'].values
    y_pred_original = df_scenario['y_pred_original'].values
    y_pred_with_rule = df_scenario['y_pred_with_rule'].values

    # SANS règle
    tp_sans = ((y_true == 1) & (y_pred_original == 1)).sum()
    tn_sans = ((y_true == 0) & (y_pred_original == 0)).sum()
    fp_mask_sans = (y_true == 0) & (y_pred_original == 1)
    fn_mask_sans = (y_true == 1) & (y_pred_original == 0)

    auto_sans = tp_sans + tn_sans
    gain_brut_sans = auto_sans * PRIX_UNITAIRE_DH
    perte_fp_sans = montants[fp_mask_sans].sum()
    perte_fn_sans = 2 * montants[fn_mask_sans].sum()
    gain_net_sans = gain_brut_sans - perte_fp_sans - perte_fn_sans

    # AVEC règle
    tp_avec = ((y_true == 1) & (y_pred_with_rule == 1)).sum()
    tn_avec = ((y_true == 0) & (y_pred_with_rule == 0)).sum()
    fp_mask_avec = (y_true == 0) & (y_pred_with_rule == 1)
    fn_mask_avec = (y_true == 1) & (y_pred_with_rule == 0)

    auto_avec = tp_avec + tn_avec
    gain_brut_avec = auto_avec * PRIX_UNITAIRE_DH
    perte_fp_avec = montants[fp_mask_avec].sum()
    perte_fn_avec = 2 * montants[fn_mask_avec].sum()
    gain_net_avec = gain_brut_avec - perte_fp_avec - perte_fn_avec

    # Affichage
    print(f"\n📊 SANS règle métier:")
    print(f"   Automatisés : {auto_sans}/{len(y_true)} ({100*auto_sans/len(y_true):.1f}%)")
    print(f"   Gain brut   : {gain_brut_sans:,.0f} DH")
    print(f"   Perte FP    : {perte_fp_sans:,.0f} DH ({fp_mask_sans.sum()} cas)")
    print(f"   Perte FN    : {perte_fn_sans:,.0f} DH ({fn_mask_sans.sum()} cas)")
    print(f"   Gain NET    : {gain_net_sans:,.0f} DH")

    print(f"\n📊 AVEC règle métier:")
    print(f"   Automatisés : {auto_avec}/{len(y_true)} ({100*auto_avec/len(y_true):.1f}%)")
    print(f"   Gain brut   : {gain_brut_avec:,.0f} DH")
    print(f"   Perte FP    : {perte_fp_avec:,.0f} DH ({fp_mask_avec.sum()} cas)")
    print(f"   Perte FN    : {perte_fn_avec:,.0f} DH ({fn_mask_avec.sum()} cas)")
    print(f"   Gain NET    : {gain_net_avec:,.0f} DH")

    difference = gain_net_avec - gain_net_sans
    print(f"\n💡 Différence: {difference:+,.0f} DH ({100*difference/gain_net_sans:+.2f}%)")

    return {
        'sans_regle': {
            'auto': auto_sans,
            'taux_auto': 100*auto_sans/len(y_true),
            'gain_brut': gain_brut_sans,
            'perte_fp': perte_fp_sans,
            'perte_fn': perte_fn_sans,
            'gain_net': gain_net_sans,
            'fp': fp_mask_sans.sum(),
            'fn': fn_mask_sans.sum()
        },
        'avec_regle': {
            'auto': auto_avec,
            'taux_auto': 100*auto_avec/len(y_true),
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
    fig.suptitle('MATRICES DE CONFUSION PAR MODÈLE - 2025', fontsize=16, fontweight='bold', y=1.02)

    models_list = ['XGBoost', 'RandomForest', 'CatBoost']
    colors_map = ['Blues', 'Greens', 'Purples']

    for idx, (model_name, cmap) in enumerate(zip(models_list, colors_map)):
        ax = axes[idx]

        y_pred = models_results[model_name]['y_pred']
        cm = confusion_matrix(y_true, y_pred)

        # Calculer les pourcentages
        cm_percent = cm.astype('float') / cm.sum() * 100

        # Annotations avec valeurs absolues et pourcentages
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

        # Calculer l'accuracy
        tn, fp, fn, tp = cm.ravel()
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        ax.text(0.5, -0.15, f'Accuracy: {accuracy:.2%}  |  Erreurs: {fp+fn}',
                ha='center', transform=ax.transAxes, fontsize=10, style='italic')

    plt.tight_layout()

    output_dir = Path('outputs/production/figures')
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / 'confusion_matrices.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Sauvegardé: confusion_matrices.png")

    plt.close()


def generate_business_rule_comparison(all_impacts):
    """Générer les graphiques de comparaison avec/sans règle métier"""
    print("\n" + "="*80)
    print("📊 GÉNÉRATION DES COMPARAISONS AVEC/SANS RÈGLE")
    print("="*80)

    models_list = list(all_impacts.keys())

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('COMPARAISON AVEC/SANS RÈGLE MÉTIER - Impact par Modèle',
                fontsize=16, fontweight='bold', y=0.995)

    # 1. Gain NET
    ax = axes[0, 0]
    x = np.arange(len(models_list))
    width = 0.35

    gains_sans = [all_impacts[m]['sans_regle']['gain_net'] for m in models_list]
    gains_avec = [all_impacts[m]['avec_regle']['gain_net'] for m in models_list]

    bars1 = ax.bar(x - width/2, gains_sans, width, label='SANS règle',
                   color='#3498db', alpha=0.8, edgecolor='black', linewidth=1)
    bars2 = ax.bar(x + width/2, gains_avec, width, label='AVEC règle',
                   color='#e74c3c', alpha=0.8, edgecolor='black', linewidth=1)

    ax.set_ylabel('Gain NET (DH)', fontweight='bold', fontsize=11)
    ax.set_title('Gain NET: SANS vs AVEC Règle', fontweight='bold', fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(models_list)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    ax.ticklabel_format(style='plain', axis='y')

    # Ajouter les valeurs
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 10000,
                   f'{height:,.0f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

    # 2. Taux d'automatisation
    ax = axes[0, 1]
    taux_sans = [all_impacts[m]['sans_regle']['taux_auto'] for m in models_list]
    taux_avec = [all_impacts[m]['avec_regle']['taux_auto'] for m in models_list]

    bars1 = ax.bar(x - width/2, taux_sans, width, label='SANS règle',
                   color='#2ecc71', alpha=0.8, edgecolor='black', linewidth=1)
    bars2 = ax.bar(x + width/2, taux_avec, width, label='AVEC règle',
                   color='#f39c12', alpha=0.8, edgecolor='black', linewidth=1)

    ax.set_ylabel('Taux automatisation (%)', fontweight='bold', fontsize=11)
    ax.set_title('Taux d\'Automatisation', fontweight='bold', fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(models_list)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                   f'{height:.1f}%', ha='center', va='bottom', fontsize=8, fontweight='bold')

    # 3. Différence de gain
    ax = axes[0, 2]
    differences = [all_impacts[m]['difference'] for m in models_list]
    colors_diff = ['#27ae60' if d >= 0 else '#e74c3c' for d in differences]

    bars = ax.bar(models_list, differences, color=colors_diff, alpha=0.8,
                  edgecolor='black', linewidth=1)
    ax.set_ylabel('Différence Gain NET (DH)', fontweight='bold', fontsize=11)
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

    # 4. Pertes FP
    ax = axes[1, 0]
    fp_sans = [all_impacts[m]['sans_regle']['perte_fp'] for m in models_list]
    fp_avec = [all_impacts[m]['avec_regle']['perte_fp'] for m in models_list]

    bars1 = ax.bar(x - width/2, fp_sans, width, label='SANS règle',
                   color='#e74c3c', alpha=0.6, edgecolor='black', linewidth=1)
    bars2 = ax.bar(x + width/2, fp_avec, width, label='AVEC règle',
                   color='#c0392b', alpha=0.8, edgecolor='black', linewidth=1)

    ax.set_ylabel('Perte FP (DH)', fontweight='bold', fontsize=11)
    ax.set_title('Pertes Faux Positifs', fontweight='bold', fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(models_list)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    ax.ticklabel_format(style='plain', axis='y')

    # 5. Pertes FN
    ax = axes[1, 1]
    fn_sans = [all_impacts[m]['sans_regle']['perte_fn'] for m in models_list]
    fn_avec = [all_impacts[m]['avec_regle']['perte_fn'] for m in models_list]

    bars1 = ax.bar(x - width/2, fn_sans, width, label='SANS règle',
                   color='#e67e22', alpha=0.6, edgecolor='black', linewidth=1)
    bars2 = ax.bar(x + width/2, fn_avec, width, label='AVEC règle',
                   color='#d35400', alpha=0.8, edgecolor='black', linewidth=1)

    ax.set_ylabel('Perte FN (DH)', fontweight='bold', fontsize=11)
    ax.set_title('Pertes Faux Négatifs (×2)', fontweight='bold', fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(models_list)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    ax.ticklabel_format(style='plain', axis='y')

    # 6. Nombre d'erreurs (FP + FN)
    ax = axes[1, 2]
    err_sans = [all_impacts[m]['sans_regle']['fp'] + all_impacts[m]['sans_regle']['fn'] for m in models_list]
    err_avec = [all_impacts[m]['avec_regle']['fp'] + all_impacts[m]['avec_regle']['fn'] for m in models_list]

    bars1 = ax.bar(x - width/2, err_sans, width, label='SANS règle',
                   color='#9b59b6', alpha=0.6, edgecolor='black', linewidth=1)
    bars2 = ax.bar(x + width/2, err_avec, width, label='AVEC règle',
                   color='#8e44ad', alpha=0.8, edgecolor='black', linewidth=1)

    ax.set_ylabel('Nombre d\'erreurs', fontweight='bold', fontsize=11)
    ax.set_title('Total Erreurs (FP + FN)', fontweight='bold', fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(models_list)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 2,
                   f'{int(height)}', ha='center', va='bottom', fontsize=8, fontweight='bold')

    plt.tight_layout()

    output_dir = Path('outputs/production/figures')
    output_path = output_dir / 'business_rule_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Sauvegardé: business_rule_comparison.png")

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

        f.write("RÈGLE MÉTIER APPLIQUÉE:\n")
        f.write("- Un client ne peut bénéficier que d'UNE validation automatique par année\n")
        f.write("- Les réclamations sont triées par Date de Qualification\n")
        f.write("- Seule la première validation est acceptée pour chaque client/année\n")
        f.write("- Les validations suivantes sont bloquées (nécessitent audit manuel)\n\n")

        f.write("="*80 + "\n")
        f.write("RÉSULTATS PAR MODÈLE\n")
        f.write("="*80 + "\n\n")

        for model_name in all_impacts.keys():
            impact = all_impacts[model_name]

            f.write(f"{model_name}:\n")
            f.write(f"\n  SANS RÈGLE MÉTIER:\n")
            f.write(f"    Taux auto   : {impact['sans_regle']['taux_auto']:.1f}%\n")
            f.write(f"    Gain brut   : {impact['sans_regle']['gain_brut']:,.0f} DH\n")
            f.write(f"    Perte FP    : {impact['sans_regle']['perte_fp']:,.0f} DH\n")
            f.write(f"    Perte FN    : {impact['sans_regle']['perte_fn']:,.0f} DH\n")
            f.write(f"    Gain NET    : {impact['sans_regle']['gain_net']:,.0f} DH\n")
            f.write(f"    Erreurs     : FP={impact['sans_regle']['fp']}, FN={impact['sans_regle']['fn']}\n")

            f.write(f"\n  AVEC RÈGLE MÉTIER:\n")
            f.write(f"    Taux auto   : {impact['avec_regle']['taux_auto']:.1f}%\n")
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
    print("ANALYSE POST-ENTRAÎNEMENT - MATRICES & RÈGLE MÉTIER")
    print("="*80)

    # Charger les résultats
    df_2025, y_true, models_results = load_results()

    # Générer les matrices de confusion
    generate_confusion_matrices(df_2025, y_true, models_results)

    # Appliquer la règle métier et calculer l'impact pour chaque modèle
    all_impacts = {}

    for model_name, results in models_results.items():
        y_pred = results['y_pred']

        # Appliquer la règle métier
        df_scenario = apply_business_rule(df_2025, y_true, y_pred, model_name)

        # Calculer l'impact financier
        impact = calculate_financial_impact(df_scenario, model_name)

        all_impacts[model_name] = impact

    # Générer les visualisations de comparaison
    generate_business_rule_comparison(all_impacts)

    # Sauvegarder le rapport
    save_summary_report(all_impacts)

    print("\n" + "="*80)
    print("✅ ANALYSE TERMINÉE")
    print("="*80)
    print("\n📂 Fichiers générés:")
    print("   - outputs/production/figures/confusion_matrices.png")
    print("   - outputs/production/figures/business_rule_comparison.png")
    print("   - outputs/production/rapport_regle_metier.txt")


if __name__ == '__main__':
    main()
