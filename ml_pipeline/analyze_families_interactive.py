"""
ANALYSE INTERACTIVE PAR FAMILLE - CatBoost
Génère des visualisations interactives Plotly pour explorer les performances par famille
Usage: python analyze_families_interactive.py
"""
import sys
sys.path.append('src')

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
import warnings
warnings.filterwarnings('ignore')

# Configuration
PRIX_UNITAIRE_DH = 169
OUTPUT_DIR = Path('outputs/production/interactive_families')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_data():
    """Charger les données et prédictions CatBoost"""
    print("\n" + "="*80)
    print("📂 CHARGEMENT DES DONNÉES")
    print("="*80)

    df_2025 = pd.read_excel('data/raw/reclamations_2025.xlsx')
    print(f"✅ Données 2025: {len(df_2025)} réclamations")

    predictions_path = Path('outputs/production/predictions/predictions_2025.pkl')

    if not predictions_path.exists():
        print("❌ Fichier de prédictions non trouvé!")
        print("   Exécutez d'abord: python model_comparison.py")
        return None, None, None

    predictions_data = joblib.load(predictions_path)

    if 'CatBoost' not in predictions_data:
        print("❌ CatBoost non trouvé dans les prédictions!")
        return None, None, None

    y_true = predictions_data['y_true']
    catboost_data = predictions_data['CatBoost']

    print(f"✅ Prédictions CatBoost chargées")

    return df_2025, y_true, catboost_data


def create_3zone_predictions(y_prob, threshold_low, threshold_high):
    """Créer les prédictions avec 3 zones"""
    y_pred = np.full(len(y_prob), -1, dtype=int)
    y_pred[y_prob <= threshold_low] = 0
    y_pred[y_prob >= threshold_high] = 1
    return y_pred


def analyze_families(df, y_true, y_prob, threshold_low, threshold_high):
    """Analyser toutes les familles en détail"""
    print("\n📊 Analyse détaillée par famille...")

    df_analysis = df.copy()
    df_analysis['y_true'] = y_true
    df_analysis['y_prob'] = y_prob
    df_analysis['y_pred'] = create_3zone_predictions(y_prob, threshold_low, threshold_high)

    # Nettoyer les montants
    df_analysis['Montant'] = pd.to_numeric(df_analysis['Montant demandé'], errors='coerce').fillna(0)
    df_analysis['Montant'] = np.clip(df_analysis['Montant'], 0, np.percentile(df_analysis['Montant'], 99))

    families = df_analysis['Famille Produit'].unique()
    results = []

    for family in families:
        mask_family = df_analysis['Famille Produit'] == family
        df_family = df_analysis[mask_family]

        if len(df_family) == 0:
            continue

        y_pred_fam = df_family['y_pred'].values
        y_true_fam = df_family['y_true'].values
        montants_fam = df_family['Montant'].values

        # Cas automatisés
        mask_auto = (y_pred_fam != -1)
        n_auto = mask_auto.sum()

        if n_auto == 0:
            continue

        y_true_auto = y_true_fam[mask_auto]
        y_pred_auto = y_pred_fam[mask_auto]
        montants_auto = montants_fam[mask_auto]

        # Métriques
        accuracy = accuracy_score(y_true_auto, y_pred_auto)
        precision = precision_score(y_true_auto, y_pred_auto, zero_division=0)
        recall = recall_score(y_true_auto, y_pred_auto, zero_division=0)

        # Confusion matrix
        cm = confusion_matrix(y_true_auto, y_pred_auto)
        tn, fp, fn, tp = cm.ravel()

        # Calculs financiers
        fp_mask = (y_true_auto == 0) & (y_pred_auto == 1)
        fn_mask = (y_true_auto == 1) & (y_pred_auto == 0)

        perte_fp = montants_auto[fp_mask].sum()
        perte_fn = 2 * montants_auto[fn_mask].sum()
        perte_totale = perte_fp + perte_fn

        gain_brut = (tp + tn) * PRIX_UNITAIRE_DH
        gain_net = gain_brut - perte_totale

        results.append({
            'Famille': str(family)[:60],
            'Volume_Total': len(df_family),
            'Volume_Auto': n_auto,
            'Volume_Audit': len(df_family) - n_auto,
            'Taux_Auto': 100 * n_auto / len(df_family),
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1_Score': 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0,
            'TP': int(tp),
            'TN': int(tn),
            'FP': int(fp),
            'FN': int(fn),
            'Erreurs_Total': int(fp + fn),
            'Taux_Erreur': 100 * (fp + fn) / n_auto if n_auto > 0 else 0,
            'Perte_FP': perte_fp,
            'Perte_FN': perte_fn,
            'Perte_Totale': perte_totale,
            'Gain_Brut': gain_brut,
            'Gain_NET': gain_net,
            'ROI': 100 * gain_net / gain_brut if gain_brut > 0 else 0
        })

    df_families = pd.DataFrame(results)
    df_families = df_families.sort_values('Volume_Total', ascending=False).reset_index(drop=True)

    print(f"✅ {len(df_families)} familles analysées")

    return df_families


def viz_1_accuracy_bar(df_families):
    """Visualisation 1: Barplot interactif d'accuracy par famille"""
    print("\n📊 Génération: Barplot accuracy par famille...")

    # Top 20 familles par volume
    df_top = df_families.head(20)

    fig = px.bar(
        df_top,
        x='Accuracy',
        y='Famille',
        orientation='h',
        title='Accuracy par Famille de Produit (Top 20 par volume)',
        labels={'Accuracy': 'Accuracy', 'Famille': 'Famille de Produit'},
        color='Accuracy',
        color_continuous_scale=['red', 'orange', 'yellow', 'lightgreen', 'green'],
        range_color=[0.90, 1.0],
        hover_data={
            'Volume_Total': ':,',
            'Volume_Auto': ':,',
            'Taux_Auto': ':.1f',
            'Erreurs_Total': True,
            'Accuracy': ':.2%'
        }
    )

    fig.update_layout(
        height=800,
        yaxis={'categoryorder': 'total ascending'},
        xaxis={'tickformat': '.0%'},
        coloraxis_colorbar={'title': 'Accuracy', 'tickformat': '.0%'}
    )

    # Lignes de référence
    fig.add_vline(x=0.95, line_dash="dash", line_color="red", opacity=0.5,
                 annotation_text="95%", annotation_position="top")
    fig.add_vline(x=0.98, line_dash="dash", line_color="green", opacity=0.5,
                 annotation_text="98%", annotation_position="top")

    output_path = OUTPUT_DIR / '01_accuracy_par_famille.html'
    fig.write_html(output_path)
    print(f"   ✅ Sauvegardé: {output_path.name}")


def viz_2_top_pertes(df_families):
    """Visualisation 2: Top 10 familles par perte financière"""
    print("\n📊 Génération: Top 10 familles en perte...")

    # Top 10 par perte totale
    df_top_pertes = df_families.nlargest(10, 'Perte_Totale')

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Top 10 Familles - Perte Totale', 'Composition des Pertes (FP vs FN)'),
        specs=[[{'type': 'bar'}, {'type': 'bar'}]]
    )

    # Graphique 1: Perte totale
    fig.add_trace(
        go.Bar(
            y=df_top_pertes['Famille'],
            x=df_top_pertes['Perte_Totale'],
            orientation='h',
            name='Perte Totale',
            marker_color='crimson',
            text=df_top_pertes['Perte_Totale'].apply(lambda x: f'{x:,.0f} DH'),
            textposition='outside',
            hovertemplate='<b>%{y}</b><br>Perte: %{x:,.0f} DH<extra></extra>'
        ),
        row=1, col=1
    )

    # Graphique 2: Composition FP vs FN
    fig.add_trace(
        go.Bar(
            y=df_top_pertes['Famille'],
            x=df_top_pertes['Perte_FP'],
            orientation='h',
            name='Perte FP',
            marker_color='orange',
            hovertemplate='<b>%{y}</b><br>Perte FP: %{x:,.0f} DH<extra></extra>'
        ),
        row=1, col=2
    )

    fig.add_trace(
        go.Bar(
            y=df_top_pertes['Famille'],
            x=df_top_pertes['Perte_FN'],
            orientation='h',
            name='Perte FN (×2)',
            marker_color='darkred',
            hovertemplate='<b>%{y}</b><br>Perte FN: %{x:,.0f} DH<extra></extra>'
        ),
        row=1, col=2
    )

    fig.update_layout(
        height=600,
        title_text="Top 10 Familles par Perte Financière",
        showlegend=True,
        barmode='stack'
    )

    fig.update_xaxes(title_text="Perte Totale (DH)", row=1, col=1)
    fig.update_xaxes(title_text="Perte (DH)", row=1, col=2)

    output_path = OUTPUT_DIR / '02_top_10_pertes.html'
    fig.write_html(output_path)
    print(f"   ✅ Sauvegardé: {output_path.name}")


def viz_3_scatter_volume_accuracy(df_families):
    """Visualisation 3: Scatter plot Volume vs Accuracy"""
    print("\n📊 Génération: Scatter Volume vs Accuracy...")

    fig = px.scatter(
        df_families,
        x='Volume_Total',
        y='Accuracy',
        size='Gain_NET',
        color='Taux_Auto',
        hover_name='Famille',
        hover_data={
            'Volume_Total': ':,',
            'Volume_Auto': ':,',
            'Accuracy': ':.2%',
            'Taux_Auto': ':.1f',
            'Erreurs_Total': True,
            'Gain_NET': ':,.0f'
        },
        title='Volume vs Accuracy par Famille (taille = Gain NET, couleur = Taux Auto)',
        labels={
            'Volume_Total': 'Volume Total',
            'Accuracy': 'Accuracy',
            'Taux_Auto': 'Taux Auto (%)',
            'Gain_NET': 'Gain NET (DH)'
        },
        color_continuous_scale='RdYlGn'
    )

    fig.update_layout(height=700)
    fig.update_yaxes(tickformat='.0%')

    # Ligne de référence accuracy 95%
    fig.add_hline(y=0.95, line_dash="dash", line_color="red", opacity=0.5,
                 annotation_text="95%", annotation_position="right")

    output_path = OUTPUT_DIR / '03_scatter_volume_accuracy.html'
    fig.write_html(output_path)
    print(f"   ✅ Sauvegardé: {output_path.name}")


def viz_4_treemap(df_families):
    """Visualisation 4: Treemap des familles"""
    print("\n📊 Génération: Treemap des familles...")

    # Ajouter une catégorie de performance
    df_families['Performance'] = df_families['Accuracy'].apply(
        lambda x: 'Excellente (≥98%)' if x >= 0.98 else 'Bonne (≥95%)' if x >= 0.95 else 'À améliorer (<95%)'
    )

    fig = px.treemap(
        df_families,
        path=['Performance', 'Famille'],
        values='Volume_Total',
        color='Accuracy',
        hover_data={
            'Volume_Total': ':,',
            'Accuracy': ':.2%',
            'Erreurs_Total': True,
            'Gain_NET': ':,.0f'
        },
        color_continuous_scale='RdYlGn',
        range_color=[0.90, 1.0],
        title='Treemap des Familles par Volume et Performance'
    )

    fig.update_layout(height=800)

    output_path = OUTPUT_DIR / '04_treemap_familles.html'
    fig.write_html(output_path)
    print(f"   ✅ Sauvegardé: {output_path.name}")


def viz_5_heatmap_metrics(df_families):
    """Visualisation 5: Heatmap des métriques par famille"""
    print("\n📊 Génération: Heatmap des métriques...")

    # Top 15 familles
    df_top = df_families.head(15)

    # Préparer les données pour heatmap
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1_Score']
    heatmap_data = df_top[metrics].T

    fig = go.Figure(data=go.Heatmap(
        z=heatmap_data.values,
        x=df_top['Famille'],
        y=metrics,
        colorscale='RdYlGn',
        text=heatmap_data.values,
        texttemplate='%{text:.2%}',
        textfont={"size": 10},
        hoverongaps=False,
        hovertemplate='Famille: %{x}<br>Métrique: %{y}<br>Valeur: %{z:.2%}<extra></extra>',
        zmin=0.90,
        zmax=1.0
    ))

    fig.update_layout(
        title='Heatmap des Métriques par Famille (Top 15)',
        xaxis_title='Famille de Produit',
        yaxis_title='Métrique',
        height=500,
        xaxis={'tickangle': -45}
    )

    output_path = OUTPUT_DIR / '05_heatmap_metrics.html'
    fig.write_html(output_path)
    print(f"   ✅ Sauvegardé: {output_path.name}")


def viz_6_erreurs_distribution(df_families):
    """Visualisation 6: Distribution des erreurs par famille"""
    print("\n📊 Génération: Distribution des erreurs...")

    # Top 15 familles par erreurs
    df_top = df_families.nlargest(15, 'Erreurs_Total')

    fig = go.Figure()

    fig.add_trace(go.Bar(
        y=df_top['Famille'],
        x=df_top['FP'],
        name='Faux Positifs (FP)',
        orientation='h',
        marker_color='orange',
        text=df_top['FP'],
        textposition='inside',
        hovertemplate='<b>%{y}</b><br>FP: %{x}<extra></extra>'
    ))

    fig.add_trace(go.Bar(
        y=df_top['Famille'],
        x=df_top['FN'],
        name='Faux Négatifs (FN)',
        orientation='h',
        marker_color='crimson',
        text=df_top['FN'],
        textposition='inside',
        hovertemplate='<b>%{y}</b><br>FN: %{x}<extra></extra>'
    ))

    fig.update_layout(
        title='Distribution des Erreurs par Famille (Top 15 en nombre d\'erreurs)',
        xaxis_title='Nombre d\'erreurs',
        yaxis_title='Famille de Produit',
        barmode='stack',
        height=600,
        yaxis={'categoryorder': 'total ascending'}
    )

    output_path = OUTPUT_DIR / '06_distribution_erreurs.html'
    fig.write_html(output_path)
    print(f"   ✅ Sauvegardé: {output_path.name}")


def viz_7_top_bottom_performance(df_families):
    """Visualisation 7: Top 10 et Bottom 10 en performance"""
    print("\n📊 Génération: Top/Bottom performance...")

    # Filtrer les familles avec volume significatif
    df_significant = df_families[df_families['Volume_Total'] >= 50]

    # Top 10 et Bottom 10
    df_top10 = df_significant.nlargest(10, 'Accuracy')
    df_bottom10 = df_significant.nsmallest(10, 'Accuracy')

    # Marquer
    df_top10 = df_top10.copy()
    df_bottom10 = df_bottom10.copy()
    df_top10['Catégorie'] = 'Top 10'
    df_bottom10['Catégorie'] = 'Bottom 10'

    df_combined = pd.concat([df_top10, df_bottom10])

    fig = px.bar(
        df_combined,
        x='Accuracy',
        y='Famille',
        color='Catégorie',
        orientation='h',
        title='Top 10 vs Bottom 10 Familles en Accuracy (volume ≥ 50)',
        labels={'Accuracy': 'Accuracy', 'Famille': 'Famille'},
        color_discrete_map={'Top 10': 'green', 'Bottom 10': 'red'},
        hover_data={
            'Volume_Total': ':,',
            'Accuracy': ':.2%',
            'Erreurs_Total': True,
            'Taux_Erreur': ':.2f'
        },
        barmode='group'
    )

    fig.update_layout(
        height=700,
        xaxis={'tickformat': '.0%'},
        yaxis={'categoryorder': 'total ascending'}
    )

    output_path = OUTPUT_DIR / '07_top_bottom_performance.html'
    fig.write_html(output_path)
    print(f"   ✅ Sauvegardé: {output_path.name}")


def viz_8_gain_net_par_famille(df_families):
    """Visualisation 8: Gain NET par famille"""
    print("\n📊 Génération: Gain NET par famille...")

    # Top 15 par volume
    df_top = df_families.head(15)

    # Créer une colonne pour la couleur (positif/négatif)
    df_top = df_top.copy()
    df_top['Gain_Color'] = df_top['Gain_NET'].apply(lambda x: 'Positif' if x >= 0 else 'Négatif')

    fig = px.bar(
        df_top,
        x='Gain_NET',
        y='Famille',
        orientation='h',
        color='Gain_Color',
        color_discrete_map={'Positif': 'green', 'Négatif': 'red'},
        title='Gain NET par Famille (Top 15 par volume)',
        labels={'Gain_NET': 'Gain NET (DH)', 'Famille': 'Famille'},
        hover_data={
            'Volume_Auto': ':,',
            'Gain_Brut': ':,.0f',
            'Perte_Totale': ':,.0f',
            'Gain_NET': ':,.0f',
            'ROI': ':.1f'
        }
    )

    fig.update_layout(
        height=600,
        yaxis={'categoryorder': 'total ascending'},
        showlegend=True
    )

    # Ligne à 0
    fig.add_vline(x=0, line_dash="solid", line_color="black", line_width=2)

    output_path = OUTPUT_DIR / '08_gain_net_famille.html'
    fig.write_html(output_path)
    print(f"   ✅ Sauvegardé: {output_path.name}")


def viz_9_table_complete(df_families):
    """Visualisation 9: Table interactive complète"""
    print("\n📊 Génération: Table interactive complète...")

    # Formater les colonnes
    df_display = df_families.copy()

    fig = go.Figure(data=[go.Table(
        columnwidth=[200, 80, 80, 80, 80, 80, 80, 80, 80, 80, 100, 100],
        header=dict(
            values=['<b>Famille</b>', '<b>Volume</b>', '<b>Auto</b>', '<b>Taux Auto</b>',
                   '<b>Accuracy</b>', '<b>Precision</b>', '<b>Recall</b>', '<b>Erreurs</b>',
                   '<b>FP</b>', '<b>FN</b>', '<b>Perte (DH)</b>', '<b>Gain NET (DH)</b>'],
            fill_color='paleturquoise',
            align='left',
            font=dict(size=12, color='black')
        ),
        cells=dict(
            values=[
                df_display['Famille'],
                df_display['Volume_Total'].apply(lambda x: f'{x:,}'),
                df_display['Volume_Auto'].apply(lambda x: f'{x:,}'),
                df_display['Taux_Auto'].apply(lambda x: f'{x:.1f}%'),
                df_display['Accuracy'].apply(lambda x: f'{x:.2%}'),
                df_display['Precision'].apply(lambda x: f'{x:.2%}'),
                df_display['Recall'].apply(lambda x: f'{x:.2%}'),
                df_display['Erreurs_Total'],
                df_display['FP'],
                df_display['FN'],
                df_display['Perte_Totale'].apply(lambda x: f'{x:,.0f}'),
                df_display['Gain_NET'].apply(lambda x: f'{x:,.0f}')
            ],
            fill_color='lavender',
            align='left',
            font=dict(size=11)
        )
    )])

    fig.update_layout(
        title='Table Complète - Toutes les Familles',
        height=800
    )

    output_path = OUTPUT_DIR / '09_table_complete.html'
    fig.write_html(output_path)
    print(f"   ✅ Sauvegardé: {output_path.name}")


def generate_summary_csv(df_families):
    """Sauvegarder un CSV récapitulatif"""
    print("\n📝 Génération: Fichier CSV récapitulatif...")

    output_path = OUTPUT_DIR / 'families_summary.csv'
    df_families.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"   ✅ Sauvegardé: {output_path.name}")


def main():
    """Fonction principale"""
    print("\n" + "="*80)
    print("ANALYSE INTERACTIVE PAR FAMILLE - CatBoost")
    print("="*80)

    # Charger les données
    df, y_true, catboost_data = load_data()

    if df is None:
        return

    y_prob = catboost_data['y_prob']
    threshold_low = catboost_data['threshold_low']
    threshold_high = catboost_data['threshold_high']

    # Analyser toutes les familles
    df_families = analyze_families(df, y_true, y_prob, threshold_low, threshold_high)

    print(f"\n📁 Dossier de sortie: {OUTPUT_DIR}")
    print("="*80)

    # Générer toutes les visualisations
    viz_1_accuracy_bar(df_families)
    viz_2_top_pertes(df_families)
    viz_3_scatter_volume_accuracy(df_families)
    viz_4_treemap(df_families)
    viz_5_heatmap_metrics(df_families)
    viz_6_erreurs_distribution(df_families)
    viz_7_top_bottom_performance(df_families)
    viz_8_gain_net_par_famille(df_families)
    viz_9_table_complete(df_families)

    # Sauvegarder CSV
    generate_summary_csv(df_families)

    print("\n" + "="*80)
    print("✅ GÉNÉRATION TERMINÉE")
    print("="*80)
    print(f"\n📂 Tous les fichiers ont été sauvegardés dans: {OUTPUT_DIR}/")
    print("\n📊 Visualisations interactives générées (HTML):")
    print("   1. 01_accuracy_par_famille.html - Barplot accuracy (Top 20)")
    print("   2. 02_top_10_pertes.html - Top 10 familles en perte")
    print("   3. 03_scatter_volume_accuracy.html - Scatter Volume vs Accuracy")
    print("   4. 04_treemap_familles.html - Treemap par volume et performance")
    print("   5. 05_heatmap_metrics.html - Heatmap des métriques (Top 15)")
    print("   6. 06_distribution_erreurs.html - Distribution FP/FN par famille")
    print("   7. 07_top_bottom_performance.html - Top 10 vs Bottom 10")
    print("   8. 08_gain_net_famille.html - Gain NET par famille")
    print("   9. 09_table_complete.html - Table interactive complète")
    print("  10. families_summary.csv - Export CSV de toutes les familles")
    print("\n💡 Ouvrez les fichiers HTML dans votre navigateur pour explorer!")
    print("💡 Survolez les graphiques avec la souris pour voir les détails!")


if __name__ == '__main__':
    main()
