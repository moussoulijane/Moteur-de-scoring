"""
SCRIPT D'INFÉRENCE V2 - Features Production-Ready
Prédit les décisions sur nouvelle base de données

Usage: python inference_v2.py --input_file chemin/vers/nouvelle_base.xlsx
"""
import sys
sys.path.append('src')

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import argparse
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import preprocessing
from preprocessor_v2 import ProductionPreprocessorV2


def load_model_and_preprocessor(model_choice='best'):
    """
    Charger le modèle V2 et le preprocessor

    Args:
        model_choice: 'best', 'xgboost', ou 'catboost'
    """
    print("\n" + "="*80)
    print("📂 CHARGEMENT DU MODÈLE V2")
    print("="*80)

    preprocessor_path = Path('outputs/production_v2/models/preprocessor_v2.pkl')
    predictions_path = Path('outputs/production_v2/predictions/predictions_2025_v2.pkl')

    # Déterminer quel modèle charger
    if model_choice == 'best':
        model_path = Path('outputs/production_v2/models/best_model_v2.pkl')
        model_name = 'Best'

        # Lire le nom du meilleur modèle
        best_info_path = Path('outputs/production_v2/models/best_model_info.txt')
        if best_info_path.exists():
            with open(best_info_path, 'r') as f:
                first_line = f.readline()
                model_name = first_line.split(': ')[1].strip()
    elif model_choice == 'xgboost':
        model_path = Path('outputs/production_v2/models/xgboost_model_v2.pkl')
        model_name = 'XGBoost'
    elif model_choice == 'catboost':
        model_path = Path('outputs/production_v2/models/catboost_model_v2.pkl')
        model_name = 'CatBoost'
    else:
        print(f"❌ Choix de modèle invalide: {model_choice}")
        print("   Options: 'best', 'xgboost', 'catboost'")
        return None, None, None, None

    if not model_path.exists():
        print(f"❌ Modèle V2 non trouvé: {model_path}")
        print("   Exécutez d'abord: python ml_pipeline_v2/model_comparison_v2.py")
        return None, None, None, None

    if not preprocessor_path.exists():
        print(f"❌ Preprocessor V2 non trouvé: {preprocessor_path}")
        print("   Exécutez d'abord: python ml_pipeline_v2/model_comparison_v2.py")
        return None, None, None, None

    # Charger modèle
    model = joblib.load(model_path)
    print(f"✅ Modèle {model_name} V2 chargé")

    # Charger preprocessor
    preprocessor = joblib.load(preprocessor_path)
    print(f"✅ Preprocessor V2 chargé")

    # Charger seuils selon le modèle choisi
    if not predictions_path.exists():
        print(f"⚠️  Fichier de prédictions non trouvé, utilisation de seuils par défaut")
        threshold_low = 0.3
        threshold_high = 0.7
    else:
        predictions_data = joblib.load(predictions_path)

        # Utiliser les seuils du meilleur modèle ou du modèle choisi
        if model_choice == 'best' and 'best_model' in predictions_data:
            best_name = predictions_data['best_model']
            threshold_low = predictions_data[best_name]['threshold_low']
            threshold_high = predictions_data[best_name]['threshold_high']
            print(f"   (Utilisation des seuils de {best_name})")
        elif model_choice == 'xgboost':
            threshold_low = predictions_data['XGBoost']['threshold_low']
            threshold_high = predictions_data['XGBoost']['threshold_high']
        else:  # catboost
            threshold_low = predictions_data['CatBoost']['threshold_low']
            threshold_high = predictions_data['CatBoost']['threshold_high']

    print(f"✅ Seuils: BAS={threshold_low:.4f}, HAUT={threshold_high:.4f}")

    # Afficher info preprocessor
    info = preprocessor.get_feature_info()
    print(f"\n📊 Informations du preprocessor:")
    print(f"   Nombre de features: {info['n_features']}")
    print(f"   Familles avec stats: {info['family_stats_count']}")
    print(f"   Catégories avec stats: {info['category_stats_count']}")

    return model, preprocessor, threshold_low, threshold_high


def verify_required_columns(df, preprocessor):
    """Vérifier que les colonnes nécessaires sont présentes"""
    print("\n" + "="*80)
    print("🔍 VÉRIFICATION DES COLONNES")
    print("="*80)

    # Colonnes de base nécessaires
    required_base_cols = [
        'Montant demandé',
        'Famille Produit'
    ]

    # Colonnes recommandées
    recommended_cols = [
        'Délai estimé',
        'Catégorie',
        'Sous-catégorie',
        'Segment',
        'Marché',
        'anciennete_annees',
        'Date de Qualification'  # Pour la règle métier
    ]

    missing_required = [col for col in required_base_cols if col not in df.columns]
    missing_recommended = [col for col in recommended_cols if col not in df.columns]

    if missing_required:
        print(f"❌ Colonnes REQUISES manquantes:")
        for col in missing_required:
            print(f"   - {col}")
        return False

    print(f"✅ Toutes les colonnes requises sont présentes")

    if missing_recommended:
        print(f"\n⚠️  Colonnes recommandées manquantes (features seront à 0):")
        for col in missing_recommended:
            print(f"   - {col}")

    print(f"\n✅ Colonnes présentes: {len([c for c in required_base_cols + recommended_cols if c in df.columns])}/{len(required_base_cols + recommended_cols)}")

    return True


def preprocess_data(df, preprocessor):
    """Préprocesser les données avec ProductionPreprocessorV2"""
    print("\n" + "="*80)
    print("⚙️ PREPROCESSING DES DONNÉES")
    print("="*80)

    try:
        # Le ProductionPreprocessorV2 gère tout le preprocessing
        X = preprocessor.transform(df)
        print(f"✅ Preprocessing effectué: shape {X.shape}")
        return X, df

    except Exception as e:
        print(f"❌ Erreur lors du preprocessing: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def make_predictions(model, X_processed, threshold_low, threshold_high):
    """Faire les prédictions avec le modèle"""
    print("\n" + "="*80)
    print("🎯 PRÉDICTIONS")
    print("="*80)

    # Probabilités
    y_prob = model.predict_proba(X_processed)[:, 1]

    # DIAGNOSTIC: Afficher la distribution des probabilités
    print(f"\n📊 Distribution des probabilités prédites:")
    print(f"   Min        : {y_prob.min():.4f}")
    print(f"   Percentile 25%: {np.percentile(y_prob, 25):.4f}")
    print(f"   Médiane    : {np.median(y_prob):.4f}")
    print(f"   Percentile 75%: {np.percentile(y_prob, 75):.4f}")
    print(f"   Max        : {y_prob.max():.4f}")
    print(f"   Moyenne    : {y_prob.mean():.4f}")
    print(f"\n🎯 Seuils utilisés:")
    print(f"   Seuil BAS  : {threshold_low:.4f}")
    print(f"   Seuil HAUT : {threshold_high:.4f}")

    # 3 zones de décision
    decisions = []
    decisions_code = []

    for prob in y_prob:
        if prob <= threshold_low:
            decisions.append('Rejet Auto')
            decisions_code.append(-1)  # -1 pour rejet
        elif prob >= threshold_high:
            decisions.append('Validation Auto')
            decisions_code.append(1)  # 1 pour fondée
        else:
            decisions.append('Audit Humain')
            decisions_code.append(0)  # 0 pour audit

    # Statistiques
    n_rejet = decisions.count('Rejet Auto')
    n_validation = decisions.count('Validation Auto')
    n_audit = decisions.count('Audit Humain')
    n_total = len(decisions)

    print(f"\n📊 Répartition des décisions:")
    print(f"   Rejet Auto       : {n_rejet:6d} ({100*n_rejet/n_total:5.1f}%)")
    print(f"   Audit Humain     : {n_audit:6d} ({100*n_audit/n_total:5.1f}%)")
    print(f"   Validation Auto  : {n_validation:6d} ({100*n_validation/n_total:5.1f}%)")
    print(f"   Total            : {n_total:6d}")

    taux_auto = 100 * (n_rejet + n_validation) / n_total
    print(f"\n✅ Taux d'automatisation: {taux_auto:.1f}%")

    return y_prob, decisions, decisions_code


def apply_business_rule(df_results):
    """
    Appliquer la règle métier: 1 validation auto par client par année

    Règle: Un client ne peut avoir qu'une seule validation automatique par an.
    Les autres validations automatiques sont converties en Audit Humain.
    """
    print("\n" + "="*80)
    print("📋 APPLICATION DE LA RÈGLE MÉTIER")
    print("="*80)

    if 'Date de Qualification' not in df_results.columns:
        print("⚠️  Colonne 'Date de Qualification' manquante, règle métier non applicable")
        return df_results

    df = df_results.copy()

    # Convertir dates
    df['Date de Qualification'] = pd.to_datetime(df['Date de Qualification'], errors='coerce')

    # Extraire année et identifiant client
    if 'Identifiant client' not in df.columns and 'ID Client' not in df.columns:
        print("⚠️  Colonne identifiant client manquante, règle métier non applicable")
        return df_results

    client_col = 'Identifiant client' if 'Identifiant client' in df.columns else 'ID Client'

    df['annee'] = df['Date de Qualification'].dt.year

    # Trier par date pour garder la première validation
    df = df.sort_values(['annee', client_col, 'Date de Qualification'])

    # Compter validations par client/année
    df['validation_number'] = df.groupby([client_col, 'annee']).cumcount() + 1

    # Appliquer la règle
    mask_to_audit = (df['Decision_Modele'] == 'Validation Auto') & (df['validation_number'] > 1)

    n_changed = mask_to_audit.sum()

    if n_changed > 0:
        df.loc[mask_to_audit, 'Decision_Modele'] = 'Audit Humain'
        df.loc[mask_to_audit, 'Decision_Code'] = 0  # 0 pour audit humain

        print(f"✅ Règle métier appliquée:")
        print(f"   {n_changed} validations converties en Audit Humain")

        # Nouvelles stats
        n_rejet = (df['Decision_Modele'] == 'Rejet Auto').sum()
        n_validation = (df['Decision_Modele'] == 'Validation Auto').sum()
        n_audit = (df['Decision_Modele'] == 'Audit Humain').sum()
        n_total = len(df)

        print(f"\n📊 Nouvelle répartition:")
        print(f"   Rejet Auto       : {n_rejet:6d} ({100*n_rejet/n_total:5.1f}%)")
        print(f"   Audit Humain     : {n_audit:6d} ({100*n_audit/n_total:5.1f}%)")
        print(f"   Validation Auto  : {n_validation:6d} ({100*n_validation/n_total:5.1f}%)")
    else:
        print("✅ Aucune modification nécessaire")

    # Supprimer colonnes temporaires
    df = df.drop(['annee', 'validation_number'], axis=1, errors='ignore')

    return df


def export_results(df_original, df_processed, y_prob, decisions, decisions_code, output_file):
    """Exporter les résultats vers Excel"""
    print("\n" + "="*80)
    print("💾 EXPORT DES RÉSULTATS")
    print("="*80)

    # Créer DataFrame de résultats
    df_results = df_original.copy()

    # Ajouter prédictions
    df_results['Probabilite_Fondee'] = y_prob
    df_results['Decision_Modele'] = decisions
    df_results['Decision_Code'] = decisions_code

    # Exporter
    df_results.to_excel(output_file, index=False, engine='openpyxl')

    print(f"✅ Résultats exportés: {output_file}")
    print(f"   Nombre de lignes: {len(df_results)}")
    print(f"   Colonnes ajoutées:")
    print(f"   - Probabilite_Fondee  : Probabilité prédite [0-1]")
    print(f"   - Decision_Modele     : Rejet Auto / Audit Humain / Validation Auto")
    print(f"   - Decision_Code       : -1 (Rejet) / 0 (Audit) / 1 (Fondée)")

    return df_results


def generate_summary_report(df_results, output_dir):
    """Générer rapport récapitulatif"""
    report_path = output_dir / f'rapport_inference_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt'

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("RAPPORT D'INFÉRENCE - MODÈLE V2\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*80 + "\n\n")

        f.write(f"Nombre de réclamations traitées: {len(df_results)}\n\n")

        f.write("RÉPARTITION DES DÉCISIONS:\n")
        f.write("-" * 80 + "\n")

        for decision in ['Rejet Auto', 'Audit Humain', 'Validation Auto']:
            count = (df_results['Decision_Modele'] == decision).sum()
            pct = 100 * count / len(df_results)
            f.write(f"  {decision:20s}: {count:6d} ({pct:5.1f}%)\n")

        taux_auto = 100 * (df_results['Decision_Modele'].isin(['Rejet Auto', 'Validation Auto'])).sum() / len(df_results)
        f.write(f"\nTaux d'automatisation: {taux_auto:.1f}%\n\n")

        # Stats par famille si disponible
        if 'Famille Produit' in df_results.columns:
            f.write("\nRÉPARTITION PAR FAMILLE DE PRODUIT:\n")
            f.write("-" * 80 + "\n")

            family_stats = df_results.groupby('Famille Produit')['Decision_Modele'].value_counts().unstack(fill_value=0)

            if 'Validation Auto' in family_stats.columns:
                family_stats = family_stats.sort_values('Validation Auto', ascending=False).head(10)

                for famille, row in family_stats.iterrows():
                    total = row.sum()
                    f.write(f"\n{famille}:\n")
                    f.write(f"  Total: {total}\n")
                    for decision in ['Rejet Auto', 'Audit Humain', 'Validation Auto']:
                        if decision in row:
                            count = row[decision]
                            pct = 100 * count / total if total > 0 else 0
                            f.write(f"  {decision}: {count} ({pct:.1f}%)\n")

    print(f"✅ Rapport généré: {report_path}")


def main():
    parser = argparse.ArgumentParser(description='Inférence avec modèle V2')
    parser.add_argument('--input_file', type=str, required=True, help='Fichier Excel avec nouvelles données')
    parser.add_argument('--output_file', type=str, help='Fichier Excel de sortie (optionnel)')
    parser.add_argument('--apply_rule', action='store_true', help='Appliquer la règle métier (1 validation/client/an)')
    parser.add_argument('--model', type=str, default='best', choices=['best', 'xgboost', 'catboost'],
                       help='Modèle à utiliser: best (meilleur basé sur gain NET), xgboost, ou catboost')

    args = parser.parse_args()

    # Charger modèle
    model, preprocessor, threshold_low, threshold_high = load_model_and_preprocessor(args.model)

    if model is None:
        return

    # Charger données
    print(f"\n📂 Chargement: {args.input_file}")
    df_original = pd.read_excel(args.input_file)
    print(f"✅ {len(df_original)} réclamations chargées")

    # Vérifier colonnes
    if not verify_required_columns(df_original, preprocessor):
        return

    # Preprocessing
    X_processed, df_processed = preprocess_data(df_original, preprocessor)

    if X_processed is None:
        return

    # Prédictions
    y_prob, decisions, decisions_code = make_predictions(model, X_processed, threshold_low, threshold_high)

    # Préparer fichier de sortie
    if args.output_file:
        output_file = Path(args.output_file)
    else:
        input_path = Path(args.input_file)
        output_file = input_path.parent / f"{input_path.stem}_predictions_v2.xlsx"

    # Exporter
    df_results = export_results(df_original, df_processed, y_prob, decisions, decisions_code, output_file)

    # Appliquer règle métier si demandé
    if args.apply_rule:
        df_results = apply_business_rule(df_results)
        # Ré-exporter
        df_results.to_excel(output_file, index=False, engine='openpyxl')
        print(f"✅ Résultats mis à jour avec règle métier: {output_file}")

    # Générer rapport
    output_dir = output_file.parent
    generate_summary_report(df_results, output_dir)

    print("\n" + "="*80)
    print("✅ INFÉRENCE TERMINÉE")
    print("="*80)
    print(f"\n📄 Fichier de sortie: {output_file}")


if __name__ == '__main__':
    main()
