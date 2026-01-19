"""
SCRIPT D'INFÉRENCE - CatBoost
Prédit les décisions (Validation Auto / Rejet Auto / Audit Humain) sur une nouvelle base de données
Usage: python inference.py --input_file chemin/vers/nouvelle_base.xlsx
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

# Import du preprocessing
from data_preprocessing import DataPreprocessor


def load_model_and_preprocessor():
    """Charger le modèle CatBoost et le preprocessor"""
    print("\n" + "="*80)
    print("📂 CHARGEMENT DU MODÈLE ET PREPROCESSOR")
    print("="*80)

    # Chemins
    model_path = Path('outputs/production/models/catboost_model.pkl')
    preprocessor_path = Path('outputs/production/models/preprocessor.pkl')
    predictions_path = Path('outputs/production/predictions/predictions_2025.pkl')

    # Vérifier que les fichiers existent
    if not model_path.exists():
        print(f"❌ Modèle non trouvé: {model_path}")
        print("   Exécutez d'abord: python model_comparison.py")
        return None, None, None, None

    if not preprocessor_path.exists():
        print(f"❌ Preprocessor non trouvé: {preprocessor_path}")
        print("   Exécutez d'abord: python model_comparison.py")
        return None, None, None, None

    # Charger le modèle
    model = joblib.load(model_path)
    print(f"✅ Modèle CatBoost chargé depuis: {model_path}")

    # Charger le preprocessor
    preprocessor = joblib.load(preprocessor_path)
    print(f"✅ Preprocessor chargé depuis: {preprocessor_path}")

    # Charger les seuils depuis les prédictions
    if predictions_path.exists():
        predictions_data = joblib.load(predictions_path)
        if 'CatBoost' in predictions_data:
            threshold_low = predictions_data['CatBoost']['threshold_low']
            threshold_high = predictions_data['CatBoost']['threshold_high']
            print(f"✅ Seuils chargés: {threshold_low:.4f} / {threshold_high:.4f}")
        else:
            print("⚠️  Seuils non trouvés, utilisation de valeurs par défaut")
            threshold_low = 0.3
            threshold_high = 0.7
    else:
        print("⚠️  Fichier de prédictions non trouvé, utilisation de seuils par défaut")
        threshold_low = 0.3
        threshold_high = 0.7

    return model, preprocessor, threshold_low, threshold_high


def load_new_data(file_path):
    """Charger la nouvelle base de données"""
    print("\n" + "="*80)
    print("📂 CHARGEMENT DES NOUVELLES DONNÉES")
    print("="*80)

    if not Path(file_path).exists():
        print(f"❌ Fichier non trouvé: {file_path}")
        return None

    # Charger les données
    df = pd.read_excel(file_path)
    print(f"✅ Données chargées: {len(df)} réclamations")
    print(f"   Colonnes: {len(df.columns)}")

    return df


def verify_required_columns(df, preprocessor):
    """Vérifier que toutes les colonnes nécessaires sont présentes"""
    print("\n" + "="*80)
    print("🔍 VÉRIFICATION DES COLONNES")
    print("="*80)

    # Colonnes de base nécessaires
    required_base_cols = [
        'Date de Qualification',
        'Montant demandé',
        'Famille Produit'
    ]

    # Vérifier les colonnes de base
    missing_cols = []
    for col in required_base_cols:
        if col not in df.columns:
            missing_cols.append(col)

    if missing_cols:
        print(f"❌ Colonnes manquantes: {missing_cols}")
        print("\nColonnes disponibles:")
        for col in df.columns:
            print(f"  - {col}")
        return False

    print(f"✅ Toutes les colonnes de base sont présentes")

    # Afficher les colonnes disponibles
    print(f"\n📋 Colonnes détectées ({len(df.columns)}):")
    for col in df.columns:
        print(f"  - {col}")

    return True


def preprocess_data(df, preprocessor):
    """Appliquer le preprocessing sur les nouvelles données"""
    print("\n" + "="*80)
    print("⚙️  PREPROCESSING DES DONNÉES")
    print("="*80)

    try:
        # Le preprocessor applique automatiquement toutes les transformations
        X_processed = preprocessor.transform(df)
        print(f"✅ Preprocessing réussi")
        print(f"   Shape après preprocessing: {X_processed.shape}")

        return X_processed

    except Exception as e:
        print(f"❌ Erreur lors du preprocessing: {str(e)}")
        return None


def make_predictions(model, X_processed, threshold_low, threshold_high):
    """Faire les prédictions avec le modèle"""
    print("\n" + "="*80)
    print("🤖 PRÉDICTION DU MODÈLE")
    print("="*80)

    # Prédire les probabilités
    y_prob = model.predict_proba(X_processed)[:, 1]
    print(f"✅ Prédictions calculées pour {len(y_prob)} réclamations")

    # Appliquer les seuils pour obtenir les 3 décisions
    decisions = []
    decisions_code = []

    for prob in y_prob:
        if prob <= threshold_low:
            decisions.append('Rejet Auto')
            decisions_code.append(0)
        elif prob >= threshold_high:
            decisions.append('Validation Auto')
            decisions_code.append(1)
        else:
            decisions.append('Audit Humain')
            decisions_code.append(-1)

    # Statistiques
    n_rejet = decisions_code.count(0)
    n_audit = decisions_code.count(-1)
    n_validation = decisions_code.count(1)
    total = len(decisions_code)

    print(f"\n📊 Répartition des décisions:")
    print(f"   Rejet Auto       : {n_rejet:6d} ({100*n_rejet/total:5.1f}%)")
    print(f"   Audit Humain     : {n_audit:6d} ({100*n_audit/total:5.1f}%)")
    print(f"   Validation Auto  : {n_validation:6d} ({100*n_validation/total:5.1f}%)")
    print(f"   TOTAL            : {total:6d}")

    print(f"\n📈 Statistiques des probabilités:")
    print(f"   Min  : {y_prob.min():.4f}")
    print(f"   Max  : {y_prob.max():.4f}")
    print(f"   Mean : {y_prob.mean():.4f}")
    print(f"   Median: {np.median(y_prob):.4f}")

    return y_prob, decisions, decisions_code


def apply_business_rule(df_results):
    """Appliquer la règle métier: 1 validation auto par client par année"""
    print("\n" + "="*80)
    print("🔒 APPLICATION DE LA RÈGLE MÉTIER")
    print("="*80)
    print("Règle: 1 validation automatique par client par année")

    df_scenario = df_results.copy()

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
        print("⚠️  Aucune colonne client trouvée, règle métier non appliquée")
        df_scenario['Décision_Finale'] = df_scenario['Décision_Modèle']
        return df_scenario

    print(f"✅ Colonne client identifiée: {client_col}")

    # Trier par client, année, puis date
    df_scenario = df_scenario.sort_values([client_col, 'Annee', 'Date de Qualification'])

    # Marquer les validations automatiques
    df_scenario['is_validation_auto'] = (df_scenario['Décision_Code'] == 1)

    # Compter les validations auto par client/année
    df_scenario['validation_rank'] = df_scenario.groupby([client_col, 'Annee'])['is_validation_auto'].cumsum()

    # Appliquer la règle: seule la première validation auto est acceptée
    df_scenario['Décision_Finale'] = df_scenario['Décision_Modèle'].copy()
    mask_blocked = (df_scenario['is_validation_auto']) & (df_scenario['validation_rank'] > 1)
    df_scenario.loc[mask_blocked, 'Décision_Finale'] = 'Audit Humain (Règle)'

    # Statistiques
    n_blocked = mask_blocked.sum()
    if n_blocked > 0:
        print(f"\n📊 Impact de la règle métier:")
        print(f"   Validations bloquées : {n_blocked}")
        print(f"   Ces réclamations sont maintenant en 'Audit Humain (Règle)'")
    else:
        print(f"\n✅ Aucune validation bloquée par la règle métier")

    return df_scenario


def save_results(df_results, output_path):
    """Sauvegarder les résultats dans un fichier Excel"""
    print("\n" + "="*80)
    print("💾 SAUVEGARDE DES RÉSULTATS")
    print("="*80)

    # Créer le dossier de sortie si nécessaire
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Colonnes à garder (toutes les colonnes d'origine + les nouvelles)
    # Supprimer les colonnes temporaires
    cols_to_drop = ['is_validation_auto', 'validation_rank', 'Annee', 'Décision_Code']
    df_final = df_results.drop(columns=[col for col in cols_to_drop if col in df_results.columns])

    # Réorganiser les colonnes pour mettre les décisions en premier
    decision_cols = ['Probabilité_Fondée', 'Décision_Modèle', 'Décision_Finale']
    other_cols = [col for col in df_final.columns if col not in decision_cols]
    df_final = df_final[decision_cols + other_cols]

    # Sauvegarder
    df_final.to_excel(output_path, index=False, engine='openpyxl')
    print(f"✅ Résultats sauvegardés: {output_path}")
    print(f"   Nombre de lignes: {len(df_final)}")
    print(f"   Nombre de colonnes: {len(df_final.columns)}")

    # Afficher les premières colonnes
    print(f"\n📋 Colonnes dans le fichier de sortie:")
    for i, col in enumerate(df_final.columns, 1):
        print(f"   {i:2d}. {col}")

    return df_final


def generate_summary_report(df_results, output_dir):
    """Générer un rapport récapitulatif"""
    print("\n" + "="*80)
    print("📝 GÉNÉRATION DU RAPPORT RÉCAPITULATIF")
    print("="*80)

    report_path = output_dir / 'rapport_inference.txt'

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("RAPPORT D'INFÉRENCE - CatBoost\n")
        f.write("="*80 + "\n\n")

        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("="*80 + "\n")
        f.write("RÉSUMÉ DES PRÉDICTIONS\n")
        f.write("="*80 + "\n\n")

        total = len(df_results)

        # Décision Modèle
        f.write("DÉCISION MODÈLE (avant règle métier):\n")
        for decision in ['Rejet Auto', 'Audit Humain', 'Validation Auto']:
            count = (df_results['Décision_Modèle'] == decision).sum()
            pct = 100 * count / total
            f.write(f"  {decision:20s}: {count:6d} ({pct:5.1f}%)\n")

        f.write(f"\n  TOTAL: {total}\n\n")

        # Décision Finale (après règle)
        f.write("DÉCISION FINALE (après règle métier):\n")
        for decision in df_results['Décision_Finale'].unique():
            count = (df_results['Décision_Finale'] == decision).sum()
            pct = 100 * count / total
            f.write(f"  {decision:25s}: {count:6d} ({pct:5.1f}%)\n")

        f.write(f"\n  TOTAL: {total}\n\n")

        # Impact de la règle
        n_blocked = ((df_results['Décision_Modèle'] == 'Validation Auto') &
                    (df_results['Décision_Finale'] == 'Audit Humain (Règle)')).sum()

        f.write("="*80 + "\n")
        f.write("IMPACT DE LA RÈGLE MÉTIER\n")
        f.write("="*80 + "\n\n")

        f.write(f"Validations bloquées: {n_blocked}\n")
        if n_blocked > 0:
            f.write(f"Impact: {100*n_blocked/total:.2f}% des réclamations\n")

        f.write("\n")

        # Statistiques des probabilités
        f.write("="*80 + "\n")
        f.write("STATISTIQUES DES PROBABILITÉS\n")
        f.write("="*80 + "\n\n")

        probs = df_results['Probabilité_Fondée']
        f.write(f"Min       : {probs.min():.4f}\n")
        f.write(f"25%       : {probs.quantile(0.25):.4f}\n")
        f.write(f"Médiane   : {probs.median():.4f}\n")
        f.write(f"Moyenne   : {probs.mean():.4f}\n")
        f.write(f"75%       : {probs.quantile(0.75):.4f}\n")
        f.write(f"Max       : {probs.max():.4f}\n\n")

        # Distribution par famille si disponible
        if 'Famille Produit' in df_results.columns:
            f.write("="*80 + "\n")
            f.write("DISTRIBUTION PAR FAMILLE DE PRODUIT\n")
            f.write("="*80 + "\n\n")

            family_stats = df_results.groupby('Famille Produit')['Décision_Finale'].value_counts().unstack(fill_value=0)

            for family in family_stats.index[:10]:  # Top 10
                total_fam = family_stats.loc[family].sum()
                f.write(f"\n{family[:50]:50s} (n={total_fam})\n")
                for decision in family_stats.columns:
                    count = family_stats.loc[family, decision]
                    pct = 100 * count / total_fam if total_fam > 0 else 0
                    f.write(f"  {decision:25s}: {count:5d} ({pct:5.1f}%)\n")

        f.write("\n" + "="*80 + "\n")
        f.write("FIN DU RAPPORT\n")
        f.write("="*80 + "\n")

    print(f"✅ Rapport sauvegardé: {report_path}")


def main():
    """Fonction principale"""
    parser = argparse.ArgumentParser(description='Inférence CatBoost sur nouvelle base de données')
    parser.add_argument('--input_file', type=str, required=True,
                       help='Chemin vers le fichier Excel d\'entrée')
    parser.add_argument('--output_file', type=str, default=None,
                       help='Chemin vers le fichier Excel de sortie (optionnel)')
    parser.add_argument('--apply_rule', action='store_true',
                       help='Appliquer la règle métier (1 validation auto par client/an)')

    args = parser.parse_args()

    print("\n" + "="*80)
    print("🚀 SCRIPT D'INFÉRENCE - CatBoost")
    print("="*80)

    # 1. Charger le modèle et preprocessor
    model, preprocessor, threshold_low, threshold_high = load_model_and_preprocessor()
    if model is None:
        return

    # 2. Charger les nouvelles données
    df = load_new_data(args.input_file)
    if df is None:
        return

    # 3. Vérifier les colonnes
    if not verify_required_columns(df, preprocessor):
        return

    # 4. Preprocessing
    X_processed = preprocess_data(df, preprocessor)
    if X_processed is None:
        return

    # 5. Faire les prédictions
    y_prob, decisions, decisions_code = make_predictions(model, X_processed, threshold_low, threshold_high)

    # 6. Ajouter les résultats au dataframe original
    df_results = df.copy()
    df_results['Probabilité_Fondée'] = y_prob
    df_results['Décision_Modèle'] = decisions
    df_results['Décision_Code'] = decisions_code

    # 7. Appliquer la règle métier si demandé
    if args.apply_rule:
        df_results = apply_business_rule(df_results)
    else:
        print("\n⚠️  Règle métier NON appliquée (utilisez --apply_rule pour l'activer)")
        df_results['Décision_Finale'] = df_results['Décision_Modèle']

    # 8. Déterminer le nom du fichier de sortie
    if args.output_file:
        output_path = Path(args.output_file)
    else:
        input_path = Path(args.input_file)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = Path('outputs/inference') / f'{input_path.stem}_predictions_{timestamp}.xlsx'

    # 9. Sauvegarder les résultats
    df_final = save_results(df_results, output_path)

    # 10. Générer le rapport
    generate_summary_report(df_results, output_path.parent)

    print("\n" + "="*80)
    print("✅ INFÉRENCE TERMINÉE")
    print("="*80)
    print(f"\n📂 Fichiers générés:")
    print(f"   - {output_path}")
    print(f"   - {output_path.parent / 'rapport_inference.txt'}")
    print("\n💡 Ouvrez le fichier Excel pour voir les prédictions!")


if __name__ == '__main__':
    main()
