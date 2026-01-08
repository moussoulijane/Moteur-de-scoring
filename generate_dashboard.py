#!/usr/bin/env python3
"""
Feature Importance + Résumé par Famille
========================================
- Feature importance du modèle XGBoost
- Résumé en nombre par Famille Produit

Usage:
    python analyse_model.py --data fichier.xlsx --output resultats.xlsx
"""

import pandas as pd
import numpy as np
import argparse
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')


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
    
    feature_names = []
    for col in X.columns:
        if pd.api.types.is_datetime64_any_dtype(X[col]) or pd.api.types.is_timedelta64_dtype(X[col]):
            X = X.drop(columns=[col])
        elif X[col].dtype == 'bool':
            X[col] = X[col].astype(int)
            feature_names.append(col)
        elif X[col].dtype == 'object':
            X[col] = LabelEncoder().fit_transform(X[col].astype(str).fillna('NA'))
            feature_names.append(col)
        elif pd.api.types.is_numeric_dtype(X[col]):
            X[col] = X[col].fillna(0)
            feature_names.append(col)
    
    X = X.select_dtypes(include=[np.number]).fillna(0)
    return X, y, list(X.columns)


def detect_family_column(df):
    """Détecte la colonne Famille Produit."""
    for col in df.columns:
        col_lower = col.lower()
        if 'famille' in col_lower or 'family' in col_lower:
            return col
        if 'produit' in col_lower and df[col].dtype == 'object':
            return col
    # Fallback: première colonne catégorielle avec peu de valeurs uniques
    for col in df.columns:
        if df[col].dtype == 'object' and 2 <= df[col].nunique() <= 50:
            return col
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', '-d', required=True)
    parser.add_argument('--output', '-o', default='analyse_model.xlsx')
    args = parser.parse_args()
    
    # Charger
    print(f"📂 Chargement: {args.data}")
    df = pd.read_excel(args.data)
    print(f"   {len(df)} lignes")
    
    # Détecter colonnes
    target_col = [c for c in df.columns if 'fond' in c.lower()][0]
    family_col = detect_family_column(df)
    print(f"   Cible: {target_col}")
    print(f"   Famille: {family_col}")
    
    # Préparer
    X, y, feature_names = prepare_data(df, target_col)
    
    # Split
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, df.index, test_size=0.2, stratify=y, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.15, stratify=y_train, random_state=42
    )
    
    # XGBoost
    print("\n🚀 Entraînement XGBoost...")
    import xgboost as xgb
    model = xgb.XGBClassifier(
        max_depth=6, learning_rate=0.05, n_estimators=300,
        random_state=42, verbosity=0
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    
    # Prédictions sur test
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # ==================== FEATURE IMPORTANCE ====================
    print("\n📊 Feature Importance...")
    df_importance = pd.DataFrame({
        'Feature': feature_names,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False).reset_index(drop=True)
    df_importance['Rang'] = range(1, len(df_importance) + 1)
    df_importance = df_importance[['Rang', 'Feature', 'Importance']]
    
    print(f"   Top 10:")
    for _, row in df_importance.head(10).iterrows():
        print(f"   {row['Rang']:2}. {row['Feature'][:30]:30} {row['Importance']:.4f}")
    
    # ==================== RÉSUMÉ PAR FAMILLE ====================
    print(f"\n📂 Résumé par {family_col}...")
    
    # Reconstruire df_test avec prédictions
    df_test = df.iloc[idx_test].copy()
    df_test['_pred'] = y_pred
    df_test['_proba'] = y_proba
    df_test['_reel'] = y_test.values
    
    # Seuils pour automatisation
    seuil_rejet = 0.30
    seuil_validation = 0.70
    
    results = []
    for famille in df_test[family_col].unique():
        mask = df_test[family_col] == famille
        sub = df_test[mask]
        
        n_total = len(sub)
        
        # Automatisés
        n_auto_rejet = ((sub['_proba'] <= seuil_rejet)).sum()
        n_auto_validation = ((sub['_proba'] >= seuil_validation)).sum()
        n_auto_total = n_auto_rejet + n_auto_validation
        n_manuel = n_total - n_auto_total
        
        # Succès (bonnes prédictions)
        n_vrai_negatif = ((sub['_pred'] == 0) & (sub['_reel'] == 0)).sum()  # Bon rejet
        n_vrai_positif = ((sub['_pred'] == 1) & (sub['_reel'] == 1)).sum()  # Bonne validation
        
        # Erreurs
        n_faux_positif = ((sub['_pred'] == 1) & (sub['_reel'] == 0)).sum()  # Fausse validation
        n_faux_negatif = ((sub['_pred'] == 0) & (sub['_reel'] == 1)).sum()  # Faux rejet
        
        results.append({
            'Famille': famille,
            'N_Total': n_total,
            'N_Auto_Rejet': n_auto_rejet,
            'N_Auto_Validation': n_auto_validation,
            'N_Automatise': n_auto_total,
            'N_Manuel': n_manuel,
            'Taux_Auto_%': round(n_auto_total / n_total * 100, 1) if n_total > 0 else 0,
            'N_Vrai_Negatif': n_vrai_negatif,
            'N_Vrai_Positif': n_vrai_positif,
            'N_Faux_Positif': n_faux_positif,
            'N_Faux_Negatif': n_faux_negatif,
            'N_Erreurs': n_faux_positif + n_faux_negatif,
            'Taux_Erreur_%': round((n_faux_positif + n_faux_negatif) / n_total * 100, 1) if n_total > 0 else 0
        })
    
    df_resume = pd.DataFrame(results).sort_values('N_Total', ascending=False)
    
    # Ajouter ligne TOTAL
    total_row = {
        'Famille': 'TOTAL',
        'N_Total': df_resume['N_Total'].sum(),
        'N_Auto_Rejet': df_resume['N_Auto_Rejet'].sum(),
        'N_Auto_Validation': df_resume['N_Auto_Validation'].sum(),
        'N_Automatise': df_resume['N_Automatise'].sum(),
        'N_Manuel': df_resume['N_Manuel'].sum(),
        'Taux_Auto_%': round(df_resume['N_Automatise'].sum() / df_resume['N_Total'].sum() * 100, 1),
        'N_Vrai_Negatif': df_resume['N_Vrai_Negatif'].sum(),
        'N_Vrai_Positif': df_resume['N_Vrai_Positif'].sum(),
        'N_Faux_Positif': df_resume['N_Faux_Positif'].sum(),
        'N_Faux_Negatif': df_resume['N_Faux_Negatif'].sum(),
        'N_Erreurs': df_resume['N_Erreurs'].sum(),
        'Taux_Erreur_%': round(df_resume['N_Erreurs'].sum() / df_resume['N_Total'].sum() * 100, 1)
    }
    df_resume = pd.concat([df_resume, pd.DataFrame([total_row])], ignore_index=True)
    
    print(f"\n   Résumé:")
    print(df_resume.to_string(index=False))
    
    # ==================== EXPORT EXCEL ====================
    print(f"\n💾 Export: {args.output}")
    with pd.ExcelWriter(args.output, engine='openpyxl') as writer:
        df_importance.to_excel(writer, sheet_name='Feature_Importance', index=False)
        df_resume.to_excel(writer, sheet_name='Resume_par_Famille', index=False)
    
    print("\n✅ Terminé")


if __name__ == "__main__":
    main()
