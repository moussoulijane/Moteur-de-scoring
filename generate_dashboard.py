#!/usr/bin/env python3
"""
Extraction simple des Faux Positifs et Faux Négatifs
=====================================================
Extrait les lignes FP et FN dans un Excel (données brutes).

Usage:
    python extract_fp_fn.py --data fichier.xlsx --output erreurs.xlsx
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
    # Target
    y = df[target_col].copy()
    if y.dtype == 'object':
        y = y.apply(lambda x: 1 if str(x).lower() in ['oui', 'yes', '1', 'fondée', 'fondee', 'true', 'o'] else 0)
    elif y.dtype == 'bool':
        y = y.astype(int)
    y = y.fillna(0).astype(int)
    
    # Features
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', '-d', required=True)
    parser.add_argument('--output', '-o', default='faux_positifs_negatifs.xlsx')
    args = parser.parse_args()
    
    # Charger
    print(f"📂 Chargement: {args.data}")
    df = pd.read_excel(args.data)
    print(f"   {len(df)} lignes")
    
    # Détecter cible
    target_col = [c for c in df.columns if 'fond' in c.lower()][0]
    print(f"   Cible: {target_col}")
    
    # Préparer
    X, y = prepare_data(df, target_col)
    
    # Split
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, df.index, test_size=0.2, stratify=y, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.15, stratify=y_train, random_state=42
    )
    
    # XGBoost
    print("🚀 Entraînement XGBoost...")
    import xgboost as xgb
    model = xgb.XGBClassifier(
        max_depth=6, learning_rate=0.05, n_estimators=300,
        random_state=42, verbosity=0
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    
    # Prédictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # Extraire lignes originales
    df_test = df.iloc[idx_test].copy()
    df_test['Prediction'] = y_pred
    df_test['Probabilite'] = y_proba
    
    # FP: prédit 1, réel 0
    # FN: prédit 0, réel 1
    y_test_arr = y_test.values
    
    df_fp = df_test[((y_pred == 1) & (y_test_arr == 0))]
    df_fn = df_test[((y_pred == 0) & (y_test_arr == 1))]
    
    print(f"\n📊 Résultats:")
    print(f"   Faux Positifs: {len(df_fp)}")
    print(f"   Faux Négatifs: {len(df_fn)}")
    
    # Export Excel
    print(f"\n💾 Export: {args.output}")
    with pd.ExcelWriter(args.output, engine='openpyxl') as writer:
        df_fp.to_excel(writer, sheet_name='Faux_Positifs', index=False)
        df_fn.to_excel(writer, sheet_name='Faux_Negatifs', index=False)
    
    print("✅ Terminé")


if __name__ == "__main__":
    main()
