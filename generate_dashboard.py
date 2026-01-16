import pandas as pd
import numpy as np
import argparse
import logging
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, brier_score_loss, classification_report
)
import warnings
warnings.filterwarnings('ignore')

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def analyze_dataframe(df: pd.DataFrame) -> dict:
    """Analyse le DataFrame et retourne un rapport détaillé."""
    report = {
        'shape': df.shape,
        'columns': {},
        'problematic': []
    }
    
    logger.info(f"\n{'='*60}")
    logger.info("ANALYSE DU DATASET")
    logger.info(f"{'='*60}")
    logger.info(f"Dimensions: {df.shape[0]} lignes x {df.shape[1]} colonnes")
    
    for col in df.columns:
        dtype = str(df[col].dtype)
        nunique = df[col].nunique()
        null_count = df[col].isna().sum()
        
        report['columns'][col] = {
            'dtype': dtype,
            'nunique': nunique,
            'nulls': null_count
        }
        
        # Identifier les colonnes problématiques
        if 'datetime' in dtype or 'timedelta' in dtype:
            report['problematic'].append((col, 'datetime'))
        elif dtype == 'bool':
            report['problematic'].append((col, 'boolean'))
        elif dtype == 'object' and nunique > 500:
            report['problematic'].append((col, 'high_cardinality'))
    
    if report['problematic']:
        logger.info(f"\nColonnes nécessitant un traitement spécial:")
        for col, reason in report['problematic']:
            logger.info(f"  - {col}: {reason}")
    
    return report


def detect_target(df: pd.DataFrame) -> str:
    """Détecte la colonne cible."""
    # Chercher par nom
    for col in df.columns:
        if 'fond' in col.lower():
            logger.info(f"Colonne cible détectée: {col}")
            return col
    
    # Chercher une colonne binaire
    for col in df.columns:
        unique_vals = df[col].dropna().unique()
        if len(unique_vals) == 2:
            if set(str(v).lower() for v in unique_vals).issubset(
                {'0', '1', 'oui', 'non', 'o', 'n', 'true', 'false', 'yes', 'no'}
            ):
                logger.info(f"Colonne cible potentielle détectée: {col}")
                return col
    
    raise ValueError("Impossible de détecter la colonne cible automatiquement")


def detect_category_columns(df: pd.DataFrame, target_col: str) -> list:
    """Détecte les colonnes catégorielles pour l'analyse."""
    cat_cols = []
    
    for col in df.columns:
        if col == target_col:
            continue
        
        col_lower = col.lower()
        
        # Chercher par nom
        if any(x in col_lower for x in ['famille', 'categ', 'produit', 'segment', 'type', 'motif']):
            if df[col].dtype == 'object' and 2 <= df[col].nunique() <= 100:
                cat_cols.append(col)
    
    logger.info(f"Colonnes catégorielles pour analyse: {cat_cols[:3]}")
    return cat_cols[:3]


def prepare_target(df: pd.DataFrame, target_col: str) -> pd.Series:
    """Prépare la variable cible en format binaire."""
    y = df[target_col].copy()
    
    if y.dtype == 'object':
        y = y.apply(lambda x: 1 if str(x).lower() in ['oui', 'yes', '1', 'fondée', 'fondee', 'true', 'o'] else 0)
    elif y.dtype == 'bool':
        y = y.astype(int)
    
    y = y.fillna(0).astype(int)
    
    logger.info(f"Distribution cible: {y.mean():.1%} positifs (fondées)")
    return y


def prepare_features(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """Prépare les features en gérant tous les types de données problématiques."""
    
    # Colonnes à exclure
    exclude_patterns = ['id', 'date', 'dt_', '_dt', 'timestamp', 'datetime', 'num_', 'numero']
    
    X = df.copy()
    
    # Supprimer la colonne cible
    if target_col in X.columns:
        X = X.drop(columns=[target_col])
    
    # Identifier et traiter chaque colonne
    cols_to_drop = []
    label_encoders = {}
    
    for col in X.columns:
        col_lower = col.lower()
        
        # Exclure les colonnes par pattern de nom
        if any(pattern in col_lower for pattern in exclude_patterns):
            cols_to_drop.append(col)
            continue
        
        try:
            # Gérer datetime
            if pd.api.types.is_datetime64_any_dtype(X[col]):
                # Extraire features utiles
                X[f'{col}_month'] = X[col].dt.month.fillna(0).astype(int)
                X[f'{col}_dow'] = X[col].dt.dayofweek.fillna(0).astype(int)
                cols_to_drop.append(col)
                continue
            
            # Gérer timedelta
            if pd.api.types.is_timedelta64_dtype(X[col]):
                X[col] = X[col].dt.total_seconds().fillna(0)
                continue
            
            # Gérer bool
            if X[col].dtype == 'bool':
                X[col] = X[col].astype(int)
                continue
            
            # Gérer object/category
            if X[col].dtype == 'object' or str(X[col].dtype) == 'category':
                # Vérifier si c'est un booléen caché
                unique_vals = set(str(v).lower() for v in X[col].dropna().unique())
                if unique_vals.issubset({'oui', 'non', 'o', 'n', 'true', 'false', '0', '1', 'yes', 'no'}):
                    X[col] = X[col].apply(
                        lambda x: 1 if str(x).lower() in ['oui', 'o', 'true', '1', 'yes'] else 0 if pd.notna(x) else 0
                    ).astype(int)
                else:
                    # Label encoding
                    le = LabelEncoder()
                    X[col] = X[col].astype(str).fillna('_MISSING_')
                    X[col] = le.fit_transform(X[col])
                    label_encoders[col] = le
                continue
            
            # Vérifier si numérique
            if pd.api.types.is_numeric_dtype(X[col]):
                X[col] = X[col].fillna(X[col].median() if X[col].notna().any() else 0)
                continue
            
            # Essayer de convertir en numérique
            try:
                X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)
            except:
                cols_to_drop.append(col)
                
        except Exception as e:
            logger.warning(f"Erreur avec colonne '{col}': {e}")
            cols_to_drop.append(col)
    
    # Supprimer les colonnes problématiques
    X = X.drop(columns=[c for c in cols_to_drop if c in X.columns])
    
    # S'assurer que tout est numérique
    for col in X.columns:
        if not pd.api.types.is_numeric_dtype(X[col]):
            try:
                X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)
            except:
                X = X.drop(columns=[col])
    
    # Remplir les NaN restants
    X = X.fillna(0)
    
    logger.info(f"Features préparées: {X.shape[1]} colonnes")
    logger.info(f"Colonnes exclues: {len(cols_to_drop)}")
    
    return X


def train_model(X_train, y_train, X_val, y_val):
    """Entraîne un modèle XGBoost."""
    import xgboost as xgb
    
    # Calculer le ratio pour le déséquilibre des classes
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'max_depth': 6,
        'learning_rate': 0.05,
        'n_estimators': 500,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 5,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'early_stopping_rounds':100,
        'scale_pos_weight': scale_pos_weight,
        'random_state': 42,
        'verbosity': 0,
        'use_label_encoder': False
    }
    
    model = xgb.XGBClassifier(**params)
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    
    logger.info(f"   Modèle XGBoost entraîné avec {model.best_iteration} itérations")
    
    return model, 'XGBoost'


def calculate_all_metrics(y_true, y_pred, y_proba=None):
    """Calcule toutes les métriques."""
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision_class_0': precision_score(y_true, y_pred, pos_label=0, zero_division=0),
        'precision_class_1': precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        'recall_class_0': recall_score(y_true, y_pred, pos_label=0, zero_division=0),
        'recall_class_1': recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        'f1_class_0': f1_score(y_true, y_pred, pos_label=0, zero_division=0),
        'f1_class_1': f1_score(y_true, y_pred, pos_label=1, zero_division=0),
        'f1_weighted': f1_score(y_true, y_pred, average='weighted', zero_division=0),
        'confusion_matrix': confusion_matrix(y_true, y_pred)
    }
    
    if y_proba is not None:
        metrics['roc_auc'] = roc_auc_score(y_true, y_proba)
        metrics['brier_score'] = brier_score_loss(y_true, y_proba)
    
    return metrics


def optimize_thresholds(y_true, y_proba):
    """Optimise les seuils de décision."""
    best_result = None
    best_score = -1
    
    for t_low in np.arange(0.10, 0.45, 0.02):
        for t_high in np.arange(0.55, 0.90, 0.02):
            if t_high <= t_low:
                continue
            
            mask_rej = y_proba <= t_low
            mask_val = y_proba >= t_high
            
            n_rej = mask_rej.sum()
            n_val = mask_val.sum()
            
            if n_rej == 0 or n_val == 0:
                continue
            
            prec_rej = (y_true[mask_rej] == 0).mean()
            prec_val = (y_true[mask_val] == 1).mean()
            automation = (n_rej + n_val) / len(y_true)
            
            # Score: privilégier précision puis automatisation
            if prec_rej >= 0.95 and prec_val >= 0.93:
                score = automation + prec_rej * 0.1 + prec_val * 0.1
                if score > best_score:
                    best_score = score
                    best_result = {
                        'threshold_low': t_low,
                        'threshold_high': t_high,
                        'precision_rejection': prec_rej,
                        'precision_validation': prec_val,
                        'automation_rate': automation,
                        'n_rejection': int(n_rej),
                        'n_validation': int(n_val),
                        'n_audit': int(len(y_true) - n_rej - n_val)
                    }
    
    if best_result is None:
        # Fallback
        t_low, t_high = 0.3, 0.7
        mask_rej = y_proba <= t_low
        mask_val = y_proba >= t_high
        
        best_result = {
            'threshold_low': t_low,
            'threshold_high': t_high,
            'precision_rejection': (y_true[mask_rej] == 0).mean() if mask_rej.sum() > 0 else 0,
            'precision_validation': (y_true[mask_val] == 1).mean() if mask_val.sum() > 0 else 0,
            'automation_rate': (mask_rej.sum() + mask_val.sum()) / len(y_true),
            'n_rejection': int(mask_rej.sum()),
            'n_validation': int(mask_val.sum()),
            'n_audit': int(len(y_true) - mask_rej.sum() - mask_val.sum())
        }
    
    return best_result


def generate_excel_summary(metrics, thresholds, model_name, output_path):
    """Génère le résumé Excel des performances."""
    from openpyxl import Workbook
    from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
    
    HEADER_FILL = PatternFill(start_color="1F4E79", end_color="1F4E79", fill_type="solid")
    HEADER_FONT = Font(color="FFFFFF", bold=True)
    SUCCESS_FILL = PatternFill(start_color="C6EFCE", end_color="C6EFCE", fill_type="solid")
    WARNING_FILL = PatternFill(start_color="FFEB9C", end_color="FFEB9C", fill_type="solid")
    ERROR_FILL = PatternFill(start_color="FFC7CE", end_color="FFC7CE", fill_type="solid")
    BORDER = Border(
        left=Side(style='thin'), right=Side(style='thin'),
        top=Side(style='thin'), bottom=Side(style='thin')
    )
    
    wb = Workbook()
    ws = wb.active
    ws.title = "Performance"
    
    # Titre
    ws['A1'] = "RAPPORT DE PERFORMANCE - CLASSIFICATION RÉCLAMATIONS"
    ws['A1'].font = Font(bold=True, size=16, color="1F4E79")
    ws.merge_cells('A1:D1')
    
    ws['A2'] = f"Modèle: {model_name} | Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}"
    
    # Métriques Classification
    row = 4
    ws.cell(row=row, column=1, value="MÉTRIQUES DE CLASSIFICATION").font = Font(bold=True, size=12)
    
    row += 1
    headers = ["Métrique", "Valeur", "Objectif", "Statut"]
    for col, h in enumerate(headers, 1):
        cell = ws.cell(row=row, column=col, value=h)
        cell.fill = HEADER_FILL
        cell.font = HEADER_FONT
        cell.border = BORDER
    
    metrics_list = [
        ("Accuracy", metrics['accuracy'], 0.90),
        ("F1-Score Weighted", metrics['f1_weighted'], 0.95),
        ("Précision Rejet (classe 0)", metrics['precision_class_0'], 0.97),
        ("Précision Validation (classe 1)", metrics['precision_class_1'], 0.95),
        ("Recall Rejet", metrics['recall_class_0'], 0.90),
        ("Recall Validation", metrics['recall_class_1'], 0.90),
        ("AUC-ROC", metrics.get('roc_auc', 0), 0.98),
    ]
    
    for name, value, threshold in metrics_list:
        row += 1
        ws.cell(row=row, column=1, value=name).border = BORDER
        cell = ws.cell(row=row, column=2, value=value)
        cell.number_format = '0.0000'
        cell.border = BORDER
        ws.cell(row=row, column=3, value=f"≥{threshold:.0%}").border = BORDER
        
        if value >= threshold:
            status = "✓ OK"
            cell.fill = SUCCESS_FILL
        elif value >= threshold - 0.05:
            status = "⚠ Proche"
            cell.fill = WARNING_FILL
        else:
            status = "✗ À améliorer"
            cell.fill = ERROR_FILL
        ws.cell(row=row, column=4, value=status).border = BORDER
    
    # Métriques Business
    row += 3
    ws.cell(row=row, column=1, value="MÉTRIQUES BUSINESS").font = Font(bold=True, size=12)
    
    row += 1
    for col, h in enumerate(["Métrique", "Valeur"], 1):
        cell = ws.cell(row=row, column=col, value=h)
        cell.fill = HEADER_FILL
        cell.font = HEADER_FONT
        cell.border = BORDER
    
    business_data = [
        ("Seuil Rejet", f"{thresholds['threshold_low']:.2f}"),
        ("Seuil Validation", f"{thresholds['threshold_high']:.2f}"),
        ("Taux Automatisation", f"{thresholds['automation_rate']:.1%}"),
        ("Précision Rejets Auto", f"{thresholds['precision_rejection']:.1%}"),
        ("Précision Validations Auto", f"{thresholds['precision_validation']:.1%}"),
        ("Nb Rejets Auto", thresholds['n_rejection']),
        ("Nb Validations Auto", thresholds['n_validation']),
        ("Nb Audits Manuels", thresholds['n_audit']),
    ]
    
    for name, value in business_data:
        row += 1
        ws.cell(row=row, column=1, value=name).border = BORDER
        ws.cell(row=row, column=2, value=value).border = BORDER
    
    # Matrice de confusion
    row += 3
    ws.cell(row=row, column=1, value="MATRICE DE CONFUSION").font = Font(bold=True, size=12)
    
    cm = metrics['confusion_matrix']
    row += 1
    ws.cell(row=row, column=2, value="Prédit: Non Fondée")
    ws.cell(row=row, column=3, value="Prédit: Fondée")
    
    row += 1
    ws.cell(row=row, column=1, value="Réel: Non Fondée")
    ws.cell(row=row, column=2, value=cm[0, 0]).fill = SUCCESS_FILL
    ws.cell(row=row, column=3, value=cm[0, 1]).fill = ERROR_FILL
    
    row += 1
    ws.cell(row=row, column=1, value="Réel: Fondée")
    ws.cell(row=row, column=2, value=cm[1, 0]).fill = WARNING_FILL
    ws.cell(row=row, column=3, value=cm[1, 1]).fill = SUCCESS_FILL
    
    # Ajuster largeurs
    ws.column_dimensions['A'].width = 35
    ws.column_dimensions['B'].width = 20
    ws.column_dimensions['C'].width = 15
    ws.column_dimensions['D'].width = 15
    
    wb.save(output_path)
    logger.info(f"✓ Résumé performance: {output_path}")


def generate_excel_by_category(df_test, y_true, y_pred, y_proba, cat_cols, output_path):
    """Génère le rapport par catégorie."""
    from openpyxl import Workbook
    from openpyxl.styles import Font, PatternFill, Border, Side
    
    HEADER_FILL = PatternFill(start_color="1F4E79", end_color="1F4E79", fill_type="solid")
    HEADER_FONT = Font(color="FFFFFF", bold=True)
    SUCCESS_FILL = PatternFill(start_color="C6EFCE", end_color="C6EFCE", fill_type="solid")
    WARNING_FILL = PatternFill(start_color="FFEB9C", end_color="FFEB9C", fill_type="solid")
    ERROR_FILL = PatternFill(start_color="FFC7CE", end_color="FFC7CE", fill_type="solid")
    BORDER = Border(
        left=Side(style='thin'), right=Side(style='thin'),
        top=Side(style='thin'), bottom=Side(style='thin')
    )
    
    wb = Workbook()
    ws = wb.active
    ws.title = "Vue ensemble"
    ws.title = "Vue ensemble"
    ws['A1'].font = Font(bold=True, size=16)
    
    for cat_col in cat_cols:
        if cat_col not in df_test.columns:
            continue
        
        ws_cat = wb.create_sheet(cat_col[:31])
        ws_cat['A1'] = f"Performance par {cat_col}"
        ws_cat['A1'].font = Font(bold=True, size=14)
        
        results = []
        for cat in df_test[cat_col].unique():
            mask = df_test[cat_col] == cat
            if mask.sum() < 5:
                continue
            
            cat_y_true = y_true[mask]
            cat_y_pred = y_pred[mask]
            
            results.append({
                'Catégorie': str(cat)[:40],
                'N': int(mask.sum()),
                'Taux Fondées': float(cat_y_true.mean()),
                'Accuracy': float(accuracy_score(cat_y_true, cat_y_pred)),
                'Précision': float(precision_score(cat_y_true, cat_y_pred, zero_division=0)),
                'Recall': float(recall_score(cat_y_true, cat_y_pred, zero_division=0)),
                'F1': float(f1_score(cat_y_true, cat_y_pred, zero_division=0))
            })
        
        if not results:
            continue
        
        results_df = pd.DataFrame(results).sort_values('F1', ascending=False)
        
        row = 3
        headers = list(results_df.columns)
        for col, h in enumerate(headers, 1):
            cell = ws_cat.cell(row=row, column=col, value=h)
            cell.fill = HEADER_FILL
            cell.font = HEADER_FONT
            cell.border = BORDER
        
        for r_idx, (_, data_row) in enumerate(results_df.iterrows(), row + 1):
            for c_idx, (col_name, value) in enumerate(data_row.items(), 1):
                cell = ws_cat.cell(row=r_idx, column=c_idx, value=value)
                cell.border = BORDER
                
                if col_name in ['Taux Fondées', 'Accuracy', 'Précision', 'Recall', 'F1']:
                    cell.number_format = '0.00%'
                    if col_name == 'F1':
                        if value >= 0.95:
                            cell.fill = SUCCESS_FILL
                        elif value >= 0.90:
                            cell.fill = WARNING_FILL
                        else:
                            cell.fill = ERROR_FILL
        
        for col in 'ABCDEFG':
            ws_cat.column_dimensions[col].width = 18
    
    wb.save(output_path)
    logger.info(f"✓ Performance par catégorie: {output_path}")


def generate_excel_errors(df_test, y_true, y_pred, y_proba, output_path):
    """Génère le fichier des erreurs (FP et FN)."""
    from openpyxl import Workbook
    from openpyxl.styles import Font, PatternFill, Border, Side
    
    HEADER_FILL = PatternFill(start_color="1F4E79", end_color="1F4E79", fill_type="solid")
    HEADER_FONT = Font(color="FFFFFF", bold=True)
    ERROR_FILL = PatternFill(start_color="FFC7CE", end_color="FFC7CE", fill_type="solid")
    WARNING_FILL = PatternFill(start_color="FFEB9C", end_color="FFEB9C", fill_type="solid")
    BORDER = Border(
        left=Side(style='thin'), right=Side(style='thin'),
        top=Side(style='thin'), bottom=Side(style='thin')
    )
    
    # Préparer les données
    df_analysis = df_test.copy()
    df_analysis['_Prediction'] = y_pred
    df_analysis['_Probabilite'] = y_proba
    df_analysis['_Reel'] = y_true
    
    # Faux Positifs et Faux Négatifs
    mask_fp = (y_pred == 1) & (y_true == 0)
    mask_fn = (y_pred == 0) & (y_true == 1)
    
    df_fp = df_analysis[mask_fp].copy()
    df_fn = df_analysis[mask_fn].copy()
    
    wb = Workbook()
    
    # Résumé
    ws = wb.active
    ws.title = "Résumé"
    ws['A1'] = "ANALYSE DES ERREURS"
    ws['A1'].font = Font(bold=True, size=16)
    
    total = len(df_fp) + len(df_fn)
    
    ws['A3'] = "Type Erreur"
    ws['B3'] = "Nombre"
    ws['C3'] = "Impact"
    for col in 'ABC':
        ws[f'{col}3'].fill = HEADER_FILL
        ws[f'{col}3'].font = HEADER_FONT
    
    ws['A4'] = "Faux Positifs (Fausses Validations)"
    ws['B4'] = len(df_fp)
    ws['C4'] = "COÛT FINANCIER"
    ws['A4'].fill = ERROR_FILL
    
    ws['A5'] = "Faux Négatifs (Faux Rejets)"
    ws['B5'] = len(df_fn)
    ws['C5'] = "INSATISFACTION CLIENT"
    ws['A5'].fill = WARNING_FILL
    
    ws['A6'] = "TOTAL"
    ws['B6'] = total
    ws['A6'].font = Font(bold=True)
    
    ws.column_dimensions['A'].width = 40
    ws.column_dimensions['B'].width = 12
    ws.column_dimensions['C'].width = 25
    
    # Faux Positifs
    ws_fp = wb.create_sheet("Faux Positifs")
    ws_fp['A1'] = f"FAUX POSITIFS - {len(df_fp)} cas"
    ws_fp['A1'].font = Font(bold=True, size=14, color="C00000")
    
    if len(df_fp) > 0:
        cols = [c for c in df_fp.columns if not c.startswith('_')] + ['_Probabilite']
        
        row = 3
        for col_idx, col in enumerate(cols, 1):
            cell = ws_fp.cell(row=row, column=col_idx, value=col)
            cell.fill = HEADER_FILL
            cell.font = HEADER_FONT
        
        for r_idx, (_, data_row) in enumerate(df_fp[cols].head(500).iterrows(), row + 1):
            for c_idx, value in enumerate(data_row, 1):
                cell = ws_fp.cell(row=r_idx, column=c_idx)
                if pd.isna(value):
                    cell.value = ""
                elif isinstance(value, (np.floating, float)):
                    cell.value = round(float(value), 4)
                elif isinstance(value, (np.integer, int)):
                    cell.value = int(value)
                else:
                    cell.value = str(value)[:100]
    
    # Faux Négatifs
    ws_fn = wb.create_sheet("Faux Négatifs")
    ws_fn['A1'] = f"FAUX NÉGATIFS - {len(df_fn)} cas"
    ws_fn['A1'].font = Font(bold=True, size=14, color="FF6600")
    
    if len(df_fn) > 0:
        cols = [c for c in df_fn.columns if not c.startswith('_')] + ['_Probabilite']
        
        row = 3
        for col_idx, col in enumerate(cols, 1):
            cell = ws_fn.cell(row=row, column=col_idx, value=col)
            cell.fill = HEADER_FILL
            cell.font = HEADER_FONT
        
        for r_idx, (_, data_row) in enumerate(df_fn[cols].head(500).iterrows(), row + 1):
            for c_idx, value in enumerate(data_row, 1):
                cell = ws_fn.cell(row=r_idx, column=c_idx)
                if pd.isna(value):
                    cell.value = ""
                elif isinstance(value, (np.floating, float)):
                    cell.value = round(float(value), 4)
                elif isinstance(value, (np.integer, int)):
                    cell.value = int(value)
                else:
                    cell.value = str(value)[:100]
    
    wb.save(output_path)
    logger.info(f"✓ Analyse erreurs: {output_path}")
    logger.info(f"  - Faux Positifs: {len(df_fp)}")
    logger.info(f"  - Faux Négatifs: {len(df_fn)}")


def main():
    parser = argparse.ArgumentParser(description='Pipeline simplifié - Classification Réclamations')
    parser.add_argument('--data', '-d', required=True, help='Fichier de données Excel')
    parser.add_argument('--output', '-o', default='outputs', help='Répertoire de sortie')
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 60)
    logger.info("PIPELINE CLASSIFICATION RÉCLAMATIONS BANCAIRES")
    logger.info("=" * 60)
    
    # 1. Charger les données
    logger.info(f"\n📂 Chargement: {args.data}")
    df = pd.read_excel(args.data)
    logger.info(f"   {df.shape[0]} lignes, {df.shape[1]} colonnes")
    
    # 2. Analyser
    analyze_dataframe(df)
    
    # 3. Détecter colonnes
    target_col = detect_target(df)
    cat_cols = detect_category_columns(df, target_col)
    
    # 4. Préparer données
    logger.info("\n🔧 Préparation des données...")
    y = prepare_target(df, target_col)
    X = prepare_features(df, target_col)
    
    # 5. Split
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, df.index, test_size=0.2, stratify=y, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.15, stratify=y_train, random_state=42
    )
    
    logger.info(f"   Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    # 6. Entraîner
    logger.info("\n🚀 Entraînement du modèle XGBoost...")
    model, model_name = train_model(X_train, y_train, X_val, y_val)
    
    # 7. Prédire
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # 8. Métriques
    logger.info("\n📊 Calcul des métriques...")
    metrics = calculate_all_metrics(y_test.values, y_pred, y_proba)
    thresholds = optimize_thresholds(y_test.values, y_proba)
    
    logger.info(f"   F1-Score: {metrics['f1_weighted']:.4f}")
    logger.info(f"   AUC-ROC: {metrics['roc_auc']:.4f}")
    logger.info(f"   Taux Auto: {thresholds['automation_rate']:.1%}")
    
    # 9. Générer livrables
    logger.info("\n📝 Génération des livrables...")
    
    df_test = df.iloc[idx_test].reset_index(drop=True)
    y_test_reset = y_test.reset_index(drop=True)
    
    generate_excel_summary(
        metrics, thresholds, model_name,
        output_dir / "1_performance_summary.xlsx"
    )
    
    generate_excel_by_category(
        df_test, y_test_reset.values, y_pred, y_proba, cat_cols,
        output_dir / "2_performance_by_category.xlsx"
    )
    
    generate_excel_errors(
        df_test, y_test_reset.values, y_pred, y_proba,
        output_dir / "3_errors_analysis.xlsx"
    )
    
    logger.info("\n" + "=" * 60)
    logger.info("✅ PIPELINE TERMINÉ AVEC SUCCÈS")
    logger.info("=" * 60)
    logger.info(f"\n📁 Livrables dans: {output_dir}")
    logger.info("   1. 1_performance_summary.xlsx")
    logger.info("   2. 2_performance_by_category.xlsx")
    logger.info("   3. 3_errors_analysis.xlsx")


if __name__ == "__main__":
    main()
