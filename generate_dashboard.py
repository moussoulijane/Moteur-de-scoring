#!/usr/bin/env python3
"""
Dashboard Complet - Classification Réclamations Bancaires
==========================================================
Génère:
1. Dashboard HTML interactif avec performances par catégorie
2. Feature Importances
3. Excel des Faux Positifs et Faux Négatifs

Usage:
    python generate_dashboard.py --data path/to/data.xlsx --output outputs/
"""

import pandas as pd
import numpy as np
import argparse
import logging
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# =============================================================================
# PRÉPARATION DES DONNÉES
# =============================================================================

def detect_target(df: pd.DataFrame) -> str:
    """Détecte la colonne cible."""
    for col in df.columns:
        if 'fond' in col.lower():
            return col
    raise ValueError("Colonne cible non trouvée")


def detect_category_columns(df: pd.DataFrame, target_col: str) -> list:
    """Détecte les colonnes catégorielles."""
    cat_cols = []
    for col in df.columns:
        if col == target_col:
            continue
        col_lower = col.lower()
        if any(x in col_lower for x in ['famille', 'categ', 'produit', 'segment', 'type', 'motif', 'canal']):
            if df[col].dtype == 'object' and 2 <= df[col].nunique() <= 100:
                cat_cols.append(col)
    return cat_cols


def prepare_target(df: pd.DataFrame, target_col: str) -> pd.Series:
    """Prépare la variable cible."""
    y = df[target_col].copy()
    if y.dtype == 'object':
        y = y.apply(lambda x: 1 if str(x).lower() in ['oui', 'yes', '1', 'fondée', 'fondee', 'true', 'o'] else 0)
    elif y.dtype == 'bool':
        y = y.astype(int)
    return y.fillna(0).astype(int)


def prepare_features(df: pd.DataFrame, target_col: str) -> tuple:
    """Prépare les features."""
    exclude_patterns = ['id', 'date', 'dt_', '_dt', 'timestamp', 'datetime', 'num_', 'numero']
    
    X = df.copy()
    if target_col in X.columns:
        X = X.drop(columns=[target_col])
    
    cols_to_drop = []
    feature_names = []
    
    for col in X.columns:
        col_lower = col.lower()
        
        if any(pattern in col_lower for pattern in exclude_patterns):
            cols_to_drop.append(col)
            continue
        
        try:
            if pd.api.types.is_datetime64_any_dtype(X[col]):
                X[f'{col}_month'] = X[col].dt.month.fillna(0).astype(int)
                X[f'{col}_dow'] = X[col].dt.dayofweek.fillna(0).astype(int)
                cols_to_drop.append(col)
                continue
            
            if pd.api.types.is_timedelta64_dtype(X[col]):
                X[col] = X[col].dt.total_seconds().fillna(0)
                continue
            
            if X[col].dtype == 'bool':
                X[col] = X[col].astype(int)
                continue
            
            if X[col].dtype == 'object' or str(X[col].dtype) == 'category':
                unique_vals = set(str(v).lower() for v in X[col].dropna().unique())
                if unique_vals.issubset({'oui', 'non', 'o', 'n', 'true', 'false', '0', '1', 'yes', 'no'}):
                    X[col] = X[col].apply(
                        lambda x: 1 if str(x).lower() in ['oui', 'o', 'true', '1', 'yes'] else 0 if pd.notna(x) else 0
                    ).astype(int)
                else:
                    le = LabelEncoder()
                    X[col] = X[col].astype(str).fillna('_MISSING_')
                    X[col] = le.fit_transform(X[col])
                continue
            
            if pd.api.types.is_numeric_dtype(X[col]):
                X[col] = X[col].fillna(X[col].median() if X[col].notna().any() else 0)
                continue
                
        except Exception as e:
            cols_to_drop.append(col)
    
    X = X.drop(columns=[c for c in cols_to_drop if c in X.columns])
    
    for col in X.columns:
        if not pd.api.types.is_numeric_dtype(X[col]):
            try:
                X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)
            except:
                X = X.drop(columns=[col])
    
    X = X.fillna(0)
    feature_names = list(X.columns)
    
    return X, feature_names


def train_xgboost(X_train, y_train, X_val, y_val):
    """Entraîne XGBoost."""
    import xgboost as xgb
    
    scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    
    model = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='auc',
        max_depth=6,
        learning_rate=0.05,
        n_estimators=500,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        verbosity=0,
        use_label_encoder=False
    )
    
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    return model


# =============================================================================
# GÉNÉRATION DU DASHBOARD HTML
# =============================================================================

def generate_html_dashboard(
    metrics: dict,
    metrics_by_category: dict,
    feature_importance: pd.DataFrame,
    thresholds: dict,
    fp_count: int,
    fn_count: int,
    output_path: str
):
    """Génère le dashboard HTML interactif."""
    
    # Préparer les données pour les graphiques
    cat_data_js = []
    for cat_name, cat_metrics in metrics_by_category.items():
        for group, m in cat_metrics.items():
            cat_data_js.append({
                'category': cat_name,
                'group': str(group)[:30],
                'n': m['n'],
                'accuracy': round(m['accuracy'] * 100, 1),
                'precision': round(m['precision'] * 100, 1),
                'recall': round(m['recall'] * 100, 1),
                'f1': round(m['f1'] * 100, 1),
                'taux_fondees': round(m['taux_fondees'] * 100, 1)
            })
    
    # Top 20 feature importance
    top_features = feature_importance.head(20).to_dict('records')
    
    cm = metrics['confusion_matrix']
    
    html_content = f'''<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Dashboard ML - Classification Réclamations</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            min-height: 100vh;
            color: #fff;
            padding: 20px;
        }}
        
        .dashboard-header {{
            text-align: center;
            padding: 30px;
            background: rgba(255,255,255,0.1);
            border-radius: 15px;
            margin-bottom: 30px;
        }}
        
        .dashboard-header h1 {{
            font-size: 2.5em;
            margin-bottom: 10px;
            background: linear-gradient(90deg, #00d4ff, #00ff88);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}
        
        .dashboard-header .date {{
            color: #888;
            font-size: 0.9em;
        }}
        
        .grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        
        .card {{
            background: rgba(255,255,255,0.05);
            border-radius: 15px;
            padding: 25px;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255,255,255,0.1);
        }}
        
        .card h3 {{
            color: #00d4ff;
            margin-bottom: 20px;
            font-size: 1.2em;
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        
        .metric-grid {{
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 15px;
        }}
        
        .metric-item {{
            background: rgba(0,0,0,0.3);
            padding: 15px;
            border-radius: 10px;
            text-align: center;
        }}
        
        .metric-item .value {{
            font-size: 2em;
            font-weight: bold;
            color: #00ff88;
        }}
        
        .metric-item .label {{
            font-size: 0.85em;
            color: #aaa;
            margin-top: 5px;
        }}
        
        .metric-item.warning .value {{
            color: #ffaa00;
        }}
        
        .metric-item.danger .value {{
            color: #ff4444;
        }}
        
        .metric-item.success .value {{
            color: #00ff88;
        }}
        
        .large-card {{
            grid-column: span 2;
        }}
        
        .chart-container {{
            position: relative;
            height: 400px;
            width: 100%;
        }}
        
        .confusion-matrix {{
            display: grid;
            grid-template-columns: auto 1fr 1fr;
            grid-template-rows: auto 1fr 1fr;
            gap: 5px;
            max-width: 400px;
            margin: 0 auto;
        }}
        
        .cm-cell {{
            padding: 20px;
            text-align: center;
            border-radius: 8px;
            font-weight: bold;
        }}
        
        .cm-header {{
            background: transparent;
            color: #888;
            font-size: 0.9em;
        }}
        
        .cm-tn {{ background: rgba(0, 255, 136, 0.3); color: #00ff88; }}
        .cm-fp {{ background: rgba(255, 68, 68, 0.3); color: #ff4444; }}
        .cm-fn {{ background: rgba(255, 170, 0, 0.3); color: #ffaa00; }}
        .cm-tp {{ background: rgba(0, 212, 255, 0.3); color: #00d4ff; }}
        
        .cm-value {{
            font-size: 1.8em;
            display: block;
        }}
        
        .cm-label {{
            font-size: 0.75em;
            opacity: 0.8;
        }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 15px;
        }}
        
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid rgba(255,255,255,0.1);
        }}
        
        th {{
            background: rgba(0,212,255,0.2);
            color: #00d4ff;
            font-weight: 600;
        }}
        
        tr:hover {{
            background: rgba(255,255,255,0.05);
        }}
        
        .progress-bar {{
            height: 8px;
            background: rgba(255,255,255,0.1);
            border-radius: 4px;
            overflow: hidden;
        }}
        
        .progress-fill {{
            height: 100%;
            border-radius: 4px;
            transition: width 0.5s ease;
        }}
        
        .progress-fill.green {{ background: linear-gradient(90deg, #00ff88, #00d4ff); }}
        .progress-fill.orange {{ background: linear-gradient(90deg, #ffaa00, #ff6600); }}
        .progress-fill.red {{ background: linear-gradient(90deg, #ff4444, #ff0000); }}
        
        .tab-container {{
            margin-top: 20px;
        }}
        
        .tab-buttons {{
            display: flex;
            gap: 10px;
            margin-bottom: 20px;
            flex-wrap: wrap;
        }}
        
        .tab-btn {{
            padding: 10px 20px;
            background: rgba(255,255,255,0.1);
            border: none;
            border-radius: 8px;
            color: #fff;
            cursor: pointer;
            transition: all 0.3s;
        }}
        
        .tab-btn:hover, .tab-btn.active {{
            background: rgba(0,212,255,0.3);
            color: #00d4ff;
        }}
        
        .tab-content {{
            display: none;
        }}
        
        .tab-content.active {{
            display: block;
        }}
        
        .error-summary {{
            display: flex;
            gap: 20px;
            justify-content: center;
            margin-top: 20px;
        }}
        
        .error-box {{
            padding: 20px 40px;
            border-radius: 10px;
            text-align: center;
        }}
        
        .error-box.fp {{
            background: rgba(255, 68, 68, 0.2);
            border: 2px solid #ff4444;
        }}
        
        .error-box.fn {{
            background: rgba(255, 170, 0, 0.2);
            border: 2px solid #ffaa00;
        }}
        
        .error-box .count {{
            font-size: 2.5em;
            font-weight: bold;
        }}
        
        .error-box.fp .count {{ color: #ff4444; }}
        .error-box.fn .count {{ color: #ffaa00; }}
        
        @media (max-width: 768px) {{
            .large-card {{
                grid-column: span 1;
            }}
            .grid {{
                grid-template-columns: 1fr;
            }}
        }}
    </style>
</head>
<body>
    <div class="dashboard-header">
        <h1>🏦 Dashboard Classification Réclamations</h1>
        <p class="date">Généré le {datetime.now().strftime('%d/%m/%Y à %H:%M')}</p>
    </div>
    
    <!-- MÉTRIQUES PRINCIPALES -->
    <div class="grid">
        <div class="card">
            <h3>📊 Métriques Globales</h3>
            <div class="metric-grid">
                <div class="metric-item {'success' if metrics['accuracy'] >= 0.90 else 'warning' if metrics['accuracy'] >= 0.85 else 'danger'}">
                    <span class="value">{metrics['accuracy']*100:.1f}%</span>
                    <span class="label">Accuracy</span>
                </div>
                <div class="metric-item {'success' if metrics['f1_weighted'] >= 0.95 else 'warning' if metrics['f1_weighted'] >= 0.90 else 'danger'}">
                    <span class="value">{metrics['f1_weighted']*100:.1f}%</span>
                    <span class="label">F1-Score</span>
                </div>
                <div class="metric-item {'success' if metrics.get('roc_auc', 0) >= 0.95 else 'warning' if metrics.get('roc_auc', 0) >= 0.90 else 'danger'}">
                    <span class="value">{metrics.get('roc_auc', 0)*100:.1f}%</span>
                    <span class="label">AUC-ROC</span>
                </div>
                <div class="metric-item">
                    <span class="value">{thresholds['automation_rate']*100:.1f}%</span>
                    <span class="label">Automatisation</span>
                </div>
            </div>
        </div>
        
        <div class="card">
            <h3>🎯 Précision par Classe</h3>
            <div class="metric-grid">
                <div class="metric-item {'success' if metrics['precision_class_0'] >= 0.97 else 'warning' if metrics['precision_class_0'] >= 0.93 else 'danger'}">
                    <span class="value">{metrics['precision_class_0']*100:.1f}%</span>
                    <span class="label">Précision Rejet</span>
                </div>
                <div class="metric-item {'success' if metrics['precision_class_1'] >= 0.95 else 'warning' if metrics['precision_class_1'] >= 0.90 else 'danger'}">
                    <span class="value">{metrics['precision_class_1']*100:.1f}%</span>
                    <span class="label">Précision Validation</span>
                </div>
                <div class="metric-item">
                    <span class="value">{metrics['recall_class_0']*100:.1f}%</span>
                    <span class="label">Recall Rejet</span>
                </div>
                <div class="metric-item">
                    <span class="value">{metrics['recall_class_1']*100:.1f}%</span>
                    <span class="label">Recall Validation</span>
                </div>
            </div>
        </div>
        
        <div class="card">
            <h3>⚙️ Seuils Optimisés</h3>
            <div class="metric-grid">
                <div class="metric-item">
                    <span class="value">{thresholds['threshold_low']:.2f}</span>
                    <span class="label">Seuil Rejet</span>
                </div>
                <div class="metric-item">
                    <span class="value">{thresholds['threshold_high']:.2f}</span>
                    <span class="label">Seuil Validation</span>
                </div>
                <div class="metric-item">
                    <span class="value">{thresholds['n_rejection']:,}</span>
                    <span class="label">Rejets Auto</span>
                </div>
                <div class="metric-item">
                    <span class="value">{thresholds['n_validation']:,}</span>
                    <span class="label">Validations Auto</span>
                </div>
            </div>
        </div>
        
        <div class="card">
            <h3>❌ Analyse des Erreurs</h3>
            <div class="error-summary">
                <div class="error-box fp">
                    <span class="count">{fp_count}</span>
                    <div>Faux Positifs</div>
                    <small>Fausses validations</small>
                </div>
                <div class="error-box fn">
                    <span class="count">{fn_count}</span>
                    <div>Faux Négatifs</div>
                    <small>Faux rejets</small>
                </div>
            </div>
        </div>
    </div>
    
    <!-- MATRICE DE CONFUSION -->
    <div class="grid">
        <div class="card">
            <h3>🔢 Matrice de Confusion</h3>
            <div class="confusion-matrix">
                <div class="cm-cell cm-header"></div>
                <div class="cm-cell cm-header">Prédit: Non Fondée</div>
                <div class="cm-cell cm-header">Prédit: Fondée</div>
                
                <div class="cm-cell cm-header">Réel: Non Fondée</div>
                <div class="cm-cell cm-tn">
                    <span class="cm-value">{cm[0,0]:,}</span>
                    <span class="cm-label">Vrais Négatifs</span>
                </div>
                <div class="cm-cell cm-fp">
                    <span class="cm-value">{cm[0,1]:,}</span>
                    <span class="cm-label">Faux Positifs</span>
                </div>
                
                <div class="cm-cell cm-header">Réel: Fondée</div>
                <div class="cm-cell cm-fn">
                    <span class="cm-value">{cm[1,0]:,}</span>
                    <span class="cm-label">Faux Négatifs</span>
                </div>
                <div class="cm-cell cm-tp">
                    <span class="cm-value">{cm[1,1]:,}</span>
                    <span class="cm-label">Vrais Positifs</span>
                </div>
            </div>
        </div>
        
        <div class="card">
            <h3>📈 Top 20 Features Importantes</h3>
            <div class="chart-container">
                <canvas id="featureChart"></canvas>
            </div>
        </div>
    </div>
    
    <!-- PERFORMANCE PAR CATÉGORIE -->
    <div class="card large-card">
        <h3>📂 Performance par Catégorie</h3>
        
        <div class="tab-container">
            <div class="tab-buttons">
                {''.join(f'<button class="tab-btn {" active" if i==0 else ""}" onclick="showTab(event, \'tab-{cat}\')">{cat}</button>' for i, cat in enumerate(metrics_by_category.keys()))}
            </div>
            
            {''.join(generate_category_tab(cat_name, cat_data) for cat_name, cat_data in metrics_by_category.items())}
        </div>
    </div>
    
    <script>
        // Données pour les graphiques
        const featureData = {top_features};
        const categoryData = {cat_data_js};
        
        // Graphique Feature Importance
        const featureCtx = document.getElementById('featureChart').getContext('2d');
        new Chart(featureCtx, {{
            type: 'bar',
            data: {{
                labels: featureData.map(f => f.feature.substring(0, 25)),
                datasets: [{{
                    label: 'Importance',
                    data: featureData.map(f => f.importance),
                    backgroundColor: 'rgba(0, 212, 255, 0.6)',
                    borderColor: 'rgba(0, 212, 255, 1)',
                    borderWidth: 1
                }}]
            }},
            options: {{
                indexAxis: 'y',
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{
                    legend: {{ display: false }}
                }},
                scales: {{
                    x: {{
                        grid: {{ color: 'rgba(255,255,255,0.1)' }},
                        ticks: {{ color: '#aaa' }}
                    }},
                    y: {{
                        grid: {{ display: false }},
                        ticks: {{ color: '#fff', font: {{ size: 10 }} }}
                    }}
                }}
            }}
        }});
        
        // Gestion des onglets
        function showTab(evt, tabId) {{
            document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
            document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
            document.getElementById(tabId).classList.add('active');
            evt.currentTarget.classList.add('active');
        }}
        
        // Activer le premier onglet
        document.querySelector('.tab-content')?.classList.add('active');
    </script>
</body>
</html>'''
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    logger.info(f"✓ Dashboard HTML généré: {output_path}")


def generate_category_tab(cat_name: str, cat_data: dict) -> str:
    """Génère le contenu HTML d'un onglet de catégorie."""
    rows = []
    for group, m in sorted(cat_data.items(), key=lambda x: x[1]['f1'], reverse=True):
        f1_class = 'green' if m['f1'] >= 0.95 else 'orange' if m['f1'] >= 0.90 else 'red'
        rows.append(f'''
            <tr>
                <td>{str(group)[:40]}</td>
                <td style="text-align:right">{m['n']:,}</td>
                <td style="text-align:right">{m['taux_fondees']*100:.1f}%</td>
                <td style="text-align:right">{m['accuracy']*100:.1f}%</td>
                <td style="text-align:right">{m['precision']*100:.1f}%</td>
                <td style="text-align:right">{m['recall']*100:.1f}%</td>
                <td>
                    <div style="display:flex;align-items:center;gap:10px">
                        <span style="width:50px">{m['f1']*100:.1f}%</span>
                        <div class="progress-bar" style="flex:1">
                            <div class="progress-fill {f1_class}" style="width:{m['f1']*100}%"></div>
                        </div>
                    </div>
                </td>
            </tr>
        ''')
    
    first_class = 'active' if cat_name == list(cat_data.keys())[0] if cat_data else '' else ''
    
    return f'''
        <div id="tab-{cat_name}" class="tab-content {first_class}">
            <table>
                <thead>
                    <tr>
                        <th>{cat_name}</th>
                        <th style="text-align:right">N</th>
                        <th style="text-align:right">Taux Fondées</th>
                        <th style="text-align:right">Accuracy</th>
                        <th style="text-align:right">Précision</th>
                        <th style="text-align:right">Recall</th>
                        <th>F1-Score</th>
                    </tr>
                </thead>
                <tbody>
                    {''.join(rows)}
                </tbody>
            </table>
        </div>
    '''


# =============================================================================
# GÉNÉRATION DES FICHIERS EXCEL
# =============================================================================

def generate_excel_errors(df_test, y_true, y_pred, y_proba, output_path):
    """Génère le fichier Excel des erreurs."""
    from openpyxl import Workbook
    from openpyxl.styles import Font, PatternFill, Border, Side, Alignment
    
    HEADER_FILL = PatternFill(start_color="1F4E79", end_color="1F4E79", fill_type="solid")
    HEADER_FONT = Font(color="FFFFFF", bold=True)
    ERROR_FILL = PatternFill(start_color="FFC7CE", end_color="FFC7CE", fill_type="solid")
    WARNING_FILL = PatternFill(start_color="FFEB9C", end_color="FFEB9C", fill_type="solid")
    BORDER = Border(
        left=Side(style='thin'), right=Side(style='thin'),
        top=Side(style='thin'), bottom=Side(style='thin')
    )
    
    df_analysis = df_test.copy()
    df_analysis['_Prediction'] = y_pred
    df_analysis['_Probabilite'] = y_proba
    df_analysis['_Reel'] = y_true
    
    mask_fp = (y_pred == 1) & (y_true == 0)
    mask_fn = (y_pred == 0) & (y_true == 1)
    
    df_fp = df_analysis[mask_fp].copy()
    df_fn = df_analysis[mask_fn].copy()
    
    wb = Workbook()
    
    # Résumé
    ws = wb.active
    ws.title = "Résumé"
    ws['A1'] = "ANALYSE DES ERREURS DE PRÉDICTION"
    ws['A1'].font = Font(bold=True, size=16, color="1F4E79")
    ws.merge_cells('A1:D1')
    
    ws['A3'] = "Type d'Erreur"
    ws['B3'] = "Nombre"
    ws['C3'] = "Pourcentage"
    ws['D3'] = "Impact Business"
    
    for col in ['A', 'B', 'C', 'D']:
        ws[f'{col}3'].fill = HEADER_FILL
        ws[f'{col}3'].font = HEADER_FONT
        ws[f'{col}3'].border = BORDER
    
    total = len(df_fp) + len(df_fn)
    
    ws['A4'] = "Faux Positifs (Fausses Validations)"
    ws['B4'] = len(df_fp)
    ws['C4'] = f"{len(df_fp)/total*100:.1f}%" if total > 0 else "0%"
    ws['D4'] = "💰 COÛT FINANCIER"
    ws['A4'].fill = ERROR_FILL
    
    ws['A5'] = "Faux Négatifs (Faux Rejets)"
    ws['B5'] = len(df_fn)
    ws['C5'] = f"{len(df_fn)/total*100:.1f}%" if total > 0 else "0%"
    ws['D5'] = "😠 INSATISFACTION CLIENT"
    ws['A5'].fill = WARNING_FILL
    
    ws['A6'] = "TOTAL ERREURS"
    ws['B6'] = total
    ws['C6'] = "100%"
    ws['A6'].font = Font(bold=True)
    
    for row in range(4, 7):
        for col in ['A', 'B', 'C', 'D']:
            ws[f'{col}{row}'].border = BORDER
    
    ws.column_dimensions['A'].width = 40
    ws.column_dimensions['B'].width = 12
    ws.column_dimensions['C'].width = 15
    ws.column_dimensions['D'].width = 30
    
    # Faux Positifs
    ws_fp = wb.create_sheet("Faux Positifs")
    ws_fp['A1'] = f"FAUX POSITIFS - {len(df_fp)} cas"
    ws_fp['A1'].font = Font(bold=True, size=14, color="C00000")
    ws_fp['A2'] = "Réclamations prédites FONDÉES mais réellement NON FONDÉES"
    
    if len(df_fp) > 0:
        cols = [c for c in df_fp.columns if not c.startswith('_')] + ['_Probabilite']
        
        row = 4
        for col_idx, col in enumerate(cols, 1):
            cell = ws_fp.cell(row=row, column=col_idx, value=col.replace('_', ''))
            cell.fill = HEADER_FILL
            cell.font = HEADER_FONT
            cell.border = BORDER
        
        for r_idx, (_, data_row) in enumerate(df_fp[cols].head(1000).iterrows(), row + 1):
            for c_idx, value in enumerate(data_row, 1):
                cell = ws_fp.cell(row=r_idx, column=c_idx)
                if pd.isna(value):
                    cell.value = ""
                elif isinstance(value, (np.floating, float)):
                    cell.value = round(float(value), 4)
                elif isinstance(value, (np.integer, int)):
                    cell.value = int(value)
                elif isinstance(value, (pd.Timestamp, datetime)):
                    cell.value = value.strftime('%Y-%m-%d') if pd.notna(value) else ""
                else:
                    cell.value = str(value)[:100]
                cell.border = BORDER
    
    # Faux Négatifs
    ws_fn = wb.create_sheet("Faux Négatifs")
    ws_fn['A1'] = f"FAUX NÉGATIFS - {len(df_fn)} cas"
    ws_fn['A1'].font = Font(bold=True, size=14, color="FF6600")
    ws_fn['A2'] = "Réclamations prédites NON FONDÉES mais réellement FONDÉES"
    
    if len(df_fn) > 0:
        cols = [c for c in df_fn.columns if not c.startswith('_')] + ['_Probabilite']
        
        row = 4
        for col_idx, col in enumerate(cols, 1):
            cell = ws_fn.cell(row=row, column=col_idx, value=col.replace('_', ''))
            cell.fill = HEADER_FILL
            cell.font = HEADER_FONT
            cell.border = BORDER
        
        for r_idx, (_, data_row) in enumerate(df_fn[cols].head(1000).iterrows(), row + 1):
            for c_idx, value in enumerate(data_row, 1):
                cell = ws_fn.cell(row=r_idx, column=c_idx)
                if pd.isna(value):
                    cell.value = ""
                elif isinstance(value, (np.floating, float)):
                    cell.value = round(float(value), 4)
                elif isinstance(value, (np.integer, int)):
                    cell.value = int(value)
                elif isinstance(value, (pd.Timestamp, datetime)):
                    cell.value = value.strftime('%Y-%m-%d') if pd.notna(value) else ""
                else:
                    cell.value = str(value)[:100]
                cell.border = BORDER
    
    wb.save(output_path)
    logger.info(f"✓ Analyse erreurs Excel: {output_path}")
    
    return len(df_fp), len(df_fn)


def generate_excel_feature_importance(feature_importance: pd.DataFrame, output_path: str):
    """Génère le fichier Excel des feature importances."""
    from openpyxl import Workbook
    from openpyxl.styles import Font, PatternFill, Border, Side, Alignment
    from openpyxl.formatting.rule import DataBarRule
    
    HEADER_FILL = PatternFill(start_color="1F4E79", end_color="1F4E79", fill_type="solid")
    HEADER_FONT = Font(color="FFFFFF", bold=True)
    BORDER = Border(
        left=Side(style='thin'), right=Side(style='thin'),
        top=Side(style='thin'), bottom=Side(style='thin')
    )
    
    wb = Workbook()
    ws = wb.active
    ws.title = "Feature Importance"
    
    ws['A1'] = "IMPORTANCE DES FEATURES - MODÈLE XGBOOST"
    ws['A1'].font = Font(bold=True, size=16, color="1F4E79")
    ws.merge_cells('A1:C1')
    
    headers = ["Rang", "Feature", "Importance"]
    for col, h in enumerate(headers, 1):
        cell = ws.cell(row=3, column=col, value=h)
        cell.fill = HEADER_FILL
        cell.font = HEADER_FONT
        cell.border = BORDER
        cell.alignment = Alignment(horizontal='center')
    
    for idx, (_, row) in enumerate(feature_importance.iterrows(), 4):
        ws.cell(row=idx, column=1, value=idx-3).border = BORDER
        ws.cell(row=idx, column=2, value=row['feature']).border = BORDER
        cell = ws.cell(row=idx, column=3, value=row['importance'])
        cell.border = BORDER
        cell.number_format = '0.0000'
    
    # Data bar pour visualisation
    rule = DataBarRule(
        start_type='min', end_type='max',
        color="00D4FF", showValue=True, minLength=None, maxLength=None
    )
    ws.conditional_formatting.add(f'C4:C{len(feature_importance)+3}', rule)
    
    ws.column_dimensions['A'].width = 8
    ws.column_dimensions['B'].width = 40
    ws.column_dimensions['C'].width = 15
    
    wb.save(output_path)
    logger.info(f"✓ Feature Importance Excel: {output_path}")


# =============================================================================
# FONCTIONS UTILITAIRES
# =============================================================================

def calculate_metrics(y_true, y_pred, y_proba=None):
    """Calcule les métriques."""
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
    
    return metrics


def calculate_metrics_by_category(df_test, y_true, y_pred, cat_cols):
    """Calcule les métriques par catégorie."""
    metrics_by_cat = {}
    
    for cat_col in cat_cols:
        if cat_col not in df_test.columns:
            continue
        
        metrics_by_cat[cat_col] = {}
        
        for group in df_test[cat_col].unique():
            mask = df_test[cat_col] == group
            if mask.sum() < 5:
                continue
            
            g_y_true = y_true[mask]
            g_y_pred = y_pred[mask]
            
            metrics_by_cat[cat_col][group] = {
                'n': int(mask.sum()),
                'taux_fondees': float(g_y_true.mean()),
                'accuracy': float(accuracy_score(g_y_true, g_y_pred)),
                'precision': float(precision_score(g_y_true, g_y_pred, zero_division=0)),
                'recall': float(recall_score(g_y_true, g_y_pred, zero_division=0)),
                'f1': float(f1_score(g_y_true, g_y_pred, zero_division=0))
            }
    
    return metrics_by_cat


def optimize_thresholds(y_true, y_proba):
    """Optimise les seuils."""
    best_result = None
    best_score = -1
    
    for t_low in np.arange(0.10, 0.45, 0.02):
        for t_high in np.arange(0.55, 0.90, 0.02):
            if t_high <= t_low:
                continue
            
            mask_rej = y_proba <= t_low
            mask_val = y_proba >= t_high
            
            n_rej, n_val = mask_rej.sum(), mask_val.sum()
            if n_rej == 0 or n_val == 0:
                continue
            
            prec_rej = (y_true[mask_rej] == 0).mean()
            prec_val = (y_true[mask_val] == 1).mean()
            automation = (n_rej + n_val) / len(y_true)
            
            if prec_rej >= 0.93 and prec_val >= 0.90:
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
        t_low, t_high = 0.3, 0.7
        mask_rej, mask_val = y_proba <= t_low, y_proba >= t_high
        best_result = {
            'threshold_low': t_low,
            'threshold_high': t_high,
            'precision_rejection': float((y_true[mask_rej] == 0).mean()) if mask_rej.sum() > 0 else 0,
            'precision_validation': float((y_true[mask_val] == 1).mean()) if mask_val.sum() > 0 else 0,
            'automation_rate': (mask_rej.sum() + mask_val.sum()) / len(y_true),
            'n_rejection': int(mask_rej.sum()),
            'n_validation': int(mask_val.sum()),
            'n_audit': int(len(y_true) - mask_rej.sum() - mask_val.sum())
        }
    
    return best_result


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Génère dashboard et analyses')
    parser.add_argument('--data', '-d', required=True, help='Fichier Excel')
    parser.add_argument('--output', '-o', default='outputs', help='Répertoire sortie')
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 60)
    logger.info("GÉNÉRATION DASHBOARD & ANALYSES")
    logger.info("=" * 60)
    
    # 1. Charger
    logger.info(f"\n📂 Chargement: {args.data}")
    df = pd.read_excel(args.data)
    logger.info(f"   {df.shape[0]} lignes, {df.shape[1]} colonnes")
    
    # 2. Préparer
    target_col = detect_target(df)
    cat_cols = detect_category_columns(df, target_col)
    logger.info(f"   Cible: {target_col}")
    logger.info(f"   Catégories: {cat_cols}")
    
    y = prepare_target(df, target_col)
    X, feature_names = prepare_features(df, target_col)
    
    # 3. Split
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, df.index, test_size=0.2, stratify=y, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.15, stratify=y_train, random_state=42
    )
    
    logger.info(f"\n📊 Split: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
    
    # 4. Entraîner
    logger.info("\n🚀 Entraînement XGBoost...")
    model = train_xgboost(X_train, y_train, X_val, y_val)
    
    # 5. Prédire
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # 6. Métriques
    logger.info("\n📈 Calcul des métriques...")
    metrics = calculate_metrics(y_test.values, y_pred, y_proba)
    thresholds = optimize_thresholds(y_test.values, y_proba)
    
    df_test = df.iloc[idx_test].reset_index(drop=True)
    y_test_reset = y_test.reset_index(drop=True)
    
    metrics_by_cat = calculate_metrics_by_category(df_test, y_test_reset.values, y_pred, cat_cols)
    
    # 7. Feature Importance
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False).reset_index(drop=True)
    
    logger.info(f"\n   F1-Score: {metrics['f1_weighted']:.4f}")
    logger.info(f"   AUC-ROC: {metrics['roc_auc']:.4f}")
    
    # 8. Générer les livrables
    logger.info("\n📝 Génération des livrables...")
    
    # Excel erreurs
    fp_count, fn_count = generate_excel_errors(
        df_test, y_test_reset.values, y_pred, y_proba,
        output_dir / "errors_analysis.xlsx"
    )
    
    # Excel feature importance
    generate_excel_feature_importance(
        feature_importance,
        output_dir / "feature_importance.xlsx"
    )
    
    # Dashboard HTML
    generate_html_dashboard(
        metrics, metrics_by_cat, feature_importance, thresholds,
        fp_count, fn_count,
        output_dir / "dashboard.html"
    )
    
    logger.info("\n" + "=" * 60)
    logger.info("✅ GÉNÉRATION TERMINÉE")
    logger.info("=" * 60)
    logger.info(f"\n📁 Fichiers créés dans: {output_dir}")
    logger.info("   • dashboard.html - Dashboard interactif")
    logger.info("   • feature_importance.xlsx - Importance des features")
    logger.info("   • errors_analysis.xlsx - Faux positifs & négatifs")


if __name__ == "__main__":
    main()
