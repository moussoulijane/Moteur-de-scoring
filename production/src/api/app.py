"""
REST API for real-time complaint classification.

Endpoints:
    POST /predict - Single complaint prediction
    POST /predict/batch - Batch prediction (small batches)
    GET /health - Health check
    GET /model/info - Model information
"""
import sys
sys.path.append('..')

from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from datetime import datetime
from pathlib import Path
import logging

from config import Config
from src.inference import Predictor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Load configuration
config = Config()

# Configure CORS
if config.api.get('cors_enabled', True):
    CORS(app)

# Configure max content length
app.config['MAX_CONTENT_LENGTH'] = config.api.get(
    'max_content_length',
    16 * 1024 * 1024  # 16MB
)

# Initialize predictor (loaded once at startup)
try:
    predictor = Predictor(
        config=config._config,
        model_name=config.models.get('default_model', 'best')
    )
    logger.info("✅ Predictor initialized successfully")
except Exception as e:
    logger.error(f"❌ Error initializing predictor: {e}")
    predictor = None


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'model_loaded': predictor is not None
    })


@app.route('/model/info', methods=['GET'])
def model_info():
    """Get information about loaded model."""
    if predictor is None:
        return jsonify({
            'error': 'Model not loaded'
        }), 503

    try:
        info = predictor.model_info
        feature_info = predictor.preprocessor.get_feature_info()

        return jsonify({
            'model': info,
            'features': {
                'count': feature_info['n_features'],
                'families_with_stats': feature_info['family_stats_count'],
                'categories_with_stats': feature_info['category_stats_count']
            },
            'thresholds': predictor.thresholds
        })
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        return jsonify({
            'error': str(e)
        }), 500


@app.route('/predict', methods=['POST'])
def predict_single():
    """
    Predict single complaint.

    Request body:
    {
        "Montant demandé": 5000,
        "Famille Produit": "Cartes",
        "Délai estimé": 30,
        "Catégorie": "...",
        ...
    }

    Optional query parameters:
    - apply_rules: true/false (default: false)

    Response:
    {
        "Probabilite_Fondee": 0.75,
        "Decision_Modele": "Validation Auto",
        "Decision_Code": 1,
        "input_data": {...},
        "timestamp": "2024-01-15T10:30:00"
    }
    """
    if predictor is None:
        return jsonify({
            'error': 'Model not loaded'
        }), 503

    # Get data
    try:
        data = request.get_json()

        if not data:
            return jsonify({
                'error': 'No data provided'
            }), 400

    except Exception as e:
        return jsonify({
            'error': f'Invalid JSON: {str(e)}'
        }), 400

    # Get query parameters
    apply_rules = request.args.get('apply_rules', 'false').lower() == 'true'

    # Validate required fields
    required_fields = ['Montant demandé', 'Famille Produit']
    missing_fields = [f for f in required_fields if f not in data]

    if missing_fields:
        return jsonify({
            'error': f'Missing required fields: {missing_fields}'
        }), 400

    # Make prediction
    try:
        result = predictor.predict_single(
            data,
            apply_business_rules=apply_rules
        )

        # Clean result (convert numpy types to Python types)
        clean_result = {}
        for key, value in result.items():
            if hasattr(value, 'item'):  # numpy types
                clean_result[key] = value.item()
            elif pd.isna(value):  # NaN values
                clean_result[key] = None
            else:
                clean_result[key] = value

        response = {
            'prediction': {
                'Probabilite_Fondee': clean_result.get('Probabilite_Fondee'),
                'Decision_Modele': clean_result.get('Decision_Modele'),
                'Decision_Code': clean_result.get('Decision_Code')
            },
            'timestamp': datetime.now().isoformat()
        }

        if apply_rules and 'Raison_Audit' in clean_result:
            response['prediction']['Raison_Audit'] = clean_result['Raison_Audit']

        return jsonify(response)

    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'error': str(e)
        }), 500


@app.route('/predict/batch', methods=['POST'])
def predict_batch():
    """
    Predict batch of complaints (small batches only).

    Request body:
    {
        "complaints": [
            {"Montant demandé": 5000, "Famille Produit": "Cartes", ...},
            {"Montant demandé": 3000, "Famille Produit": "Comptes", ...}
        ]
    }

    Optional query parameters:
    - apply_rules: true/false (default: false)

    Response:
    {
        "predictions": [
            {"Probabilite_Fondee": 0.75, "Decision_Modele": "Validation Auto", ...},
            ...
        ],
        "summary": {
            "total": 2,
            "Rejet Auto": 0,
            "Audit Humain": 0,
            "Validation Auto": 2
        },
        "timestamp": "2024-01-15T10:30:00"
    }
    """
    if predictor is None:
        return jsonify({
            'error': 'Model not loaded'
        }), 503

    # Get data
    try:
        data = request.get_json()

        if not data or 'complaints' not in data:
            return jsonify({
                'error': 'Missing "complaints" field in request body'
            }), 400

        complaints = data['complaints']

        if not isinstance(complaints, list):
            return jsonify({
                'error': '"complaints" must be a list'
            }), 400

        if len(complaints) == 0:
            return jsonify({
                'error': 'Empty complaints list'
            }), 400

        if len(complaints) > 100:
            return jsonify({
                'error': 'Batch size too large (max 100). Use batch_inference.py for larger batches.'
            }), 400

    except Exception as e:
        return jsonify({
            'error': f'Invalid JSON: {str(e)}'
        }), 400

    # Get query parameters
    apply_rules = request.args.get('apply_rules', 'false').lower() == 'true'

    # Convert to DataFrame
    try:
        df = pd.DataFrame(complaints)
    except Exception as e:
        return jsonify({
            'error': f'Error creating DataFrame: {str(e)}'
        }), 400

    # Make predictions
    try:
        df_results = predictor.predict(
            df,
            apply_business_rules=apply_rules
        )

        # Convert to list of dictionaries
        predictions = []
        for _, row in df_results.iterrows():
            pred = {}
            for key, value in row.items():
                if hasattr(value, 'item'):  # numpy types
                    pred[key] = value.item()
                elif pd.isna(value):  # NaN values
                    pred[key] = None
                else:
                    pred[key] = value
            predictions.append(pred)

        # Compute summary
        summary = {
            'total': len(predictions),
            'Rejet Auto': (df_results['Decision_Modele'] == 'Rejet Auto').sum(),
            'Audit Humain': (df_results['Decision_Modele'] == 'Audit Humain').sum(),
            'Validation Auto': (df_results['Decision_Modele'] == 'Validation Auto').sum()
        }

        # Convert numpy int64 to Python int
        for key in ['total', 'Rejet Auto', 'Audit Humain', 'Validation Auto']:
            if hasattr(summary[key], 'item'):
                summary[key] = summary[key].item()

        return jsonify({
            'predictions': predictions,
            'summary': summary,
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        logger.error(f"Error during batch prediction: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'error': str(e)
        }), 500


@app.errorhandler(413)
def request_entity_too_large(error):
    """Handle request too large error."""
    return jsonify({
        'error': 'Request too large. Use batch_inference.py for large files.'
    }), 413


@app.errorhandler(404)
def not_found(error):
    """Handle not found error."""
    return jsonify({
        'error': 'Endpoint not found'
    }), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle internal server error."""
    logger.error(f"Internal server error: {error}")
    return jsonify({
        'error': 'Internal server error'
    }), 500


def main():
    """Run the API server."""
    host = config.api.get('host', '0.0.0.0')
    port = config.api.get('port', 5000)
    debug = config.api.get('debug', False)

    logger.info("="*80)
    logger.info("🚀 STARTING API SERVER")
    logger.info("="*80)
    logger.info(f"Host: {host}")
    logger.info(f"Port: {port}")
    logger.info(f"Debug: {debug}")
    logger.info("="*80)

    app.run(host=host, port=port, debug=debug)


if __name__ == '__main__':
    main()
