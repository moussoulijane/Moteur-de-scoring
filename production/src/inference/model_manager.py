"""
Model management with versioning.
"""
import joblib
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
from datetime import datetime
import json
import logging

logger = logging.getLogger(__name__)


class ModelManager:
    """
    Manage model loading, saving, and versioning.

    Features:
    - Model loading with versioning
    - Metadata tracking
    - Version history
    """

    def __init__(self, models_dir: str = "production/models"):
        """
        Initialize model manager.

        Args:
            models_dir: Directory containing models
        """
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)

        self.current_model = None
        self.current_preprocessor = None
        self.current_thresholds = None
        self.current_metadata = {}

    def load_model(
        self,
        model_name: str = "best",
        version: Optional[str] = None
    ) -> Tuple[Any, Any, Dict[str, float]]:
        """
        Load model, preprocessor, and thresholds.

        Args:
            model_name: Model name ('best', 'xgboost', 'catboost')
            version: Model version (directory name). If None, uses latest.

        Returns:
            Tuple of (model, preprocessor, thresholds_dict)
        """
        logger.info("="*80)
        logger.info("📂 LOADING MODEL")
        logger.info("="*80)

        # Determine model directory
        if version:
            model_dir = self.models_dir / version
        else:
            model_dir = self.models_dir

        if not model_dir.exists():
            raise FileNotFoundError(f"Model directory not found: {model_dir}")

        # Determine model file
        if model_name == "best":
            model_path = model_dir / "best_model.pkl"
            # Read actual model name from info file
            info_path = model_dir / "model_info.txt"
            if info_path.exists():
                with open(info_path, 'r') as f:
                    first_line = f.readline()
                    if first_line.startswith("Best model:"):
                        actual_name = first_line.split(': ')[1].strip()
                        logger.info(f"   Best model is: {actual_name.upper()}")
        else:
            model_path = model_dir / f"{model_name}_model.pkl"

        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        # Load model
        model = joblib.load(model_path)
        logger.info(f"✅ Model loaded: {model_path.name}")

        # Load preprocessor
        preprocessor_path = model_dir / "preprocessor.pkl"
        if not preprocessor_path.exists():
            raise FileNotFoundError(f"Preprocessor not found: {preprocessor_path}")

        preprocessor = joblib.load(preprocessor_path)
        logger.info(f"✅ Preprocessor loaded")

        # Display preprocessor info
        info = preprocessor.get_feature_info()
        logger.info(f"\n📊 Preprocessor info:")
        logger.info(f"   Features: {info['n_features']}")
        logger.info(f"   Families with stats: {info['family_stats_count']}")

        # Load thresholds
        thresholds_path = model_dir / "thresholds.pkl"
        if thresholds_path.exists():
            all_thresholds = joblib.load(thresholds_path)
            # Get thresholds for this model
            if model_name == "best":
                # Use first available model's thresholds
                thresholds_dict = next(iter(all_thresholds.values()))
            elif model_name in all_thresholds:
                thresholds_dict = all_thresholds[model_name]
            else:
                thresholds_dict = {}
        else:
            thresholds_dict = {}

        if thresholds_dict:
            logger.info(f"\n✅ Thresholds loaded:")
            logger.info(f"   Low  : {thresholds_dict.get('threshold_low', 0.30):.4f}")
            logger.info(f"   High : {thresholds_dict.get('threshold_high', 0.70):.4f}")
        else:
            logger.warning("⚠️ Thresholds not found, using defaults")
            thresholds_dict = {'threshold_low': 0.30, 'threshold_high': 0.70}

        # Store current model
        self.current_model = model
        self.current_preprocessor = preprocessor
        self.current_thresholds = thresholds_dict
        self.current_metadata = {
            'model_name': model_name,
            'version': version,
            'loaded_at': datetime.now().isoformat()
        }

        return model, preprocessor, thresholds_dict

    def save_model_version(
        self,
        model: Any,
        preprocessor: Any,
        thresholds: Dict[str, Any],
        version_name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Path:
        """
        Save model as a new version.

        Args:
            model: Trained model
            preprocessor: Fitted preprocessor
            thresholds: Threshold dictionary
            version_name: Version name (default: timestamp)
            metadata: Additional metadata

        Returns:
            Path to version directory
        """
        # Create version name
        if version_name is None:
            version_name = f"v_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        version_dir = self.models_dir / version_name
        version_dir.mkdir(parents=True, exist_ok=True)

        # Save model
        model_path = version_dir / "best_model.pkl"
        joblib.dump(model, model_path)
        logger.info(f"✅ Model saved: {model_path}")

        # Save preprocessor
        preprocessor_path = version_dir / "preprocessor.pkl"
        joblib.dump(preprocessor, preprocessor_path)
        logger.info(f"✅ Preprocessor saved: {preprocessor_path}")

        # Save thresholds
        thresholds_path = version_dir / "thresholds.pkl"
        joblib.dump({'best': thresholds}, thresholds_path)
        logger.info(f"✅ Thresholds saved: {thresholds_path}")

        # Save metadata
        if metadata:
            metadata_path = version_dir / "metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            logger.info(f"✅ Metadata saved: {metadata_path}")

        logger.info(f"\n📂 Model version saved: {version_dir}")

        return version_dir

    def list_versions(self) -> list:
        """
        List all available model versions.

        Returns:
            List of version directory names
        """
        versions = []
        for path in self.models_dir.iterdir():
            if path.is_dir() and (path / "best_model.pkl").exists():
                versions.append(path.name)

        return sorted(versions, reverse=True)

    def get_version_info(self, version: str) -> Dict[str, Any]:
        """
        Get information about a specific version.

        Args:
            version: Version name

        Returns:
            Dictionary with version information
        """
        version_dir = self.models_dir / version

        if not version_dir.exists():
            raise ValueError(f"Version not found: {version}")

        info = {
            'version': version,
            'path': str(version_dir),
            'created': datetime.fromtimestamp(
                version_dir.stat().st_mtime
            ).isoformat()
        }

        # Load metadata if available
        metadata_path = version_dir / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                info['metadata'] = json.load(f)

        # Check files
        info['files'] = {
            'model': (version_dir / "best_model.pkl").exists(),
            'preprocessor': (version_dir / "preprocessor.pkl").exists(),
            'thresholds': (version_dir / "thresholds.pkl").exists(),
            'metadata': metadata_path.exists()
        }

        return info

    @property
    def current_model_info(self) -> Dict[str, Any]:
        """Get information about currently loaded model."""
        return self.current_metadata.copy()
