"""
Predictor for complaint classification.
"""
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, Any
import logging

from .model_manager import ModelManager
from .business_rules import BusinessRulesEngine

logger = logging.getLogger(__name__)


class Predictor:
    """
    Main predictor for complaint classification.

    Features:
    - Model loading
    - Preprocessing
    - Prediction with probabilities
    - Decision classification (3 zones)
    - Business rules application
    """

    def __init__(
        self,
        config: Dict[str, Any],
        model_name: str = "best",
        version: Optional[str] = None
    ):
        """
        Initialize predictor.

        Args:
            config: Configuration dictionary
            model_name: Model name to load
            version: Model version (optional)
        """
        self.config = config
        self.model_manager = ModelManager(
            models_dir=config.get('paths', {}).get('models_dir', 'production/models')
        )

        # Load model
        self.model, self.preprocessor, self.thresholds = self.model_manager.load_model(
            model_name=model_name,
            version=version
        )

        # Initialize business rules engine
        self.business_rules = BusinessRulesEngine(config)

    def predict(
        self,
        df: pd.DataFrame,
        apply_business_rules: bool = True
    ) -> pd.DataFrame:
        """
        Make predictions on new data.

        Args:
            df: Input DataFrame
            apply_business_rules: Whether to apply business rules

        Returns:
            DataFrame with predictions
        """
        logger.info("="*80)
        logger.info("🎯 MAKING PREDICTIONS")
        logger.info("="*80)

        # Validate input
        self._validate_input(df)

        # Preprocess
        logger.info("\n⚙️ Preprocessing...")
        X_processed = self.preprocessor.transform(df)
        logger.info(f"✅ Preprocessing done: shape {X_processed.shape}")

        # Predict probabilities
        logger.info("\n🎯 Computing probabilities...")
        y_prob = self.model.predict_proba(X_processed)[:, 1]

        # Display probability distribution
        self._log_probability_distribution(y_prob)

        # Create decisions
        decisions, decisions_code = self._create_decisions(y_prob)

        # Create results DataFrame
        df_results = df.copy()
        df_results['Probabilite_Fondee'] = y_prob
        df_results['Decision_Modele'] = decisions
        df_results['Decision_Code'] = decisions_code

        # Log decision statistics
        self._log_decision_statistics(df_results)

        # Apply business rules if requested
        if apply_business_rules:
            df_before = df_results.copy()
            df_results = self.business_rules.apply_rules(df_results)

            # Log rules impact
            summary = self.business_rules.get_rules_summary(df_before, df_results)
            if summary['total_changes'] > 0:
                logger.info(f"\n📊 Business Rules Impact:")
                logger.info(f"   Validations: {summary['validations_before']} → {summary['validations_after']}")
                logger.info(f"   Audits: {summary['audits_before']} → {summary['audits_after']}")

        return df_results

    def predict_single(
        self,
        data: Dict[str, Any],
        apply_business_rules: bool = False
    ) -> Dict[str, Any]:
        """
        Make prediction for a single complaint.

        Args:
            data: Dictionary with complaint data
            apply_business_rules: Whether to apply business rules

        Returns:
            Dictionary with prediction results
        """
        # Convert to DataFrame
        df = pd.DataFrame([data])

        # Predict
        df_results = self.predict(df, apply_business_rules=apply_business_rules)

        # Convert to dictionary
        result = df_results.iloc[0].to_dict()

        return result

    def _validate_input(self, df: pd.DataFrame) -> None:
        """Validate input DataFrame."""
        required_cols = ['Montant demandé', 'Famille Produit']

        missing_cols = [col for col in required_cols if col not in df.columns]

        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

    def _create_decisions(
        self,
        y_prob: np.ndarray
    ) -> Tuple[list, list]:
        """
        Create decisions based on probabilities and thresholds.

        Args:
            y_prob: Predicted probabilities

        Returns:
            Tuple of (decisions_list, decisions_code_list)
        """
        threshold_low = self.thresholds.get('threshold_low', 0.30)
        threshold_high = self.thresholds.get('threshold_high', 0.70)

        decisions = []
        decisions_code = []

        for prob in y_prob:
            if prob <= threshold_low:
                decisions.append('Rejet Auto')
                decisions_code.append(-1)
            elif prob >= threshold_high:
                decisions.append('Validation Auto')
                decisions_code.append(1)
            else:
                decisions.append('Audit Humain')
                decisions_code.append(0)

        return decisions, decisions_code

    def _log_probability_distribution(self, y_prob: np.ndarray) -> None:
        """Log probability distribution statistics."""
        logger.info(f"\n📊 Probability distribution:")
        logger.info(f"   Min       : {y_prob.min():.4f}")
        logger.info(f"   25th pct  : {np.percentile(y_prob, 25):.4f}")
        logger.info(f"   Median    : {np.median(y_prob):.4f}")
        logger.info(f"   75th pct  : {np.percentile(y_prob, 75):.4f}")
        logger.info(f"   Max       : {y_prob.max():.4f}")
        logger.info(f"   Mean      : {y_prob.mean():.4f}")

        threshold_low = self.thresholds.get('threshold_low', 0.30)
        threshold_high = self.thresholds.get('threshold_high', 0.70)

        logger.info(f"\n🎯 Thresholds:")
        logger.info(f"   Low  : {threshold_low:.4f}")
        logger.info(f"   High : {threshold_high:.4f}")

    def _log_decision_statistics(self, df: pd.DataFrame) -> None:
        """Log decision statistics."""
        n_total = len(df)
        n_rejet = (df['Decision_Modele'] == 'Rejet Auto').sum()
        n_audit = (df['Decision_Modele'] == 'Audit Humain').sum()
        n_validation = (df['Decision_Modele'] == 'Validation Auto').sum()

        logger.info(f"\n📊 Decision distribution:")
        logger.info(f"   Reject Auto    : {n_rejet:6d} ({100*n_rejet/n_total:5.1f}%)")
        logger.info(f"   Human Audit    : {n_audit:6d} ({100*n_audit/n_total:5.1f}%)")
        logger.info(f"   Validate Auto  : {n_validation:6d} ({100*n_validation/n_total:5.1f}%)")
        logger.info(f"   Total          : {n_total:6d}")

        taux_auto = 100 * (n_rejet + n_validation) / n_total
        logger.info(f"\n✅ Automation rate: {taux_auto:.1f}%")

    @property
    def model_info(self) -> Dict[str, Any]:
        """Get information about loaded model."""
        return self.model_manager.current_model_info
