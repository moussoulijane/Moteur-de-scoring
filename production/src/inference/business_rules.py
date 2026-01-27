"""
Business rules engine for complaint processing.
"""
import pandas as pd
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


class BusinessRulesEngine:
    """
    Apply business rules to model predictions.

    Rules:
    1. Maximum 1 automatic validation per client per year
    2. Validated amount must be ≤ PNB from last year
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize business rules engine.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.rules_config = config.get('business_rules', {})

    def apply_rules(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply all business rules to predictions.

        Args:
            df: DataFrame with predictions (must have 'Decision_Modele' column)

        Returns:
            DataFrame with rules applied
        """
        logger.info("="*80)
        logger.info("📋 APPLYING BUSINESS RULES")
        logger.info("="*80)

        if 'Decision_Modele' not in df.columns:
            raise ValueError("DataFrame must have 'Decision_Modele' column")

        df_result = df.copy()

        # Initialize audit reason column
        df_result['Raison_Audit'] = ''

        # Apply rules
        df_result = self._apply_max_validations_rule(df_result)
        df_result = self._apply_amount_vs_pnb_rule(df_result)

        return df_result

    def _apply_max_validations_rule(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Rule #1: Maximum 1 automatic validation per client per year.

        Args:
            df: DataFrame with predictions

        Returns:
            DataFrame with rule applied
        """
        max_validations = self.rules_config.get('max_validations_per_client_per_year', 1)

        if max_validations <= 0:
            logger.info("Rule #1 disabled (max_validations <= 0)")
            return df

        logger.info(f"\n🔍 Rule #1: Max {max_validations} validation(s) per client per year")

        if 'Date de Qualification' not in df.columns:
            logger.warning("   ⚠️ 'Date de Qualification' column missing, rule skipped")
            return df

        # Check for client identifier
        client_col = None
        for col_name in ['Identifiant client', 'ID Client', 'Client ID']:
            if col_name in df.columns:
                client_col = col_name
                break

        if not client_col:
            logger.warning("   ⚠️ Client identifier column missing, rule skipped")
            return df

        df_work = df.copy()

        # Convert dates
        df_work['Date de Qualification'] = pd.to_datetime(
            df_work['Date de Qualification'],
            errors='coerce'
        )

        # Extract year
        df_work['annee'] = df_work['Date de Qualification'].dt.year

        # Sort by date to keep first validation
        df_work = df_work.sort_values(
            ['annee', client_col, 'Date de Qualification']
        )

        # Count validations per client/year
        df_work['validation_number'] = df_work.groupby(
            [client_col, 'annee']
        ).cumcount() + 1

        # Apply rule: convert excess validations to audit
        mask_rule1 = (
            (df_work['Decision_Modele'] == 'Validation Auto')
            & (df_work['validation_number'] > max_validations)
        )

        n_changed = mask_rule1.sum()

        if n_changed > 0:
            df_work.loc[mask_rule1, 'Decision_Modele'] = 'Audit Humain'
            df_work.loc[mask_rule1, 'Decision_Code'] = 0
            df_work.loc[mask_rule1, 'Raison_Audit'] = (
                f'Rule #1: >{max_validations} validation(s)/client/year'
            )
            logger.info(f"   ✅ {n_changed} validations converted to Audit")
        else:
            logger.info("   ✅ No conversion needed")

        # Remove temporary columns
        df_work = df_work.drop(['annee', 'validation_number'], axis=1, errors='ignore')

        return df_work

    def _apply_amount_vs_pnb_rule(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Rule #2: Validated amount must be ≤ PNB from last year.

        Args:
            df: DataFrame with predictions

        Returns:
            DataFrame with rule applied
        """
        if not self.rules_config.get('check_amount_vs_pnb', True):
            logger.info("\nRule #2 disabled")
            return df

        logger.info("\n🔍 Rule #2: Validated amount ≤ PNB from last year")

        # Check required columns
        required_cols = [
            'Montant demandé',
            'PNB analytique (vision commerciale) cumulé'
        ]

        if not all(col in df.columns for col in required_cols):
            logger.warning(
                f"   ⚠️ Required columns missing ({required_cols}), rule skipped"
            )
            return df

        df_work = df.copy()

        # Apply rule: amount > PNB → convert to audit
        mask_rule2 = (
            (df_work['Decision_Modele'] == 'Validation Auto')
            & (df_work['Montant demandé'] > df_work['PNB analytique (vision commerciale) cumulé'])
            & (df_work['PNB analytique (vision commerciale) cumulé'] > 0)
        )

        n_changed = mask_rule2.sum()

        if n_changed > 0:
            df_work.loc[mask_rule2, 'Decision_Modele'] = 'Audit Humain'
            df_work.loc[mask_rule2, 'Decision_Code'] = 0

            # Add reason (may append to existing reason)
            existing_reason = df_work.loc[mask_rule2, 'Raison_Audit']
            df_work.loc[mask_rule2, 'Raison_Audit'] = existing_reason.apply(
                lambda x: (x + ' | ' if x else '') + 'Rule #2: Amount > PNB'
            )

            logger.info(f"   ✅ {n_changed} validations converted to Audit")
        else:
            logger.info("   ✅ No conversion needed")

        return df_work

    def get_rules_summary(self, df_before: pd.DataFrame, df_after: pd.DataFrame) -> Dict[str, int]:
        """
        Get summary of rules impact.

        Args:
            df_before: DataFrame before rules
            df_after: DataFrame after rules

        Returns:
            Dictionary with impact counts
        """
        # Count changes
        changes = (
            (df_before['Decision_Modele'] == 'Validation Auto')
            & (df_after['Decision_Modele'] == 'Audit Humain')
        )

        return {
            'total_changes': changes.sum(),
            'validations_before': (df_before['Decision_Modele'] == 'Validation Auto').sum(),
            'validations_after': (df_after['Decision_Modele'] == 'Validation Auto').sum(),
            'audits_before': (df_before['Decision_Modele'] == 'Audit Humain').sum(),
            'audits_after': (df_after['Decision_Modele'] == 'Audit Humain').sum()
        }
