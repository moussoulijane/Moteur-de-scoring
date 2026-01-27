"""
Threshold optimizer for decision making.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
import logging

logger = logging.getLogger(__name__)


class ThresholdOptimizer:
    """
    Optimize decision thresholds for 3-zone classification.

    Zones:
    - Zone 1: prob ≤ threshold_low → AUTO-REJECT
    - Zone 2: threshold_low < prob < threshold_high → AUDIT
    - Zone 3: prob ≥ threshold_high → AUTO-VALIDATE
    """

    def __init__(self, config: Dict[str, Any], y_test: np.ndarray, df_test: pd.DataFrame):
        """
        Initialize optimizer.

        Args:
            config: Configuration dictionary
            y_test: True labels
            df_test: Test DataFrame with amounts
        """
        self.config = config
        self.y_test = y_test
        self.df_test = df_test

        # Get price per unit
        self.prix_unitaire = config.get('metrics', {}).get('prix_unitaire_dh', 169)

        # Get threshold optimization config
        opt_config = config.get('thresholds', {}).get('optimization', {})
        self.threshold_low_range = opt_config.get('threshold_low_range', [0.05, 0.50])
        self.threshold_high_range = opt_config.get('threshold_high_range', [0.50, 0.98])
        self.step = opt_config.get('step', 0.02)
        self.min_prec_reject = opt_config.get('min_precision_reject', 0.93)
        self.min_prec_validate = opt_config.get('min_precision_validate', 0.90)
        self.min_prec_reject_relaxed = opt_config.get('min_precision_reject_relaxed', 0.90)
        self.min_prec_validate_relaxed = opt_config.get('min_precision_validate_relaxed', 0.87)

    def optimize_thresholds(
        self,
        y_prob: np.ndarray
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Optimize thresholds to maximize net gain with high automation rate.

        Args:
            y_prob: Predicted probabilities

        Returns:
            Tuple of (best_result, all_results)
        """
        logger.info("\n🎯 Optimizing decision thresholds (3 zones - high automation)...")

        montants = self.df_test['Montant demandé'].values
        best_result = None
        best_gain_net = -float('inf')
        best_result_relaxed = None
        best_score_relaxed = -float('inf')
        threshold_results = []

        # Test different threshold combinations
        for t_low in np.arange(
            self.threshold_low_range[0],
            self.threshold_low_range[1],
            self.step
        ):
            for t_high in np.arange(
                self.threshold_high_range[0],
                self.threshold_high_range[1],
                self.step
            ):
                if t_high <= t_low:
                    continue

                # Define 3 zones
                mask_rejet = y_prob <= t_low
                mask_audit = (y_prob > t_low) & (y_prob < t_high)
                mask_validation = y_prob >= t_high

                # Predictions
                y_pred = np.zeros(len(y_prob), dtype=int)
                y_pred[mask_validation] = 1

                # Automated cases
                mask_auto = mask_rejet | mask_validation

                if mask_auto.sum() == 0:
                    continue

                # Metrics
                y_pred_auto = y_pred[mask_auto]
                y_true_auto = self.y_test[mask_auto]
                montants_auto = montants[mask_auto]

                fp_mask = (y_true_auto == 0) & (y_pred_auto == 1)
                fn_mask = (y_true_auto == 1) & (y_pred_auto == 0)

                # Financial calculation
                montants_auto_clean = np.nan_to_num(
                    montants_auto, nan=0.0, posinf=0.0, neginf=0.0
                )
                if len(montants_auto_clean) > 0 and montants_auto_clean.max() > 0:
                    montants_auto_clean = np.clip(
                        montants_auto_clean,
                        0,
                        np.percentile(montants_auto_clean, 99)
                    )
                else:
                    montants_auto_clean = montants_auto_clean.clip(0)

                perte_fp = montants_auto_clean[fp_mask].sum()
                perte_fn = 2 * montants_auto_clean[fn_mask].sum()

                auto_count = mask_auto.sum()
                gain_brut = auto_count * self.prix_unitaire
                gain_net = gain_brut - perte_fp - perte_fn

                # Precision by zone
                prec_rejet = (
                    (self.y_test[mask_rejet] == 0).mean()
                    if mask_rejet.sum() > 0 else 0
                )
                prec_validation = (
                    (self.y_test[mask_validation] == 1).mean()
                    if mask_validation.sum() > 0 else 0
                )

                automation_rate = mask_auto.sum() / len(y_prob)

                threshold_results.append({
                    'threshold_low': t_low,
                    'threshold_high': t_high,
                    'gain_net': gain_net,
                    'auto': auto_count,
                    'fp': fp_mask.sum(),
                    'fn': fn_mask.sum(),
                    'n_rejet': mask_rejet.sum(),
                    'n_audit': mask_audit.sum(),
                    'n_validation': mask_validation.sum(),
                    'prec_rejet': prec_rejet,
                    'prec_validation': prec_validation,
                    'automation_rate': automation_rate
                })

                # Primary criterion: Maximize net gain with strict precision
                if (
                    prec_rejet >= self.min_prec_reject
                    and prec_validation >= self.min_prec_validate
                ):
                    if gain_net > best_gain_net:
                        best_gain_net = gain_net
                        best_result = threshold_results[-1].copy()

                # Secondary criterion: Favor automation with acceptable precision
                if (
                    prec_rejet >= self.min_prec_reject_relaxed
                    and prec_validation >= self.min_prec_validate_relaxed
                ):
                    # Composite score: net gain + automation bonus
                    score = gain_net + (automation_rate * 100000)
                    if score > best_score_relaxed:
                        best_score_relaxed = score
                        best_result_relaxed = threshold_results[-1].copy()

        # Choose best result
        if best_result is None and best_result_relaxed is not None:
            best_result = best_result_relaxed
            logger.info("   ⚠️ Using relaxed criteria to maximize automation")
        elif best_result is None and threshold_results:
            # Fallback: best compromise net gain / automation
            best_result = max(
                threshold_results,
                key=lambda x: x['gain_net'] + (x['automation_rate'] * 50000)
            )
            logger.info("   ⚠️ Using best compromise net gain / automation")

        if best_result:
            logger.info(f"   ✅ Low threshold (Reject): {best_result['threshold_low']:.2f}")
            logger.info(f"   ✅ High threshold (Validate): {best_result['threshold_high']:.2f}")
            logger.info(f"   ✅ Automation rate: {best_result['automation_rate']:.1%}")
            logger.info(f"   ✅ Reject precision: {best_result['prec_rejet']:.1%}")
            logger.info(f"   ✅ Validate precision: {best_result['prec_validation']:.1%}")
            logger.info(f"   ✅ Net Gain: {best_result['gain_net']:,.0f} DH")
        else:
            # Ultimate fallback
            default_config = self.config.get('thresholds', {}).get('default', {})
            best_result = {
                'threshold_low': default_config.get('low', 0.30),
                'threshold_high': default_config.get('high', 0.70),
                'gain_net': 0,
                'auto': 0,
                'fp': 0,
                'fn': 0,
                'n_rejet': 0,
                'n_audit': len(y_prob),
                'n_validation': 0,
                'prec_rejet': 0,
                'prec_validation': 0,
                'automation_rate': 0
            }
            logger.warning("   ⚠️ Using default thresholds")

        return best_result, threshold_results
