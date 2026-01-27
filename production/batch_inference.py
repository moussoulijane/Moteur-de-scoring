"""
Batch inference script for Excel files.

Usage:
    python batch_inference.py --input data/complaints.xlsx --output results.xlsx
"""
import sys
sys.path.append('src')

import pandas as pd
import argparse
from pathlib import Path
from datetime import datetime
import logging

from config import Config
from src.inference import Predictor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description='Batch inference on Excel file'
    )
    parser.add_argument(
        '--input',
        '-i',
        type=str,
        required=True,
        help='Input Excel file path'
    )
    parser.add_argument(
        '--output',
        '-o',
        type=str,
        help='Output Excel file path (optional, will auto-generate if not provided)'
    )
    parser.add_argument(
        '--model',
        '-m',
        type=str,
        default='best',
        choices=['best', 'xgboost', 'catboost'],
        help='Model to use (default: best)'
    )
    parser.add_argument(
        '--version',
        '-v',
        type=str,
        help='Model version (optional, uses latest if not provided)'
    )
    parser.add_argument(
        '--apply-rules',
        '-r',
        action='store_true',
        help='Apply business rules'
    )
    parser.add_argument(
        '--config',
        '-c',
        type=str,
        help='Path to config file (optional)'
    )

    args = parser.parse_args()

    # Load configuration
    if args.config:
        config = Config(args.config)
    else:
        config = Config()

    logger.info("="*80)
    logger.info("🚀 BATCH INFERENCE - EXCEL MODE")
    logger.info("="*80)

    # Validate input file
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"❌ Input file not found: {input_path}")
        sys.exit(1)

    # Load data
    logger.info(f"\n📂 Loading data from: {input_path}")
    try:
        df = pd.read_excel(input_path)
        logger.info(f"✅ Loaded {len(df)} complaints")
    except Exception as e:
        logger.error(f"❌ Error loading Excel file: {e}")
        sys.exit(1)

    # Initialize predictor
    try:
        predictor = Predictor(
            config=config._config,
            model_name=args.model,
            version=args.version
        )
    except Exception as e:
        logger.error(f"❌ Error loading model: {e}")
        sys.exit(1)

    # Make predictions
    try:
        df_results = predictor.predict(
            df,
            apply_business_rules=args.apply_rules
        )
    except Exception as e:
        logger.error(f"❌ Error during prediction: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Prepare output file
    if args.output:
        output_path = Path(args.output)
    else:
        # Auto-generate output filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = input_path.parent / f"{input_path.stem}_predictions_{timestamp}.xlsx"

    # Save results
    logger.info("="*80)
    logger.info("💾 SAVING RESULTS")
    logger.info("="*80)

    try:
        df_results.to_excel(output_path, index=False, engine='openpyxl')
        logger.info(f"✅ Results saved to: {output_path}")
        logger.info(f"   Rows: {len(df_results)}")
        logger.info(f"   Columns added:")
        logger.info(f"   - Probabilite_Fondee  : Predicted probability [0-1]")
        logger.info(f"   - Decision_Modele     : Rejet Auto / Audit Humain / Validation Auto")
        logger.info(f"   - Decision_Code       : -1 (Reject) / 0 (Audit) / 1 (Validate)")
        if args.apply_rules:
            logger.info(f"   - Raison_Audit        : Reason for audit (if applicable)")
    except Exception as e:
        logger.error(f"❌ Error saving results: {e}")
        sys.exit(1)

    # Generate summary report
    logger.info("="*80)
    logger.info("📊 SUMMARY")
    logger.info("="*80)

    summary_stats = {
        'Total complaints': len(df_results),
        'Rejet Auto': (df_results['Decision_Modele'] == 'Rejet Auto').sum(),
        'Audit Humain': (df_results['Decision_Modele'] == 'Audit Humain').sum(),
        'Validation Auto': (df_results['Decision_Modele'] == 'Validation Auto').sum()
    }

    for key, value in summary_stats.items():
        if key == 'Total complaints':
            logger.info(f"{key}: {value}")
        else:
            pct = 100 * value / summary_stats['Total complaints']
            logger.info(f"{key}: {value} ({pct:.1f}%)")

    auto_rate = 100 * (
        summary_stats['Rejet Auto'] + summary_stats['Validation Auto']
    ) / summary_stats['Total complaints']
    logger.info(f"\nAutomation rate: {auto_rate:.1f}%")

    # Save summary report
    report_path = output_path.parent / f"summary_{output_path.stem}.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("BATCH INFERENCE SUMMARY\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*80 + "\n\n")

        f.write(f"Input file: {input_path}\n")
        f.write(f"Output file: {output_path}\n")
        f.write(f"Model: {args.model}\n")
        if args.version:
            f.write(f"Version: {args.version}\n")
        f.write(f"Business rules applied: {args.apply_rules}\n\n")

        f.write("STATISTICS:\n")
        f.write("-"*80 + "\n")
        for key, value in summary_stats.items():
            if key == 'Total complaints':
                f.write(f"{key}: {value}\n")
            else:
                pct = 100 * value / summary_stats['Total complaints']
                f.write(f"{key}: {value} ({pct:.1f}%)\n")

        f.write(f"\nAutomation rate: {auto_rate:.1f}%\n")

    logger.info(f"\n✅ Summary report saved to: {report_path}")

    logger.info("\n" + "="*80)
    logger.info("✅ BATCH INFERENCE COMPLETED")
    logger.info("="*80)


if __name__ == '__main__':
    main()
