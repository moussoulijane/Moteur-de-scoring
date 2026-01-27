"""
Main training script for production model.

Usage:
    python train_model.py --train data/reclamations_2024.xlsx --test data/reclamations_2025.xlsx
"""
import sys
sys.path.append('src')

import argparse
from pathlib import Path
import logging

from config import Config
from src.training import ModelTrainer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description='Train production model'
    )
    parser.add_argument(
        '--train',
        '-t',
        type=str,
        required=True,
        help='Training data Excel file'
    )
    parser.add_argument(
        '--test',
        '-e',
        type=str,
        help='Test data Excel file (optional)'
    )
    parser.add_argument(
        '--output',
        '-o',
        type=str,
        help='Output directory for models (optional)'
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
    logger.info("🚀 PRODUCTION MODEL TRAINING")
    logger.info("="*80)

    # Validate training file
    train_path = Path(args.train)
    if not train_path.exists():
        logger.error(f"❌ Training file not found: {train_path}")
        sys.exit(1)

    # Validate test file if provided
    test_path = None
    if args.test:
        test_path = Path(args.test)
        if not test_path.exists():
            logger.error(f"❌ Test file not found: {test_path}")
            sys.exit(1)

    # Initialize trainer
    trainer = ModelTrainer(config._config)

    # Run training pipeline
    try:
        trainer.run(
            train_file=str(train_path),
            test_file=str(test_path) if test_path else None,
            output_dir=args.output
        )

        logger.info("\n" + "="*80)
        logger.info("✅ TRAINING COMPLETED SUCCESSFULLY")
        logger.info("="*80)

        if trainer.results:
            best_model = max(
                trainer.results.items(),
                key=lambda x: x[1]['gain_net']
            )
            logger.info(f"\n🏆 Best model: {best_model[0].upper()}")
            logger.info(f"   Net Gain: {best_model[1]['gain_net']:,.0f} DH")
            logger.info(f"   F1-Score: {best_model[1]['f1']:.4f}")
            logger.info(f"   ROC-AUC: {best_model[1]['auc']:.4f}")

    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
