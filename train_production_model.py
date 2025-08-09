#!/usr/bin/env python3
"""
train_production_model.py - Production Model Training Script

This script configures and runs the Enhanced Model Trainer for production-grade
model training using the full l2_training_data_practical dataset.

Sprint 2 - Priority 1: Train Production ML Model
"""

import os
import sys
import logging
import time
from datetime import datetime
from typing import Dict, Any

# Configure logging for production training
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f'production_training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    ]
)
logger = logging.getLogger(__name__)


def get_production_training_config() -> Dict[str, Any]:
    """Get optimized configuration for production model training."""
    
    return {
        # Basic configuration
        'symbol': 'BTCUSDT',
        'db_path': './trading_bot.db',
        'base_dir': './trading_bot_data',
        
        # Training parameters - optimized for production
        'n_time_splits': 8,        # Increased from 5 for better cross-validation
        'test_size': 0.15,         # Reduced to use more data for training
        'optuna_trials': 200,      # Increased from 5 for thorough hyperparameter optimization
        
        # Feature engineering parameters
        'feature_engineering': {
            'use_advanced_features': True,
            'feature_selection': True,
            'max_features': 45,    # Use top 45 most important features
        },
        
        # Model parameters
        'model_config': {
            'enable_early_stopping': True,
            'early_stopping_rounds': 50,
            'max_depth_range': (4, 12),
            'num_leaves_range': (20, 200),
            'learning_rate_range': (0.01, 0.3),
        },
        
        # Target engineering
        'target_config': {
            'horizons': [10, 30, 50, 100, 300],  # Multiple prediction horizons
            'volatility_adjustment': True,
            'outlier_handling': 'clip',
            'clip_quantiles': (0.001, 0.999),
        },
        
        # Performance optimization
        'performance': {
            'n_jobs': -1,              # Use all CPU cores
            'verbose_optuna': True,
            'save_intermediate': True,  # Save models during training
        },
        
        # Validation settings
        'validation': {
            'walk_forward': True,
            'min_train_samples': 50000,  # Minimum samples per fold
            'validation_split': 0.2,
        }
    }


def validate_database_access():
    """Validate database access and data quality before training."""
    
    try:
        from modeltrainer_enhanced import EnhancedModelTrainer
        
        # Test configuration
        test_config = {
            'symbol': 'BTCUSDT',
            'db_path': './trading_bot.db',
            'base_dir': './trading_bot_data'
        }
        
        trainer = EnhancedModelTrainer(test_config)
        
        # Test database connection and data loading
        logger.info("Validating database access...")
        
        # Load small sample to test
        sample_data = trainer.load_training_data_from_db('BTCUSDT', limit=1000)
        
        if sample_data.empty:
            raise ValueError("No training data available in database")
        
        logger.info(f"✓ Database validation successful:")
        logger.info(f"  Sample data shape: {sample_data.shape}")
        logger.info(f"  Available columns: {len(sample_data.columns)}")
        logger.info(f"  Date range: {sample_data.index.min()} to {sample_data.index.max()}")
        
        # Check for target columns
        target_cols = [col for col in sample_data.columns if 'target' in col.lower()]
        logger.info(f"  Target columns found: {target_cols}")
        
        return True
        
    except Exception as e:
        logger.error(f"Database validation failed: {e}")
        return False


def run_production_training():
    """Execute production model training with full dataset."""
    
    logger.info("🚀 STARTING PRODUCTION MODEL TRAINING")
    logger.info("=" * 80)
    
    # Get production configuration
    config = get_production_training_config()
    
    logger.info("Production Training Configuration:")
    for key, value in config.items():
        if isinstance(value, dict):
            logger.info(f"  {key}:")
            for subkey, subvalue in value.items():
                logger.info(f"    {subkey}: {subvalue}")
        else:
            logger.info(f"  {key}: {value}")
    
    logger.info("=" * 80)
    
    try:
        # Validate database before starting
        if not validate_database_access():
            logger.error("Database validation failed - aborting training")
            return False
        
        # Import and initialize trainer
        from modeltrainer_enhanced import EnhancedModelTrainer
        
        trainer = EnhancedModelTrainer(config)
        
        # Record training start time
        training_start = time.time()
        logger.info(f"Training started at: {datetime.now()}")
        
        # Execute training on FULL dataset (no limit)
        logger.info("🎯 TRAINING ON FULL DATASET - NO LIMITS")
        logger.info("This may take 30-60 minutes depending on system performance...")
        
        ensemble_result = trainer.train_from_database(
            symbol='BTCUSDT',
            limit=None  # 🔥 NO LIMIT - Use full 519k+ row dataset
        )
        
        # Calculate training duration
        training_duration = time.time() - training_start
        logger.info(f"Training completed in {training_duration:.2f} seconds ({training_duration/60:.1f} minutes)")
        
        # Log results
        logger.info("=" * 80)
        logger.info("🎉 PRODUCTION TRAINING COMPLETED SUCCESSFULLY!")
        logger.info("=" * 80)
        
        logger.info("Training Results:")
        logger.info(f"  Trained models for horizons: {list(ensemble_result['models'].keys())}")
        logger.info(f"  Ensemble weights: {ensemble_result['weights']}")
        
        if 'metrics' in ensemble_result:
            logger.info("  Validation Metrics:")
            for horizon, metrics in ensemble_result['metrics'].items():
                if isinstance(metrics, dict):
                    mae = metrics.get('mae', 'N/A')
                    correlation = metrics.get('correlation', 'N/A')
                    logger.info(f"    Horizon {horizon}: MAE={mae}, Correlation={correlation}")
        
        # Verify model files were saved
        model_path = trainer.ensemble_path
        if os.path.exists(model_path):
            model_size = os.path.getsize(model_path) / (1024 * 1024)  # MB
            logger.info(f"  Model saved: {model_path} ({model_size:.1f} MB)")
        else:
            logger.warning(f"  Model file not found: {model_path}")
        
        # Verify features file
        features_path = trainer.features_path
        if os.path.exists(features_path):
            logger.info(f"  Features saved: {features_path}")
        
        # Verify metrics file
        metrics_path = trainer.metrics_path
        if os.path.exists(metrics_path):
            logger.info(f"  Metrics saved: {metrics_path}")
        
        logger.info("=" * 80)
        logger.info("✅ PRODUCTION MODEL READY FOR DEPLOYMENT!")
        
        return True
        
    except Exception as e:
        logger.error(f"Production training failed: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False


def main():
    """Main function for production model training."""
    
    print("BTC Production Model Training")
    print("=" * 60)
    print("Sprint 2 - Priority 1: Train Production ML Model")
    print("Training on full l2_training_data_practical dataset")
    print("=" * 60)
    
    # Check if we have enough disk space (rough estimate)
    try:
        import shutil
        free_space_gb = shutil.disk_usage('.').free / (1024**3)
        if free_space_gb < 2:
            print(f"⚠️  Warning: Low disk space ({free_space_gb:.1f} GB free)")
            print("Model training may require additional space for temporary files")
    except:
        pass
    
    # Run production training
    success = run_production_training()
    
    if success:
        print("\n🎉 Production model training completed successfully!")
        print("The trained model is ready for integration with the paper trading system.")
        sys.exit(0)
    else:
        print("\n❌ Production model training failed!")
        print("Check the logs for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()