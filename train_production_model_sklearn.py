#!/usr/bin/env python3
"""
train_production_model_sklearn.py - Production Model Training Script (Sklearn Version)

This script configures and runs the Enhanced Model Trainer for production-grade
model training using sklearn instead of LightGBM due to dependency issues.

Sprint 2 - Priority 1: Train Production ML Model
"""

import os
import sys
import logging
import time
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Any, Optional, Tuple
from contextlib import contextmanager
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
import pickle
import json

# Configure logging for production training
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f'production_training_sklearn_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    ]
)
logger = logging.getLogger(__name__)


class SklearnModelTrainer:
    """Simplified model trainer using sklearn instead of LightGBM."""
    
    def __init__(self, config: dict):
        """Initialize sklearn model trainer."""
        self.config = config
        self.base_dir = config.get('base_dir', './trading_bot_data')
        self.db_path = config.get('db_path', 'trading_bot.db')
        os.makedirs(self.base_dir, exist_ok=True)
        
        safe_symbol = config.get('symbol', 'SYMBOL').replace('/', '_').replace(':', '')
        
        # Model paths
        self.ensemble_path = os.path.join(
            self.base_dir, f"enhanced_ensemble_{safe_symbol}.pkl"
        )
        self.features_path = os.path.join(
            self.base_dir, f"enhanced_features_{safe_symbol}.json"
        )
        self.metrics_path = os.path.join(
            self.base_dir, f"enhanced_metrics_{safe_symbol}.json"
        )
        
        # Training parameters
        self.n_folds = config.get('n_time_splits', 5)
        self.test_size = config.get('test_size', 0.2)
        
        # Target engineering parameters
        self.target_horizons = [10, 50, 100, 300]  # 1s, 5s, 10s, 30s at 100ms
        self.volatility_lookback = 100
        self.target_clip_quantiles = (0.001, 0.999)
        
        # Define basic L2 features (same as fallback from enhanced trainer)
        self.l2_features = [
            'spread', 'spread_bps', 'mid_price_return', 'order_book_imbalance',
            'order_book_imbalance_2', 'total_bid_volume_5', 'total_ask_volume_5',
            'bid_pressure', 'ask_pressure', 'l2_volatility_10', 'l2_volatility_50',
            'microprice'
        ]
        
        logger.info(f"SklearnModelTrainer initialized")
        logger.info(f"Database path: {self.db_path}")
        logger.info(f"L2 features: {len(self.l2_features)}")

    @contextmanager
    def get_db_connection(self):
        """Context manager for database connections."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        except Exception as e:
            logger.error(f"Database error: {e}")
            raise
        finally:
            conn.close()

    def load_training_data_from_db(self, symbol: str = 'BTCUSDT', limit: Optional[int] = None) -> pd.DataFrame:
        """Load training data from database and create basic features."""
        logger.info(f"Loading training data from database for {symbol}")
        
        try:
            with self.get_db_connection() as conn:
                # Load raw L2 data
                query = """
                    SELECT 
                        timestamp, symbol, mid_price, spread, microprice, weighted_bid_price, weighted_ask_price,
                        order_book_imbalance, bid_price_1, bid_size_1, bid_price_2, bid_size_2, bid_price_3, bid_size_3,
                        bid_price_4, bid_size_4, bid_price_5, bid_size_5, ask_price_1, ask_size_1, ask_price_2, ask_size_2,
                        ask_price_3, ask_size_3, ask_price_4, ask_size_4, ask_price_5, ask_size_5
                    FROM l2_training_data_practical 
                    WHERE symbol = ?
                        AND timestamp IS NOT NULL
                        AND mid_price IS NOT NULL
                        AND spread IS NOT NULL
                    ORDER BY timestamp DESC
                """
                
                if limit:
                    query += f" LIMIT {limit}"
                
                df = pd.read_sql_query(query, conn, params=(symbol,))
                
                if df.empty:
                    logger.warning(f"No data found for {symbol}")
                    return pd.DataFrame()
                
                # Convert timestamp to datetime and sort
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df = df.sort_values('timestamp').reset_index(drop=True)
                
                # Create basic features
                df_basic = self._create_basic_features(df)
                
                logger.info(f"Loaded {len(df_basic)} rows from database")
                logger.info(f"Date range: {df_basic['timestamp'].min()} to {df_basic['timestamp'].max()}")
                
                return df_basic
                
        except Exception as e:
            logger.error(f"Error loading training data: {e}")
            return pd.DataFrame()

    def _create_basic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create basic L2 features from raw data."""
        try:
            df_basic = df.copy()
            
            # Basic spread features
            df_basic['spread_bps'] = (df_basic['spread'] / df_basic['mid_price']) * 10000
            
            # Basic volume features  
            df_basic['total_bid_volume_5'] = (
                df_basic['bid_size_1'] + df_basic['bid_size_2'] + df_basic['bid_size_3'] + 
                df_basic['bid_size_4'] + df_basic['bid_size_5']
            )
            df_basic['total_ask_volume_5'] = (
                df_basic['ask_size_1'] + df_basic['ask_size_2'] + df_basic['ask_size_3'] + 
                df_basic['ask_size_4'] + df_basic['ask_size_5']
            )
            
            # Basic volatility - rolling std of returns
            df_basic['mid_price_return'] = df_basic['mid_price'].pct_change()
            df_basic['l2_volatility_10'] = df_basic['mid_price_return'].rolling(10, min_periods=2).std()
            df_basic['l2_volatility_50'] = df_basic['mid_price_return'].rolling(50, min_periods=5).std()
            
            # Basic imbalance features
            df_basic['order_book_imbalance_2'] = (
                (df_basic['bid_size_1'] + df_basic['bid_size_2'] - df_basic['ask_size_1'] - df_basic['ask_size_2']) /
                (df_basic['bid_size_1'] + df_basic['bid_size_2'] + df_basic['ask_size_1'] + df_basic['ask_size_2'] + 1e-8)
            )
            
            # Basic pressure features
            df_basic['bid_pressure'] = df_basic['total_bid_volume_5'] / (df_basic['total_bid_volume_5'] + df_basic['total_ask_volume_5'] + 1e-8)
            df_basic['ask_pressure'] = df_basic['total_ask_volume_5'] / (df_basic['total_bid_volume_5'] + df_basic['total_ask_volume_5'] + 1e-8)
            
            # Fill NaN values
            df_basic = df_basic.ffill().bfill().fillna(0)
            
            logger.info(f"Created {len(self.l2_features)} basic L2 features")
            
            return df_basic
            
        except Exception as e:
            logger.error(f"Error creating basic features: {e}")
            return df.copy()

    def _engineer_targets(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer multiple targets for different prediction horizons."""
        logger.info("Engineering multi-horizon targets...")
        
        df_targets = df.copy()
        
        # Use microprice if available, otherwise mid_price
        price_col = 'microprice' if 'microprice' in df.columns else 'mid_price'
        logger.info(f"Using {price_col} as price series for target engineering")
        
        # Calculate returns for each horizon
        for horizon in self.target_horizons:
            df_targets[f'return_{horizon}'] = (
                df[price_col].shift(-horizon) / df[price_col] - 1
            )
        
        # Calculate adaptive volatility
        returns = df[price_col].pct_change()
        ewma_span = self.volatility_lookback // 2
        df_targets['volatility_ewma'] = returns.ewm(span=ewma_span, min_periods=10).std()
        df_targets['volatility_realized'] = returns.rolling(self.volatility_lookback, min_periods=20).std()
        df_targets['volatility'] = df_targets['volatility_ewma'].fillna(
            df_targets['volatility_realized']).fillna(returns.std())
        
        # Create normalized targets
        for horizon in self.target_horizons:
            return_col = f'return_{horizon}'
            target_col = f'target_{horizon}'
            
            # Volatility-normalized return
            df_targets[target_col] = (
                df_targets[return_col] / (df_targets['volatility'] + 1e-8)
            )
            
            # Clip extreme values
            lower = df_targets[target_col].quantile(self.target_clip_quantiles[0])
            upper = df_targets[target_col].quantile(self.target_clip_quantiles[1])
            df_targets[target_col] = df_targets[target_col].clip(lower, upper)
        
        # Drop rows with NaN targets
        target_cols = [f'target_{h}' for h in self.target_horizons]
        df_targets = df_targets.dropna(subset=target_cols)
        
        logger.info(f"Engineered targets for {len(df_targets)} samples")
        
        return df_targets

    def train_from_database(self, symbol: str = 'BTCUSDT', limit: Optional[int] = None) -> Dict:
        """Train ensemble model using data loaded directly from database."""
        logger.info(f"Starting database-driven training for {symbol}")
        
        # Load data from database
        df_features = self.load_training_data_from_db(symbol, limit)
        
        if df_features.empty:
            raise ValueError("No training data available in database")
        
        # Use the loaded data for training
        return self.train(df_features)

    def train(self, df_labeled: pd.DataFrame) -> Dict:
        """Train ensemble model with multiple prediction horizons using sklearn."""
        logger.info("Starting sklearn-based model training...")
        
        # Generate multiple targets for different horizons
        df_multi_target = self._engineer_targets(df_labeled)
        
        if df_multi_target.empty:
            raise ValueError("Target engineering failed")
        
        # Train models for each horizon
        ensemble_models = {}
        feature_importance = {}
        validation_metrics = {}
        
        for horizon in self.target_horizons:
            logger.info(f"Training model for {horizon}-tick horizon...")
            
            # Prepare data for this horizon
            target_col = f'target_{horizon}'
            if target_col not in df_multi_target.columns:
                logger.warning(f"Target {target_col} not found, skipping")
                continue
            
            # Train model with walk-forward validation
            model_result = self._train_single_horizon(
                df_multi_target, 
                target_col, 
                horizon
            )
            
            if model_result:
                ensemble_models[horizon] = model_result['model']
                feature_importance[horizon] = model_result['importance']
                validation_metrics[horizon] = model_result['metrics']
        
        # Simple ensemble weights based on performance
        ensemble_weights = {}
        total_inverse_mae = 0
        
        for horizon, metrics in validation_metrics.items():
            mae = metrics['mae']
            inverse_mae = 1 / (mae + 1e-8)
            ensemble_weights[horizon] = inverse_mae
            total_inverse_mae += inverse_mae
        
        # Normalize weights
        for horizon in ensemble_weights:
            ensemble_weights[horizon] /= total_inverse_mae
        
        logger.info(f"Ensemble weights: {ensemble_weights}")
        
        # Save everything
        ensemble_result = {
            'models': ensemble_models,
            'features': self.l2_features,
            'weights': ensemble_weights,
            'thresholds': {},  # Simplified - no dynamic thresholds
            'importance': feature_importance,
            'metrics': validation_metrics,
            'config': {
                'horizons': self.target_horizons,
                'volatility_lookback': self.volatility_lookback,
                'clip_quantiles': self.target_clip_quantiles
            }
        }
        
        self._save_ensemble(ensemble_result)
        
        return ensemble_result

    def _train_single_horizon(self, df: pd.DataFrame, target_col: str, horizon: int) -> Optional[Dict]:
        """Train model for a single prediction horizon with walk-forward validation."""
        
        # Prepare features
        X, y = self._prepare_features(df, target_col)
        
        if X.empty or y.empty:
            logger.warning(f"No data for horizon {horizon}")
            return None
        
        # Walk-forward validation
        tscv = TimeSeriesSplit(n_splits=self.n_folds)
        val_scores = []
        
        logger.info(f"Training sklearn GradientBoostingRegressor for {horizon}-tick horizon...")
        
        # Use simple sklearn gradient boosting
        model = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            random_state=42,
            subsample=0.8
        )
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # Train model
            fold_model = GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42,
                subsample=0.8
            )
            fold_model.fit(X_train, y_train)
            
            # Validate
            y_pred = fold_model.predict(X_val)
            val_score = mean_absolute_error(y_val, y_pred)
            val_scores.append(val_score)
            
            logger.info(f"Fold {fold+1}/{self.n_folds} MAE: {val_score:.6f}")
        
        # Train final model on all data
        final_model = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            random_state=42,
            subsample=0.8
        )
        final_model.fit(X, y)
        
        # Calculate metrics
        y_pred_all = final_model.predict(X)
        
        metrics = {
            'mae': np.mean(val_scores),
            'mae_std': np.std(val_scores),
            'correlation': np.corrcoef(y_pred_all, y.values)[0, 1],
            'profitable_ratio': (y_pred_all * y.values > 0).mean()
        }
        
        # Get feature importance
        importance = pd.DataFrame({
            'feature': self.l2_features,
            'importance': final_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return {
            'model': final_model,
            'importance': importance,
            'metrics': metrics
        }

    def _prepare_features(self, df: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare features using the basic L2 features."""
        
        # Use only the predefined L2 features available
        available_features = [f for f in self.l2_features if f in df.columns]
        missing_features = [f for f in self.l2_features if f not in df.columns]
        
        if missing_features:
            logger.warning(f"Missing L2 features: {missing_features}")
        
        if not available_features:
            raise ValueError("No L2 features available for training")
        
        logger.info(f"Using {len(available_features)} L2 features for training")
        
        X = df[available_features].copy()
        y = df[target_col].copy()
        
        # Handle any remaining NaN values
        X = X.ffill().bfill().fillna(0)
        
        return X, y

    def _save_ensemble(self, ensemble_result: Dict):
        """Save ensemble models and metadata."""
        
        # Save models
        with open(self.ensemble_path, 'wb') as f:
            pickle.dump(ensemble_result, f)
        
        # Save features
        with open(self.features_path, 'w') as f:
            json.dump({
                'features': ensemble_result['features'],
                'horizons': list(ensemble_result['models'].keys()),
                'weights': ensemble_result['weights']
            }, f, indent=2)
        
        # Save metrics
        metrics_summary = {
            'training_date': datetime.now().isoformat(),
            'horizons': list(ensemble_result['models'].keys()),
            'ensemble_weights': ensemble_result['weights'],
            'validation_metrics': ensemble_result['metrics'],
            'model_type': 'sklearn_gradient_boosting'
        }
        
        with open(self.metrics_path, 'w') as f:
            json.dump(metrics_summary, f, indent=2)
        
        logger.info(f"Ensemble saved to {self.ensemble_path}")


def main():
    """Main function to run the sklearn-based model trainer."""
    
    print("BTC Production Model Training (Sklearn Version)")
    print("=" * 60)
    print("Sprint 2 - Priority 1: Train Production ML Model")
    print("Training on full l2_training_data_practical dataset")
    print("=" * 60)
    
    # Configuration for PRODUCTION TRAINING
    config = {
        'symbol': 'BTCUSDT',
        'db_path': 'trading_bot.db',
        'base_dir': './trading_bot_data',
        'n_time_splits': 8,     # Increased for better cross-validation
        'test_size': 0.15,      # Use more data for training
    }
    
    try:
        # Initialize trainer
        trainer = SklearnModelTrainer(config)
        
        # Train model using database data
        logger.info("Starting model training with sklearn backend...")
        
        # PRODUCTION TRAINING - NO LIMITS, FULL DATASET
        logger.info("🚀 STARTING PRODUCTION TRAINING ON FULL DATASET")
        logger.info("Expected training time: 10-20 minutes on full dataset with sklearn")
        
        training_start = time.time()
        
        ensemble_result = trainer.train_from_database(
            symbol='BTCUSDT',
            limit=None  # 🔥 NO LIMIT - Use full dataset
        )
        
        training_duration = time.time() - training_start
        logger.info(f"Training completed in {training_duration:.2f} seconds ({training_duration/60:.1f} minutes)")
        
        logger.info("Training completed successfully!")
        logger.info(f"Trained models for horizons: {list(ensemble_result['models'].keys())}")
        logger.info(f"Ensemble weights: {ensemble_result['weights']}")
        
        # Print feature importance summary
        if 'importance' in ensemble_result:
            for horizon, importance_df in ensemble_result['importance'].items():
                logger.info(f"Top 5 features for {horizon}-tick horizon:")
                logger.info(f"{importance_df.head(5).to_string()}")
                
        # Print validation metrics
        if 'metrics' in ensemble_result:
            for horizon, metrics in ensemble_result['metrics'].items():
                logger.info(f"Validation metrics for {horizon}-tick horizon:")
                for metric_name, metric_value in metrics.items():
                    logger.info(f"  {metric_name}: {metric_value:.6f}")
        
        logger.info(f"Models and metadata saved to: {trainer.ensemble_path}")
        logger.info("✅ PRODUCTION MODEL READY FOR DEPLOYMENT!")
        
        return True
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)