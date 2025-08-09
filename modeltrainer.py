# modeltrainer.py - Consolidated Model Training Module
# Combines all improvements from various training scripts

import os
import json
import logging
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
import optuna
from datetime import datetime
import traceback
from typing import Tuple, List, Dict, Optional

# Setup logging
logger = logging.getLogger(__name__)


class ModelTrainer:
    """
    Consolidated model trainer for L2-based Bitcoin trading system.
    Handles LightGBM training with Optuna optimization and proper validation.
    """
    
    def __init__(self, config: dict):
        """Initialize model trainer with configuration."""
        self.config = config
        self.trained_features = []
        self.base_dir = config.get('base_dir', './trading_bot_data')
        
        # Ensure base directory exists
        os.makedirs(self.base_dir, exist_ok=True)
        
        # Safe symbol handling for filenames
        safe_symbol = config.get('symbol', 'SYMBOL').replace('/', '_').replace(':', '')
        
        # Model output paths
        self.model_path = os.path.join(
            self.base_dir, f"lgbm_model_{safe_symbol}_l2_only.txt"
        )
        self.features_json_path = os.path.join(
            self.base_dir, f"model_features_{safe_symbol}_l2_only.json"
        )
        self.scaling_json_path = os.path.join(
            self.base_dir, f"lgbm_model_{safe_symbol}_l2_only_scaling.json"
        )
        
        # Training parameters
        self.optuna_trials = config.get('optuna_trials', 50)
        self.test_size = config.get('test_size', 0.2)
        self.use_time_series_split = config.get('use_time_series_split', True)
        self.n_splits = config.get('n_time_splits', 5)
        
        logger.info(f"ModelTrainer initialized. Model path: {self.model_path}")

    def _prepare_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare data for training, excluding all non-predictive columns.
        Prevents target leakage by removing future information.
        """
        # Comprehensive list of columns to exclude
        exclude_columns = [
            # Target column
            'target',
            
            # Timestamp and identifiers
            'timestamp', 'symbol', 'exchange', 'id',
            
            # Target leakage columns (future information)
            'target_return_1min', 'target_return_5min', 'target_volatility',
            'target_direction', 'target_return', 'target_price',
            'future_price', 'future_return', 'future_volatility',
            
            # Metadata columns
            'update_id', 'sequence_id', 'data_quality_score',
            'received_timestamp_ms', 'timestamp_ms',
            
            # Redundant columns in L2-only mode
            'close', 'open', 'high', 'low', 'volume',
            
            # Any column containing 'future' or 'target'
            *[col for col in df.columns if 'future' in col.lower()],
            *[col for col in df.columns if 'target' in col.lower() and col != 'target']
        ]
        
        # Select features
        features = [c for c in df.columns if c not in exclude_columns]
        
        logger.info(f"Preparing {len(features)} features for training")
        logger.info(f"Excluded {len(df.columns) - len(features) - 1} columns")
        logger.debug(f"Excluded columns: {[c for c in df.columns if c in exclude_columns]}")
        
        # Validate we have features
        if not features:
            raise ValueError("No features available after excluding columns")
        
        # Validate target exists
        if 'target' not in df.columns:
            raise ValueError("Target column not found in dataframe")
        
        X = df[features].copy()
        y = df['target'].copy()
        
        # Store feature names
        self.trained_features = features
        
        # Check for NaN values
        nan_features = X.columns[X.isnull().any()].tolist()
        if nan_features:
            logger.warning(f"Features with NaN values: {nan_features}")
            # Option to handle NaNs (LightGBM can handle them)
            # X = X.fillna(X.mean())
        
        return X, y

    def train(self, df_labeled: pd.DataFrame) -> Tuple[lgb.Booster, List[str]]:
        """
        Train model with Optuna hyperparameter optimization.
        Uses time series split for proper validation.
        """
        logger.info("Starting model training...")
        
        # Prepare data
        X, y = self._prepare_data(df_labeled)
        
        if X.empty or y.empty:
            raise ValueError("Data preparation failed - empty features or target")
        
        # Log data statistics
        logger.info(f"Training data shape: {X.shape}")
        logger.info(f"Target statistics - Mean: {y.mean():.6f}, Std: {y.std():.6f}")
        
        # Split data - use time series split if enabled
        if self.use_time_series_split:
            # For time series, use most recent data for validation
            split_idx = int(len(X) * (1 - self.test_size))
            X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
            y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]
            logger.info("Using time series split for validation")
        else:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=self.test_size, random_state=42
            )
            logger.info("Using random split for validation")

        # Define Optuna objective
        def objective(trial):
            params = {
                'objective': 'regression_l1',
                'metric': 'mae',
                'boosting_type': 'gbdt',
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.1, log=True),
                'num_leaves': trial.suggest_int('num_leaves', 20, 150),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
                'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
                'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
                'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
                'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
                'verbosity': -1,
                'n_jobs': -1,
                'random_state': 42
            }
            
            # Train model with early stopping
            model = lgb.LGBMRegressor(**params)
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                eval_metric='mae',
                callbacks=[lgb.early_stopping(25, verbose=False)]
            )
            
            # Return validation score
            y_pred = model.predict(X_val)
            return mean_absolute_error(y_val, y_pred)

        # Run Optuna optimization
        logger.info(f"Starting Optuna optimization with {self.optuna_trials} trials...")
        study = optuna.create_study(direction='minimize', study_name='lgbm_optimization')
        study.optimize(objective, n_trials=self.optuna_trials, show_progress_bar=True)
        
        # Get best parameters
        best_params = study.best_params
        logger.info(f"Best Optuna params: {best_params}")
        logger.info(f"Best validation MAE: {study.best_value:.6f}")
        
        # Train final model on full data with best parameters
        logger.info("Training final model on full dataset...")
        final_params = {
            'objective': 'regression_l1',
            'metric': 'mae',
            'boosting_type': 'gbdt',
            'verbosity': -1,
            'n_jobs': -1,
            'random_state': 42,
            **best_params
        }
        
        final_model = lgb.LGBMRegressor(**final_params)
        final_model.fit(X, y)
        
        # Save model and metadata
        self.save_model(final_model, y.mean(), y.std())
        
        # Log feature importance
        self._log_feature_importance(final_model)
        
        return final_model.booster_, self.trained_features

    def save_model(self, model: lgb.LGBMRegressor, target_mean: float, target_std: float):
        """Save model, features, and scaling parameters."""
        try:
            # Save LightGBM model
            model.booster_.save_model(self.model_path)
            logger.info(f"Model saved to {self.model_path}")
            
            # Save feature names
            features_info = {
                'trained_features': self.trained_features,
                'n_features': len(self.trained_features),
                'model_type': 'LightGBM',
                'training_date': datetime.now().isoformat()
            }
            with open(self.features_json_path, 'w') as f:
                json.dump(features_info, f, indent=4)
            logger.info(f"Features saved to {self.features_json_path}")
            
            # Save scaling parameters
            scaling_info = {
                'target_mean': float(target_mean),
                'target_std': float(target_std),
                'features': self.trained_features
            }
            with open(self.scaling_json_path, 'w') as f:
                json.dump(scaling_info, f, indent=4)
            logger.info(f"Scaling parameters saved to {self.scaling_json_path}")
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            traceback.print_exc()
            raise

    def _log_feature_importance(self, model: lgb.LGBMRegressor):
        """Log feature importance for model interpretation."""
        importance = model.feature_importances_
        feature_imp = pd.DataFrame({
            'feature': self.trained_features,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        logger.info("\nTop 10 most important features:")
        for idx, row in feature_imp.head(10).iterrows():
            logger.info(f"  {row['feature']}: {row['importance']:.2f}")

    def train_model(self, df_labeled_features: pd.DataFrame, save: bool = True) -> Tuple[lgb.Booster, List[str]]:
        """
        Public interface for model training.
        Maintains compatibility with existing code.
        """
        return self.train(df_labeled_features)

    def validate_model(self, model_path: str, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """
        Validate a trained model on test data.
        Returns performance metrics.
        """
        # Load model
        booster = lgb.Booster(model_file=model_path)
        
        # Make predictions
        y_pred = booster.predict(X_test)
        
        # Calculate metrics
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        
        # Calculate directional accuracy
        y_test_direction = np.sign(y_test)
        y_pred_direction = np.sign(y_pred)
        directional_accuracy = (y_test_direction == y_pred_direction).mean()
        
        metrics = {
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'directional_accuracy': directional_accuracy
        }
        
        logger.info(f"Validation metrics: {metrics}")
        return metrics


# Backward compatibility function
def train_model_simple(df: pd.DataFrame, config: dict) -> Tuple[lgb.Booster, List[str]]:
    """Simple training function for backward compatibility."""
    trainer = ModelTrainer(config)
    return trainer.train_model(df)