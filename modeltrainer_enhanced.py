# modeltrainer_enhanced.py
# Enhanced Model Training with Ensemble Methods and Dynamic Thresholds
# Updated to use 51 L2 features directly from database

import os
import json
import logging
import sqlite3
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.metrics import mean_absolute_error, precision_recall_curve, roc_auc_score
from sklearn.preprocessing import StandardScaler
import optuna
from datetime import datetime
import pickle
from typing import Tuple, List, Dict, Optional
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class EnhancedModelTrainer:
    """
    Enhanced model trainer implementing:
    - Multi-timeframe ensemble models
    - Better target engineering
    - Dynamic threshold optimization
    - Feature importance tracking
    - Walk-forward validation
    """
    
    def __init__(self, config: dict):
        """Initialize enhanced model trainer with database integration."""
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
        self.optuna_trials = config.get('optuna_trials', 100)
        
        # Target engineering parameters
        self.target_horizons = [10, 50, 100, 300]  # 1s, 5s, 10s, 30s at 100ms
        self.volatility_lookback = 100
        self.target_clip_quantiles = (0.001, 0.999)  # More aggressive clipping
        
        # Ensemble parameters
        self.ensemble_weights = None
        self.dynamic_thresholds = {}
        self.feature_importance_history = []
        
        # Define the 51 L2 features available in database
        self.l2_features = self._define_database_features()
        
        logger.info(f"Enhanced ModelTrainer initialized with database integration")
        logger.info(f"Database path: {self.db_path}")
        logger.info(f"L2 features available: {len(self.l2_features)}")

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

    def _define_database_features(self) -> List[str]:
        """Define the L2 features actually available in the database."""
        return [
            # Raw order book features (40 features) - bid/ask prices and sizes for 10 levels
            'bid_price_1', 'bid_size_1', 'bid_price_2', 'bid_size_2', 'bid_price_3', 'bid_size_3',
            'bid_price_4', 'bid_size_4', 'bid_price_5', 'bid_size_5', 'bid_price_6', 'bid_size_6',
            'bid_price_7', 'bid_size_7', 'bid_price_8', 'bid_size_8', 'bid_price_9', 'bid_size_9',
            'bid_price_10', 'bid_size_10',
            'ask_price_1', 'ask_size_1', 'ask_price_2', 'ask_size_2', 'ask_price_3', 'ask_size_3',
            'ask_price_4', 'ask_size_4', 'ask_price_5', 'ask_size_5', 'ask_price_6', 'ask_size_6',
            'ask_price_7', 'ask_size_7', 'ask_price_8', 'ask_size_8', 'ask_price_9', 'ask_size_9',
            'ask_price_10', 'ask_size_10',
            
            # Computed features (11 features)
            'mid_price', 'spread', 'spread_bps',
            'total_bid_volume_10', 'total_ask_volume_10',
            'weighted_bid_price', 'weighted_ask_price',
            'order_book_imbalance', 'microprice',
            'price_impact_bid', 'price_impact_ask'
        ]

    def load_raw_l2_data_from_db(self, symbol: str = 'BTCUSDT', limit: Optional[int] = None) -> pd.DataFrame:
        """
        Load raw L2 order book data from database for feature engineering.
        
        Args:
            symbol: Trading symbol to load data for
            limit: Optional limit on number of rows to load (most recent first)
            
        Returns:
            DataFrame with raw L2 order book data
        """
        logger.info(f"Loading raw L2 data from database for {symbol}")
        
        try:
            with self.get_db_connection() as conn:
                # Query to get raw L2 order book data
                query = """
                    SELECT 
                        timestamp,
                        symbol,
                        bid_price_1, bid_size_1, bid_price_2, bid_size_2, bid_price_3, bid_size_3,
                        bid_price_4, bid_size_4, bid_price_5, bid_size_5, bid_price_6, bid_size_6,
                        bid_price_7, bid_size_7, bid_price_8, bid_size_8, bid_price_9, bid_size_9,
                        bid_price_10, bid_size_10,
                        ask_price_1, ask_size_1, ask_price_2, ask_size_2, ask_price_3, ask_size_3,
                        ask_price_4, ask_size_4, ask_price_5, ask_size_5, ask_price_6, ask_size_6,
                        ask_price_7, ask_size_7, ask_price_8, ask_size_8, ask_price_9, ask_size_9,
                        ask_price_10, ask_size_10,
                        mid_price, spread, microprice, weighted_bid_price, weighted_ask_price,
                        order_book_imbalance
                    FROM l2_training_data_practical 
                    WHERE symbol = ? AND data_source = 'live_trading'
                        AND timestamp IS NOT NULL
                        AND bid_price_1 IS NOT NULL
                        AND ask_price_1 IS NOT NULL
                        AND data_quality_score > 0.5
                    ORDER BY timestamp DESC
                """
                
                if limit:
                    query += f" LIMIT {limit}"
                
                df = pd.read_sql_query(query, conn, params=(symbol,))
                
                if df.empty:
                    logger.warning(f"No raw L2 data found for {symbol}")
                    return pd.DataFrame()
                
                # Convert timestamp to datetime
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                
                # Sort by timestamp ascending for feature engineering
                df = df.sort_values('timestamp').reset_index(drop=True)
                
                logger.info(f"Loaded {len(df)} rows of raw L2 data")
                logger.info(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
                
                return df
                
        except Exception as e:
            logger.error(f"Error loading raw L2 data from database: {e}")
            return pd.DataFrame()

    def load_training_data_from_db(self, symbol: str = 'BTCUSDT', limit: Optional[int] = None) -> pd.DataFrame:
        """
        Load training data with features - generates L2 features if not available.
        
        Args:
            symbol: Trading symbol to load data for
            limit: Optional limit on number of rows to load (most recent first)
            
        Returns:
            DataFrame with L2 features and basic price data
        """
        logger.info(f"Loading training data from database for {symbol}")
        
        # Load from l2_training_data_practical table (our main dataset)
        try:
            with self.get_db_connection() as conn:
                # Use available features from l2_training_data_practical
                available_features = [
                    'bid_price_1', 'bid_size_1', 'bid_price_2', 'bid_size_2', 'bid_price_3', 'bid_size_3',
                    'bid_price_4', 'bid_size_4', 'bid_price_5', 'bid_size_5', 'bid_price_6', 'bid_size_6',
                    'bid_price_7', 'bid_size_7', 'bid_price_8', 'bid_size_8', 'bid_price_9', 'bid_size_9',
                    'bid_price_10', 'bid_size_10',
                    'ask_price_1', 'ask_size_1', 'ask_price_2', 'ask_size_2', 'ask_price_3', 'ask_size_3',
                    'ask_price_4', 'ask_size_4', 'ask_price_5', 'ask_size_5', 'ask_price_6', 'ask_size_6',
                    'ask_price_7', 'ask_size_7', 'ask_price_8', 'ask_size_8', 'ask_price_9', 'ask_size_9',
                    'ask_price_10', 'ask_size_10',
                    'mid_price', 'spread', 'spread_bps', 'total_bid_volume_10', 'total_ask_volume_10',
                    'weighted_bid_price', 'weighted_ask_price', 'order_book_imbalance', 'microprice',
                    'price_impact_bid', 'price_impact_ask'
                ]
                
                feature_columns = ', '.join(available_features)
                
                query = f"""
                    SELECT 
                        timestamp,
                        symbol,
                        {feature_columns}
                    FROM l2_training_data_practical 
                    WHERE symbol = ? AND data_source = 'live_trading'
                        AND timestamp IS NOT NULL
                        AND mid_price IS NOT NULL
                        AND spread IS NOT NULL
                    ORDER BY timestamp DESC
                """
                
                if limit:
                    query += f" LIMIT {limit}"
                
                df = pd.read_sql_query(query, conn, params=(symbol,))
                
                if not df.empty:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    df = df.sort_values('timestamp').reset_index(drop=True)
                    logger.info(f"Loaded {len(df)} rows from l2_training_data_practical table")
                    logger.info(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
                    return df
                    
        except Exception as e:
            logger.error(f"Could not load from l2_training_data_practical table: {e}")
        
        # Fallback to old method if needed
        logger.info("Falling back to raw data loading...")
        
        raw_df = self.load_raw_l2_data_from_db(symbol, limit)
        if raw_df.empty:
            return pd.DataFrame()
        
        # For now, use the basic features available in the raw data table
        # This gives us a working baseline while the full feature engineering is debugged
        logger.info("Using available basic features from raw L2 data...")
        
        try:
            # Create basic L2 features from raw order book data
            df_basic = raw_df.copy()
            
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
            
            # Define a minimal feature set that we've actually created
            basic_l2_features = [
                'spread', 'spread_bps', 'mid_price_return', 'order_book_imbalance',
                'order_book_imbalance_2', 'total_bid_volume_5', 'total_ask_volume_5',
                'bid_pressure', 'ask_pressure', 'l2_volatility_10', 'l2_volatility_50',
                'microprice'
            ]
            
            # Filter to available features
            available_features = [f for f in basic_l2_features if f in df_basic.columns]
            basic_columns = ['timestamp', 'symbol', 'mid_price']  # Include mid_price for target engineering
            
            all_columns = basic_columns + available_features
            df_final = df_basic[all_columns].copy()
            
            # Fill NaN values
            df_final = df_final.ffill().bfill().fillna(0)
            
            logger.info(f"Prepared {len(available_features)} basic L2 features for training: {available_features}")
            
            # Update the trainer's feature list to match what we actually have
            self.l2_features = available_features
            
            return df_final
            
        except Exception as e:
            logger.error(f"Error creating basic L2 features: {e}")
            return pd.DataFrame()

    def train_from_database(self, symbol: str = 'BTCUSDT', limit: Optional[int] = None) -> Dict:
        """
        Train ensemble model using data loaded directly from database.
        
        Args:
            symbol: Trading symbol to train on
            limit: Optional limit on training data rows
            
        Returns:
            dict: Contains ensemble models, features, metrics, and thresholds
        """
        logger.info(f"Starting database-driven training for {symbol}")
        
        # Load data from database
        df_features = self.load_training_data_from_db(symbol, limit)
        
        if df_features.empty:
            raise ValueError("No training data available in database")
        
        # Use the loaded data for training
        return self.train(df_features)

    def train(self, df_labeled: pd.DataFrame) -> Dict:
        """
        Train ensemble model with multiple prediction horizons.
        
        Returns:
            dict: Contains ensemble models, features, metrics, and thresholds
        """
        logger.info("Starting enhanced model training...")
        
        # Generate multiple targets for different horizons
        df_multi_target = self._engineer_targets(df_labeled)
        
        if df_multi_target.empty:
            raise ValueError("Target engineering failed")
        
        # Train models for each horizon
        ensemble_models = {}
        feature_importance = {}
        validation_metrics = {}
        model_results = {}  # Store complete model results for threshold calculation
        
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
                model_results[horizon] = model_result  # Store complete result
        
        # Optimize ensemble weights
        self.ensemble_weights = self._optimize_ensemble_weights(
            validation_metrics
        )
        
        # Calculate dynamic thresholds using complete model results
        self.dynamic_thresholds = self._calculate_dynamic_thresholds(
            model_results,
            df_multi_target
        )
        
        # Save everything
        ensemble_result = {
            'models': ensemble_models,
            'features': self.trained_features,
            'weights': self.ensemble_weights,
            'thresholds': self.dynamic_thresholds,
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

    def _engineer_targets(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer multiple targets for different prediction horizons.
        Uses volatility-adjusted returns with better normalization.
        For database-loaded data, reconstructs price from basic data.
        """
        logger.info("Engineering multi-horizon targets...")
        
        df_targets = df.copy()
        
        # For database data, we need to reconstruct/derive price
        # Priority: microprice from features, then derive from basic data
        price_col = None
        
        # Try to get price from L2 features
        if 'microprice' in df.columns:
            price_col = 'microprice'
        elif 'weighted_mid_price' in df.columns:
            price_col = 'weighted_mid_price'
        else:
            # Need to reconstruct price from basic bid/ask data if available
            if 'bid_price_1' in df.columns and 'ask_price_1' in df.columns:
                df_targets['mid_price'] = (df['bid_price_1'] + df['ask_price_1']) / 2
                price_col = 'mid_price'
                logger.info("Reconstructed mid_price from bid/ask prices")
            else:
                raise ValueError("No price data available for target engineering")
        
        logger.info(f"Using {price_col} as price series for target engineering")
        
        # Calculate returns for each horizon
        for horizon in self.target_horizons:
            # Forward returns
            df_targets[f'return_{horizon}'] = (
                df[price_col].shift(-horizon) / df[price_col] - 1
            )
        
        # Calculate adaptive volatility
        returns = df[price_col].pct_change()
        
        # Use EWMA volatility for faster adaptation
        ewma_span = self.volatility_lookback // 2
        df_targets['volatility_ewma'] = returns.ewm(
            span=ewma_span, 
            min_periods=10
        ).std()
        
        # Also calculate realized volatility
        df_targets['volatility_realized'] = returns.rolling(
            self.volatility_lookback,
            min_periods=20
        ).std()
        
        # Combine volatilities (prefer EWMA but use realized as fallback)
        df_targets['volatility'] = df_targets['volatility_ewma'].fillna(
            df_targets['volatility_realized']
        ).fillna(returns.std())
        
        # Create normalized targets
        for horizon in self.target_horizons:
            return_col = f'return_{horizon}'
            target_col = f'target_{horizon}'
            
            # Volatility-normalized return
            df_targets[target_col] = (
                df_targets[return_col] / 
                (df_targets['volatility'] + 1e-8)
            )
            
            # Clip extreme values
            lower = df_targets[target_col].quantile(self.target_clip_quantiles[0])
            upper = df_targets[target_col].quantile(self.target_clip_quantiles[1])
            df_targets[target_col] = df_targets[target_col].clip(lower, upper)
            
            # Add categorical target for classification ensemble member
            # Handle NaN values before converting to categorical
            target_values = df_targets[target_col].fillna(0)
            df_targets[f'target_cat_{horizon}'] = pd.cut(
                target_values,
                bins=[-np.inf, -0.5, 0.5, np.inf],
                labels=[-1, 0, 1]
            ).astype(int)
        
        # Drop rows with NaN targets
        target_cols = [f'target_{h}' for h in self.target_horizons]
        df_targets = df_targets.dropna(subset=target_cols)
        
        logger.info(f"Engineered targets for {len(df_targets)} samples")
        
        return df_targets

    def _train_single_horizon(
        self, 
        df: pd.DataFrame, 
        target_col: str, 
        horizon: int
    ) -> Optional[Dict]:
        """Train model for a single prediction horizon with walk-forward validation."""
        
        # Prepare features
        X, y = self._prepare_features(df, target_col)
        
        if X.empty or y.empty:
            logger.warning(f"No data for horizon {horizon}")
            return None
        
        # Walk-forward validation
        tscv = TimeSeriesSplit(n_splits=self.n_folds)
        val_scores = []
        val_predictions = []
        val_indices = []
        
        # Optuna optimization on first fold
        logger.info(f"Optimizing hyperparameters for {horizon}-tick horizon...")
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            if fold == 0:
                # Run Optuna on first fold
                best_params = self._optimize_hyperparameters(
                    X_train, y_train, X_val, y_val
                )
            
            # Train model with best params
            model = lgb.LGBMRegressor(**best_params)
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                eval_metric='mae',
                callbacks=[lgb.early_stopping(50, verbose=False)]
            )
            
            # Validate
            y_pred = model.predict(X_val)
            val_score = mean_absolute_error(y_val, y_pred)
            val_scores.append(val_score)
            val_predictions.extend(y_pred)
            val_indices.extend(val_idx)
            
            logger.info(f"Fold {fold+1}/{self.n_folds} MAE: {val_score:.6f}")
        
        # Train final model on all data
        final_model = lgb.LGBMRegressor(**best_params)
        final_model.fit(
            X, y,
            eval_metric='mae',
            callbacks=[lgb.log_evaluation(100)]
        )
        
        # Calculate metrics
        val_predictions = np.array(val_predictions)
        val_actuals = y.iloc[val_indices].values
        
        metrics = {
            'mae': np.mean(val_scores),
            'mae_std': np.std(val_scores),
            'correlation': np.corrcoef(val_predictions, val_actuals)[0, 1],
            'profitable_ratio': (val_predictions * val_actuals > 0).mean()
        }
        
        # Get feature importance
        importance = pd.DataFrame({
            'feature': self.trained_features,
            'importance': final_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return {
            'model': final_model,
            'importance': importance,
            'metrics': metrics,
            'val_predictions': (val_indices, val_predictions)
        }

    def _prepare_features(self, df: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare features using the 51 L2 features from database."""
        
        # Use only the predefined L2 features available in database
        available_features = [f for f in self.l2_features if f in df.columns]
        missing_features = [f for f in self.l2_features if f not in df.columns]
        
        if missing_features:
            logger.warning(f"Missing L2 features: {missing_features}")
        
        if not available_features:
            raise ValueError("No L2 features available for training")
        
        logger.info(f"Using {len(available_features)} L2 features for training")
        
        # Store features used in training
        self.trained_features = available_features
        
        X = df[available_features].copy()
        y = df[target_col].copy()
        
        # Handle any remaining NaN values
        X = X.ffill().bfill().fillna(0)
        
        return X, y

    def _optimize_hyperparameters(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series
    ) -> Dict:
        """Optimize hyperparameters using Optuna."""
        
        def objective(trial):
            params = {
                'objective': 'regression',
                'metric': 'mae',
                'boosting_type': 'gbdt',
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'num_leaves': trial.suggest_int('num_leaves', 20, 200),
                'max_depth': trial.suggest_int('max_depth', 3, 12),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
                'min_split_gain': trial.suggest_float('min_split_gain', 0.0, 1.0),
                'verbosity': -1,
                'n_jobs': -1,
                'random_state': 42
            }
            
            model = lgb.LGBMRegressor(**params)
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                eval_metric='mae',
                callbacks=[
                    lgb.early_stopping(25, verbose=False),
                    lgb.log_evaluation(0)
                ]
            )
            
            y_pred = model.predict(X_val)
            return mean_absolute_error(y_val, y_pred)
        
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=self.optuna_trials, show_progress_bar=True)
        
        best_params = study.best_params
        best_params.update({
            'objective': 'regression',
            'metric': 'mae',
            'boosting_type': 'gbdt',
            'verbosity': -1,
            'n_jobs': -1,
            'random_state': 42
        })
        
        return best_params

    def _optimize_ensemble_weights(
        self, 
        validation_metrics: Dict
    ) -> Dict[int, float]:
        """Optimize ensemble weights using validation metrics."""
        
        # Simple inverse-variance weighting based on validation performance
        weights = {}
        total_inverse_mae = 0
        
        for horizon, metrics in validation_metrics.items():
            mae = metrics['mae']
            inverse_mae = 1 / (mae + 1e-8)
            weights[horizon] = inverse_mae
            total_inverse_mae += inverse_mae
        
        # Normalize weights
        for horizon in weights:
            weights[horizon] /= total_inverse_mae
        
        logger.info(f"Ensemble weights: {weights}")
        
        return weights

    def _calculate_dynamic_thresholds(
        self,
        models: Dict,
        df: pd.DataFrame
    ) -> Dict[str, Dict[str, float]]:
        """Calculate dynamic thresholds based on market volatility."""
        
        thresholds = {}
        
        # Calculate base thresholds from validation predictions
        for horizon, model_data in models.items():
            val_indices, val_preds = model_data['val_predictions']
            
            # Calculate thresholds for different volatility regimes
            volatility = df.iloc[val_indices]['volatility'].values
            vol_percentiles = [25, 50, 75]
            
            thresholds[horizon] = {}
            
            for pctl in vol_percentiles:
                vol_threshold = np.percentile(volatility, pctl)
                mask = volatility <= vol_threshold
                
                if mask.sum() > 100:  # Enough samples
                    preds_subset = val_preds[mask]
                    
                    # Find threshold that gives ~30% signal rate
                    sorted_preds = np.sort(np.abs(preds_subset))
                    threshold_idx = int(0.7 * len(sorted_preds))
                    threshold = sorted_preds[threshold_idx]
                    
                    thresholds[horizon][f'vol_p{pctl}'] = threshold
            
            # Also store overall threshold
            sorted_all = np.sort(np.abs(val_preds))
            thresholds[horizon]['overall'] = sorted_all[int(0.7 * len(sorted_all))]
        
        logger.info(f"Dynamic thresholds calculated: {thresholds}")
        
        return thresholds

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
            'thresholds': ensemble_result['thresholds'],
            'validation_metrics': ensemble_result['metrics']
        }
        
        with open(self.metrics_path, 'w') as f:
            json.dump(metrics_summary, f, indent=2)
        
        logger.info(f"Ensemble saved to {self.ensemble_path}")


def main():
    """Main function to run the enhanced model trainer with database integration."""
    import logging
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Configuration for PRODUCTION TRAINING
    config = {
        'symbol': 'BTCUSDT',
        'db_path': 'trading_bot.db',
        'base_dir': './trading_bot_data',
        'n_time_splits': 8,     # Increased for better cross-validation
        'test_size': 0.15,      # Use more data for training
        'optuna_trials': 200    # Thorough hyperparameter optimization
    }
    
    try:
        # Initialize trainer
        trainer = EnhancedModelTrainer(config)
        
        # Train model using database data
        logger.info("Starting model training with 51 L2 features from database...")
        
        # PRODUCTION TRAINING - NO LIMITS, FULL DATASET
        logger.info("🚀 STARTING PRODUCTION TRAINING ON FULL DATASET")
        logger.info("Expected training time: 30-60 minutes on full 519k+ row dataset")
        
        ensemble_result = trainer.train_from_database(
            symbol='BTCUSDT',
            limit=None  # 🔥 NO LIMIT - Use full 519k+ row dataset
        )
        
        logger.info("Training completed successfully!")
        logger.info(f"Trained models for horizons: {list(ensemble_result['models'].keys())}")
        logger.info(f"Ensemble weights: {ensemble_result['weights']}")
        
        # Print feature importance summary
        if 'importance' in ensemble_result:
            for horizon, importance_df in ensemble_result['importance'].items():
                logger.info(f"Top 10 features for {horizon}-tick horizon:")
                logger.info(f"{importance_df.head(10).to_string()}")
                
        # Print validation metrics
        if 'metrics' in ensemble_result:
            for horizon, metrics in ensemble_result['metrics'].items():
                logger.info(f"Validation metrics for {horizon}-tick horizon:")
                for metric_name, metric_value in metrics.items():
                    logger.info(f"  {metric_name}: {metric_value:.6f}")
        
        logger.info(f"Models and metadata saved to: {trainer.ensemble_path}")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


if __name__ == "__main__":
    main()