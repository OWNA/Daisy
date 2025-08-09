# modeltrainer_enhanced.py
# Enhanced Model Training with Ensemble Methods and Dynamic Thresholds

import os
import json
import logging
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
        """Initialize enhanced model trainer."""
        self.config = config
        self.base_dir = config.get('base_dir', './trading_bot_data')
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
        
        logger.info("Enhanced ModelTrainer initialized")

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
        
        # Optimize ensemble weights
        self.ensemble_weights = self._optimize_ensemble_weights(
            ensemble_models, 
            df_multi_target
        )
        
        # Calculate dynamic thresholds
        self.dynamic_thresholds = self._calculate_dynamic_thresholds(
            ensemble_models,
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
        """
        logger.info("Engineering multi-horizon targets...")
        
        # Get price series (prefer microprice for precision)
        price_col = None
        for col in ['microprice', 'weighted_mid_price', 'mid_price']:
            if col in df.columns:
                price_col = col
                break
        
        if not price_col:
            raise ValueError("No price column found for target engineering")
        
        df_targets = df.copy()
        
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
            df_targets[f'target_cat_{horizon}'] = pd.cut(
                df_targets[target_col],
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
        """Prepare features excluding targets and non-predictive columns."""
        
        exclude_columns = [
            target_col,
            'timestamp', 'symbol', 'exchange',
            *[col for col in df.columns if 'target' in col],
            *[col for col in df.columns if 'return_' in col],
            *[col for col in df.columns if 'future' in col],
            'volatility', 'volatility_ewma', 'volatility_realized'
        ]
        
        features = [c for c in df.columns if c not in exclude_columns]
        self.trained_features = features
        
        X = df[features].copy()
        y = df[target_col].copy()
        
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
        models: Dict, 
        df: pd.DataFrame
    ) -> Dict[int, float]:
        """Optimize ensemble weights using validation predictions."""
        
        # Simple inverse-variance weighting based on validation performance
        weights = {}
        total_inverse_mae = 0
        
        for horizon, model_data in models.items():
            mae = model_data['metrics']['mae']
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