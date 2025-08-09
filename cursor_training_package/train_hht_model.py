#!/usr/bin/env python3
"""
Minimal HHT Model Training Script for Cursor
Streamlined version focusing on L2+HHT feature generation and model training
"""

import pandas as pd
import numpy as np
import sqlite3
import json
import yaml
import time
import logging
from pathlib import Path
import argparse

# Core ML imports
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
import optuna

# HHT imports
from PyEMD import EMD
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MinimalHHTProcessor:
    """Streamlined HHT processor for training"""
    
    def __init__(self, window_size=500, max_imfs=3):
        self.window_size = window_size
        self.max_imfs = max_imfs
        self.emd = EMD(max_imf=max_imfs)
        
    def process_price_series(self, prices):
        """Process entire price series and return HHT features"""
        logger.info(f"Processing {len(prices)} price points with HHT...")
        
        # Initialize feature arrays
        n_points = len(prices)
        hht_features = {
            'hht_trend_strength': np.zeros(n_points),
            'hht_trend_slope': np.zeros(n_points),
            'hht_cycle_phase': np.zeros(n_points),
            'hht_dominant_freq': np.zeros(n_points),
            'hht_inst_amplitude': np.zeros(n_points),
            'hht_regime_classifier': np.zeros(n_points),
            'hht_energy_high_freq': np.zeros(n_points),
            'hht_energy_mid_freq': np.zeros(n_points),
            'hht_energy_residual': np.zeros(n_points),
        }
        
        # Process in chunks for efficiency
        chunk_size = self.window_size
        for i in range(0, n_points, chunk_size // 4):  # 75% overlap
            end_idx = min(i + chunk_size, n_points)
            if end_idx - i < 100:  # Skip too small chunks
                continue
                
            chunk_prices = prices[i:end_idx]
            chunk_features = self._process_chunk(chunk_prices)
            
            # Fill features for this chunk
            for feature_name, values in chunk_features.items():
                if len(values) > 0:
                    # Use the last value for the entire chunk (point-in-time)
                    hht_features[feature_name][i:end_idx] = values[-1]
            
            if i % 10000 == 0:
                logger.info(f"Processed {i}/{n_points} points...")
        
        return hht_features
    
    def _process_chunk(self, prices):
        """Process a chunk of prices with EMD"""
        try:
            # EMD decomposition
            imfs = self.emd(np.array(prices, dtype=np.float32))
            
            if len(imfs) < 2:
                return self._default_features()
            
            # Calculate features
            trend_strength, trend_slope = self._calc_trend_metrics(imfs)
            cycle_phase, dominant_freq = self._calc_cycle_metrics(imfs)
            inst_amplitude = self._calc_amplitude(imfs)
            energy_dist = self._calc_energy_distribution(imfs)
            regime = self._classify_regime(imfs, trend_strength, trend_slope, energy_dist)
            
            return {
                'hht_trend_strength': [trend_strength],
                'hht_trend_slope': [trend_slope],
                'hht_cycle_phase': [cycle_phase],
                'hht_dominant_freq': [dominant_freq],
                'hht_inst_amplitude': [inst_amplitude],
                'hht_regime_classifier': [regime],
                'hht_energy_high_freq': [energy_dist[0]],
                'hht_energy_mid_freq': [energy_dist[1]],
                'hht_energy_residual': [energy_dist[2]],
            }
            
        except Exception as e:
            logger.warning(f"HHT processing failed: {e}")
            return self._default_features()
    
    def _default_features(self):
        return {
            'hht_trend_strength': [0.0],
            'hht_trend_slope': [0.0],
            'hht_cycle_phase': [0.0],
            'hht_dominant_freq': [0.0],
            'hht_inst_amplitude': [0.0],
            'hht_regime_classifier': [0],
            'hht_energy_high_freq': [0.0],
            'hht_energy_mid_freq': [0.0],
            'hht_energy_residual': [0.0],
        }
    
    def _calc_trend_metrics(self, imfs):
        residual = imfs[-1]
        residual_energy = np.var(residual)
        total_energy = sum(np.var(imf) for imf in imfs)
        trend_strength = residual_energy / (total_energy + 1e-8)
        
        if len(residual) >= 10:
            x = np.arange(len(residual))
            slope = np.polyfit(x, residual, 1)[0]
            mean_price = np.mean(residual)
            trend_slope = slope / (abs(mean_price) + 1e-8)
        else:
            trend_slope = 0.0
        
        return float(trend_strength), float(trend_slope)
    
    def _calc_cycle_metrics(self, imfs):
        if len(imfs) < 2:
            return 0.0, 0.0
        
        cycle_imf = imfs[1] if len(imfs) > 1 else imfs[0]
        try:
            from scipy.signal import hilbert
            analytic_signal = hilbert(cycle_imf)
            instantaneous_phase = np.angle(analytic_signal)
            current_phase = np.sin(instantaneous_phase[-1])
            
            if len(instantaneous_phase) > 10:
                phase_diff = np.diff(np.unwrap(instantaneous_phase))
                dominant_freq = np.mean(phase_diff) / (2 * np.pi)
            else:
                dominant_freq = 0.0
                
        except ImportError:
            current_phase = 0.0
            dominant_freq = 0.0
        
        return float(current_phase), float(abs(dominant_freq))
    
    def _calc_amplitude(self, imfs):
        if len(imfs) < 2:
            return 0.0
        
        energies = [np.var(imf) for imf in imfs[:-1]]
        if not energies:
            return 0.0
        
        dominant_idx = np.argmax(energies)
        dominant_imf = imfs[dominant_idx]
        
        recent_window = min(50, len(dominant_imf))
        recent_data = dominant_imf[-recent_window:]
        amplitude = np.sqrt(np.mean(recent_data**2))
        
        return float(amplitude)
    
    def _calc_energy_distribution(self, imfs):
        if len(imfs) < 2:
            return [0.0, 0.0, 1.0]
        
        energies = [np.var(imf) for imf in imfs]
        total_energy = sum(energies) + 1e-8
        
        high_freq_energy = energies[0] / total_energy if len(energies) > 0 else 0.0
        
        mid_freq_energy = 0.0
        if len(energies) > 2:
            mid_freq_energy = sum(energies[1:-1]) / total_energy
        
        residual_energy = energies[-1] / total_energy if len(energies) > 1 else 0.0
        
        return [float(high_freq_energy), float(mid_freq_energy), float(residual_energy)]
    
    def _classify_regime(self, imfs, trend_strength, trend_slope, energy_dist):
        high_freq_energy, mid_freq_energy, residual_energy = energy_dist
        
        if high_freq_energy > 0.6:
            return 2  # Noisy
        
        if residual_energy > 0.4 and trend_strength > 0.3:
            if abs(trend_slope) > 0.0001:
                return 1 if trend_slope > 0 else -1  # Trending
            else:
                return 0  # Ranging
        
        if mid_freq_energy > 0.4:
            return 0  # Cyclical
        
        return 0  # Default ranging

class MinimalL2FeatureEngineer:
    """Streamlined L2 feature engineer"""
    
    def __init__(self):
        self.ofi_windows = {'10s': 100, '30s': 300, '1m': 600, '5m': 3000}
    
    def generate_features(self, df):
        """Generate L2 microstructure features"""
        logger.info(f"Generating L2 features for {len(df)} rows...")
        
        # Basic spread and mid-price
        df['spread'] = df['ask_price_1'] - df['bid_price_1']
        df['mid_price'] = (df['ask_price_1'] + df['bid_price_1']) / 2
        df['spread_bps'] = (df['spread'] / df['mid_price']) * 10000
        
        # Weighted mid price
        total_volume = df['bid_size_1'] + df['ask_size_1']
        df['weighted_mid_price'] = (
            df['bid_price_1'] * df['bid_size_1'] + 
            df['ask_price_1'] * df['ask_size_1']
        ) / total_volume
        df['weighted_mid_price'] = df['weighted_mid_price'].fillna(df['mid_price'])
        
        # Order book imbalance
        df['order_book_imbalance'] = (df['bid_size_1'] - df['ask_size_1']) / total_volume
        
        # Order book imbalance at different levels
        for level in [2, 3, 5]:
            bid_sum = df[[f'bid_size_{i}' for i in range(1, level+1)]].sum(axis=1)
            ask_sum = df[[f'ask_size_{i}' for i in range(1, level+1)]].sum(axis=1)
            total_sum = bid_sum + ask_sum
            df[f'order_book_imbalance_{level}'] = (bid_sum - ask_sum) / total_sum
            df[f'order_book_imbalance_{level}'] = df[f'order_book_imbalance_{level}'].fillna(0)
        
        # Volatility features
        df['mid_price_return'] = df['mid_price'].pct_change()
        for window in [10, 50, 200]:
            df[f'l2_volatility_{window}'] = df['mid_price_return'].rolling(window=window).std()
        
        # Order flow imbalance
        self._add_ofi_features(df)
        
        # Handle missing values
        df = df.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        logger.info(f"Generated {len([c for c in df.columns if c not in ['timestamp', 'symbol']])} L2 features")
        return df
    
    def _add_ofi_features(self, df):
        """Add order flow imbalance features"""
        # Calculate volume changes
        for level in range(1, 6):
            df[f'bid_volume_change_{level}'] = df[f'bid_size_{level}'].diff()
            df[f'ask_volume_change_{level}'] = df[f'ask_size_{level}'].diff()
        
        # Calculate OFI for each window
        for window_name, window_size in self.ofi_windows.items():
            if window_size > len(df):
                df[f'ofi_{window_name}'] = 0.0
                continue
                
            bid_changes = df[[f'bid_volume_change_{i}' for i in range(1, 6)]].sum(axis=1)
            ask_changes = df[[f'ask_volume_change_{i}' for i in range(1, 6)]].sum(axis=1)
            
            df[f'ofi_{window_name}'] = (
                bid_changes.rolling(window=window_size, min_periods=1).sum() - 
                ask_changes.rolling(window=window_size, min_periods=1).sum()
            )
        
        # Drop intermediate columns
        cols_to_drop = [f'bid_volume_change_{i}' for i in range(1, 6)] + \
                      [f'ask_volume_change_{i}' for i in range(1, 6)]
        df.drop(columns=cols_to_drop, inplace=True)

def load_l2_data_from_db(db_path, limit=50000):
    """Load L2 data from SQLite database"""
    logger.info(f"Loading L2 data from {db_path}")
    
    conn = sqlite3.connect(db_path)
    query = f"""
    SELECT * FROM l2_training_data_practical 
    ORDER BY timestamp DESC 
    LIMIT {limit}
    """
    
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    logger.info(f"Loaded {len(df)} L2 records")
    return df

def generate_labels(df):
    """Generate trading labels"""
    logger.info("Generating labels...")
    
    # Use weighted_mid_price for labeling
    price_col = 'weighted_mid_price'
    returns = df[price_col].pct_change()
    
    # Calculate volatility
    vol_window = 50
    volatility = returns.rolling(window=vol_window).std()
    
    # Volatility-normalized returns with forward shift
    target = returns.shift(-1) / (volatility.fillna(method='ffill').fillna(1e-9) + 1e-9)
    
    # Clip extreme values
    target = target.clip(target.quantile(0.005), target.quantile(0.995))
    
    df['target'] = target
    df = df.dropna(subset=['target'])
    
    logger.info(f"Generated labels. Target mean: {target.mean():.6f}, std: {target.std():.6f}")
    return df, target.mean(), target.std()

def train_model(df, target_mean, target_std, n_trials=20):
    """Train LightGBM model with Optuna optimization"""
    logger.info("Starting model training...")
    
    # Prepare features
    exclude_cols = ['timestamp', 'symbol', 'target']
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    X = df[feature_cols].values
    y = df['target'].values
    
    logger.info(f"Training with {len(feature_cols)} features, {len(X)} samples")
    
    # Define objective function
    def objective(trial):
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.1, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 20, 150),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10, log=True),
            'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10, log=True),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
            'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
            'verbose': -1
        }
        
        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=3)
        rmse_scores = []
        
        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            model = lgb.LGBMRegressor(**params)
            model.fit(X_train, y_train)
            
            y_pred = model.predict(X_val)
            rmse = np.sqrt(mean_squared_error(y_val, y_pred))
            rmse_scores.append(rmse)
        
        return np.mean(rmse_scores)
    
    # Optimize
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials)
    
    # Train final model
    best_params = study.best_params
    best_params.update({'objective': 'regression', 'metric': 'rmse', 'verbose': -1})
    
    final_model = lgb.LGBMRegressor(**best_params)
    final_model.fit(X, y)
    
    logger.info(f"Training complete. Best RMSE: {study.best_value:.6f}")
    
    return final_model, feature_cols, study.best_params

def main():
    parser = argparse.ArgumentParser(description='HHT Model Training for Cursor')
    parser.add_argument('--db_path', default='trading_bot.db', help='Path to SQLite database')
    parser.add_argument('--trials', type=int, default=20, help='Optuna trials')
    parser.add_argument('--samples', type=int, default=50000, help='Number of samples to use')
    parser.add_argument('--output_dir', default='./models', help='Output directory for models')
    
    args = parser.parse_args()
    
    # Create output directory
    Path(args.output_dir).mkdir(exist_ok=True)
    
    start_time = time.time()
    
    # Load L2 data
    df = load_l2_data_from_db(args.db_path, args.samples)
    
    # Generate L2 features
    fe = MinimalL2FeatureEngineer()
    df = fe.generate_features(df)
    
    # Generate HHT features
    hht = MinimalHHTProcessor(window_size=500, max_imfs=3)
    hht_features = hht.process_price_series(df['weighted_mid_price'].values)
    
    # Add HHT features to dataframe
    for feature_name, values in hht_features.items():
        df[feature_name] = values
    
    logger.info(f"Total features: {len([c for c in df.columns if c not in ['timestamp', 'symbol']])}")
    
    # Generate labels
    df, target_mean, target_std = generate_labels(df)
    
    # Train model
    model, features, best_params = train_model(df, target_mean, target_std, args.trials)
    
    # Save model and metadata
    model_path = Path(args.output_dir) / 'hht_l2_model.txt'
    features_path = Path(args.output_dir) / 'model_features.json'
    scaling_path = Path(args.output_dir) / 'scaling_params.json'
    params_path = Path(args.output_dir) / 'best_params.json'
    
    model.booster_.save_model(str(model_path))
    
    with open(features_path, 'w') as f:
        json.dump({'features': features}, f, indent=2)
    
    with open(scaling_path, 'w') as f:
        json.dump({
            'target_mean': float(target_mean),
            'target_std': float(target_std),
            'features': features
        }, f, indent=2)
    
    with open(params_path, 'w') as f:
        json.dump(best_params, f, indent=2)
    
    total_time = time.time() - start_time
    logger.info(f"Training completed in {total_time/60:.1f} minutes")
    logger.info(f"Model saved to: {model_path}")
    logger.info(f"Features: {len(features)}")

if __name__ == "__main__":
    main()