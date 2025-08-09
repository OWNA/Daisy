#!/usr/bin/env python3
"""
Train Model on Live WebSocket Data

Uses the fresh live data we collected to train a production-ready model.
"""
import sqlite3
import pandas as pd
import numpy as np
import pickle
import os
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import json
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

class LiveModelTrainer:
    """Train models on live WebSocket data."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.db_path = './trading_bot_live.db'
        self.model_dir = './trading_bot_data'
        
        # Ensure model directory exists
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Model configurations
        self.prediction_horizons = [10, 30, 100, 300]  # Future price prediction horizons
        self.feature_columns = []
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        
        print("LiveModelTrainer initialized")
        print(f"Database: {self.db_path}")
        print(f"Model directory: {self.model_dir}")
    
    def load_live_data(self, hours_back: int = 2) -> pd.DataFrame:
        """Load recent live WebSocket data."""
        try:
            print(f"Loading live data from last {hours_back} hours...")
            
            # Calculate time threshold - load most recent data
            time_threshold = datetime.now() - timedelta(hours=hours_back)
            
            conn = sqlite3.connect(self.db_path)
            
            query = """
                SELECT * FROM l2_training_data_practical 
                WHERE data_source = 'websocket_live' 
                AND timestamp >= ? 
                ORDER BY timestamp DESC
                LIMIT 2000
            """
            
            df = pd.read_sql_query(query, conn, params=(time_threshold.isoformat(),))
            conn.close()
            
            if df.empty:
                print("No live data found!")
                return pd.DataFrame()
            
            # Convert timestamp
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            print(f"Loaded {len(df)} rows of live data")
            print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
            print(f"Price range: ${df['mid_price'].min():.2f} to ${df['mid_price'].max():.2f}")
            
            return df
            
        except Exception as e:
            print(f"Error loading live data: {e}")
            return pd.DataFrame()
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer features from raw L2 data."""
        try:
            print("Engineering features from live data...")
            
            df_features = df.copy()
            
            # Basic price features
            df_features['price_return_1'] = df_features['mid_price'].pct_change(1)
            df_features['price_return_5'] = df_features['mid_price'].pct_change(5)
            df_features['price_return_10'] = df_features['mid_price'].pct_change(10)
            
            # Volatility features
            df_features['volatility_10'] = df_features['price_return_1'].rolling(10).std()
            df_features['volatility_30'] = df_features['price_return_1'].rolling(30).std()
            df_features['volatility_100'] = df_features['price_return_1'].rolling(100).std()
            
            # Spread features
            df_features['spread_bps'] = (df_features['spread'] / df_features['mid_price']) * 10000
            df_features['spread_ma_10'] = df_features['spread_bps'].rolling(10).mean()
            df_features['spread_ma_30'] = df_features['spread_bps'].rolling(30).mean()
            
            # Volume features
            df_features['total_bid_volume'] = (
                df_features['bid_size_1'] + df_features['bid_size_2'] + 
                df_features['bid_size_3'] + df_features['bid_size_4'] + df_features['bid_size_5']
            )
            df_features['total_ask_volume'] = (
                df_features['ask_size_1'] + df_features['ask_size_2'] + 
                df_features['ask_size_3'] + df_features['ask_size_4'] + df_features['ask_size_5']
            )
            
            # Imbalance features
            total_volume = df_features['total_bid_volume'] + df_features['total_ask_volume']
            df_features['volume_imbalance'] = (
                (df_features['total_bid_volume'] - df_features['total_ask_volume']) / 
                (total_volume + 1e-8)
            )
            
            # Price level features
            df_features['bid_ask_ratio'] = df_features['bid_price_1'] / df_features['ask_price_1']
            df_features['depth_ratio'] = df_features['total_bid_volume'] / df_features['total_ask_volume']
            
            # Moving averages
            df_features['price_ma_10'] = df_features['mid_price'].rolling(10).mean()
            df_features['price_ma_30'] = df_features['mid_price'].rolling(30).mean()
            df_features['price_ma_100'] = df_features['mid_price'].rolling(100).mean()
            
            # Price momentum
            df_features['momentum_10'] = df_features['mid_price'] / df_features['price_ma_10'] - 1
            df_features['momentum_30'] = df_features['mid_price'] / df_features['price_ma_30'] - 1
            
            # Volume momentum
            df_features['volume_ma_10'] = total_volume.rolling(10).mean()
            df_features['volume_momentum'] = total_volume / df_features['volume_ma_10'] - 1
            
            # Weighted price features
            df_features['weighted_price_diff'] = (
                df_features['weighted_bid_price'] - df_features['weighted_ask_price']
            )
            
            # Microstructure features
            df_features['microprice_diff'] = df_features['microprice'] - df_features['mid_price']
            df_features['microprice_momentum'] = df_features['microprice'].pct_change(5)
            
            # Fill NaN values and replace infinites
            df_features = df_features.replace([np.inf, -np.inf], np.nan)
            df_features = df_features.ffill().bfill().fillna(0)
            
            # Select feature columns (exclude target and metadata columns)
            exclude_cols = [
                'id', 'timestamp', 'symbol', 'exchange', 'data_source', 
                'created_at', 'data_quality_score'
            ]
            
            feature_cols = [col for col in df_features.columns if col not in exclude_cols]
            self.feature_columns = feature_cols
            
            print(f"Engineered {len(feature_cols)} features")
            print("Feature columns:", feature_cols[:10], "..." if len(feature_cols) > 10 else "")
            
            return df_features
            
        except Exception as e:
            print(f"Error engineering features: {e}")
            return df
    
    def create_targets(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create prediction targets for different horizons."""
        try:
            print("Creating prediction targets...")
            
            df_targets = df.copy()
            
            # Create targets for different horizons
            for horizon in self.prediction_horizons:
                # Future price return
                future_price = df_targets['mid_price'].shift(-horizon)
                df_targets[f'target_return_{horizon}'] = (
                    (future_price - df_targets['mid_price']) / df_targets['mid_price']
                )
                
                # Future volatility
                future_returns = df_targets['price_return_1'].shift(-horizon)
                df_targets[f'target_volatility_{horizon}'] = (
                    future_returns.rolling(horizon//2, min_periods=1).std().shift(-horizon//2)
                )
                
                # Direction (classification target)
                df_targets[f'target_direction_{horizon}'] = np.where(
                    df_targets[f'target_return_{horizon}'] > 0.0005, 1,  # Up > 0.05%
                    np.where(df_targets[f'target_return_{horizon}'] < -0.0005, -1, 0)  # Down < -0.05%
                )
            
            print(f"Created targets for horizons: {self.prediction_horizons}")
            
            return df_targets
            
        except Exception as e:
            print(f"Error creating targets: {e}")
            return df
    
    def train_models(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Train models for each prediction horizon."""
        try:
            print("Training models...")
            
            # Prepare features - ensure numeric types only
            feature_df = df[self.feature_columns].select_dtypes(include=[np.number])
            self.feature_columns = feature_df.columns.tolist()  # Update feature columns to numeric only
            
            X = feature_df.values.astype(np.float64)  # Ensure float64
            
            # Clean data: replace infinites and extreme values
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Remove rows with NaN targets  
            valid_rows = ~np.isnan(X).any(axis=1)
            for horizon in self.prediction_horizons:
                target_col = f'target_return_{horizon}'
                if target_col in df.columns:
                    valid_rows &= ~np.isnan(df[target_col].values)
            
            X = X[valid_rows]
            
            if len(X) < 100:
                print(f"Warning: Only {len(X)} valid samples - may not be enough for training")
            
            results = {}
            
            for horizon in self.prediction_horizons:
                print(f"\nTraining model for {horizon}-step horizon...")
                
                target_col = f'target_return_{horizon}'
                if target_col not in df.columns:
                    continue
                
                y = df[target_col].values[valid_rows]
                
                # Skip if not enough data
                if len(y) < 50:
                    print(f"Skipping horizon {horizon} - insufficient data")
                    continue
                
                # Split data
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42, shuffle=False
                )
                
                # Scale features
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                # Train ensemble of models
                models = {
                    'rf': RandomForestRegressor(
                        n_estimators=100, 
                        max_depth=10, 
                        min_samples_split=10,
                        random_state=42,
                        n_jobs=-1
                    ),
                    'gbm': GradientBoostingRegressor(
                        n_estimators=100,
                        max_depth=6,
                        learning_rate=0.1,
                        random_state=42
                    )
                }
                
                horizon_results = {}
                
                for model_name, model in models.items():
                    print(f"  Training {model_name}...")
                    
                    # Train model
                    model.fit(X_train_scaled, y_train)
                    
                    # Evaluate
                    y_pred_train = model.predict(X_train_scaled)
                    y_pred_test = model.predict(X_test_scaled)
                    
                    train_r2 = r2_score(y_train, y_pred_train)
                    test_r2 = r2_score(y_test, y_pred_test)
                    test_mse = mean_squared_error(y_test, y_pred_test)
                    
                    # Cross validation
                    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=3, scoring='r2')
                    
                    horizon_results[model_name] = {
                        'model': model,
                        'scaler': scaler,
                        'train_r2': train_r2,
                        'test_r2': test_r2,
                        'test_mse': test_mse,
                        'cv_mean': cv_scores.mean(),
                        'cv_std': cv_scores.std()
                    }
                    
                    print(f"    Train R²: {train_r2:.4f}")
                    print(f"    Test R²: {test_r2:.4f}")
                    print(f"    CV R²: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
                    
                    # Feature importance
                    if hasattr(model, 'feature_importances_'):
                        importance = dict(zip(self.feature_columns, model.feature_importances_))
                        horizon_results[model_name]['feature_importance'] = importance
                
                results[horizon] = horizon_results
            
            return results
            
        except Exception as e:
            print(f"Error training models: {e}")
            return {}
    
    def save_models(self, models_dict: Dict[str, Any]) -> bool:
        """Save trained models and metadata."""
        try:
            print("Saving models...")
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            for horizon, horizon_models in models_dict.items():
                for model_name, model_data in horizon_models.items():
                    # Save model
                    model_filename = f"model_{model_name}_h{horizon}_{timestamp}.joblib"
                    model_path = os.path.join(self.model_dir, model_filename)
                    joblib.dump(model_data['model'], model_path)
                    
                    # Save scaler
                    scaler_filename = f"scaler_{model_name}_h{horizon}_{timestamp}.joblib"
                    scaler_path = os.path.join(self.model_dir, scaler_filename)
                    joblib.dump(model_data['scaler'], scaler_path)
                    
                    print(f"Saved {model_name} model for horizon {horizon}")
            
            # Save metadata
            metadata = {
                'timestamp': timestamp,
                'feature_columns': self.feature_columns,
                'prediction_horizons': self.prediction_horizons,
                'model_performance': {}
            }
            
            # Add performance metrics
            for horizon, horizon_models in models_dict.items():
                metadata['model_performance'][str(horizon)] = {}
                for model_name, model_data in horizon_models.items():
                    metadata['model_performance'][str(horizon)][model_name] = {
                        'train_r2': model_data['train_r2'],
                        'test_r2': model_data['test_r2'],
                        'test_mse': model_data['test_mse'],
                        'cv_mean': model_data['cv_mean'],
                        'cv_std': model_data['cv_std']
                    }
            
            # Save metadata
            metadata_path = os.path.join(self.model_dir, f"enhanced_features_BTCUSDT.json")
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            
            print(f"Models and metadata saved to {self.model_dir}")
            print(f"Metadata file: enhanced_features_BTCUSDT.json")
            
            return True
            
        except Exception as e:
            print(f"Error saving models: {e}")
            return False
    
    def run_full_training(self) -> bool:
        """Run complete training pipeline."""
        try:
            print("="*60)
            print("LIVE MODEL TRAINING PIPELINE")
            print("="*60)
            
            # 1. Load live data
            df_raw = self.load_live_data(hours_back=24)  # Use last 24 hours of data
            if df_raw.empty:
                print("No data available for training")
                return False
            
            # 2. Engineer features
            df_features = self.engineer_features(df_raw)
            
            # 3. Create targets
            df_targets = self.create_targets(df_features)
            
            # 4. Train models
            models_dict = self.train_models(df_targets)
            
            if not models_dict:
                print("No models were trained successfully")
                return False
            
            # 5. Save models
            success = self.save_models(models_dict)
            
            if success:
                print("\n" + "="*60)
                print("TRAINING COMPLETE!")
                print("="*60)
                print(f"Models trained on {len(df_raw)} samples")
                print(f"Features: {len(self.feature_columns)}")
                print(f"Horizons: {self.prediction_horizons}")
                print(f"Saved to: {self.model_dir}")
                print("\nReady for paper trading!")
            
            return success
            
        except Exception as e:
            print(f"Training pipeline failed: {e}")
            return False


def main():
    """Main training function."""
    trainer = LiveModelTrainer()
    success = trainer.run_full_training()
    
    if success:
        print("\\nModels ready for paper trading!")
        print("Run: ./venv/Scripts/python.exe run.py")
    else:
        print("\\nTraining failed")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())