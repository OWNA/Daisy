#!/usr/bin/env python3
"""
Direct training script that works with L2 data files
"""

import sys
import os
import json
import gzip
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import lightgbm as lgb
from sklearn.model_selection import train_test_split
import optuna
from optuna.samplers import TPESampler

def load_l2_data_direct(file_path):
    """Load L2 data directly from file"""
    print(f"\nLoading L2 data from {file_path}...")
    
    records = []
    with gzip.open(file_path, 'rt') as f:
        for i, line in enumerate(f):
            if i % 10000 == 0:
                print(f"  Loaded {i} records...")
            
            record = json.loads(line)
            records.append(record)
    
    print(f"✓ Loaded {len(records)} records")
    return records

def extract_l2_features(records):
    """Extract L2 features from records"""
    print("\nExtracting L2 features...")
    
    features_list = []
    
    for i, record in enumerate(records):
        if i % 10000 == 0:
            print(f"  Processing record {i}/{len(records)}...")
        
        # Get bids and asks
        bids = record.get('b', record.get('bids', []))
        asks = record.get('a', record.get('asks', []))
        
        if not bids or not asks:
            continue
        
        # Extract features
        features = {
            'timestamp': record.get('timestamp', ''),
            'bid_price_1': float(bids[0][0]) if len(bids) > 0 else np.nan,
            'bid_size_1': float(bids[0][1]) if len(bids) > 0 else np.nan,
            'ask_price_1': float(asks[0][0]) if len(asks) > 0 else np.nan,
            'ask_size_1': float(asks[0][1]) if len(asks) > 0 else np.nan,
        }
        
        # Add more levels
        for level in range(2, 6):  # Levels 2-5
            if len(bids) >= level:
                features[f'bid_price_{level}'] = float(bids[level-1][0])
                features[f'bid_size_{level}'] = float(bids[level-1][1])
            else:
                features[f'bid_price_{level}'] = np.nan
                features[f'bid_size_{level}'] = np.nan
                
            if len(asks) >= level:
                features[f'ask_price_{level}'] = float(asks[level-1][0])
                features[f'ask_size_{level}'] = float(asks[level-1][1])
            else:
                features[f'ask_price_{level}'] = np.nan
                features[f'ask_size_{level}'] = np.nan
        
        # Calculate microstructure features
        features['mid_price'] = (features['bid_price_1'] + features['ask_price_1']) / 2
        features['spread'] = features['ask_price_1'] - features['bid_price_1']
        features['spread_pct'] = features['spread'] / features['mid_price'] * 100
        
        # Imbalance
        total_bid_size = sum(float(bid[1]) for bid in bids[:5])
        total_ask_size = sum(float(ask[1]) for ask in asks[:5])
        features['imbalance'] = (total_bid_size - total_ask_size) / (total_bid_size + total_ask_size) if (total_bid_size + total_ask_size) > 0 else 0
        
        # Depth
        features['bid_depth'] = total_bid_size
        features['ask_depth'] = total_ask_size
        features['total_depth'] = total_bid_size + total_ask_size
        
        features_list.append(features)
    
    # Convert to DataFrame
    df = pd.DataFrame(features_list)
    
    # Sort by timestamp
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')
        df = df.set_index('timestamp')
    
    print(f"✓ Extracted features for {len(df)} samples")
    print(f"  Features: {list(df.columns)}")
    
    return df

def add_technical_features(df):
    """Add technical indicators"""
    print("\nAdding technical features...")
    
    # Price changes
    df['price_change_1m'] = df['mid_price'].diff()
    df['price_change_5m'] = df['mid_price'].diff(5)
    df['price_change_pct_1m'] = df['mid_price'].pct_change()
    df['price_change_pct_5m'] = df['mid_price'].pct_change(5)
    
    # Moving averages
    df['ma_5'] = df['mid_price'].rolling(5).mean()
    df['ma_20'] = df['mid_price'].rolling(20).mean()
    df['ma_diff'] = df['ma_5'] - df['ma_20']
    
    # Volatility
    df['volatility_5m'] = df['price_change_1m'].rolling(5).std()
    df['volatility_20m'] = df['price_change_1m'].rolling(20).std()
    
    # Volume features
    df['volume_imbalance_ma5'] = df['imbalance'].rolling(5).mean()
    df['spread_ma5'] = df['spread_pct'].rolling(5).mean()
    
    return df

def generate_labels(df):
    """Generate target labels for prediction"""
    print("\nGenerating labels...")
    
    # Predict 5-minute future return
    df['future_return'] = df['mid_price'].shift(-5) / df['mid_price'] - 1
    
    # Remove NaN values
    df = df.dropna()
    
    print(f"✓ Generated labels for {len(df)} samples")
    print(f"  Target mean: {df['future_return'].mean():.6f}")
    print(f"  Target std: {df['future_return'].std():.6f}")
    
    return df

def train_model(df, optuna_trials=50):
    """Train LightGBM model with Optuna"""
    print("\nPreparing for training...")
    
    # Separate features and target
    feature_cols = [col for col in df.columns if col not in ['future_return', 'mid_price']]
    X = df[feature_cols]
    y = df['future_return']
    
    print(f"  Features: {len(feature_cols)}")
    print(f"  Samples: {len(X)}")
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"  Training samples: {len(X_train)}")
    print(f"  Validation samples: {len(X_val)}")
    
    # Optuna optimization
    print(f"\nRunning Optuna optimization ({optuna_trials} trials)...")
    
    def objective(trial):
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': trial.suggest_int('num_leaves', 20, 300),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
            'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            'verbosity': -1
        }
        
        # Train model
        train_data = lgb.Dataset(X_train, label=y_train)
        valid_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        model = lgb.train(
            params,
            train_data,
            valid_sets=[valid_data],
            num_boost_round=100,
            callbacks=[lgb.early_stopping(10), lgb.log_evaluation(0)]
        )
        
        # Predict and calculate R²
        y_pred = model.predict(X_val, num_iteration=model.best_iteration)
        mse = np.mean((y_val - y_pred) ** 2)
        r2 = 1 - (mse / np.var(y_val))
        
        return r2
    
    # Run optimization
    study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42))
    study.optimize(objective, n_trials=optuna_trials, show_progress_bar=True)
    
    print(f"\n✓ Optimization complete!")
    print(f"  Best R²: {study.best_value:.4f}")
    print(f"  Best params: {study.best_params}")
    
    # Train final model with best params
    print("\nTraining final model...")
    best_params = study.best_params
    best_params.update({
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'verbosity': -1
    })
    
    train_data = lgb.Dataset(X_train, label=y_train)
    valid_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
    final_model = lgb.train(
        best_params,
        train_data,
        valid_sets=[valid_data],
        num_boost_round=200,
        callbacks=[lgb.early_stopping(20), lgb.log_evaluation(10)]
    )
    
    # Feature importance
    importance = final_model.feature_importance(importance_type='gain')
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': importance
    }).sort_values('importance', ascending=False)
    
    print("\nTop 10 features:")
    for idx, row in feature_importance.head(10).iterrows():
        print(f"  {row['feature']:<30} {row['importance']:>10.2f}")
    
    return final_model, feature_cols, study.best_value

def save_model(model, features, r2_score, data_file):
    """Save model and metadata"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save model
    model_filename = f"lgbm_model_direct_{timestamp}.txt"
    model.save_model(model_filename)
    print(f"\n✓ Model saved: {model_filename}")
    
    # Save features
    features_filename = f"model_features_direct_{timestamp}.json"
    metadata = {
        'features': features,
        'total_features': len(features),
        'r2_score': r2_score,
        'training_file': data_file,
        'timestamp': timestamp
    }
    
    with open(features_filename, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✓ Features saved: {features_filename}")
    
    return model_filename, features_filename

def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True, help='L2 data file')
    parser.add_argument('--trials', type=int, default=50, help='Optuna trials')
    parser.add_argument('--sample', type=int, help='Sample N records (for testing)')
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("DIRECT L2 MODEL TRAINING")
    print("="*60)
    
    # Load data
    records = load_l2_data_direct(args.data)
    
    if args.sample:
        records = records[:args.sample]
        print(f"\nUsing sample of {len(records)} records")
    
    # Extract features
    df = extract_l2_features(records)
    
    # Add technical features
    df = add_technical_features(df)
    
    # Generate labels
    df = generate_labels(df)
    
    # Train model
    model, features, r2_score = train_model(df, args.trials)
    
    # Save model
    model_file, features_file = save_model(model, features, r2_score, args.data)
    
    print("\n" + "="*60)
    print("✅ TRAINING COMPLETE!")
    print("="*60)
    print(f"Model: {model_file}")
    print(f"R² Score: {r2_score:.4f}")
    print(f"Features: {len(features)}")

if __name__ == "__main__":
    main()