#!/usr/bin/env python3
"""
Direct training that ACTUALLY WORKS with your L2 data
"""

import os
import sys
import json
import gzip
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import lightgbm as lgb
from sklearn.model_selection import train_test_split
import optuna

# Add current directory
sys.path.insert(0, '.')

from featureengineer import FeatureEngineer
from labelgenerator import LabelGenerator

def train_your_l2_data(data_file, trials=50):
    print("\n" + "="*60)
    print("TRAINING YOUR L2 DATA - DIRECT METHOD")
    print("="*60)
    
    # Step 1: Load L2 data directly
    print(f"\nLoading {data_file}...")
    records = []
    
    with gzip.open(data_file, 'rt') as f:
        for i, line in enumerate(f):
            if i % 10000 == 0:
                print(f"  Loaded {i} records...")
            record = json.loads(line)
            
            # Convert format if needed
            if 'bids' in record and 'b' not in record:
                record['b'] = record['bids']
                record['a'] = record['asks']
            
            records.append(record)
    
    print(f"✓ Loaded {len(records)} records")
    
    # Step 2: Convert to DataFrame format expected by FeatureEngineer
    print("\nConverting to DataFrame...")
    rows = []
    
    for i, record in enumerate(records):
        if i % 10000 == 0:
            print(f"  Processing {i}/{len(records)}...")
            
        if 'b' not in record or 'a' not in record:
            continue
            
        bids = record['b']
        asks = record['a']
        
        if not bids or not asks:
            continue
        
        # Create row with L2 data
        row = {
            'timestamp': record.get('timestamp', datetime.now().isoformat()),
        }
        
        # Add bid/ask levels
        for level in range(1, 11):  # 10 levels
            if len(bids) >= level:
                row[f'bid_price_{level}'] = float(bids[level-1][0])
                row[f'bid_size_{level}'] = float(bids[level-1][1])
            else:
                row[f'bid_price_{level}'] = np.nan
                row[f'bid_size_{level}'] = np.nan
                
            if len(asks) >= level:
                row[f'ask_price_{level}'] = float(asks[level-1][0])
                row[f'ask_size_{level}'] = float(asks[level-1][1])
            else:
                row[f'ask_price_{level}'] = np.nan
                row[f'ask_size_{level}'] = np.nan
        
        # Add mid price
        row['mid_price'] = (float(bids[0][0]) + float(asks[0][0])) / 2
        row['close'] = row['mid_price']  # FeatureEngineer expects 'close'
        
        rows.append(row)
    
    df = pd.DataFrame(rows)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.set_index('timestamp')
    
    print(f"✓ Created DataFrame with {len(df)} rows")
    
    # Step 3: Generate features
    print("\nGenerating features...")
    config = {
        'symbol': 'BTC/USDT:USDT',
        'use_l2_features': True,
        'use_hht_features': True,
        'l2_only_mode': True
    }
    
    feature_engineer = FeatureEngineer(config)
    df_features = feature_engineer.generate_all_features(df)
    
    print(f"✓ Generated {len(df_features.columns)} features")
    
    # Show feature breakdown
    l2_features = [col for col in df_features.columns if any(x in col for x in ['spread', 'imbalance', 'depth', 'slope'])]
    hht_features = [col for col in df_features.columns if 'hht' in col]
    
    print(f"  L2 features: {len(l2_features)}")
    print(f"  HHT features: {len(hht_features)}")
    
    # Step 4: Generate labels
    print("\nGenerating labels...")
    label_generator = LabelGenerator(config)
    df_labeled, target_mean, target_std = label_generator.generate_labels(df_features)
    
    print(f"✓ Generated labels")
    print(f"  Samples: {len(df_labeled)}")
    print(f"  Target mean: {target_mean:.6f}")
    print(f"  Target std: {target_std:.6f}")
    
    # Step 5: Train model
    print(f"\nTraining model with {trials} Optuna trials...")
    
    # Prepare data
    feature_cols = [col for col in df_labeled.columns if col not in ['target', 'timestamp', 'close', 'mid_price']]
    X = df_labeled[feature_cols]
    y = df_labeled['target']
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Optuna optimization
    def objective(trial):
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': trial.suggest_int('num_leaves', 20, 300),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
            'verbosity': -1
        }
        
        train_data = lgb.Dataset(X_train, label=y_train)
        valid_data = lgb.Dataset(X_val, label=y_val)
        
        model = lgb.train(
            params,
            train_data,
            valid_sets=[valid_data],
            num_boost_round=100,
            callbacks=[lgb.early_stopping(10), lgb.log_evaluation(0)]
        )
        
        predictions = model.predict(X_val)
        mse = np.mean((y_val - predictions) ** 2)
        return mse
    
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=trials, show_progress_bar=True)
    
    print(f"\n✓ Optimization complete!")
    print(f"  Best MSE: {study.best_value:.6f}")
    
    # Train final model
    best_params = study.best_params
    best_params.update({'objective': 'regression', 'metric': 'rmse', 'verbosity': -1})
    
    train_data = lgb.Dataset(X_train, label=y_train)
    final_model = lgb.train(
        best_params,
        train_data,
        num_boost_round=200,
        callbacks=[lgb.log_evaluation(10)]
    )
    
    # Calculate R²
    y_pred = final_model.predict(X_val)
    r2 = 1 - (np.mean((y_val - y_pred) ** 2) / np.var(y_val))
    print(f"  R² score: {r2:.4f}")
    
    # Feature importance
    importance = final_model.feature_importance(importance_type='gain')
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': importance
    }).sort_values('importance', ascending=False)
    
    print("\nTop 10 features:")
    for idx, row in feature_importance.head(10).iterrows():
        print(f"  {row['feature']:<40} {row['importance']:>10.2f}")
    
    # Check HHT contribution
    hht_importance = feature_importance[feature_importance['feature'].str.contains('hht')]['importance'].sum()
    total_importance = feature_importance['importance'].sum()
    hht_pct = (hht_importance / total_importance * 100) if total_importance > 0 else 0
    print(f"\nHHT features contribution: {hht_pct:.1f}%")
    
    # Save model
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_file = f"lgbm_model_l2_hht_{timestamp}.txt"
    features_file = f"model_features_l2_hht_{timestamp}.json"
    
    final_model.save_model(model_file)
    
    with open(features_file, 'w') as f:
        json.dump({
            'features': feature_cols,
            'r2_score': r2,
            'mse': study.best_value,
            'hht_contribution_pct': hht_pct,
            'total_features': len(feature_cols),
            'l2_features': len(l2_features),
            'hht_features': len(hht_features)
        }, f, indent=2)
    
    print(f"\n✅ TRAINING COMPLETE!")
    print(f"Model saved: {model_file}")
    print(f"Features saved: {features_file}")
    
    return model_file

if __name__ == "__main__":
    # Train on your specific file
    if len(sys.argv) > 1:
        data_file = sys.argv[1]
    else:
        # Default to your file
        data_file = "l2_data/l2_data_BTC_USDT_USDT_20250707_040413.jsonl.gz"
    
    if not os.path.exists(data_file):
        # Try the converted version
        data_file = "l2_data/l2_data_040413_converted.jsonl.gz"
    
    if not os.path.exists(data_file):
        print(f"Error: Data file not found: {data_file}")
        sys.exit(1)
    
    trials = int(sys.argv[2]) if len(sys.argv) > 2 else 50
    
    train_your_l2_data(data_file, trials)