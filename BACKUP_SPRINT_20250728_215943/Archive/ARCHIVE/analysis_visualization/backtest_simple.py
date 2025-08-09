#!/usr/bin/env python3
"""
Simple backtest that works with your trained models
"""

import os
import sys
import json
import gzip
import pandas as pd
import numpy as np
import lightgbm as lgb
from datetime import datetime
from pathlib import Path

def run_simple_backtest(model_file, data_file=None, max_rows=10000):
    """Run a simple backtest"""
    print("\n" + "="*60)
    print("SIMPLE BACKTEST")
    print("="*60)
    
    # Load model
    print(f"\nLoading model: {model_file}")
    model = lgb.Booster(model_file=model_file)
    
    # Load feature list
    features_file = model_file.replace('lgbm_model_', 'model_features_').replace('.txt', '.json')
    if os.path.exists(features_file):
        with open(features_file, 'r') as f:
            metadata = json.load(f)
        feature_cols = metadata.get('features', [])
        print(f"Features: {len(feature_cols)}")
    else:
        print("Warning: No features file found")
        feature_cols = model.feature_name()
    
    # Load L2 data
    if not data_file:
        # Use most recent L2 data
        l2_files = sorted(Path('l2_data').glob('*.jsonl.gz'), 
                         key=lambda x: x.stat().st_mtime, reverse=True)
        if l2_files:
            data_file = l2_files[0]
        else:
            print("Error: No L2 data files found")
            return
    
    print(f"\nLoading data: {data_file}")
    
    # Load and process data (same as training)
    records = []
    with gzip.open(data_file, 'rt') as f:
        for i, line in enumerate(f):
            if i >= max_rows:
                break
            record = json.loads(line)
            if 'bids' in record and 'b' not in record:
                record['b'] = record['bids']
                record['a'] = record['asks']
            records.append(record)
    
    print(f"Loaded {len(records)} records")
    
    # Convert to features
    from featureengineer import FeatureEngineer
    
    # Create DataFrame
    rows = []
    for record in records:
        if 'b' not in record or 'a' not in record:
            continue
        
        bids = record['b']
        asks = record['a']
        
        if not bids or not asks:
            continue
        
        row = {'timestamp': record.get('timestamp', datetime.now().isoformat())}
        
        # Add L2 data
        for level in range(1, 11):
            if len(bids) >= level:
                row[f'bid_price_{level}'] = float(bids[level-1][0])
                row[f'bid_size_{level}'] = float(bids[level-1][1])
            if len(asks) >= level:
                row[f'ask_price_{level}'] = float(asks[level-1][0])
                row[f'ask_size_{level}'] = float(asks[level-1][1])
        
        row['mid_price'] = (float(bids[0][0]) + float(asks[0][0])) / 2
        row['close'] = row['mid_price']
        rows.append(row)
    
    df = pd.DataFrame(rows)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.set_index('timestamp')
    
    # Generate features
    config = {'symbol': 'BTC/USDT:USDT', 'use_l2_features': True, 
              'use_hht_features': True, 'l2_only_mode': True}
    feature_engineer = FeatureEngineer(config)
    df_features = feature_engineer.generate_all_features(df)
    
    print(f"Generated {len(df_features)} feature rows")
    
    # Run backtest
    print("\nRunning backtest...")
    
    # Initialize
    position = 0
    balance = 10000
    trades = []
    results = []
    
    # Make predictions
    for i in range(100, len(df_features)):  # Start after warmup
        try:
            # Get features for prediction
            X = df_features.iloc[i:i+1][feature_cols]
            
            # Handle missing features
            for col in feature_cols:
                if col not in X.columns:
                    X[col] = 0
            
            # Make prediction
            pred = model.predict(X)[0]
            
            # Current price
            current_price = df_features.iloc[i]['close']
            timestamp = df_features.index[i]
            
            # Trading logic
            action = ''
            if pred > 0.0001 and position == 0:
                # Buy signal
                position = 0.001  # Buy 0.001 BTC
                action = 'buy'
                trades.append({
                    'timestamp': timestamp,
                    'action': 'buy',
                    'price': current_price,
                    'prediction': pred
                })
            elif pred < -0.0001 and position > 0:
                # Sell signal
                action = 'sell'
                buy_price = trades[-1]['price']
                pnl = (current_price - buy_price) * position
                balance += pnl
                trades.append({
                    'timestamp': timestamp,
                    'action': 'sell',
                    'price': current_price,
                    'prediction': pred,
                    'pnl': pnl
                })
                position = 0
            
            # Record result
            results.append({
                'timestamp': timestamp,
                'price': current_price,
                'position': position,
                'balance': balance,
                'equity': balance + (position * current_price if position > 0 else 0),
                'action': action,
                'prediction': pred
            })
            
            if i % 1000 == 0:
                print(f"  Processed {i}/{len(df_features)} | Trades: {len(trades)} | Balance: ${balance:.2f}")
                
        except Exception as e:
            if i == 100:  # Only print error once
                print(f"Warning: {e}")
            continue
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    # Calculate metrics
    total_trades = len([t for t in trades if t['action'] == 'sell'])
    if total_trades > 0:
        wins = len([t for t in trades if t.get('pnl', 0) > 0])
        win_rate = wins / total_trades * 100
    else:
        win_rate = 0
    
    final_balance = balance
    total_return = (final_balance - 10000) / 10000 * 100
    
    print(f"\n" + "="*60)
    print("BACKTEST RESULTS")
    print("="*60)
    print(f"Total trades: {total_trades}")
    print(f"Win rate: {win_rate:.1f}%")
    print(f"Final balance: ${final_balance:,.2f}")
    print(f"Total return: {total_return:.2f}%")
    
    # Save results
    output_dir = Path('backtest_results')
    output_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = output_dir / f"backtest_{timestamp}.csv"
    
    results_df.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")
    
    return output_file

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python backtest_simple.py <model_file> [data_file] [max_rows]")
        # Use latest model
        models = sorted(Path('.').glob('lgbm_model_*.txt'), 
                       key=lambda x: x.stat().st_mtime, reverse=True)
        if models:
            model_file = str(models[0])
            print(f"\nUsing latest model: {model_file}")
        else:
            print("No models found!")
            sys.exit(1)
    else:
        model_file = sys.argv[1]
    
    data_file = sys.argv[2] if len(sys.argv) > 2 else None
    max_rows = int(sys.argv[3]) if len(sys.argv) > 3 else 10000
    
    run_simple_backtest(model_file, data_file, max_rows)