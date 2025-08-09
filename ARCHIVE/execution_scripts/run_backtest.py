#!/usr/bin/env python3
"""
L2-Only Backtest Simulation
Run backtest simulation with L2-only trained model
"""

import os
import sys
import yaml
import pandas as pd
import warnings
import json
import sqlite3
import argparse
from strategybacktester import StrategyBacktester
from modelpredictor import ModelPredictor
from featureengineer import FeatureEngineer
from advancedriskmanager import AdvancedRiskManager

# Suppress warnings
warnings.filterwarnings('ignore')

# Setup
sys.path.append('./')
os.environ['BOT_BASE_DIR'] = './'

# Parse arguments
parser = argparse.ArgumentParser(description='L2-Only Backtest Simulation')
parser.add_argument(
    '--config', type=str, default='config.yaml', help='Path to config file'
)
parser.add_argument(
    '--l2-only',
    action='store_true',
    default=True,
    help='Force L2-only mode (default: True)'
)
args = parser.parse_args()

print("🚀 L2-Only Backtest Simulation")
print("="*60)

# Load config
with open(args.config, 'r') as f:
    config = yaml.safe_load(f)
print(f"📋 Using config: {args.config}")

# Validate L2-only mode
if not config.get('l2_only_mode', False) and not args.l2_only:
    print("❌ ERROR: This script requires L2-only mode!")
    msg = "   Set 'l2_only_mode: true' in config or use --l2-only flag"
    print(msg)
    sys.exit(1)

# Force L2-only mode
config['l2_only_mode'] = True
config['use_l2_features'] = True
config['use_l2_features_for_training'] = True

print("✅ L2-only mode enabled")

# Check if L2-only model exists
safe_symbol = config.get(
    'symbol', 'BTC/USDT:USDT'
).replace('/', '_').replace(':', '')
model_path = f'lgbm_model_{safe_symbol}_l2_only.txt'
features_path = f'model_features_{safe_symbol}_l2_only.json'

# Fallback to original model if L2-specific doesn't exist
if not os.path.exists(model_path):
    model_path = f'lgbm_model_{safe_symbol}_1m.txt'
    features_path = f'model_features_{safe_symbol}_1m.json'

if not os.path.exists(model_path):
    print(f"❌ Model not found: {model_path}")
    msg = (
        "Please run L2-only training first: "
        "python run_training_simple.py --l2-only"
    )
    print(msg)
    sys.exit(1)

if not os.path.exists(features_path):
    print(f"❌ Features config not found: {features_path}")
    sys.exit(1)

print(f"✅ Using model: {model_path}")

# Load L2 data from database
print("\n📂 Loading L2 data from database...")
try:
    db_path = config.get('database_path', './trading_bot.db')
    
    with sqlite3.connect(db_path) as conn:
        # Try the new practical table first, fallback to original
        try:
            query = """
            SELECT * FROM l2_training_data_practical 
            ORDER BY timestamp 
            LIMIT 100000
            """
            df = pd.read_sql_query(query, conn)
            print("✅ Using l2_training_data_practical table")
        except sqlite3.Error:
            query = """
            SELECT * FROM l2_training_data 
            ORDER BY timestamp 
            LIMIT 100000
            """
            df = pd.read_sql_query(query, conn)
            print("✅ Using l2_training_data table")
    
    if df is None or df.empty:
        print("❌ No L2 data found in database!")
        msg = "   Please upload L2 data first using data_upload_manager.py"
        print(msg)
        sys.exit(1)
        
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    print(f"✅ Loaded {len(df)} L2 rows")
    date_range = f"{df['timestamp'].min()} to {df['timestamp'].max()}"
    print(f"📅 Date range: {date_range}")
    
    # Check for required L2 columns
    required_l2_cols = ['bid_ask_spread', 'weighted_mid_price', 
                        'order_book_imbalance_2']
    missing_cols = [col for col in required_l2_cols if col not in df.columns]
    if missing_cols:
        print(f"⚠️  Missing L2 columns: {missing_cols}")
    else:
        print("✅ Required L2 columns present")
        
except Exception as e:
    print(f"❌ Error loading L2 data: {e}")
    sys.exit(1)

# Load feature configuration
with open(features_path, 'r') as f:
    feature_info = json.load(f)
    trained_features = feature_info.get('trained_features', [])
    print(f"✅ Loaded {len(trained_features)} trained features")

# Initialize components
print("\n🔧 Initializing L2-only components...")

# Feature Engineer - Use L2-only config
fe = FeatureEngineer(config)

# Model Predictor
mp = ModelPredictor(config)
# Load the model
if not mp.load_model_and_features(model_path, features_path):
    print("❌ Failed to load model")
    sys.exit(1)
print("✅ L2-only model loaded")

# Generate L2 features
print("\n📊 Generating L2 features...")
df_features = fe.generate_features(df)
print(f"✅ L2 features generated: {df_features.shape}")

# Validate L2 features
l2_terms = ['l2_', 'bid_ask', 'weighted_mid', 'microprice', 
            'order_book', 'price_impact', 'slope', 'volatility']
l2_feature_count = len([col for col in df_features.columns 
                       if any(l2_term in col.lower() 
                              for l2_term in l2_terms)])
print(f"📈 L2-specific features: {l2_feature_count}")

# Check if we have the required features
missing_features = [
    f for f in trained_features if f not in df_features.columns
]
if missing_features:
    # Show first 10
    print(f"⚠️  Missing features: {missing_features[:10]}...")
    print("This may cause prediction failures")

# Initialize L2-aware backtester
print("\n🔄 Initializing L2-aware backtester...")
risk_manager = AdvancedRiskManager(config)
backtester = StrategyBacktester(config, risk_manager)

# Run L2-only backtest
print("\n📈 Running L2-only backtest simulation...")
print("This may take a few minutes...")

# Prepare data for backtesting
print("🤖 Generating L2-based predictions...")

# Make predictions using the L2-only model
predictions = []
successful_predictions = 0
feature_window = config.get('feature_window', 100)

for i in range(len(df_features)):
    if i < feature_window:
        predictions.append(0)  # No prediction for initial rows
    else:
        try:
            # Get features for this row - only use features that exist
            available_features = [
                f for f in trained_features if f in df_features.columns
            ]
            # Need at least 80% of features
            if len(available_features) < len(trained_features) * 0.8:
                predictions.append(0)
                continue
                
            row_features = df_features.iloc[i][available_features].values
            if hasattr(row_features, 'reshape'):
                row_features = row_features.reshape(1, -1)

            pred = mp.predict(row_features)
            if isinstance(pred, (list, pd.Series)):
                predictions.append(float(pred[0]))
            else:
                predictions.append(float(pred))
            successful_predictions += 1
        except Exception as e:
            if i < 170:  # Only show first few errors
                print(f"Warning: L2 prediction failed for row {i}: {e}")
            predictions.append(0.0)

df_features['prediction'] = predictions
total_preds = len(predictions)
print(
    f"✅ Generated {total_preds} L2 predictions "
    f"({successful_predictions} successful)"
)

# Add required columns for L2-aware backtester
threshold = config.get('prediction_threshold', 0.15)
df_features['signal'] = df_features['prediction'].apply(
    lambda x: 1 if x > threshold else -1 if x < -threshold else 0
)

# Use L2-derived price for backtesting
if 'weighted_mid_price' in df_features.columns:
    df_features['close'] = df_features['weighted_mid_price']
elif 'microprice' in df_features.columns:
    df_features['close'] = df_features['microprice']
else:
    print("⚠️  No L2 price column found, using first numeric column")
    numeric_cols = df_features.select_dtypes(include=['number']).columns
    df_features['close'] = df_features[numeric_cols[0]]

# Run L2-only backtest
try:
    equity_curve, trades = backtester.run_backtest(df_features)
    
    # Display L2-specific results
    if equity_curve is not None and not equity_curve.empty:
        print("\n📊 L2-Only Backtest Results:")
        print("="*60)
        
        # Calculate basic metrics from equity curve
        if 'equity' in equity_curve.columns:
            initial_balance = equity_curve['equity'].iloc[0]
            final_balance = equity_curve['equity'].iloc[-1]
            total_return = (final_balance - initial_balance) / initial_balance
            print(f"Initial Balance: ${initial_balance:.2f}")
            print(f"Final Balance: ${final_balance:.2f}")
            print(f"Total Return: {total_return:.2%}")
            
        print(f"Total Trades: {len(trades) if trades is not None else 0}")
        print(f"L2 Features Used: {l2_feature_count}")
        print(f"Data Points: {len(equity_curve)}")
        
        final_equity = final_balance if 'equity' in equity_curve.columns else 1.0
        initial_balance_config = config.get('initial_balance', 10000)
        print(f"\nFinal Equity: ${final_equity:.2f} "
              f"(started with ${initial_balance_config:.2f})")
        
        # Save L2-specific results
        os.makedirs('backtest_results', exist_ok=True)
        results_file = os.path.join(
            'backtest_results',
            'l2_only_backtest_results.json'
        )
        
        # Create results dictionary
        results_dict = {
            'backtest_summary': {
                'initial_balance': (
                    initial_balance
                    if 'equity' in equity_curve.columns
                    else initial_balance_config
                ),
                'final_balance': (
                    final_balance
                    if 'equity' in equity_curve.columns
                    else initial_balance_config
                ),
                'total_return': (
                    total_return if 'equity' in equity_curve.columns else 0.0
                ),
                'total_trades': len(trades) if trades is not None else 0,
                'data_points': len(equity_curve)
            },
            'l2_metadata': {
                'l2_only_mode': True,
                'l2_feature_count': l2_feature_count,
                'model_path': model_path,
                'data_source': 'l2_training_data',
                'prediction_threshold': threshold
            }
        }
        
        with open(results_file, 'w') as f:
            json.dump(results_dict, f, indent=2, default=str)
        print(f"\n💾 L2-only results saved to {results_file}")
        
    else:
        print("\n❌ L2-only backtest failed - no results returned")
        
except Exception as e:
    print(f"\n❌ L2-only backtest failed with error: {e}")
    print("This might be due to insufficient L2 data or configuration issues")

print("\n" + "="*60)
print("🎯 L2-Only Backtest completed!")
print("📋 Next steps:")
print(
    "   1. Analyze results: "
    "python analyze_backtest_results.py"
)
print(
    "   2. Run live simulation: "
    "python run_simulation.py --config config.yaml"
)
print(
    "   3. Start live trading: "
    "python run_trading_bot.py --config config.yaml"
)
print("="*60)
