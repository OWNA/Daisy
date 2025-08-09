#!/usr/bin/env python3
"""
L2-Only Live Simulation
Simulates live trading with L2-only trained model using real-time L2 data
"""

import os
import sys
import yaml
import pandas as pd
import numpy as np
import warnings
import time
import sqlite3
import argparse
from datetime import datetime, timedelta

# Suppress warnings
warnings.filterwarnings('ignore')

# Setup
sys.path.append('./')
os.environ['BOT_BASE_DIR'] = './'

# Import modules
from datahandler import DataHandler
from featureengineer import FeatureEngineer
from modelpredictor import ModelPredictor
from utils import load_model_and_features

# Parse arguments
parser = argparse.ArgumentParser(description='L2-Only Live Simulation')
parser.add_argument('--config', type=str, default='config.yaml',
                    help='Path to config file')
parser.add_argument('--l2-only', action='store_true', default=True,
                    help='Force L2-only mode (default: True)')
args = parser.parse_args()

print("🚀 L2-Only Live Trading Simulation")
print("="*60)

# Load config
config = yaml.safe_load(open(args.config, 'r'))
print(f"📋 Using config: {args.config}")

# Validate L2-only mode
if not config.get('l2_only_mode', False) and not args.l2_only:
    print("❌ ERROR: This script requires L2-only mode!")
    print("   Set 'l2_only_mode: true' in config or use --l2-only flag")
    sys.exit(1)

# Force L2-only mode
config['l2_only_mode'] = True
config['use_l2_features'] = True
config['use_l2_features_for_training'] = True

print("✅ L2-only mode enabled")

# Check if L2-only model exists
safe_symbol = config.get('symbol', 'BTC/USDT:USDT').replace('/', '_').replace(':', '')
model_path = f'lgbm_model_{safe_symbol}_l2_only.txt'
features_path = f'model_features_{safe_symbol}_l2_only.json'

# Fallback to original model if L2-specific doesn't exist
if not os.path.exists(model_path):
    model_path = f'lgbm_model_{safe_symbol}_1m.txt'
    features_path = f'model_features_{safe_symbol}_1m.json'

if not os.path.exists(model_path):
    print(f"❌ Model not found: {model_path}")
    print("Please run L2-only training first: python run_training_simple.py --l2-only")
    sys.exit(1)

print(f"✅ Using model: {model_path}")

# Initialize exchange (for L2 data streaming)
import ccxt
exchange = ccxt.bybit({
    'enableRateLimit': True,
    'options': {'defaultType': 'linear'},
    'testnet': config.get('exchange_testnet', True)
})

# Initialize components for L2-only mode
data_handler = DataHandler(config, exchange)
feature_engineer = FeatureEngineer(config)

# Load L2-only model
model, trained_features = load_model_and_features(model_path, features_path)

if model is None:
    print("❌ Failed to load L2-only model. Exiting.")
    sys.exit(1)

model_predictor = ModelPredictor(config, model, trained_features)

# L2 simulation parameters
initial_balance = config.get('initial_balance', 10000)
balance = initial_balance
position = 0
position_size = 0
entry_price = 0
trades = []
equity_curve = []
l2_metrics = []

# L2-specific settings
l2_buffer_size = config.get('l2_buffer_size', 1000)
l2_sampling_freq = config.get('l2_sampling_frequency_ms', 100)
feature_window = config.get('feature_window', 100)

# Simulation duration
sim_duration = config.get('simulation_duration_seconds', 1800)  # 30 minutes default
end_time = datetime.now() + timedelta(seconds=sim_duration)

print(f"📊 L2-Only Simulation Parameters:")
print(f"  Initial Balance: ${initial_balance:,.2f}")
print(f"  Duration: {sim_duration} seconds")
print(f"  Threshold: ±{config.get('simulation_threshold', 0.15)}")
print(f"  Symbol: {config['symbol']}")
print(f"  L2 Sampling: {l2_sampling_freq}ms")
print(f"  L2 Buffer Size: {l2_buffer_size}")

# Initialize L2 data buffer
l2_data_buffer = []

print("\n🔄 Starting L2-only simulation loop...")
print("Press Ctrl+C to stop early\n")

try:
    while datetime.now() < end_time:
        loop_start = time.time()
        
        # Fetch latest L2 data
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Fetching L2 data...")
        
        try:
            # Get L2 order book data
            order_book = exchange.fetch_order_book(config['symbol'], limit=50)
            
            # Convert to L2 data format
            l2_row = {
                'timestamp': datetime.now(),
                'bid_ask_spread': order_book['bids'][0][0] - order_book['asks'][0][0] if order_book['bids'] and order_book['asks'] else 0,
                'weighted_mid_price': (order_book['bids'][0][0] + order_book['asks'][0][0]) / 2 if order_book['bids'] and order_book['asks'] else 0,
                'order_book_imbalance_2': 0,  # Will be calculated by feature engineer
                'total_bid_volume_2': sum([bid[1] for bid in order_book['bids'][:2]]) if len(order_book['bids']) >= 2 else 0,
                'total_ask_volume_2': sum([ask[1] for ask in order_book['asks'][:2]]) if len(order_book['asks']) >= 2 else 0,
            }
            
            # Add to buffer
            l2_data_buffer.append(l2_row)
            
            # Maintain buffer size
            if len(l2_data_buffer) > l2_buffer_size:
                l2_data_buffer = l2_data_buffer[-l2_buffer_size:]
                
        except Exception as e:
            print(f"⚠️  Error fetching L2 data: {e}")
            time.sleep(1)
            continue
        
        # Check if we have enough L2 data
        if len(l2_data_buffer) < feature_window:
            print(f"⚠️  Insufficient L2 data: {len(l2_data_buffer)}/{feature_window}")
            time.sleep(l2_sampling_freq / 1000)
            continue
        
        # Convert buffer to DataFrame
        df_l2 = pd.DataFrame(l2_data_buffer)
        
        # Generate L2-only features
        try:
            df_features = feature_engineer.generate_all_features(df_l2)
            
            if df_features.empty or len(df_features) < feature_window:
                print("⚠️  Insufficient L2 features generated")
                time.sleep(l2_sampling_freq / 1000)
                continue
                
        except Exception as e:
            print(f"⚠️  Error generating L2 features: {e}")
            time.sleep(l2_sampling_freq / 1000)
            continue
        
        # Make L2-based prediction
        latest_features = df_features.iloc[-1]
        current_price = l2_row['weighted_mid_price']
        
        try:
            prediction = model_predictor.predict_single(latest_features)
        except Exception as e:
            print(f"⚠️  Prediction error: {e}")
            prediction = 0
        
        # L2-specific metrics
        spread = l2_row['bid_ask_spread']
        spread_pct = spread / current_price * 100 if current_price > 0 else 0
        
        # Trading logic with L2-aware thresholds
        threshold = config.get('simulation_threshold', 0.15)
        
        print(f"  Price: ${current_price:.4f}")
        print(f"  Spread: {spread:.4f} ({spread_pct:.4f}%)")
        print(f"  Prediction: {prediction:.4f}")
        print(f"  Position: {position}")
        print(f"  Balance: ${balance:.2f}")
        
        # Check for signals (L2-aware)
        if position == 0:  # No position
            if prediction > threshold and spread_pct < 0.1:  # L2: Check spread
                # Long signal
                position = 1
                position_size = balance * 0.95  # Use 95% of balance
                entry_price = current_price
                print(f"  📈 LONG signal! Entry at ${entry_price:.4f}")
                
            elif prediction < -threshold and spread_pct < 0.1:  # L2: Check spread
                # Short signal
                position = -1
                position_size = balance * 0.95
                entry_price = current_price
                print(f"  📉 SHORT signal! Entry at ${entry_price:.4f}")
                
        else:  # Have position
            # Calculate P&L
            if position == 1:
                pnl = (current_price - entry_price) / entry_price * position_size
            else:  # position == -1
                pnl = (entry_price - current_price) / entry_price * position_size
            
            pnl_pct = pnl / position_size * 100
            
            print(f"  P&L: ${pnl:.2f} ({pnl_pct:.2f}%)")
            
            # L2-aware exit logic
            exit_signal = False
            
            # Exit on opposite signal
            if (position == 1 and prediction < -threshold/2) or \
               (position == -1 and prediction > threshold/2):
                exit_signal = True
                print(f"  🔄 Exit signal (opposite direction)")
            
            # Exit on stop loss (1.5% for L2 - tighter)
            elif pnl_pct < -1.5:
                exit_signal = True
                print(f"  🛑 Stop loss triggered")
            
            # Exit on take profit (2.5% for L2 - tighter)
            elif pnl_pct > 2.5:
                exit_signal = True
                print(f"  ✅ Take profit triggered")
            
            # Exit on wide spread (liquidity concern)
            elif spread_pct > 0.2:
                exit_signal = True
                print(f"  ⚠️  Wide spread exit ({spread_pct:.4f}%)")
            
            if exit_signal:
                # Close position
                balance += pnl
                trades.append({
                    'entry_time': datetime.now() - timedelta(seconds=30),  # Approximate
                    'exit_time': datetime.now(),
                    'entry_price': entry_price,
                    'exit_price': current_price,
                    'position': position,
                    'pnl': pnl,
                    'pnl_pct': pnl_pct,
                    'spread_at_exit': spread_pct
                })
                
                print(f"  💰 Position closed. New balance: ${balance:.2f}")
                position = 0
                position_size = 0
                entry_price = 0
        
        # Record equity and L2 metrics
        current_equity = balance
        if position != 0:
            # Add unrealized P&L
            if position == 1:
                unrealized_pnl = (current_price - entry_price) / entry_price * position_size
            else:
                unrealized_pnl = (entry_price - current_price) / entry_price * position_size
            current_equity += unrealized_pnl
        
        equity_curve.append({
            'timestamp': datetime.now(),
            'equity': current_equity,
            'balance': balance,
            'position': position,
            'price': current_price,
            'spread_pct': spread_pct
        })
        
        l2_metrics.append({
            'timestamp': datetime.now(),
            'spread': spread,
            'spread_pct': spread_pct,
            'prediction': prediction,
            'l2_buffer_size': len(l2_data_buffer)
        })
        
        # L2-specific sleep (higher frequency)
        loop_duration = time.time() - loop_start
        sleep_time = max(0, l2_sampling_freq / 1000 - loop_duration)
        if sleep_time > 0:
            time.sleep(sleep_time)
            
except KeyboardInterrupt:
    print("\n⚠️  L2 simulation interrupted by user")

# Close any open position
if position != 0:
    current_price = l2_data_buffer[-1]['weighted_mid_price'] if l2_data_buffer else entry_price
    if position == 1:
        pnl = (current_price - entry_price) / entry_price * position_size
    else:
        pnl = (entry_price - current_price) / entry_price * position_size
    
    balance += pnl
    trades.append({
        'entry_time': datetime.now() - timedelta(seconds=30),
        'exit_time': datetime.now(),
        'entry_price': entry_price,
        'exit_price': current_price,
        'position': position,
        'pnl': pnl,
        'pnl_pct': pnl / position_size * 100,
        'spread_at_exit': 0
    })
    print(f"\n💰 Closed final position. P&L: ${pnl:.2f}")

# L2-Only Results summary
print("\n" + "="*60)
print("📊 L2-ONLY SIMULATION RESULTS")
print("="*60)

final_balance = balance
total_return = (final_balance - initial_balance) / initial_balance * 100

print(f"Initial Balance: ${initial_balance:,.2f}")
print(f"Final Balance: ${final_balance:,.2f}")
print(f"Total Return: {total_return:.2f}%")
print(f"Total Trades: {len(trades)}")

# L2-specific metrics
if l2_metrics:
    avg_spread = np.mean([m['spread_pct'] for m in l2_metrics])
    max_spread = np.max([m['spread_pct'] for m in l2_metrics])
    print(f"Average Spread: {avg_spread:.4f}%")
    print(f"Max Spread: {max_spread:.4f}%")

if trades:
    winning_trades = [t for t in trades if t['pnl'] > 0]
    losing_trades = [t for t in trades if t['pnl'] < 0]
    
    print(f"Winning Trades: {len(winning_trades)}")
    print(f"Losing Trades: {len(losing_trades)}")
    
    if len(trades) > 0:
        win_rate = len(winning_trades) / len(trades) * 100
        print(f"Win Rate: {win_rate:.1f}%")
    
    if winning_trades:
        avg_win = sum(t['pnl'] for t in winning_trades) / len(winning_trades)
        print(f"Average Win: ${avg_win:.2f}")
    
    if losing_trades:
        avg_loss = sum(t['pnl'] for t in losing_trades) / len(losing_trades)
        print(f"Average Loss: ${avg_loss:.2f}")

# Save L2-only results
print("\n💾 Saving L2-only results...")

# Ensure directories exist
os.makedirs('backtest_results', exist_ok=True)
os.makedirs('paper_trading_results', exist_ok=True)

# Save trades with L2 metadata
if trades:
    trades_df = pd.DataFrame(trades)
    trades_path = os.path.join('paper_trading_results', 'l2_simulation_trades.csv')
    trades_df.to_csv(trades_path, index=False)
    print(f"✅ L2 trades saved to {trades_path}")

# Save equity curve with L2 data
if equity_curve:
    equity_df = pd.DataFrame(equity_curve)
    equity_path = os.path.join('paper_trading_results', 'l2_simulation_equity.csv')
    equity_df.to_csv(equity_path, index=False)
    print(f"✅ L2 equity curve saved to {equity_path}")

# Save L2 metrics
if l2_metrics:
    l2_metrics_df = pd.DataFrame(l2_metrics)
    l2_metrics_path = os.path.join('paper_trading_results', 'l2_simulation_metrics.csv')
    l2_metrics_df.to_csv(l2_metrics_path, index=False)
    print(f"✅ L2 metrics saved to {l2_metrics_path}")

print("\n🎉 L2-Only Simulation complete!")
print("📋 Next steps:")
print("   1. Analyze results: python analyze_predictions.py")
print("   2. Start live trading: python run_trading_bot.py --config config.yaml")
print("="*60) 