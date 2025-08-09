#!/usr/bin/env python3
"""
Unified Trading System - Complete workflow that actually works
"""

import os
import sys
import json
import gzip
import yaml
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import time
import threading
import subprocess
import lightgbm as lgb
from sklearn.model_selection import train_test_split
import optuna
import websocket
import ccxt
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()

class UnifiedTradingSystem:
    def __init__(self):
        self.config = self.load_config()
        self.console = Console()
        
    def load_config(self):
        """Load or create config"""
        if os.path.exists('config.yaml'):
            with open('config.yaml', 'r') as f:
                return yaml.safe_load(f)
        else:
            # Create default config
            config = {
                'symbol': 'BTC/USDT:USDT',
                'exchange': 'bybit',
                'initial_balance': 10000,
                'risk_per_trade': 0.01,
                'l2_levels': 10
            }
            with open('config.yaml', 'w') as f:
                yaml.dump(config, f)
            return config
    
    # ========== DATA COLLECTION ==========
    
    def collect_l2_data(self, duration_seconds=300):
        """Collect L2 data via WebSocket"""
        console.print(f"\n[cyan]Collecting L2 data for {duration_seconds} seconds...[/cyan]")
        
        # Create output filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = f"l2_data/l2_data_{timestamp}.jsonl.gz"
        
        # Ensure directory exists
        os.makedirs('l2_data', exist_ok=True)
        
        collected_count = 0
        start_time = time.time()
        
        def on_message(ws, message):
            nonlocal collected_count
            try:
                data = json.loads(message)
                
                # Handle Bybit orderbook format
                if 'topic' in data and 'orderbook' in data['topic'] and 'data' in data:
                    orderbook = data['data']
                    
                    # Convert to our standard format
                    record = {
                        'timestamp': datetime.now().isoformat(),
                        'exchange': 'bybit',
                        'symbol': self.config['symbol'],
                        'bids': orderbook.get('b', []),
                        'asks': orderbook.get('a', [])
                    }
                    
                    # Save to file
                    with gzip.open(output_file, 'at') as f:
                        json.dump(record, f)
                        f.write('\n')
                    
                    collected_count += 1
                    
                    # Progress update
                    elapsed = time.time() - start_time
                    if collected_count % 100 == 0:
                        console.print(f"  Collected: {collected_count} | Elapsed: {elapsed:.1f}s", end='\r')
                    
                    # Check if done
                    if elapsed >= duration_seconds:
                        ws.close()
                        
            except Exception as e:
                console.print(f"[red]Error: {e}[/red]")
        
        def on_error(ws, error):
            console.print(f"[red]WebSocket error: {error}[/red]")
        
        def on_close(ws, close_status_code, close_msg):
            console.print(f"\n[green]✓ Collection complete![/green]")
        
        def on_open(ws):
            # Subscribe to orderbook
            subscribe_msg = {
                "op": "subscribe",
                "args": [f"orderbook.50.BTCUSDT"]
            }
            ws.send(json.dumps(subscribe_msg))
            console.print("[green]Connected to Bybit WebSocket[/green]")
        
        # Connect to WebSocket
        ws_url = "wss://stream.bybit.com/v5/public/linear"
        ws = websocket.WebSocketApp(ws_url,
                                    on_open=on_open,
                                    on_message=on_message,
                                    on_error=on_error,
                                    on_close=on_close)
        
        # Run WebSocket
        ws.run_forever()
        
        # Summary
        file_size = os.path.getsize(output_file) / 1024 / 1024
        console.print(f"\n[cyan]Summary:[/cyan]")
        console.print(f"  File: {output_file}")
        console.print(f"  Records: {collected_count}")
        console.print(f"  Size: {file_size:.2f} MB")
        console.print(f"  Rate: {collected_count/duration_seconds:.1f} records/sec")
        
        return output_file
    
    # ========== DATA PROCESSING ==========
    
    def load_and_process_l2_data(self, file_path):
        """Load L2 data and extract features"""
        console.print(f"\n[cyan]Loading L2 data from {file_path}...[/cyan]")
        
        records = []
        with gzip.open(file_path, 'rt') as f:
            for line in f:
                records.append(json.loads(line))
        
        console.print(f"  Loaded {len(records)} records")
        
        # Extract features
        features_list = []
        for record in records:
            bids = record.get('bids', [])
            asks = record.get('asks', [])
            
            if not bids or not asks:
                continue
            
            # Basic L2 features
            features = {
                'timestamp': record['timestamp'],
                'bid_price_1': float(bids[0][0]),
                'bid_size_1': float(bids[0][1]),
                'ask_price_1': float(asks[0][0]),
                'ask_size_1': float(asks[0][1]),
                'mid_price': (float(bids[0][0]) + float(asks[0][0])) / 2,
                'spread': float(asks[0][0]) - float(bids[0][0]),
            }
            
            # More levels
            for i in range(2, min(6, len(bids), len(asks))):
                features[f'bid_price_{i}'] = float(bids[i-1][0])
                features[f'bid_size_{i}'] = float(bids[i-1][1])
                features[f'ask_price_{i}'] = float(asks[i-1][0])
                features[f'ask_size_{i}'] = float(asks[i-1][1])
            
            # Microstructure features
            total_bid_size = sum(float(b[1]) for b in bids[:5])
            total_ask_size = sum(float(a[1]) for a in asks[:5])
            features['imbalance'] = (total_bid_size - total_ask_size) / (total_bid_size + total_ask_size)
            features['spread_pct'] = features['spread'] / features['mid_price'] * 100
            
            features_list.append(features)
        
        # Convert to DataFrame
        df = pd.DataFrame(features_list)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp').sort_index()
        
        # Add technical features
        df['returns_1m'] = df['mid_price'].pct_change()
        df['returns_5m'] = df['mid_price'].pct_change(5)
        df['volatility'] = df['returns_1m'].rolling(20).std()
        df['spread_ma'] = df['spread_pct'].rolling(10).mean()
        df['imbalance_ma'] = df['imbalance'].rolling(10).mean()
        
        # Generate labels (predict 5-min returns)
        df['target'] = df['mid_price'].shift(-5) / df['mid_price'] - 1
        
        # Clean data
        df = df.dropna()
        
        console.print(f"  Processed {len(df)} samples with {len(df.columns)} features")
        
        return df
    
    # ========== MODEL TRAINING ==========
    
    def train_model(self, df, trials=50):
        """Train model with Optuna"""
        console.print(f"\n[cyan]Training model with {trials} Optuna trials...[/cyan]")
        
        # Prepare data
        feature_cols = [col for col in df.columns if col not in ['target', 'mid_price']]
        X = df[feature_cols]
        y = df['target']
        
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        
        console.print(f"  Features: {len(feature_cols)}")
        console.print(f"  Train samples: {len(X_train)}")
        console.print(f"  Val samples: {len(X_val)}")
        
        # Optuna optimization
        def objective(trial):
            params = {
                'objective': 'regression',
                'metric': 'rmse',
                'num_leaves': trial.suggest_int('num_leaves', 20, 200),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
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
        
        # Train final model
        best_params = study.best_params
        best_params.update({'objective': 'regression', 'metric': 'rmse', 'verbosity': -1})
        
        train_data = lgb.Dataset(X_train, label=y_train)
        final_model = lgb.train(
            best_params,
            train_data,
            num_boost_round=200,
            callbacks=[lgb.log_evaluation(0)]
        )
        
        # Save model
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_file = f"model_unified_{timestamp}.txt"
        final_model.save_model(model_file)
        
        # Save features
        feature_file = f"features_unified_{timestamp}.json"
        with open(feature_file, 'w') as f:
            json.dump({
                'features': feature_cols,
                'mse': study.best_value,
                'params': best_params
            }, f, indent=2)
        
        console.print(f"\n[green]✓ Model saved: {model_file}[/green]")
        console.print(f"[green]✓ Features saved: {feature_file}[/green]")
        console.print(f"[green]✓ Best MSE: {study.best_value:.6f}[/green]")
        
        return model_file, feature_cols
    
    # ========== LIVE SIMULATION ==========
    
    def run_live_simulation(self, model_file, feature_cols, duration=300, output_file=None):
        """Run live trading simulation"""
        console.print(f"\n[cyan]Running live simulation for {duration} seconds...[/cyan]")
        
        # Load model
        model = lgb.Booster(model_file=model_file)
        
        # Initialize trading state
        position = 0
        balance = self.config['initial_balance']
        trades = []
        
        # Output file for monitoring
        if output_file:
            with open(output_file, 'w') as f:
                f.write('timestamp,price,position,balance,action\n')
        
        # Feature buffer
        feature_buffer = []
        
        def on_message(ws, message):
            nonlocal position, balance, feature_buffer
            
            try:
                data = json.loads(message)
                
                if 'topic' in data and 'orderbook' in data['topic'] and 'data' in data:
                    orderbook = data['data']
                    bids = orderbook.get('b', [])
                    asks = orderbook.get('a', [])
                    
                    if not bids or not asks:
                        return
                    
                    # Extract current features
                    current_features = {
                        'bid_price_1': float(bids[0][0]),
                        'bid_size_1': float(bids[0][1]),
                        'ask_price_1': float(asks[0][0]),
                        'ask_size_1': float(asks[0][1]),
                        'mid_price': (float(bids[0][0]) + float(asks[0][0])) / 2,
                        'spread': float(asks[0][0]) - float(bids[0][0]),
                    }
                    
                    # Add to buffer
                    feature_buffer.append(current_features)
                    if len(feature_buffer) > 100:
                        feature_buffer.pop(0)
                    
                    # Need enough data for features
                    if len(feature_buffer) < 20:
                        return
                    
                    # Calculate features from buffer
                    df_buffer = pd.DataFrame(feature_buffer)
                    
                    # Add all required features
                    features_dict = {}
                    for col in feature_cols:
                        if col in df_buffer.columns:
                            features_dict[col] = df_buffer[col].iloc[-1]
                        else:
                            # Calculate or set default
                            if 'returns' in col:
                                features_dict[col] = df_buffer['mid_price'].pct_change().iloc[-1]
                            elif 'volatility' in col:
                                features_dict[col] = df_buffer['mid_price'].pct_change().std()
                            elif 'ma' in col:
                                features_dict[col] = df_buffer['spread'].mean() if 'spread' in col else df_buffer['imbalance'].mean()
                            else:
                                features_dict[col] = 0
                    
                    # Make prediction
                    X = pd.DataFrame([features_dict])
                    prediction = model.predict(X)[0]
                    
                    # Trading logic
                    current_price = current_features['mid_price']
                    action = ''
                    
                    if prediction > 0.001 and position <= 0:  # Buy signal
                        position = 0.01  # Buy 0.01 BTC
                        action = 'buy'
                        trades.append({
                            'time': datetime.now(),
                            'action': 'buy',
                            'price': current_price,
                            'size': position
                        })
                    elif prediction < -0.001 and position > 0:  # Sell signal
                        action = 'sell'
                        trades.append({
                            'time': datetime.now(),
                            'action': 'sell',
                            'price': current_price,
                            'size': position
                        })
                        position = 0
                    
                    # Update balance
                    if position > 0:
                        balance = self.config['initial_balance'] + (current_price - trades[-1]['price']) * position
                    
                    # Save to output
                    if output_file:
                        with open(output_file, 'a') as f:
                            f.write(f"{datetime.now()},{current_price},{position},{balance},{action}\n")
                    
                    # Display update
                    console.print(f"Price: ${current_price:,.2f} | Position: {position:.4f} | Balance: ${balance:,.2f} | Pred: {prediction:.6f}", end='\r')
                    
            except Exception as e:
                console.print(f"\n[red]Error: {e}[/red]")
        
        # WebSocket handlers
        def on_error(ws, error):
            console.print(f"[red]WebSocket error: {error}[/red]")
        
        def on_close(ws, close_status_code, close_msg):
            console.print(f"\n[green]Simulation complete![/green]")
        
        def on_open(ws):
            subscribe_msg = {
                "op": "subscribe",
                "args": [f"orderbook.50.BTCUSDT"]
            }
            ws.send(json.dumps(subscribe_msg))
        
        # Run simulation
        ws_url = "wss://stream.bybit.com/v5/public/linear"
        ws = websocket.WebSocketApp(ws_url,
                                    on_open=on_open,
                                    on_message=on_message,
                                    on_error=on_error,
                                    on_close=on_close)
        
        # Run for specified duration
        timer = threading.Timer(duration, lambda: ws.close())
        timer.start()
        
        ws.run_forever()
        
        # Summary
        console.print(f"\n[cyan]Simulation Summary:[/cyan]")
        console.print(f"  Total trades: {len(trades)}")
        console.print(f"  Final balance: ${balance:,.2f}")
        console.print(f"  P&L: ${balance - self.config['initial_balance']:,.2f}")
        
        if output_file:
            console.print(f"  Results saved to: {output_file}")
    
    # ========== MAIN WORKFLOW ==========
    
    def run_complete_workflow(self, collect_duration=300, train_trials=50, sim_duration=300):
        """Run complete workflow: collect → train → simulate"""
        console.print("\n[bold cyan]COMPLETE TRADING WORKFLOW[/bold cyan]")
        console.print("="*50)
        
        # Step 1: Collect data
        console.print("\n[yellow]Step 1: Collecting L2 data...[/yellow]")
        data_file = self.collect_l2_data(collect_duration)
        
        # Step 2: Process data
        console.print("\n[yellow]Step 2: Processing data...[/yellow]")
        df = self.load_and_process_l2_data(data_file)
        
        # Step 3: Train model
        console.print("\n[yellow]Step 3: Training model...[/yellow]")
        model_file, feature_cols = self.train_model(df, train_trials)
        
        # Step 4: Run simulation
        console.print("\n[yellow]Step 4: Running live simulation...[/yellow]")
        output_file = f"simulation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        self.run_live_simulation(model_file, feature_cols, sim_duration, output_file)
        
        console.print("\n[bold green]✓ WORKFLOW COMPLETE![/bold green]")

# CLI Interface
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Unified Trading System')
    parser.add_argument('command', choices=['collect', 'train', 'simulate', 'workflow'],
                       help='Command to run')
    parser.add_argument('--duration', type=int, default=300, help='Duration in seconds')
    parser.add_argument('--data', help='Data file for training')
    parser.add_argument('--model', help='Model file for simulation')
    parser.add_argument('--trials', type=int, default=50, help='Optuna trials')
    parser.add_argument('--output', help='Output file for results')
    
    args = parser.parse_args()
    
    system = UnifiedTradingSystem()
    
    if args.command == 'collect':
        system.collect_l2_data(args.duration)
    
    elif args.command == 'train':
        if not args.data:
            console.print("[red]Please specify --data file[/red]")
        else:
            df = system.load_and_process_l2_data(args.data)
            system.train_model(df, args.trials)
    
    elif args.command == 'simulate':
        if not args.model:
            console.print("[red]Please specify --model file[/red]")
        else:
            # Load features from model metadata
            feature_file = args.model.replace('.txt', '.json').replace('model_', 'features_')
            with open(feature_file, 'r') as f:
                metadata = json.load(f)
            system.run_live_simulation(args.model, metadata['features'], args.duration, args.output)
    
    elif args.command == 'workflow':
        system.run_complete_workflow(args.duration, args.trials, args.duration)