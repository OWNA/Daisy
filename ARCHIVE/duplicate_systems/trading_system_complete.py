#!/usr/bin/env python3
"""
Complete Trading System - Everything in one file that ACTUALLY WORKS
"""

import os
import sys
import json
import gzip
import time
import threading
import subprocess
from datetime import datetime
from pathlib import Path

# Data processing
import pandas as pd
import numpy as np

# Machine learning
import lightgbm as lgb
from sklearn.model_selection import train_test_split

# WebSocket for data collection
import websocket

# Rich for UI
from rich.console import Console
from rich.prompt import Prompt, IntPrompt, Confirm
from rich.panel import Panel

console = Console()

class CompleteTradingSystem:
    def __init__(self):
        self.console = console
        self.symbol = 'BTCUSDT'
        self.ws_url = "wss://stream.bybit.com/v5/public/linear"
        
        # Create directories
        Path('l2_data').mkdir(exist_ok=True)
        Path('models').mkdir(exist_ok=True)
        Path('results').mkdir(exist_ok=True)
        
    def clear_screen(self):
        os.system('cls' if os.name == 'nt' else 'clear')
        
    # ============ MENU SYSTEM ============
    
    def run(self):
        """Main entry point"""
        while True:
            self.clear_screen()
            console.print(Panel.fit(
                "[bold cyan]🤖 Complete Trading System[/bold cyan]\n" +
                "[dim]Everything works from collection to trading[/dim]",
                border_style="cyan"
            ))
            
            console.print("\n[bold]Choose an option:[/bold]\n")
            console.print("1. 📊 Collect L2 Data (WebSocket)")
            console.print("2. 🧠 Train Model on Collected Data")
            console.print("3. 🤖 Run Trading Simulation")
            console.print("4. ⚡ Quick Demo (30 sec each step)")
            console.print("5. 📁 View Your Files")
            console.print("6. ❌ Exit\n")
            
            choice = Prompt.ask("[yellow]Select[/yellow]", choices=["1","2","3","4","5","6"])
            
            try:
                if choice == "1":
                    self.collect_data_menu()
                elif choice == "2":
                    self.train_model_menu()
                elif choice == "3":
                    self.run_simulation_menu()
                elif choice == "4":
                    self.quick_demo()
                elif choice == "5":
                    self.view_files()
                elif choice == "6":
                    console.print("\n[green]Goodbye! 👋[/green]")
                    break
            except Exception as e:
                console.print(f"\n[red]Error: {e}[/red]")
                Prompt.ask("\nPress Enter to continue")
    
    # ============ DATA COLLECTION ============
    
    def collect_data_menu(self):
        """Menu for data collection"""
        self.clear_screen()
        console.print("[bold cyan]📊 Collect L2 Data[/bold cyan]\n")
        
        duration = IntPrompt.ask("Collection duration in seconds", default=300)
        
        console.print(f"\n[yellow]Starting L2 data collection for {duration} seconds...[/yellow]")
        console.print("[dim]Connecting to Bybit WebSocket...[/dim]\n")
        
        filename = self.collect_l2_data(duration)
        
        if filename:
            console.print(f"\n[green]✓ Data saved to: {filename}[/green]")
        
        Prompt.ask("\nPress Enter to continue")
    
    def collect_l2_data(self, duration_seconds):
        """Actually collect L2 data from Bybit"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = f"l2_data/l2_data_{timestamp}.jsonl.gz"
        
        self.collection_count = 0
        self.collection_start = time.time()
        self.collection_file = output_file
        self.collection_running = True
        
        def on_message(ws, message):
            if not self.collection_running:
                return
                
            try:
                data = json.loads(message)
                
                # Look for orderbook data
                if 'topic' in data and 'orderbook' in data['topic'] and 'data' in data:
                    orderbook = data['data']
                    
                    # Save in simple format
                    record = {
                        'timestamp': datetime.now().isoformat(),
                        'bids': orderbook.get('b', [])[:10],  # Top 10 bids
                        'asks': orderbook.get('a', [])[:10]   # Top 10 asks
                    }
                    
                    # Write to file
                    with gzip.open(self.collection_file, 'at') as f:
                        json.dump(record, f)
                        f.write('\n')
                    
                    self.collection_count += 1
                    
                    # Progress update
                    elapsed = time.time() - self.collection_start
                    if self.collection_count % 50 == 0:
                        rate = self.collection_count / elapsed
                        console.print(f"Collected: {self.collection_count} records | Rate: {rate:.1f}/sec | Elapsed: {elapsed:.0f}s", end='\r')
                    
                    # Check if done
                    if elapsed >= duration_seconds:
                        self.collection_running = False
                        ws.close()
                        
            except Exception as e:
                console.print(f"\n[red]Message error: {e}[/red]")
        
        def on_error(ws, error):
            console.print(f"\n[red]WebSocket error: {error}[/red]")
            self.collection_running = False
        
        def on_close(ws, close_status_code, close_msg):
            console.print(f"\n[green]Collection complete! Total records: {self.collection_count}[/green]")
        
        def on_open(ws):
            # Subscribe to orderbook
            subscribe_msg = {
                "op": "subscribe",
                "args": ["orderbook.50.BTCUSDT"]
            }
            ws.send(json.dumps(subscribe_msg))
            console.print("[green]Connected to Bybit[/green]")
        
        try:
            # Connect and collect
            ws = websocket.WebSocketApp(self.ws_url,
                                      on_open=on_open,
                                      on_message=on_message,
                                      on_error=on_error,
                                      on_close=on_close)
            
            ws.run_forever()
            
            # Check file size
            if os.path.exists(output_file):
                size_mb = os.path.getsize(output_file) / 1024 / 1024
                console.print(f"File size: {size_mb:.2f} MB")
                return output_file
            
        except Exception as e:
            console.print(f"\n[red]Collection failed: {e}[/red]")
            return None
    
    # ============ MODEL TRAINING ============
    
    def train_model_menu(self):
        """Menu for model training"""
        self.clear_screen()
        console.print("[bold cyan]🧠 Train Model[/bold cyan]\n")
        
        # List data files
        data_files = sorted(Path('l2_data').glob('*.jsonl.gz'), 
                          key=lambda x: x.stat().st_mtime, reverse=True)
        
        if not data_files:
            console.print("[red]No data files found! Collect data first.[/red]")
            Prompt.ask("\nPress Enter to continue")
            return
        
        # Show files
        console.print("Available data files:\n")
        for i, file in enumerate(data_files[:10], 1):
            size_mb = file.stat().st_size / 1024 / 1024
            mtime = datetime.fromtimestamp(file.stat().st_mtime).strftime('%Y-%m-%d %H:%M')
            console.print(f"{i}. {file.name} ({size_mb:.1f} MB) - {mtime}")
        
        # Select file
        file_num = IntPrompt.ask("\nSelect file number", default=1)
        if 1 <= file_num <= len(data_files):
            selected_file = data_files[file_num - 1]
        else:
            console.print("[red]Invalid selection[/red]")
            return
        
        console.print(f"\n[green]Selected: {selected_file.name}[/green]")
        
        # Train
        console.print("\n[yellow]Loading and processing data...[/yellow]")
        model_file = self.train_model(selected_file)
        
        if model_file:
            console.print(f"\n[green]✓ Model saved to: {model_file}[/green]")
        
        Prompt.ask("\nPress Enter to continue")
    
    def train_model(self, data_file):
        """Actually train the model"""
        try:
            # Load data
            console.print("Loading L2 data...")
            records = []
            with gzip.open(data_file, 'rt') as f:
                for line in f:
                    records.append(json.loads(line))
            
            console.print(f"Loaded {len(records)} records")
            
            # Convert to features
            console.print("Extracting features...")
            features_list = []
            
            for i, record in enumerate(records):
                if i % 1000 == 0 and i > 0:
                    console.print(f"  Processed {i}/{len(records)} records...", end='\r')
                
                bids = record.get('bids', [])
                asks = record.get('asks', [])
                
                if len(bids) < 3 or len(asks) < 3:
                    continue
                
                # Simple features
                feature = {
                    'bid1': float(bids[0][0]),
                    'bid1_size': float(bids[0][1]),
                    'ask1': float(asks[0][0]),
                    'ask1_size': float(asks[0][1]),
                    'mid': (float(bids[0][0]) + float(asks[0][0])) / 2,
                    'spread': float(asks[0][0]) - float(bids[0][0]),
                    'imbalance': (float(bids[0][1]) - float(asks[0][1])) / (float(bids[0][1]) + float(asks[0][1]))
                }
                
                # Add depth
                bid_depth = sum(float(b[1]) for b in bids[:5])
                ask_depth = sum(float(a[1]) for a in asks[:5])
                feature['bid_depth'] = bid_depth
                feature['ask_depth'] = ask_depth
                feature['depth_imbalance'] = (bid_depth - ask_depth) / (bid_depth + ask_depth)
                
                features_list.append(feature)
            
            # Create DataFrame
            df = pd.DataFrame(features_list)
            console.print(f"\n\nCreated {len(df)} feature rows")
            
            # Add target (future price movement)
            df['target'] = df['mid'].shift(-5).pct_change(5)  # 5-step ahead return
            df = df.dropna()
            
            console.print(f"Final dataset: {len(df)} samples")
            
            # Train model
            console.print("\nTraining model...")
            feature_cols = ['spread', 'imbalance', 'bid_depth', 'ask_depth', 'depth_imbalance',
                          'bid1_size', 'ask1_size']
            
            X = df[feature_cols]
            y = df['target']
            
            # Split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Train simple model
            model = lgb.LGBMRegressor(
                n_estimators=100,
                learning_rate=0.1,
                num_leaves=31,
                random_state=42,
                verbosity=-1
            )
            
            model.fit(X_train, y_train)
            
            # Evaluate
            train_score = model.score(X_train, y_train)
            test_score = model.score(X_test, y_test)
            
            console.print(f"\nTrain R²: {train_score:.4f}")
            console.print(f"Test R²: {test_score:.4f}")
            
            # Save model
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            model_file = f"models/model_{timestamp}.txt"
            model.booster_.save_model(model_file)
            
            # Save metadata
            metadata = {
                'features': feature_cols,
                'train_score': train_score,
                'test_score': test_score,
                'samples': len(df),
                'data_file': str(data_file)
            }
            
            with open(f"models/model_{timestamp}_meta.json", 'w') as f:
                json.dump(metadata, f, indent=2)
            
            return model_file
            
        except Exception as e:
            console.print(f"\n[red]Training failed: {e}[/red]")
            import traceback
            traceback.print_exc()
            return None
    
    # ============ SIMULATION ============
    
    def run_simulation_menu(self):
        """Menu for running simulation"""
        self.clear_screen()
        console.print("[bold cyan]🤖 Trading Simulation[/bold cyan]\n")
        
        # List models
        model_files = sorted(Path('models').glob('model_*.txt'),
                           key=lambda x: x.stat().st_mtime, reverse=True)
        
        if not model_files:
            console.print("[red]No models found! Train a model first.[/red]")
            Prompt.ask("\nPress Enter to continue")
            return
        
        # Show models
        console.print("Available models:\n")
        for i, file in enumerate(model_files[:10], 1):
            # Try to load metadata
            meta_file = str(file).replace('.txt', '_meta.json')
            if os.path.exists(meta_file):
                with open(meta_file, 'r') as f:
                    meta = json.load(f)
                console.print(f"{i}. {file.name} (R²: {meta.get('test_score', 0):.3f})")
            else:
                console.print(f"{i}. {file.name}")
        
        # Select model
        model_num = IntPrompt.ask("\nSelect model number", default=1)
        if 1 <= model_num <= len(model_files):
            selected_model = model_files[model_num - 1]
        else:
            console.print("[red]Invalid selection[/red]")
            return
        
        duration = IntPrompt.ask("\nSimulation duration (seconds)", default=300)
        
        console.print(f"\n[yellow]Starting simulation for {duration} seconds...[/yellow]")
        self.run_simulation(selected_model, duration)
        
        Prompt.ask("\nPress Enter to continue")
    
    def run_simulation(self, model_file, duration):
        """Actually run the simulation"""
        try:
            # Load model
            model = lgb.Booster(model_file=str(model_file))
            
            # Load metadata
            meta_file = str(model_file).replace('.txt', '_meta.json')
            with open(meta_file, 'r') as f:
                metadata = json.load(f)
            
            feature_cols = metadata['features']
            
            # Initialize simulation state
            self.sim_position = 0
            self.sim_balance = 10000
            self.sim_trades = []
            self.sim_start = time.time()
            self.sim_running = True
            
            # Output file
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = f"results/simulation_{timestamp}.csv"
            
            # Write header
            with open(output_file, 'w') as f:
                f.write('timestamp,price,position,balance,action\n')
            
            def on_message(ws, message):
                if not self.sim_running:
                    return
                    
                try:
                    data = json.loads(message)
                    
                    if 'topic' in data and 'orderbook' in data['topic'] and 'data' in data:
                        orderbook = data['data']
                        bids = orderbook.get('b', [])
                        asks = orderbook.get('a', [])
                        
                        if len(bids) < 5 or len(asks) < 5:
                            return
                        
                        # Calculate features
                        bid1 = float(bids[0][0])
                        ask1 = float(asks[0][0])
                        mid = (bid1 + ask1) / 2
                        
                        features = {
                            'spread': ask1 - bid1,
                            'imbalance': (float(bids[0][1]) - float(asks[0][1])) / (float(bids[0][1]) + float(asks[0][1])),
                            'bid_depth': sum(float(b[1]) for b in bids[:5]),
                            'ask_depth': sum(float(a[1]) for a in asks[:5]),
                            'depth_imbalance': 0,  # Will calculate
                            'bid1_size': float(bids[0][1]),
                            'ask1_size': float(asks[0][1])
                        }
                        
                        features['depth_imbalance'] = (features['bid_depth'] - features['ask_depth']) / (features['bid_depth'] + features['ask_depth'])
                        
                        # Make prediction
                        X = pd.DataFrame([features])[feature_cols]
                        prediction = model.predict(X)[0]
                        
                        # Simple trading logic
                        action = ''
                        if prediction > 0.0001 and self.sim_position == 0:
                            self.sim_position = 0.001  # Buy 0.001 BTC
                            action = 'buy'
                            self.sim_trades.append({'action': 'buy', 'price': mid, 'size': 0.001})
                        elif prediction < -0.0001 and self.sim_position > 0:
                            action = 'sell'
                            self.sim_trades.append({'action': 'sell', 'price': mid, 'size': self.sim_position})
                            # Calculate P&L
                            buy_price = self.sim_trades[-2]['price']
                            pnl = (mid - buy_price) * self.sim_position
                            self.sim_balance += pnl
                            self.sim_position = 0
                        
                        # Save to file
                        with open(output_file, 'a') as f:
                            f.write(f"{datetime.now().isoformat()},{mid},{self.sim_position},{self.sim_balance},{action}\n")
                        
                        # Display
                        elapsed = time.time() - self.sim_start
                        console.print(f"Price: ${mid:,.2f} | Pos: {self.sim_position:.4f} | Bal: ${self.sim_balance:,.2f} | Pred: {prediction:.6f} | Time: {elapsed:.0f}s", end='\r')
                        
                        # Check if done
                        if elapsed >= duration:
                            self.sim_running = False
                            ws.close()
                            
                except Exception as e:
                    console.print(f"\n[red]Sim error: {e}[/red]")
            
            def on_error(ws, error):
                console.print(f"\n[red]WebSocket error: {error}[/red]")
                self.sim_running = False
            
            def on_close(ws, close_status_code, close_msg):
                console.print(f"\n\n[green]Simulation complete![/green]")
                console.print(f"Total trades: {len(self.sim_trades)}")
                console.print(f"Final balance: ${self.sim_balance:,.2f}")
                console.print(f"P&L: ${self.sim_balance - 10000:,.2f}")
                console.print(f"Results saved to: {output_file}")
            
            def on_open(ws):
                # Subscribe
                subscribe_msg = {
                    "op": "subscribe",
                    "args": ["orderbook.50.BTCUSDT"]
                }
                ws.send(json.dumps(subscribe_msg))
            
            # Run simulation
            ws = websocket.WebSocketApp(self.ws_url,
                                      on_open=on_open,
                                      on_message=on_message,
                                      on_error=on_error,
                                      on_close=on_close)
            
            ws.run_forever()
            
        except Exception as e:
            console.print(f"\n[red]Simulation failed: {e}[/red]")
            import traceback
            traceback.print_exc()
    
    # ============ UTILITIES ============
    
    def quick_demo(self):
        """Quick demo workflow"""
        self.clear_screen()
        console.print("[bold cyan]⚡ Quick Demo[/bold cyan]\n")
        console.print("This will run a complete demo:")
        console.print("  1. Collect data for 30 seconds")
        console.print("  2. Train a model")
        console.print("  3. Run simulation for 30 seconds\n")
        
        if Confirm.ask("Continue?", default=True):
            # Collect
            console.print("\n[yellow]Step 1: Collecting data...[/yellow]")
            data_file = self.collect_l2_data(30)
            
            if data_file:
                # Train
                console.print("\n[yellow]Step 2: Training model...[/yellow]")
                model_file = self.train_model(data_file)
                
                if model_file:
                    # Simulate
                    console.print("\n[yellow]Step 3: Running simulation...[/yellow]")
                    self.run_simulation(model_file, 30)
        
        Prompt.ask("\nPress Enter to continue")
    
    def view_files(self):
        """View collected files"""
        self.clear_screen()
        console.print("[bold cyan]📁 Your Files[/bold cyan]\n")
        
        # Data files
        data_files = list(Path('l2_data').glob('*.jsonl.gz'))
        console.print(f"[yellow]Data files:[/yellow] {len(data_files)}")
        for file in sorted(data_files)[-5:]:
            size_mb = file.stat().st_size / 1024 / 1024
            console.print(f"  {file.name} ({size_mb:.1f} MB)")
        
        # Models
        model_files = list(Path('models').glob('model_*.txt'))
        console.print(f"\n[yellow]Models:[/yellow] {len(model_files)}")
        for file in sorted(model_files)[-5:]:
            console.print(f"  {file.name}")
        
        # Results
        result_files = list(Path('results').glob('*.csv'))
        console.print(f"\n[yellow]Results:[/yellow] {len(result_files)}")
        for file in sorted(result_files)[-5:]:
            console.print(f"  {file.name}")
        
        Prompt.ask("\nPress Enter to continue")

# ============ MAIN ============

if __name__ == "__main__":
    system = CompleteTradingSystem()
    system.run()