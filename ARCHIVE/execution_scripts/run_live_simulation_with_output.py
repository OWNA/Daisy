#!/usr/bin/env python3
"""
Live simulation wrapper that outputs data for real-time monitoring
"""

import sys
import os
import time
import pandas as pd
import numpy as np
from datetime import datetime
import json
import argparse
import subprocess
from pathlib import Path

def run_simulation_with_output(duration, model_path, output_file):
    """Run simulation and write results to file for monitoring"""
    
    print(f"Starting live simulation for {duration} seconds...")
    print(f"Output file: {output_file}")
    print(f"Model: {model_path}")
    
    # Create initial CSV with headers
    df_template = pd.DataFrame(columns=[
        'timestamp', 'price', 'position', 'equity', 'action', 'size', 'pnl'
    ])
    df_template.to_csv(output_file, index=False)
    
    # Run the actual simulation in a subprocess
    cmd = f"python start_live_simulation.py"
    
    # Check if start_live_simulation.py exists
    if not os.path.exists('start_live_simulation.py'):
        print("Warning: start_live_simulation.py not found, running basic simulation")
        
        # Run a basic simulation that generates sample data
        start_time = time.time()
        initial_price = 100000  # Starting BTC price
        position = 0
        equity = 10000
        
        while time.time() - start_time < duration:
            # Generate simulated data
            timestamp = datetime.now().isoformat()
            
            # Simulate price movement
            price_change = np.random.normal(0, 50)
            current_price = initial_price + price_change
            
            # Simulate trading decision (random for demo)
            action = ''
            size = 0
            
            if np.random.random() > 0.95:  # 5% chance of trade
                if position == 0:
                    action = 'buy'
                    size = 0.001
                    position = size
                elif position > 0 and np.random.random() > 0.5:
                    action = 'sell'
                    size = position
                    position = 0
            
            # Calculate equity
            if position > 0:
                equity = 10000 + (current_price - initial_price) * position
            
            # Write row to CSV
            row_data = {
                'timestamp': timestamp,
                'price': current_price,
                'position': position,
                'equity': equity,
                'action': action,
                'size': size,
                'pnl': equity - 10000
            }
            
            # Append to CSV
            pd.DataFrame([row_data]).to_csv(output_file, mode='a', header=False, index=False)
            
            # Sleep briefly
            time.sleep(1)
            
            # Print progress
            elapsed = int(time.time() - start_time)
            print(f"\rSimulation progress: {elapsed}/{duration}s", end='', flush=True)
    
    else:
        # Run the actual simulation script
        process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Monitor the process and capture output
        start_time = time.time()
        while process.poll() is None and time.time() - start_time < duration:
            time.sleep(1)
            elapsed = int(time.time() - start_time)
            print(f"\rSimulation progress: {elapsed}/{duration}s", end='', flush=True)
        
        # Terminate if still running
        if process.poll() is None:
            process.terminate()
    
    print("\nSimulation complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--duration', type=int, default=300, help='Duration in seconds')
    parser.add_argument('--model', default='lgbm_model_BTC_USDTUSDT_optuna_l2_hht.txt', help='Model path')
    parser.add_argument('--output', required=True, help='Output CSV file')
    
    args = parser.parse_args()
    
    run_simulation_with_output(args.duration, args.model, args.output)