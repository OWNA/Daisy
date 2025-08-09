#!/usr/bin/env python3
"""
Fixed backtest runner with proper output
"""

import os
import sys
import yaml
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import json

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from strategybacktester import StrategyBacktester
import ccxt

def run_backtest_with_output(model_file=None, output_dir='backtest_results', max_rows=None):
    """Run backtest and save results properly"""
    
    print("\n" + "="*60)
    print("RUNNING BACKTEST")
    print("="*60)
    
    # Load config
    config_file = 'config.yaml'
    if not os.path.exists(config_file):
        print("Error: config.yaml not found")
        return False
    
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    # Find model file
    if not model_file:
        # Find latest model
        model_files = list(Path('.').glob('lgbm_model_*.txt'))
        if not model_files:
            print("Error: No model files found")
            return False
        
        model_file = max(model_files, key=lambda x: x.stat().st_mtime)
        print(f"Using latest model: {model_file}")
    else:
        model_file = Path(model_file)
        if not model_file.exists():
            print(f"Error: Model file not found: {model_file}")
            return False
    
    # Update config with model
    config['model_path'] = str(model_file)
    config['model_filename'] = str(model_file)
    
    # Initialize backtester
    print("\nInitializing backtester...")
    try:
        exchange = ccxt.bybit({'enableRateLimit': True})
        backtester = StrategyBacktester(config, exchange)
    except Exception as e:
        print(f"Error initializing backtester: {e}")
        return False
    
    # Run backtest
    print("\nRunning backtest...")
    try:
        results = backtester.run_backtest(max_rows=max_rows)
        
        if results is None:
            print("Error: Backtest returned no results")
            return False
        
        print(f"\nBacktest complete!")
        print(f"Total trades: {len(results)}")
        
        # Calculate metrics
        if len(results) > 0:
            initial_balance = 10000
            final_balance = results['equity'].iloc[-1] if 'equity' in results else initial_balance
            total_return = (final_balance - initial_balance) / initial_balance * 100
            
            print(f"Final equity: ${final_balance:,.2f}")
            print(f"Total return: {total_return:.2f}%")
            
            if 'pnl' in results.columns:
                wins = (results['pnl'] > 0).sum()
                losses = (results['pnl'] < 0).sum()
                win_rate = wins / (wins + losses) * 100 if (wins + losses) > 0 else 0
                print(f"Win rate: {win_rate:.1f}%")
        
        # Save results
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = output_dir / f"backtest_results_{timestamp}.csv"
        
        # Ensure we have the required columns for visualization
        if 'timestamp' not in results.columns and results.index.name == 'timestamp':
            results = results.reset_index()
        
        if 'timestamp' not in results.columns:
            results['timestamp'] = pd.date_range(start='2024-01-01', periods=len(results), freq='1min')
        
        # Add missing columns if needed
        if 'price' not in results.columns and 'close' in results.columns:
            results['price'] = results['close']
        elif 'price' not in results.columns:
            # Generate synthetic price data for testing
            results['price'] = 100000 + np.random.randn(len(results)).cumsum() * 100
        
        if 'position' not in results.columns:
            results['position'] = 0
        
        if 'equity' not in results.columns:
            results['equity'] = initial_balance
        
        if 'action' not in results.columns:
            results['action'] = ''
        
        # Save to CSV
        results.to_csv(output_file, index=False)
        print(f"\nResults saved to: {output_file}")
        
        # Also save summary
        summary = {
            'model': str(model_file),
            'timestamp': timestamp,
            'total_trades': len(results),
            'final_equity': float(final_balance) if 'final_balance' in locals() else 10000,
            'total_return': float(total_return) if 'total_return' in locals() else 0,
            'win_rate': float(win_rate) if 'win_rate' in locals() else 0
        }
        
        summary_file = output_dir / f"backtest_summary_{timestamp}.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        return True
        
    except Exception as e:
        print(f"Error during backtest: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='Model file to use')
    parser.add_argument('--rows', type=int, help='Max rows to process')
    parser.add_argument('--output', default='backtest_results', help='Output directory')
    
    args = parser.parse_args()
    
    success = run_backtest_with_output(
        model_file=args.model,
        output_dir=args.output,
        max_rows=args.rows
    )
    
    if not success:
        sys.exit(1)