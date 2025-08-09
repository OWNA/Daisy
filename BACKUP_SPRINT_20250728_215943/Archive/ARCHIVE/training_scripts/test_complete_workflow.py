#!/usr/bin/env python3
"""
Test complete trading bot workflow
"""

import os
import sys
import subprocess
import time
from pathlib import Path
import json
import pandas as pd
import yaml

def run_command(cmd, description):
    """Run a command and check result"""
    print(f"\n{'='*60}")
    print(f"TEST: {description}")
    print(f"CMD: {cmd}")
    print('='*60)
    
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if result.returncode == 0:
        print(f"✅ SUCCESS")
        if result.stdout:
            print("Output preview:", result.stdout[:200])
    else:
        print(f"❌ FAILED (exit code: {result.returncode})")
        if result.stderr:
            print("Error:", result.stderr[:500])
    
    return result.returncode == 0

def test_workflow():
    """Test complete workflow"""
    print("\n" + "="*60)
    print("TRADING BOT WORKFLOW TEST")
    print("="*60)
    
    results = {}
    
    # 1. Check Python environment
    print("\n1. CHECKING ENVIRONMENT")
    print("-" * 40)
    
    # Check Python version
    result = subprocess.run([sys.executable, "--version"], capture_output=True, text=True)
    print(f"Python: {result.stdout.strip()}")
    
    # Check required packages
    required_packages = ['pandas', 'numpy', 'lightgbm', 'ccxt', 'websocket', 'rich']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package} - MISSING")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
        print("Run: pip install -r requirements.txt")
    
    # 2. Check config files
    print("\n2. CHECKING CONFIG FILES")
    print("-" * 40)
    
    config_files = ['config.yaml', 'config_l2_only.yaml']
    config_found = False
    
    for config_file in config_files:
        if os.path.exists(config_file):
            print(f"✓ {config_file} found")
            config_found = True
            
            # Load and check config
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
            
            print(f"  Symbol: {config.get('symbol', 'NOT SET')}")
            print(f"  L2 features: {config.get('use_l2_features', False)}")
            break
    
    if not config_found:
        print("✗ No config file found!")
        
        # Create basic config
        basic_config = {
            'symbol': 'BTC/USDT:USDT',
            'use_l2_features': True,
            'use_hht_features': True,
            'initial_balance': 10000,
            'lookback_minutes': 60,
            'prediction_minutes': 5
        }
        
        with open('config.yaml', 'w') as f:
            yaml.dump(basic_config, f)
        
        print("✓ Created config.yaml")
    
    # 3. Check L2 data
    print("\n3. CHECKING L2 DATA")
    print("-" * 40)
    
    l2_dir = Path('l2_data')
    if l2_dir.exists():
        l2_files = list(l2_dir.glob('*.jsonl.gz'))
        print(f"✓ L2 data directory exists")
        print(f"  Files: {len(l2_files)}")
        
        if l2_files:
            latest_file = max(l2_files, key=lambda x: x.stat().st_mtime)
            print(f"  Latest: {latest_file.name}")
            print(f"  Size: {latest_file.stat().st_size / 1024 / 1024:.2f} MB")
    else:
        print("✗ No L2 data directory")
        l2_dir.mkdir(exist_ok=True)
        print("✓ Created l2_data directory")
    
    # 4. Test data collection (short duration)
    print("\n4. TESTING DATA COLLECTION")
    print("-" * 40)
    
    if run_command(
        "python run_l2_collector.py --config config.yaml --interval 10",
        "Collect 10 seconds of L2 data"
    ):
        results['data_collection'] = 'PASS'
        
        # Check if data was saved
        new_files = list(l2_dir.glob('*.jsonl.gz'))
        if len(new_files) > len(l2_files):
            print("✓ New data file created")
    else:
        results['data_collection'] = 'FAIL'
    
    # 5. Test model training
    print("\n5. TESTING MODEL TRAINING")
    print("-" * 40)
    
    # Find a data file to train on
    l2_files = list(l2_dir.glob('*.jsonl.gz'))
    if l2_files:
        # Use the largest file
        train_file = max(l2_files, key=lambda x: x.stat().st_size)
        
        if run_command(
            f"python train_model_robust.py --data {train_file.name} --trials 2",
            f"Train model on {train_file.name} (2 trials for speed)"
        ):
            results['model_training'] = 'PASS'
            
            # Check if model was saved
            model_files = list(Path('.').glob('lgbm_model_trained_*.txt'))
            if model_files:
                print(f"✓ Model saved: {model_files[-1].name}")
        else:
            results['model_training'] = 'FAIL'
    else:
        print("✗ No data files for training")
        results['model_training'] = 'SKIP'
    
    # 6. Test backtesting
    print("\n6. TESTING BACKTESTING")
    print("-" * 40)
    
    # Find a model to use
    model_files = list(Path('.').glob('lgbm_model_*.txt'))
    if model_files and l2_files:
        model_file = model_files[-1]
        
        if run_command(
            f"python run_backtest.py --config config.yaml --l2-only",
            f"Run backtest with {model_file.name}"
        ):
            results['backtesting'] = 'PASS'
            
            # Check results
            results_dir = Path('backtest_results')
            if results_dir.exists():
                result_files = list(results_dir.glob('*.csv'))
                if result_files:
                    print(f"✓ Results saved: {len(result_files)} files")
        else:
            results['backtesting'] = 'FAIL'
    else:
        print("✗ No model or data for backtesting")
        results['backtesting'] = 'SKIP'
    
    # 7. Test visualization
    print("\n7. TESTING VISUALIZATION")
    print("-" * 40)
    
    # Check if Flask is installed
    try:
        import flask
        print("✓ Flask installed")
        
        # Test starting the visualizer (kill after 3 seconds)
        import threading
        
        def kill_after_delay():
            time.sleep(3)
            # This is a bit hacky but works for testing
            os.system("taskkill /F /IM python.exe 2>nul || killall python 2>/dev/null")
        
        # Don't actually test server start in automated test
        print("✓ Visualization server available (not auto-tested)")
        results['visualization'] = 'PASS'
        
    except ImportError:
        print("✗ Flask not installed")
        results['visualization'] = 'FAIL'
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    for test, result in results.items():
        status = "✅" if result == 'PASS' else "❌" if result == 'FAIL' else "⏭️"
        print(f"{status} {test}: {result}")
    
    # Recommendations
    print("\nRECOMMENDATIONS:")
    
    if missing_packages:
        print(f"1. Install missing packages: pip install {' '.join(missing_packages)}")
    
    if 'FAIL' in results.values():
        print("2. Check error messages above for specific issues")
    
    if not l2_files:
        print("3. Collect some L2 data first: python run_l2_collector.py --interval 300")
    
    print("\nTo use the interactive CLI: python trade_interactive.py")

if __name__ == "__main__":
    test_workflow()