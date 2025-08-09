#!/usr/bin/env python3
"""
Check what model files exist after training
"""

import os
import glob

print("Checking for model files...\n")

# List all .txt files that might be models
model_files = glob.glob("lgbm_model_*.txt")
if model_files:
    print("Found model files:")
    for f in model_files:
        size = os.path.getsize(f) / 1024
        print(f"  - {f} ({size:.1f} KB)")
else:
    print("No model files found")

print("\n")

# List all .json files that might be scaling/features
json_files = glob.glob("*_scaling.json") + glob.glob("model_features_*.json")
if json_files:
    print("Found config files:")
    for f in json_files:
        print(f"  - {f}")
else:
    print("No config files found")

print("\n")

# Show expected filenames based on config
import yaml
try:
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    symbol = config.get('symbol', 'BTC/USDT:USDT')
    print(f"Config symbol: {symbol}")
    
    # Show different sanitization methods
    safe1 = symbol.replace('/', '_').replace(':', '_')
    safe2 = symbol.replace('/', '_').replace(':', '')
    
    print(f"\nPossible model filenames:")
    print(f"  - lgbm_model_{safe1}.txt")
    print(f"  - lgbm_model_{safe1}_l2_only.txt")
    print(f"  - lgbm_model_{safe2}.txt")
    print(f"  - lgbm_model_{safe2}_l2_only.txt")
    
except Exception as e:
    print(f"Could not read config: {e}")