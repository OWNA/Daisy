#!/usr/bin/env python3
"""
Train on specific L2 data file
"""

import os
import sys
from pathlib import Path

# The file you want to train on
target_file = "l2_data_BTC_USDT_USDT_20250707_040413.jsonl.gz"

# Check if it exists
file_path = Path('l2_data') / target_file
if file_path.exists():
    print(f"✅ Found file: {target_file}")
    print(f"   Size: {file_path.stat().st_size / 1024 / 1024:.2f} MB")
    
    # Run training
    print("\nStarting training with 50 Optuna trials...")
    cmd = f"python train_model_robust.py --data {target_file} --features all --trials 50"
    print(f"Command: {cmd}\n")
    
    os.system(cmd)
else:
    print(f"❌ File not found: {target_file}")
    print("\nAvailable files in l2_data:")
    
    l2_dir = Path('l2_data')
    if l2_dir.exists():
        files = sorted(l2_dir.glob('*.jsonl.gz'), key=lambda x: x.stat().st_mtime, reverse=True)
        for i, file in enumerate(files[:10], 1):
            size_mb = file.stat().st_size / 1024 / 1024
            print(f"{i}. {file.name} ({size_mb:.2f} MB)")
    else:
        print("No l2_data directory found!")