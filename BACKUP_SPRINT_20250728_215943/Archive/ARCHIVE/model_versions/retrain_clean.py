#!/usr/bin/env python3
"""
Clean retraining script without target leakage
"""

import os
import shutil
from datetime import datetime

print("=== CLEAN MODEL RETRAINING ===")
print(f"Starting at {datetime.now()}")

# Backup old model files
backup_dir = f"model_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
os.makedirs(backup_dir, exist_ok=True)

files_to_backup = [
    "trading_bot_data/lgbm_model_BTC_USDTUSDT_l2_only.txt",
    "trading_bot_data/model_features_BTC_USDTUSDT_l2_only.json",
    "lgbm_model_BTC_USDTUSDT_l2_only_scaling.json"
]

for file in files_to_backup:
    if os.path.exists(file):
        shutil.copy(file, backup_dir)
        print(f"Backed up {file}")

print(f"\nOld model files backed up to {backup_dir}/")
print("\nStarting clean training...")

# Run training
import subprocess
result = subprocess.run(["python", "main.py", "train"], capture_output=True, text=True)

if result.returncode == 0:
    print("\n✅ Training completed successfully!")
    print("\nYou can now run paper trading with the clean model:")
    print("python main.py trade --paper")
else:
    print("\n❌ Training failed!")
    print("Error:", result.stderr)
