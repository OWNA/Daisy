#!/usr/bin/env python3
"""
Fix ModelTrainer to exclude target features and prepare for retraining
"""

import shutil
import os

print("Fixing ModelTrainer to prevent target leakage...")

# Backup original
shutil.copy('modeltrainer.py', 'modeltrainer_original.py')

# Read the file
with open('modeltrainer.py', 'r') as f:
    content = f.read()

# Find and replace the _prepare_data method
old_prepare = """    def _prepare_data(self, df):
        features = [
            c for c in df.columns
            if c not in ['target', 'timestamp', 'symbol', 'exchange']
        ]
        X = df[features]
        y = df['target']
        self.trained_features = features
        return X, y"""

new_prepare = """    def _prepare_data(self, df):
        # Exclude ALL non-predictive columns to prevent target leakage
        exclude_columns = [
            'target', 'timestamp', 'symbol', 'exchange', 'id',
            # CRITICAL: Exclude all target-related columns that leak future information
            'target_return_1min', 'target_return_5min', 'target_volatility', 
            'target_direction', 'target_return', 'target_price',
            # Also exclude metadata that's not useful for prediction
            'update_id', 'sequence_id', 'data_quality_score',
            # Close is redundant with mid_price in L2-only mode
            'close'
        ]
        
        features = [
            c for c in df.columns
            if c not in exclude_columns
        ]
        
        print(f"Preparing {len(features)} features for training (excluded {len(df.columns) - len(features) - 1} columns)")
        print(f"Excluded columns: {[c for c in df.columns if c in exclude_columns]}")
        
        X = df[features]
        y = df['target']
        self.trained_features = features
        return X, y"""

# Replace the method
content = content.replace(old_prepare, new_prepare)

# Write back
with open('modeltrainer.py', 'w') as f:
    f.write(content)

print("✅ ModelTrainer fixed to exclude target leakage")

# Create a clean retraining script
retrain_script = '''#!/usr/bin/env python3
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

print(f"\\nOld model files backed up to {backup_dir}/")
print("\\nStarting clean training...")

# Run training
import subprocess
result = subprocess.run(["python", "main.py", "train"], capture_output=True, text=True)

if result.returncode == 0:
    print("\\n✅ Training completed successfully!")
    print("\\nYou can now run paper trading with the clean model:")
    print("python main.py trade --paper")
else:
    print("\\n❌ Training failed!")
    print("Error:", result.stderr)
'''

with open('retrain_clean.py', 'w') as f:
    f.write(retrain_script)

print("\n✅ Created retrain_clean.py")
print("\n" + "="*50)
print("NEXT STEPS:")
print("="*50)
print("\n1. Run the clean retraining:")
print("   python retrain_clean.py")
print("\n2. This will:")
print("   - Backup your current (flawed) model")
print("   - Train a new model WITHOUT target leakage")
print("   - Use only legitimate L2 microstructure features")
print("\n3. After training completes, run paper trading:")
print("   python main.py trade --paper")
print("\n4. The new model will:")
print("   - Make predictions based only on L2 order book features")
print("   - Not have access to future information")
print("   - Give realistic performance estimates")
print("\nEstimated time: 5-10 minutes")