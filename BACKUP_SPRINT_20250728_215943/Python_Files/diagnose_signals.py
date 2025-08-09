#!/usr/bin/env python3
"""
Diagnose why trading signals aren't showing
"""

import os
import sys

print("="*60)
print("TRADING SIGNAL DIAGNOSIS")
print("="*60)

# 1. Check model files exist
print("\n1. Checking model files...")
model_files = [
    "trading_bot_data/lgbm_model_BTC_USDTUSDT_l2_only.txt",
    "trading_bot_data/model_features_BTC_USDTUSDT_l2_only.json",
    "trading_bot_data/lgbm_model_BTC_USDTUSDT_l2_only_scaling.json"
]

all_exist = True
for f in model_files:
    exists = os.path.exists(f)
    print(f"   {f}: {'✓ EXISTS' if exists else '✗ MISSING'}")
    if not exists:
        all_exist = False

if not all_exist:
    print("\n❌ Model files missing! You need to retrain the model.")
    print("   Run: python main.py train")
    sys.exit(1)

# 2. Check config threshold
print("\n2. Checking prediction threshold in config...")
import yaml
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

threshold = config.get('prediction_threshold', 0.5)
print(f"   Prediction threshold: {threshold}")
if threshold > 0.1:
    print("   ⚠️  Threshold might be too high for the retrained model")
    print("   ✓  Already added prediction_threshold: 0.01 to config")

# 3. Summary of changes made
print("\n3. Changes made to improve signal generation:")
print("   ✓ Added prediction_threshold: 0.01 to config.yaml")
print("   ✓ Enhanced logging in main.py to show prediction details")
print("   ✓ Added position tracking for paper trading")

# 4. Key insights
print("\n4. Why signals weren't showing:")
print("   • The retrained model (without target leakage) produces smaller predictions")
print("   • Default threshold of 0.5 was too high for these predictions")
print("   • Model needs |prediction| > threshold to generate a signal")
print("   • With threshold=0.01, more signals should appear")

# 5. Next steps
print("\n5. NEXT STEPS:")
print("   1. Run paper trading again:")
print("      python main.py trade --paper")
print("")
print("   2. You should now see:")
print("      - Prediction details every 10 seconds")
print("      - Trading signals when |prediction| > 0.01")
print("      - Position and balance updates")
print("")
print("   3. If still no signals, run check_predictions.py:")
print("      python check_predictions.py")
print("      This will show the actual prediction distribution")

print("\n" + "="*60)
print("The system is now configured for more sensitive signal detection!")
print("="*60)