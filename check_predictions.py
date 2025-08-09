#!/usr/bin/env python3
"""
Check what predictions the model is actually making
"""

import sys
import os
import yaml
import json
import numpy as np
import lightgbm as lgb

# Load config
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Load model directly
model_path = "trading_bot_data/lgbm_model_BTC_USDTUSDT_l2_only.txt"
features_path = "trading_bot_data/model_features_BTC_USDTUSDT_l2_only.json"

print(f"Loading model from: {model_path}")
model = lgb.Booster(model_file=model_path)

# Load features list
with open(features_path, 'r') as f:
    features_data = json.load(f)
    feature_names = features_data['trained_features']

print(f"\nModel expects {len(feature_names)} features")
print(f"First 5 features: {feature_names[:5]}")

# Create dummy data with realistic L2 values
dummy_data = np.zeros((1, len(feature_names)))

# Set some realistic values based on feature names
for i, feat in enumerate(feature_names):
    if 'price' in feat and 'impact' not in feat:
        # Price features around 50000 (BTC price)
        dummy_data[0, i] = 50000 + np.random.randn() * 10
    elif 'size' in feat or 'volume' in feat:
        # Size features
        dummy_data[0, i] = 10 + np.random.rand() * 50
    elif 'spread' in feat:
        # Spread features
        dummy_data[0, i] = 10 if 'spread' == feat else 0.0002  # bps
    elif 'imbalance' in feat:
        # Imbalance features between -1 and 1
        dummy_data[0, i] = np.random.randn() * 0.3
    elif 'volatility' in feat:
        # Volatility features
        dummy_data[0, i] = 0.001 + np.random.rand() * 0.01
    elif 'return' in feat:
        # Return features (small)
        dummy_data[0, i] = np.random.randn() * 0.001
    else:
        # Default small random values
        dummy_data[0, i] = np.random.randn() * 0.1

# Make predictions
raw_pred = model.predict(dummy_data)
print(f"\nRaw prediction: {raw_pred[0]:.6f}")

# Load scaling params if available
scaling_path = model_path.replace('.txt', '_scaling.json')
if os.path.exists(scaling_path):
    with open(scaling_path, 'r') as f:
        scaling = json.load(f)
    print(f"\nScaling params found:")
    print(f"  target_mean: {scaling['target_mean']:.6f}")
    print(f"  target_std: {scaling['target_std']:.6f}")
    
    # Unscale prediction
    unscaled = raw_pred[0] * scaling['target_std'] + scaling['target_mean']
    print(f"\nUnscaled prediction: {unscaled:.6f}")
else:
    print("\nNo scaling params found")

# Check signal generation with different thresholds
print("\nSignal generation with different thresholds:")
for threshold in [0.5, 0.1, 0.05, 0.01, 0.001]:
    if raw_pred[0] > threshold:
        signal = 1
    elif raw_pred[0] < -threshold:
        signal = -1
    else:
        signal = 0
    print(f"  Threshold {threshold:.3f}: signal = {signal}")

# Make multiple predictions to see distribution
print("\nMaking 100 predictions with random data to see distribution:")
predictions = []
for _ in range(100):
    # Generate random L2-like data
    test_data = np.zeros((1, len(feature_names)))
    for i, feat in enumerate(feature_names):
        if 'price' in feat and 'impact' not in feat:
            test_data[0, i] = 50000 + np.random.randn() * 50
        elif 'size' in feat or 'volume' in feat:
            test_data[0, i] = 10 + np.random.rand() * 100
        else:
            test_data[0, i] = np.random.randn() * 0.1
    
    pred = model.predict(test_data)[0]
    predictions.append(pred)

predictions = np.array(predictions)
print(f"\nPrediction statistics:")
print(f"  Mean: {np.mean(predictions):.6f}")
print(f"  Std: {np.std(predictions):.6f}")
print(f"  Min: {np.min(predictions):.6f}")
print(f"  Max: {np.max(predictions):.6f}")
print(f"  Predictions > 0.5: {np.sum(predictions > 0.5)}")
print(f"  Predictions < -0.5: {np.sum(predictions < -0.5)}")
print(f"  Predictions > 0.1: {np.sum(predictions > 0.1)}")
print(f"  Predictions < -0.1: {np.sum(predictions < -0.1)}")
print(f"  Predictions > 0.01: {np.sum(predictions > 0.01)}")
print(f"  Predictions < -0.01: {np.sum(predictions < -0.01)}")