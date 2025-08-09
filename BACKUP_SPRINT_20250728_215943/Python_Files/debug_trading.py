#!/usr/bin/env python3
"""
Debug why no trading signals are showing
"""

import pandas as pd
import numpy as np
from modelpredictor import ModelPredictor
from featureengineer import FeatureEngineer
import yaml

# Load config
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Initialize components
predictor = ModelPredictor(config)
engineer = FeatureEngineer(config)

# Load model
model_path = "trading_bot_data/lgbm_model_BTC_USDTUSDT_l2_only.txt"
features_path = "trading_bot_data/model_features_BTC_USDTUSDT_l2_only.json"

success = predictor.load_model_and_features(
    model_file_path=model_path,
    features_file_path=features_path
)

print(f"Model loaded: {success}")
print(f"Number of features: {len(predictor.trained_features)}")

# Create dummy L2 data
dummy_data = {
    'timestamp': pd.Timestamp.now(tz='UTC'),
    'mid_price': 50000,
    'spread': 10,
    'spread_bps': 2
}

# Add price/size levels
for i in range(1, 11):
    dummy_data[f'bid_price_{i}'] = 50000 - i * 5
    dummy_data[f'bid_size_{i}'] = 10 + i
    dummy_data[f'ask_price_{i}'] = 50000 + i * 5
    dummy_data[f'ask_size_{i}'] = 10 + i

# Generate features
df = pd.DataFrame([dummy_data])
features = engineer.generate_features(df)
print(f"\nGenerated features: {features.shape}")

# Set scaling params
predictor.set_scaling_params(-4.0, 786.9)

# Make prediction
signals = predictor.predict_signals(features)
print(f"\nSignals shape: {signals.shape if signals is not None else 'None'}")

if signals is not None:
    print("\nSignal columns:", list(signals.columns))
    print("\nFirst row of signals:")
    print(signals.iloc[0])
    
    # Check signal values
    if 'signal' in signals.columns:
        signal_val = signals['signal'].iloc[0]
        print(f"\nSignal value: {signal_val}")
        print(f"Signal type: {type(signal_val)}")
        print(f"Would trade: {signal_val != 0}")
    else:
        print("\nNo 'signal' column found")
        print("Available columns:", list(signals.columns))