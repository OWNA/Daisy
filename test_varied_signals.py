#!/usr/bin/env python3
"""
Test signal generation with varied market conditions
"""

import pandas as pd
import numpy as np
import json
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

# Load scaling params
scaling_path = "trading_bot_data/lgbm_model_BTC_USDTUSDT_l2_only_scaling.json"
with open(scaling_path, 'r') as f:
    scaling_params = json.load(f)
predictor.set_scaling_params(scaling_params['target_mean'], scaling_params['target_std'])

print("Testing multiple market scenarios...")

# Test scenarios: (description, mid_price, spread_multiplier, bid_ask_imbalance)
scenarios = [
    ("Balanced market", 50000, 1.0, 1.0),
    ("Tight spread", 50000, 0.3, 1.0),
    ("Wide spread", 50000, 3.0, 1.0),
    ("Bid heavy", 50000, 1.0, 2.0),  # More bid volume
    ("Ask heavy", 50000, 1.0, 0.5),  # More ask volume
    ("High price", 70000, 1.0, 1.0),
    ("Low price", 30000, 1.0, 1.0),
]

for description, mid_price, spread_mult, volume_ratio in scenarios:
    # Create market data
    base_spread = 10 * spread_mult
    
    dummy_data = {
        'timestamp': pd.Timestamp.now(tz='UTC'),
        'mid_price': mid_price,
        'spread': base_spread,
        'spread_bps': (base_spread / mid_price) * 10000
    }
    
    # Add price/size levels with imbalance
    for i in range(1, 11):
        dummy_data[f'bid_price_{i}'] = mid_price - (base_spread/2) - (i-1) * 5
        dummy_data[f'bid_size_{i}'] = (10 + i * 2) * volume_ratio  # Bid volume affected by ratio
        dummy_data[f'ask_price_{i}'] = mid_price + (base_spread/2) + (i-1) * 5
        dummy_data[f'ask_size_{i}'] = (10 + i * 2) / volume_ratio  # Ask volume inversely affected
    
    # Generate features and prediction
    df = pd.DataFrame([dummy_data])
    features = engineer.generate_features(df)
    signals = predictor.predict_signals(features)
    
    if signals is not None and 'signal' in signals.columns:
        signal_val = signals['signal'].iloc[0]
        pred_val = signals['pred_unscaled_target'].iloc[0]
        print(f"{description:15} | Signal: {signal_val:2.0f} | Prediction: {pred_val:8.6f} | Trade: {'Yes' if signal_val != 0 else 'No'}")
    else:
        print(f"{description:15} | ERROR: No signals generated")

print("\nTesting complete. The system should now generate varied signals based on market conditions.")