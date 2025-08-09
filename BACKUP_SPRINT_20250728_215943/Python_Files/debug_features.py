#!/usr/bin/env python3
"""
Debug feature mismatch between training and prediction
"""

import json
import pandas as pd
import yaml

# Load config
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Load saved features list
with open('trading_bot_data/model_features_BTC_USDTUSDT_l2_only.json', 'r') as f:
    saved_features = json.load(f)['trained_features']

print(f"Saved features count: {len(saved_features)}")
print("Saved features:", saved_features[:10], "...")

# Generate features from dummy data to see what's created
from featureengineer import FeatureEngineer
from datahandler import DataHandler

# Create dummy L2 data
dummy_data = {
    'timestamp': pd.Timestamp.now(tz='UTC'),
    'mid_price': 50000,
    'spread': 10,
    'spread_bps': 2,
    'bid_volume_5': 100,
    'ask_volume_5': 100,
    'order_book_imbalance': 0.1,
    'best_bid': 49995,
    'best_ask': 50005
}

# Add bid/ask levels
for i in range(1, 11):
    dummy_data[f'bid_price_{i}'] = 50000 - i * 5
    dummy_data[f'bid_size_{i}'] = 10 * i
    dummy_data[f'ask_price_{i}'] = 50000 + i * 5
    dummy_data[f'ask_size_{i}'] = 10 * i

# Create DataFrame
df = pd.DataFrame([dummy_data])

# Generate features
fe = FeatureEngineer(config)
df_features = fe.generate_features(df)

print(f"\nGenerated features count: {len(df_features.columns)}")
print("Generated features:", list(df_features.columns)[:10], "...")

# Find missing features
missing_in_generated = set(saved_features) - set(df_features.columns)
extra_in_generated = set(df_features.columns) - set(saved_features)

print(f"\nFeatures in saved list but not generated: {len(missing_in_generated)}")
if missing_in_generated:
    print("Missing:", list(missing_in_generated))

print(f"\nFeatures generated but not in saved list: {len(extra_in_generated)}")
if extra_in_generated:
    print("Extra:", list(extra_in_generated))

# The issue might be that some features are calculated during training but not during prediction
print("\n\nTo fix this issue:")
print("1. We need to ensure the feature generation creates exactly the same features")
print("2. The model was trained with 82 features but we're only providing 73")
print("3. Missing features might include calculated features that aren't being generated")