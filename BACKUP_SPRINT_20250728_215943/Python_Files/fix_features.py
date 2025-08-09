#!/usr/bin/env python3
"""
Fix the features JSON file to only include actual predictive features
"""

import json

# Load current features
with open('trading_bot_data/model_features_BTC_USDTUSDT_l2_only.json', 'r') as f:
    data = json.load(f)

# Get all features
all_features = data['trained_features']

# Filter out non-predictive columns
exclude_columns = [
    'id', 'timestamp', 'symbol', 'exchange',
    'target', 'target_return_1min', 'target_return_5min', 
    'target_volatility', 'target_direction',
    'update_id', 'sequence_id', 'data_quality_score',
    'close'  # This is derived from mid_price in L2-only mode
]

# Keep only actual features
predictive_features = [f for f in all_features if f not in exclude_columns]

print(f"Original features: {len(all_features)}")
print(f"Filtered features: {len(predictive_features)}")

# Save cleaned features
data['trained_features'] = predictive_features

with open('trading_bot_data/model_features_BTC_USDTUSDT_l2_only_fixed.json', 'w') as f:
    json.dump(data, f, indent=4)

# Also create a backup of original
import shutil
shutil.copy(
    'trading_bot_data/model_features_BTC_USDTUSDT_l2_only.json',
    'trading_bot_data/model_features_BTC_USDTUSDT_l2_only_original.json'
)

# Replace original with fixed version
shutil.copy(
    'trading_bot_data/model_features_BTC_USDTUSDT_l2_only_fixed.json',
    'trading_bot_data/model_features_BTC_USDTUSDT_l2_only.json'
)

print("\nFixed features file saved!")
print(f"Predictive features: {predictive_features[:5]}...")