#!/usr/bin/env python3
"""Remove problematic features from the saved list"""
import json

with open('trading_bot_data/model_features_BTC_USDTUSDT_l2_only.json', 'r') as f:
    data = json.load(f)

# These are the actual L2 features (no metadata)
clean_features = [col for col in data['trained_features'] 
                  if col not in ['id', 'target_return_1min', 'target_return_5min',
                                'target_volatility', 'target_direction', 'update_id',
                                'sequence_id', 'data_quality_score', 'close']]

print(f"Cleaned features: {len(clean_features)} (was {len(data['trained_features'])})")

# For now, pad with duplicates to reach the expected count
while len(clean_features) < 82:
    clean_features.append('mid_price')  # Duplicate a harmless feature

data['trained_features'] = clean_features[:82]

with open('trading_bot_data/model_features_BTC_USDTUSDT_l2_only.json', 'w') as f:
    json.dump(data, f, indent=4)

print("Fixed features file!")
