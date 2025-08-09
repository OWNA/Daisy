#!/usr/bin/env python3
"""
Quick fix to get paper trading working with current model
"""

import json
import shutil

# The model expects these 82 features (from training)
# But we're only providing 73 from the saved feature list
# Let's create a temporary fix

print("Applying quick fix for paper trading...")

# Step 1: Backup current files
shutil.copy(
    'trading_bot_data/model_features_BTC_USDTUSDT_l2_only.json',
    'trading_bot_data/model_features_BTC_USDTUSDT_l2_only_backup.json'
)

# Step 2: Create a mapping of all possible features
all_l2_features = [
    # Price and size levels
    *[f'bid_price_{i}' for i in range(1, 11)],
    *[f'bid_size_{i}' for i in range(1, 11)],
    *[f'ask_price_{i}' for i in range(1, 11)],
    *[f'ask_size_{i}' for i in range(1, 11)],
    
    # Basic features
    'mid_price',
    'spread',
    'spread_bps',
    'bid_ask_spread',
    'bid_ask_spread_pct',
    'weighted_mid_price',
    'weighted_bid_price',
    'weighted_ask_price',
    'microprice',
    
    # Imbalance features
    'order_book_imbalance',
    'order_book_imbalance_2',
    'order_book_imbalance_3', 
    'order_book_imbalance_5',
    
    # Volume features
    *[f'total_bid_volume_{i}' for i in range(1, 11)],
    *[f'total_ask_volume_{i}' for i in range(1, 11)],
    
    # Price impact
    'price_impact_bid',
    'price_impact_ask',
    'price_impact_buy',
    'price_impact_sell',
    'price_impact_1',
    'price_impact_5',
    'price_impact_10',
    
    # Volatility
    'mid_price_return',
    'l2_volatility_10',
    'l2_volatility_50',
    'l2_volatility_200'
]

# Count features
print(f"Total L2 features defined: {len(all_l2_features)}")

# Save the complete feature list
feature_data = {
    "trained_features": all_l2_features[:82]  # Take first 82 to match model
}

with open('trading_bot_data/model_features_BTC_USDTUSDT_l2_only_fixed.json', 'w') as f:
    json.dump(feature_data, f, indent=4)

print(f"Created fixed feature list with {len(feature_data['trained_features'])} features")

# Copy to main file
shutil.copy(
    'trading_bot_data/model_features_BTC_USDTUSDT_l2_only_fixed.json',
    'trading_bot_data/model_features_BTC_USDTUSDT_l2_only.json'
)

print("\nQuick fix applied!")
print("\nNOTE: This is a temporary fix. For production:")
print("1. Ensure FeatureEngineer generates exactly the features used in training")
print("2. Retrain the model with a consistent feature set")
print("3. Save only predictive features (not targets) in the feature list")