#!/usr/bin/env python3
"""
Test that L2 price processing is fixed
"""

import yaml
import pandas as pd
from datahandler import DataHandler

# Load config
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Create DataHandler
data_handler = DataHandler(config, None)

# Create a mock orderbook with realistic BTC data
mock_orderbook = {
    'bids': [
        [67500, 0.5],    # Best bid
        [67499, 1.2],
        [67498, 0.8],
        [67497, 2.1],
        [67496, 1.5]     # Only 5 levels
    ],
    'asks': [
        [67501, 0.4],    # Best ask
        [67502, 1.1],
        [67503, 0.9],
        [67504, 1.8],
        [67505, 2.0]     # Only 5 levels
    ]
}

print("Testing L2 snapshot processing...")
print(f"Mock orderbook has {len(mock_orderbook['bids'])} bid levels and {len(mock_orderbook['asks'])} ask levels")

# Process the snapshot
features = data_handler._process_l2_snapshot(mock_orderbook)

print(f"\nProcessed features:")
print(f"  Mid price: ${features['mid_price']:.2f}")
print(f"  Spread: ${features['spread']:.2f}")
print(f"  Spread (bps): {features['spread_bps']:.2f}")

# Check all 10 levels
print("\nPrice levels (should extrapolate missing levels):")
for i in range(1, 11):
    bid_price = features[f'bid_price_{i}']
    ask_price = features[f'ask_price_{i}']
    print(f"  Level {i}: Bid=${bid_price:.2f}, Ask=${ask_price:.2f}")

# Verify no zero prices
zero_prices = []
for i in range(1, 11):
    if features[f'bid_price_{i}'] == 0:
        zero_prices.append(f'bid_price_{i}')
    if features[f'ask_price_{i}'] == 0:
        zero_prices.append(f'ask_price_{i}')

if zero_prices:
    print(f"\n❌ ERROR: Found zero prices in: {zero_prices}")
else:
    print(f"\n✅ SUCCESS: No zero prices found, all levels have realistic values")

# Test with empty orderbook
print("\n\nTesting with minimal orderbook...")
minimal_orderbook = {
    'bids': [[67500, 1.0]],
    'asks': [[67501, 1.0]]
}

features2 = data_handler._process_l2_snapshot(minimal_orderbook)
print(f"Mid price: ${features2['mid_price']:.2f}")
print(f"Bid level 10: ${features2['bid_price_10']:.2f} (should be ~${features2['mid_price'] * 0.99:.2f})")
print(f"Ask level 10: ${features2['ask_price_10']:.2f} (should be ~${features2['mid_price'] * 1.01:.2f})")