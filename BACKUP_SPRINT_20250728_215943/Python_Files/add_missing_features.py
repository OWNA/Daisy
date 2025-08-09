#!/usr/bin/env python3
"""
Add missing features that the model expects
"""

# The model was trained with these additional features that aren't being generated:
# These are likely calculated features that need to be added

missing_features = [
    'bid_ask_spread',        # This is just 'spread' with different name
    'bid_ask_spread_pct',    # This is 'spread_bps' / 10000
    'weighted_mid_price',    # Weighted average of bid/ask
    'order_book_imbalance_2', # Imbalance at level 2
    'order_book_imbalance_3', # Imbalance at level 3 
    'order_book_imbalance_5', # Imbalance at level 5
    'price_impact_buy',      # Price impact for buying
    'price_impact_sell',     # Price impact for selling
    'price_impact_1',        # Price impact at level 1
]

print("Missing features that need to be added:")
for f in missing_features:
    print(f"  - {f}")

print("\nThese features are in the config.yaml l2_features list but not being generated!")
print("The FeatureEngineer needs to calculate these additional features.")