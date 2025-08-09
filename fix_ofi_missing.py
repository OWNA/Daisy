"""
Quick fix to add missing OFI features when insufficient historical data
"""

import pandas as pd

def add_missing_ofi_features(df):
    """Add OFI features with zeros when not enough historical data"""
    
    ofi_features = [
        'ofi_10s', 'ofi_normalized_10s', 'ofi_weighted_10s',
        'ofi_30s', 'ofi_normalized_30s', 'ofi_weighted_30s', 
        'ofi_1m', 'ofi_normalized_1m', 'ofi_weighted_1m',
        'ofi_5m', 'ofi_normalized_5m', 'ofi_weighted_5m'
    ]
    
    for feature in ofi_features:
        if feature not in df.columns:
            df[feature] = 0.0
            
    return df

if __name__ == "__main__":
    print("This fix has been applied to the feature engineer.")
    print("OFI features will be set to 0 when insufficient data.")
    print("They will calculate properly once enough historical data accumulates.")