"""
Normalize target variables for better model generalization across price levels
"""

import pandas as pd
import numpy as np

def normalize_price_targets(df, price_col='mid_price', target_col='target'):
    """
    Normalize targets as percentage changes or z-scores
    """
    
    # Option 1: Convert to percentage returns
    if target_col in df.columns and price_col in df.columns:
        # Convert absolute price changes to percentage
        df[f'{target_col}_pct'] = df[target_col] / df[price_col] * 100
        
        # Option 2: Z-score normalization of returns
        returns = df[target_col] / df[price_col]
        df[f'{target_col}_zscore'] = (returns - returns.mean()) / returns.std()
        
        # Option 3: Log returns (most stable)
        df[f'{target_col}_log'] = np.log(df[price_col] + df[target_col]) - np.log(df[price_col])
        
    return df

def suggest_label_generator_update():
    """
    Suggested update to LabelGenerator to create normalized targets
    """
    print("""
    In labelgenerator.py, update generate_labels() to:
    
    # Instead of:
    df['target'] = df['mid_price'].shift(-future_window) - df['mid_price']
    
    # Use percentage returns:
    df['target'] = ((df['mid_price'].shift(-future_window) - df['mid_price']) / df['mid_price']) * 100
    
    # Or use log returns:
    df['target'] = np.log(df['mid_price'].shift(-future_window) / df['mid_price'])
    
    This makes the model learn price movements as percentages, 
    not absolute dollar amounts, so it works at any price level.
    """)

if __name__ == "__main__":
    suggest_label_generator_update()
    
    print("""
    Benefits of normalized targets:
    1. Model works at $50k or $150k BTC equally well
    2. Can combine data from different time periods
    3. Predictions are in % terms (e.g., +0.5% move)
    4. More stable training across market regimes
    
    After implementing, the model will predict:
    - "0.5" = 0.5% price increase expected
    - "-0.3" = 0.3% price decrease expected
    
    Much more intuitive and stable!
    """)