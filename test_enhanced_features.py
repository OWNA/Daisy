"""
Test script for enhanced FeatureEngineer with order flow imbalance features.
This demonstrates the new features added to improve signal quality.
"""

import pandas as pd
import numpy as np
import yaml
from featureengineer import FeatureEngineer
from datetime import datetime, timedelta

def create_sample_l2_data(n_rows=1000):
    """Create sample L2 order book data for testing."""
    np.random.seed(42)
    
    # Base price
    base_price = 50000
    
    # Generate timestamps (100ms intervals)
    timestamps = pd.date_range(
        start=datetime.now() - timedelta(seconds=n_rows * 0.1),
        periods=n_rows,
        freq='100ms'
    )
    
    data = {'timestamp': timestamps}
    
    # Generate realistic L2 data with some patterns
    for i in range(n_rows):
        # Add some trend and volatility
        trend = np.sin(i / 100) * 50
        volatility = np.random.normal(0, 10)
        
        mid_price = base_price + trend + volatility
        spread = np.random.uniform(0.5, 2.0)
        
        # Generate bid prices (descending)
        for level in range(1, 11):
            data.setdefault(f'bid_price_{level}', []).append(
                mid_price - spread/2 - (level-1) * np.random.uniform(0.5, 1.5)
            )
            # Generate bid sizes with some correlation to price distance
            size_base = np.random.uniform(0.1, 2.0)
            size_variation = np.random.exponential(0.5)
            data.setdefault(f'bid_size_{level}', []).append(
                size_base * (1 + size_variation) / level
            )
        
        # Generate ask prices (ascending)
        for level in range(1, 11):
            data.setdefault(f'ask_price_{level}', []).append(
                mid_price + spread/2 + (level-1) * np.random.uniform(0.5, 1.5)
            )
            # Generate ask sizes
            size_base = np.random.uniform(0.1, 2.0)
            size_variation = np.random.exponential(0.5)
            data.setdefault(f'ask_size_{level}', []).append(
                size_base * (1 + size_variation) / level
            )
    
    return pd.DataFrame(data)

def analyze_new_features(df_features):
    """Analyze and summarize the new features."""
    
    # Identify new feature categories
    ofi_features = [col for col in df_features.columns if 'ofi_' in col]
    pressure_features = [col for col in df_features.columns if 'pressure' in col]
    stability_features = [col for col in df_features.columns if 'stability' in col or 'lifetime' in col or 'resilience' in col]
    
    print("\n=== ENHANCED FEATURE ANALYSIS ===\n")
    
    print(f"1. Order Flow Imbalance Features ({len(ofi_features)}):")
    for feat in sorted(ofi_features):
        print(f"   - {feat}")
    
    print(f"\n2. Book Pressure Features ({len(pressure_features)}):")
    for feat in sorted(pressure_features):
        print(f"   - {feat}")
    
    print(f"\n3. Stability Indicators ({len(stability_features)}):")
    for feat in sorted(stability_features):
        print(f"   - {feat}")
    
    # Show sample statistics
    print("\n=== FEATURE STATISTICS (last 100 rows) ===\n")
    
    key_features = [
        'ofi_10s', 'ofi_normalized_10s', 'ofi_weighted_10s',
        'pressure_imbalance_weighted', 'book_depth_asymmetry',
        'quote_lifetime', 'book_resilience', 'spread_stability_50'
    ]
    
    for feat in key_features:
        if feat in df_features.columns:
            values = df_features[feat].iloc[-100:]
            print(f"{feat:30s} | Mean: {values.mean():8.4f} | Std: {values.std():8.4f} | "
                  f"Min: {values.min():8.4f} | Max: {values.max():8.4f}")
    
    return ofi_features, pressure_features, stability_features

def main():
    """Test the enhanced feature engineer."""
    
    # Load config
    try:
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)
    except:
        config = {'l2_features': []}
    
    # Create sample data
    print("Creating sample L2 order book data...")
    df_l2 = create_sample_l2_data(n_rows=5000)  # 8.3 minutes of data
    print(f"Sample data shape: {df_l2.shape}")
    
    # Initialize enhanced feature engineer
    feature_engineer = FeatureEngineer(config)
    
    # Generate features
    print("\nGenerating enhanced features...")
    df_features = feature_engineer.generate_features(df_l2)
    
    # Analyze new features
    ofi_features, pressure_features, stability_features = analyze_new_features(df_features)
    
    # Summary
    print("\n=== FEATURE SUMMARY ===")
    print(f"Total features generated: {len(df_features.columns)}")
    print(f"Original L2 columns: {len(df_l2.columns)}")
    print(f"New features added: {len(df_features.columns) - len(df_l2.columns)}")
    
    # Check for any NaN values
    nan_counts = df_features.isna().sum()
    if nan_counts.sum() > 0:
        print("\nWarning: NaN values found in:")
        print(nan_counts[nan_counts > 0])
    else:
        print("\nNo NaN values found - all features properly calculated!")
    
    # Save a sample for inspection
    sample_file = 'enhanced_features_sample.csv'
    df_features.iloc[-500:].to_csv(sample_file, index=False)
    print(f"\nSample of enhanced features saved to: {sample_file}")
    
    return df_features

if __name__ == "__main__":
    df_with_features = main()