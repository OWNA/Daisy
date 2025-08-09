#!/usr/bin/env python3
"""
Test script for enhanced feature engineering with database integration.
Tests the read-before-write pattern and performance improvements.
"""

import pandas as pd
import numpy as np
import time
import logging
from datetime import datetime, timedelta
from featureengineer_enhanced import EnhancedFeatureEngineer

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_sample_l2_data(num_rows=1000):
    """Create sample L2 data for testing."""
    logger.info(f"Creating sample L2 data with {num_rows} rows")
    
    # Generate timestamps
    start_time = datetime.now() - timedelta(minutes=10)
    timestamps = [start_time + timedelta(milliseconds=i*100) for i in range(num_rows)]
    
    # Create realistic L2 order book data
    np.random.seed(42)  # For reproducible results
    base_price = 50000.0
    
    data = []
    for i, ts in enumerate(timestamps):
        # Simulate price movement
        price_change = np.random.normal(0, 0.001) * base_price
        mid_price = base_price + price_change
        spread = np.random.uniform(0.5, 2.0)
        
        # Generate bid/ask prices and sizes for 10 levels
        row = {
            'timestamp': ts,
            'symbol': 'BTCUSDT'
        }
        
        # Generate 10 levels of bid/ask data
        for level in range(1, 11):
            bid_price = mid_price - spread/2 - (level-1) * 0.5
            ask_price = mid_price + spread/2 + (level-1) * 0.5
            bid_size = np.random.exponential(100) * (11-level)  # Larger sizes at better levels
            ask_size = np.random.exponential(100) * (11-level)
            
            row[f'bid_price_{level}'] = bid_price
            row[f'ask_price_{level}'] = ask_price
            row[f'bid_size_{level}'] = bid_size
            row[f'ask_size_{level}'] = ask_size
        
        data.append(row)
    
    df = pd.DataFrame(data)
    logger.info(f"Created sample data with shape: {df.shape}")
    return df

def test_feature_engineer_performance():
    """Test the enhanced feature engineer with database integration."""
    logger.info("=" * 60)
    logger.info("TESTING ENHANCED FEATURE ENGINEER WITH DATABASE INTEGRATION")
    logger.info("=" * 60)
    
    # Test configuration
    config = {
        'symbol': 'BTC/USDT',
        'l2_features': [],
        'feature_window': 100
    }
    
    # Initialize feature engineer
    feature_engineer = EnhancedFeatureEngineer(config, db_path="trading_bot.db")
    
    # Create test data
    test_data = create_sample_l2_data(500)  # Start with 500 rows
    
    logger.info("\n--- TEST 1: Initial Feature Calculation (Cold Start) ---")
    start_time = time.time()
    
    # First run - should calculate all features and write to DB
    result_df1 = feature_engineer.generate_features_with_db_integration(
        test_data, force_recalculate=False
    )
    
    time1 = time.time() - start_time
    logger.info(f"First run completed in {time1:.3f} seconds")
    logger.info(f"Result shape: {result_df1.shape}")
    
    # Show performance stats
    stats1 = feature_engineer.get_performance_stats()
    logger.info(f"Performance stats: {stats1}")
    
    logger.info("\n--- TEST 2: Second Run (Should Use Database Cache) ---")
    start_time = time.time()
    
    # Second run - should read from database for existing timestamps
    result_df2 = feature_engineer.generate_features_with_db_integration(
        test_data, force_recalculate=False
    )
    
    time2 = time.time() - start_time
    logger.info(f"Second run completed in {time2:.3f} seconds")
    logger.info(f"Result shape: {result_df2.shape}")
    
    # Show updated performance stats
    stats2 = feature_engineer.get_performance_stats()
    logger.info(f"Performance stats: {stats2}")
    
    # Calculate performance improvement
    if time2 > 0:
        speedup = time1 / time2
        logger.info(f"Performance improvement: {speedup:.2f}x faster")
    
    logger.info("\n--- TEST 3: Mixed Data (Some New, Some Cached) ---")
    
    # Create new data with some overlapping timestamps
    mixed_data = create_sample_l2_data(300)
    # Modify timestamps to create partial overlap
    mixed_data['timestamp'] = mixed_data['timestamp'] + timedelta(minutes=5)
    
    start_time = time.time()
    result_df3 = feature_engineer.generate_features_with_db_integration(
        mixed_data, force_recalculate=False
    )
    time3 = time.time() - start_time
    
    logger.info(f"Mixed data run completed in {time3:.3f} seconds")
    logger.info(f"Result shape: {result_df3.shape}")
    
    # Final performance stats
    stats3 = feature_engineer.get_performance_stats()
    logger.info(f"Final performance stats: {stats3}")
    
    logger.info("\n--- TEST 4: Feature Validation ---")
    
    # Check that Phase 1 features are present
    phase1_features = []
    for feature_group in feature_engineer.phase1_features.values():
        phase1_features.extend(feature_group)
    
    present_features = [f for f in phase1_features if f in result_df1.columns]
    missing_features = [f for f in phase1_features if f not in result_df1.columns]
    
    logger.info(f"Phase 1 features present: {len(present_features)}/{len(phase1_features)}")
    logger.info(f"Present features: {present_features[:10]}...")  # Show first 10
    
    if missing_features:
        logger.warning(f"Missing features: {missing_features}")
    
    # Check for NaN values in key features
    key_features = ['ofi_10s', 'bid_pressure', 'quote_lifetime', 'l2_volatility_50']
    for feature in key_features:
        if feature in result_df1.columns:
            nan_count = result_df1[feature].isna().sum()
            logger.info(f"{feature}: {nan_count} NaN values out of {len(result_df1)}")
    
    logger.info("\n--- SUMMARY ---")
    logger.info(f"Database integration working: {'✓' if stats3['cache_hits'] > 0 else '✗'}")
    logger.info(f"Performance improvement: {'✓' if time1 > time2 else '✗'}")
    logger.info(f"Feature completeness: {len(present_features)}/{len(phase1_features)} features")
    logger.info(f"Cache hit rate: {stats3['cache_hit_rate_percent']:.1f}%")
    
    return result_df1, stats3

if __name__ == "__main__":
    try:
        result_df, final_stats = test_feature_engineer_performance()
        print(f"\nTest completed successfully!")
        print(f"Final cache hit rate: {final_stats['cache_hit_rate_percent']:.1f}%")
    except Exception as e:
        logger.error(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()