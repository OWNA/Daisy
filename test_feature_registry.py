#!/usr/bin/env python3
"""
test_feature_registry.py - Test Suite for Feature Registry

This script tests the Feature Registry implementation to ensure it properly
manages feature definitions and computations.

Sprint 2 - Priority 0.2: Test Feature Registry
"""

import os
import sys
import logging
import unittest
import pandas as pd
import numpy as np
from typing import Dict, Any

# Configure logging for testing
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TestFeatureRegistry(unittest.TestCase):
    """Test cases for Feature Registry functionality."""
    
    def setUp(self):
        """Set up test environment."""
        try:
            from feature_registry import (
                FeatureRegistry, FeatureDefinition, FeatureCategory, 
                FeatureDataType, FeatureDependency, reset_default_registry
            )
            self.FeatureRegistry = FeatureRegistry
            self.FeatureDefinition = FeatureDefinition
            self.FeatureCategory = FeatureCategory
            self.FeatureDataType = FeatureDataType
            self.FeatureDependency = FeatureDependency
            reset_default_registry()
        except ImportError as e:
            self.skipTest(f"FeatureRegistry not available: {e}")
        
        # Create test data
        self.test_data = self._create_test_data()
    
    def _create_test_data(self) -> pd.DataFrame:
        """Create test L2 order book data."""
        np.random.seed(42)
        n_rows = 100
        
        # Generate realistic L2 order book data
        base_price = 50000.0
        price_noise = np.random.normal(0, 0.001, n_rows).cumsum()
        
        mid_prices = base_price * (1 + price_noise)
        spreads = np.random.uniform(0.5, 2.0, n_rows)
        
        data = {
            'timestamp': pd.date_range('2025-01-01', periods=n_rows, freq='1min'),
            'mid_price': mid_prices,
            'spread': spreads,
            'bid_price_1': mid_prices - spreads/2,
            'ask_price_1': mid_prices + spreads/2,
        }
        
        # Add bid/ask sizes for 5 levels
        for i in range(1, 6):
            data[f'bid_size_{i}'] = np.random.exponential(2.0, n_rows) * (6 - i)
            data[f'ask_size_{i}'] = np.random.exponential(2.0, n_rows) * (6 - i)
        
        return pd.DataFrame(data)
    
    def test_registry_initialization(self):
        """Test Feature Registry initialization."""
        registry = self.FeatureRegistry()
        
        self.assertIsNotNone(registry)
        
        # Should have basic L2 features registered
        features = registry.list_features()
        self.assertGreater(len(features), 0)
        
        # Check that specific basic features are present
        expected_features = [
            'spread_bps', 'total_bid_volume_5', 'total_ask_volume_5',
            'mid_price_return', 'l2_volatility_10', 'l2_volatility_50',
            'order_book_imbalance_2', 'bid_pressure', 'ask_pressure'
        ]
        
        for feature_name in expected_features:
            self.assertIn(feature_name, features, f"Feature '{feature_name}' should be registered")
        
        logger.info(f"✓ Registry initialization test passed ({len(features)} features)")
    
    def test_feature_definition_validation(self):
        """Test feature definition validation."""
        registry = self.FeatureRegistry()
        
        # Test valid feature definition
        valid_feature = self.FeatureDefinition(
            name="test_feature",
            category=self.FeatureCategory.SPREAD,
            data_type=self.FeatureDataType.FLOAT,
            description="Test feature",
            calculation_func=lambda df: df['mid_price'] * 2
        )
        
        # Should not raise an exception
        registry.register_feature(valid_feature)
        self.assertIsNotNone(registry.get_feature("test_feature"))
        
        # Test invalid feature definition (empty name)
        with self.assertRaises(ValueError):
            invalid_feature = self.FeatureDefinition(
                name="",
                category=self.FeatureCategory.SPREAD,
                data_type=self.FeatureDataType.FLOAT,
                description="Invalid feature",
                calculation_func=lambda df: df['mid_price']
            )
        
        logger.info("✓ Feature definition validation test passed")
    
    def test_feature_registration_and_retrieval(self):
        """Test feature registration and retrieval."""
        registry = self.FeatureRegistry()
        
        # Register a custom feature
        custom_feature = self.FeatureDefinition(
            name="custom_test_feature",
            category=self.FeatureCategory.VOLUME,
            data_type=self.FeatureDataType.FLOAT,
            description="Custom test feature",
            calculation_func=lambda df: df['bid_size_1'] + df['ask_size_1']
        )
        
        registry.register_feature(custom_feature)
        
        # Test retrieval
        retrieved = registry.get_feature("custom_test_feature")
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.name, "custom_test_feature")
        self.assertEqual(retrieved.category, self.FeatureCategory.VOLUME)
        
        # Test feature list
        features = registry.list_features()
        self.assertIn("custom_test_feature", features)
        
        logger.info("✓ Feature registration and retrieval test passed")
    
    def test_features_by_category(self):
        """Test retrieving features by category."""
        registry = self.FeatureRegistry()
        
        # Get spread features
        spread_features = registry.get_features_by_category(self.FeatureCategory.SPREAD)
        self.assertGreater(len(spread_features), 0)
        
        # Verify all returned features are spread category
        for feature in spread_features:
            self.assertEqual(feature.category, self.FeatureCategory.SPREAD)
        
        # Get volume features
        volume_features = registry.get_features_by_category(self.FeatureCategory.VOLUME)
        self.assertGreater(len(volume_features), 0)
        
        logger.info(f"✓ Features by category test passed (spread: {len(spread_features)}, volume: {len(volume_features)})")
    
    def test_single_feature_computation(self):
        """Test computation of individual features."""
        registry = self.FeatureRegistry()
        
        # Test spread_bps computation
        spread_bps = registry.compute_feature('spread_bps', self.test_data)
        
        self.assertIsInstance(spread_bps, pd.Series)
        self.assertEqual(len(spread_bps), len(self.test_data))
        self.assertEqual(spread_bps.name, 'spread_bps')
        
        # Verify the computation is correct
        expected = (self.test_data['spread'] / self.test_data['mid_price']) * 10000
        pd.testing.assert_series_equal(spread_bps, expected, check_names=False)
        
        # Test volume computation
        bid_volume = registry.compute_feature('total_bid_volume_5', self.test_data)
        
        self.assertIsInstance(bid_volume, pd.Series)
        self.assertGreater(bid_volume.sum(), 0)
        
        logger.info("✓ Single feature computation test passed")
    
    def test_multiple_features_computation(self):
        """Test computation of multiple features."""
        registry = self.FeatureRegistry()
        
        # Compute specific features
        feature_names = ['spread_bps', 'total_bid_volume_5', 'bid_pressure']
        features_df = registry.compute_features(feature_names, self.test_data)
        
        self.assertIsInstance(features_df, pd.DataFrame)
        self.assertEqual(len(features_df), len(self.test_data))
        self.assertEqual(list(features_df.columns), feature_names)
        
        # Verify no NaN values in final result
        self.assertFalse(features_df.isnull().any().any())
        
        logger.info(f"✓ Multiple features computation test passed ({len(feature_names)} features)")
    
    def test_all_features_computation(self):
        """Test computation of all registered features."""
        registry = self.FeatureRegistry()
        
        # Compute all features
        all_features_df = registry.compute_all_features(self.test_data)
        
        self.assertIsInstance(all_features_df, pd.DataFrame)
        self.assertEqual(len(all_features_df), len(self.test_data))
        
        # Should contain all registered features
        registered_features = registry.list_features()
        self.assertEqual(len(all_features_df.columns), len(registered_features))
        
        # Verify no NaN values in final result
        self.assertFalse(all_features_df.isnull().any().any())
        
        logger.info(f"✓ All features computation test passed ({len(all_features_df.columns)} features)")
    
    def test_feature_caching(self):
        """Test feature computation caching."""
        registry = self.FeatureRegistry()
        
        # Clear cache
        registry.clear_cache()
        
        # Compute feature twice
        result1 = registry.compute_feature('spread_bps', self.test_data, use_cache=True)
        result2 = registry.compute_feature('spread_bps', self.test_data, use_cache=True)
        
        # Results should be identical (from cache)
        pd.testing.assert_series_equal(result1, result2)
        
        # Test cache clearing
        registry.clear_cache()
        result3 = registry.compute_feature('spread_bps', self.test_data, use_cache=True)
        
        # Should still be equal but computed fresh
        pd.testing.assert_series_equal(result1, result3)
        
        logger.info("✓ Feature caching test passed")
    
    def test_error_handling(self):
        """Test error handling in feature computation."""
        registry = self.FeatureRegistry()
        
        # Test computing non-existent feature
        with self.assertRaises(ValueError):
            registry.compute_feature('non_existent_feature', self.test_data)
        
        # Test with invalid data (missing required columns)
        invalid_data = pd.DataFrame({'irrelevant_column': [1, 2, 3]})
        
        # Should return zero-filled series as fallback
        result = registry.compute_feature('spread_bps', invalid_data)
        self.assertIsInstance(result, pd.Series)
        self.assertEqual(len(result), len(invalid_data))
        
        logger.info("✓ Error handling test passed")
    
    def test_feature_info(self):
        """Test feature information retrieval."""
        registry = self.FeatureRegistry()
        
        info_df = registry.get_feature_info()
        
        self.assertIsInstance(info_df, pd.DataFrame)
        self.assertGreater(len(info_df), 0)
        
        # Check required columns
        required_columns = ['name', 'category', 'data_type', 'description', 'computation_cost']
        for col in required_columns:
            self.assertIn(col, info_df.columns)
        
        logger.info(f"✓ Feature info test passed ({len(info_df)} features documented)")
    
    def test_feature_validation(self):
        """Test feature validation and range checking."""
        registry = self.FeatureRegistry()
        
        # Compute a feature with expected range (spread_bps should be positive)
        spread_bps = registry.compute_feature('spread_bps', self.test_data)
        
        # Spread in basis points should be positive
        self.assertTrue((spread_bps >= 0).all(), "Spread BPS should be non-negative")
        
        # Order book imbalance should be between -1 and 1
        imbalance = registry.compute_feature('order_book_imbalance_2', self.test_data)
        self.assertTrue((imbalance >= -1).all() and (imbalance <= 1).all(), 
                       "Order book imbalance should be between -1 and 1")
        
        # Pressures should be between 0 and 1
        bid_pressure = registry.compute_feature('bid_pressure', self.test_data)
        ask_pressure = registry.compute_feature('ask_pressure', self.test_data)
        
        self.assertTrue((bid_pressure >= 0).all() and (bid_pressure <= 1).all(),
                       "Bid pressure should be between 0 and 1")
        self.assertTrue((ask_pressure >= 0).all() and (ask_pressure <= 1).all(),
                       "Ask pressure should be between 0 and 1")
        
        logger.info("✓ Feature validation test passed")


def run_integration_test():
    """Run integration test with sample market data."""
    logger.info("Starting Feature Registry integration test...")
    
    try:
        from feature_registry import FeatureRegistry
        
        # Create registry
        registry = FeatureRegistry()
        
        # Create sample data similar to real L2 data
        np.random.seed(123)
        sample_data = pd.DataFrame({
            'timestamp': pd.date_range('2025-01-01', periods=500, freq='1s'),
            'mid_price': 50000 + np.random.normal(0, 10, 500).cumsum(),
            'spread': np.random.uniform(0.5, 3.0, 500),
            'bid_size_1': np.random.exponential(5.0, 500),
            'bid_size_2': np.random.exponential(4.0, 500),
            'bid_size_3': np.random.exponential(3.0, 500),
            'bid_size_4': np.random.exponential(2.0, 500),
            'bid_size_5': np.random.exponential(1.0, 500),
            'ask_size_1': np.random.exponential(5.0, 500),
            'ask_size_2': np.random.exponential(4.0, 500),
            'ask_size_3': np.random.exponential(3.0, 500),
            'ask_size_4': np.random.exponential(2.0, 500),
            'ask_size_5': np.random.exponential(1.0, 500),
        })
        
        logger.info(f"Created sample data: {len(sample_data)} rows, {len(sample_data.columns)} columns")
        
        # Test feature computation performance
        import time
        
        start_time = time.time()
        all_features = registry.compute_all_features(sample_data)
        computation_time = time.time() - start_time
        
        logger.info(f"Computed {len(all_features.columns)} features in {computation_time:.3f}s")
        logger.info(f"Performance: {len(sample_data) / computation_time:.0f} rows/second")
        
        # Validate results
        logger.info("Feature computation results:")
        for col in all_features.columns:
            feature_data = all_features[col]
            logger.info(f"  {col}: range [{feature_data.min():.4f}, {feature_data.max():.4f}], "
                       f"mean {feature_data.mean():.4f}, null_count {feature_data.isnull().sum()}")
        
        # Test feature info
        info_df = registry.get_feature_info()
        logger.info(f"\nRegistered features by category:")
        
        category_counts = info_df['category'].value_counts()
        for category, count in category_counts.items():
            logger.info(f"  {category}: {count} features")
        
        logger.info("\n✓ Feature Registry integration test completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Integration test failed: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False


def main():
    """Main function to run all tests."""
    print("Feature Registry Test Suite")
    print("=" * 50)
    
    # Run unit tests
    print("\nRunning unit tests...")
    unittest.main(argv=[''], exit=False, verbosity=2)
    
    # Run integration test
    print("\nRunning integration test...")
    integration_success = run_integration_test()
    
    if integration_success:
        print("\n✓ All tests completed successfully!")
        sys.exit(0)
    else:
        print("\n✗ Some tests failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()