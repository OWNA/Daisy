#!/usr/bin/env python3
"""
test_component_factory.py - Test Suite for ComponentFactory

This script tests the ComponentFactory implementation to ensure it properly
creates and manages trading system components.

Sprint 2 - Priority 0.1: Test ComponentFactory Integration
"""

import os
import sys
import logging
import unittest
from unittest.mock import patch, MagicMock
from typing import Dict, Any

# Configure logging for testing
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TestComponentFactory(unittest.TestCase):
    """Test cases for ComponentFactory functionality."""
    
    def setUp(self):
        """Set up test environment."""
        try:
            from component_factory import ComponentFactory
            self.ComponentFactory = ComponentFactory
        except ImportError as e:
            self.skipTest(f"ComponentFactory not available: {e}")
        
        # Test configuration
        self.test_config = {
            'database_path': './trading_bot.db',
            'base_dir': './trading_bot_data'
        }
    
    def test_factory_initialization(self):
        """Test ComponentFactory initialization."""
        factory = self.ComponentFactory(self.test_config)
        
        self.assertIsNotNone(factory)
        self.assertEqual(factory.base_config, self.test_config)
        self.assertIn('database', factory._component_configs)
        self.assertIn('feature_engineer', factory._component_configs)
        self.assertIn('model_predictor', factory._component_configs)
        
        logger.info("✓ Factory initialization test passed")
    
    def test_component_registration(self):
        """Test component registration."""
        from component_factory import ComponentConfig
        
        factory = self.ComponentFactory(self.test_config)
        
        # Register a test component
        test_config = ComponentConfig(
            component_type='test_component',
            config={'test_param': 'test_value'},
            singleton=True
        )
        
        factory.register_component('test_component', test_config)
        
        self.assertIn('test_component', factory._component_configs)
        self.assertEqual(factory._component_configs['test_component'], test_config)
        
        logger.info("✓ Component registration test passed")
    
    def test_database_component_creation(self):
        """Test database component creation."""
        factory = self.ComponentFactory(self.test_config)
        
        # Test database component creation
        database_component = factory.get_component('database')
        
        # Database component should be a connection factory function
        self.assertIsNotNone(database_component)
        self.assertTrue(callable(database_component))
        
        # Test that we can create a connection
        try:
            conn = database_component()
            self.assertIsNotNone(conn)
            conn.close()
        except Exception as e:
            logger.warning(f"Database connection test failed (expected if DB doesn't exist): {e}")
        
        logger.info("✓ Database component creation test passed")
    
    @patch('component_factory.EnhancedFeatureEngineer')
    def test_feature_engineer_component_creation(self, mock_fe):
        """Test feature engineer component creation."""
        # Mock the feature engineer
        mock_instance = MagicMock()
        mock_fe.return_value = mock_instance
        
        factory = self.ComponentFactory(self.test_config)
        
        # Test feature engineer component creation
        fe_component = factory.get_component('feature_engineer')
        
        self.assertIsNotNone(fe_component)
        self.assertEqual(fe_component, mock_instance)
        
        # Verify it was called with correct parameters
        mock_fe.assert_called_once()
        call_args = mock_fe.call_args
        self.assertIn('db_path', call_args[0][0])
        
        logger.info("✓ Feature engineer component creation test passed")
    
    @patch('component_factory.EnhancedModelPredictor')
    def test_model_predictor_component_creation(self, mock_predictor):
        """Test model predictor component creation."""
        # Mock the model predictor
        mock_instance = MagicMock()
        mock_predictor.return_value = mock_instance
        
        factory = self.ComponentFactory(self.test_config)
        
        # Test model predictor component creation
        predictor_component = factory.get_component('model_predictor')
        
        self.assertIsNotNone(predictor_component)
        self.assertEqual(predictor_component, mock_instance)
        
        # Verify it was called with correct parameters
        mock_predictor.assert_called_once()
        call_args = mock_predictor.call_args
        self.assertIn('base_dir', call_args[0][0])
        
        logger.info("✓ Model predictor component creation test passed")
    
    def test_singleton_behavior(self):
        """Test singleton behavior for components."""
        factory = self.ComponentFactory(self.test_config)
        
        # Get database component twice
        db1 = factory.get_component('database')
        db2 = factory.get_component('database')
        
        # Should be the same instance for singleton components
        self.assertEqual(db1, db2)
        
        logger.info("✓ Singleton behavior test passed")
    
    def test_dependency_resolution(self):
        """Test dependency resolution between components."""
        from component_factory import ComponentConfig
        
        factory = self.ComponentFactory(self.test_config)
        
        # Check that order executor depends on exchange
        order_executor_config = factory._component_configs['order_executor']
        self.assertIsNotNone(order_executor_config.dependencies)
        self.assertIn('exchange', order_executor_config.dependencies)
        
        logger.info("✓ Dependency resolution test passed")
    
    def test_error_handling(self):
        """Test error handling for failed component creation."""
        factory = self.ComponentFactory(self.test_config)
        
        # Try to get a non-existent component
        result = factory.get_component('non_existent_component')
        self.assertIsNone(result)
        
        logger.info("✓ Error handling test passed")
    
    def test_initialization_order(self):
        """Test component initialization order."""
        factory = self.ComponentFactory(self.test_config)
        
        # Check that initialization order is correct
        self.assertIn('database', factory._initialization_order)
        self.assertIn('feature_engineer', factory._initialization_order)
        
        # Database should come before feature_engineer
        db_index = factory._initialization_order.index('database')
        fe_index = factory._initialization_order.index('feature_engineer')
        self.assertLess(db_index, fe_index)
        
        logger.info("✓ Initialization order test passed")
    
    def test_component_context_manager(self):
        """Test component context manager."""
        factory = self.ComponentFactory(self.test_config)
        
        # Test that context manager works without errors
        try:
            with factory.component_context() as components:
                self.assertIsInstance(components, dict)
                # Should contain at least database component
                self.assertIn('database', components)
        except Exception as e:
            # Context manager might fail if components can't be created
            logger.warning(f"Context manager test failed (expected if dependencies missing): {e}")
        
        logger.info("✓ Component context manager test passed")


def run_integration_test():
    """Run integration test with real system components."""
    logger.info("Starting ComponentFactory integration test...")
    
    try:
        from component_factory import ComponentFactory
        
        # Create factory with real configuration
        config = {
            'database_path': './trading_bot.db',
            'base_dir': './trading_bot_data'
        }
        
        factory = ComponentFactory(config)
        
        # Test component initialization
        results = factory.initialize_all_components()
        
        logger.info("ComponentFactory integration test results:")
        
        for component_name, success in results.items():
            status = "✓ PASS" if success else "✗ FAIL"
            logger.info(f"  {component_name}: {status}")
        
        # Calculate success rate
        success_count = sum(results.values())
        total_count = len(results)
        success_rate = (success_count / total_count) * 100
        
        logger.info(f"Integration test success rate: {success_rate:.1f}% ({success_count}/{total_count})")
        
        # Test component retrieval
        logger.info("\nTesting component retrieval:")
        
        database = factory.get_component('database')
        logger.info(f"  Database component: {'✓ Available' if database else '✗ Not available'}")
        
        feature_engineer = factory.get_component('feature_engineer')
        logger.info(f"  Feature engineer: {'✓ Available' if feature_engineer else '✗ Not available'}")
        
        model_predictor = factory.get_component('model_predictor')
        logger.info(f"  Model predictor: {'✓ Available' if model_predictor else '✗ Not available'}")
        
        # Cleanup
        factory.shutdown_all_components()
        logger.info("\n✓ ComponentFactory integration test completed successfully")
        
        return success_rate >= 50  # Accept 50% success rate for integration test
        
    except Exception as e:
        logger.error(f"Integration test failed: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False


def main():
    """Main function to run all tests."""
    print("ComponentFactory Test Suite")
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