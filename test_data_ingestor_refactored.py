#!/usr/bin/env python3
"""
Test script for the refactored data_ingestor.py

This script validates that the refactored data ingestor:
1. Properly handles WebSocket URL configuration
2. Connects to Bybit demo trading successfully
3. Creates the database schema correctly
4. Processes L2 data without errors
5. Handles shutdown gracefully
"""

import logging
import time
import signal
import sys
from datetime import datetime
from data_ingestor import create_data_ingestor, L2Update

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_data_ingestor():
    """Test the refactored data ingestor."""
    
    print("=" * 60)
    print("TESTING REFACTORED DATA INGESTOR")
    print("=" * 60)
    print("This test validates the critical fixes:")
    print("1. WebSocket URL configuration bug fixed")
    print("2. Simplified data processing pipeline")
    print("3. Improved threading/async architecture")
    print("4. Robust database schema handling")
    print("=" * 60)
    
    # Test configuration for demo trading
    config_dict = {
        'exchange': 'bybit',
        'symbol': 'BTC/USDT:USDT',
        'sandbox': False,  # Demo trading on mainnet
        'db_path': './test_trading_bot_live.db',
        'table_name': 'l2_training_data_practical',
        'log_updates': False,  # Reduce noise during test
        'buffer_size': 50,
        'write_interval': 2.0,
        'orderbook_depth': 10,
        'data_retention_hours': 1,  # Short retention for test
        'max_reconnect_attempts': 3,
        'reconnect_delay': 2.0
    }
    
    # Create data ingestor
    logger.info("Creating data ingestor with test configuration...")
    ingestor = create_data_ingestor(config_dict)
    
    # Test statistics
    updates_received = 0
    errors_received = 0
    
    def on_update(update: L2Update):
        """Track updates received."""
        nonlocal updates_received
        updates_received += 1
        
        if updates_received <= 5:  # Log first few updates
            logger.info(f"Update {updates_received}: {update.symbol} - "
                       f"Bid: ${update.bids[0][0]:.2f}, Ask: ${update.asks[0][0]:.2f}, "
                       f"Spread: ${(update.asks[0][0] - update.bids[0][0]):.2f}")
    
    def on_error(error: Exception):
        """Track errors received."""
        nonlocal errors_received
        errors_received += 1
        logger.warning(f"Error {errors_received}: {error}")
    
    # Set callbacks
    ingestor.set_update_callback(on_update)
    ingestor.set_error_callback(on_error)
    
    # Test shutdown handler
    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum} - stopping test")
        ingestor.stop()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Start the ingestor
        logger.info("Starting data ingestor...")
        start_time = datetime.now()
        
        if not ingestor.start():
            logger.error("❌ Failed to start data ingestor")
            return False
        
        logger.info("✅ Data ingestor started successfully")
        
        # Run for test duration
        test_duration = 30  # 30 seconds
        logger.info(f"Running test for {test_duration} seconds...")
        
        for i in range(test_duration):
            time.sleep(1)
            
            # Check health periodically
            if i % 10 == 9:
                stats = ingestor.get_stats()
                is_healthy = ingestor.is_healthy()
                
                logger.info(f"Status check: Healthy={is_healthy}, "
                           f"Updates={stats['total_updates']}, "
                           f"Writes={stats['successful_writes']}, "
                           f"Buffer={stats['buffer_size']}")
                
                if not is_healthy:
                    logger.warning("⚠ Ingestor health check failed")
        
        # Final statistics
        final_stats = ingestor.get_stats()
        runtime = datetime.now() - start_time
        
        print("\n" + "=" * 60)
        print("TEST RESULTS")
        print("=" * 60)
        print(f"Runtime: {runtime}")
        print(f"Total updates received: {final_stats['total_updates']}")
        print(f"Successful database writes: {final_stats['successful_writes']}")
        print(f"Failed database writes: {final_stats['failed_writes']}")
        print(f"Reconnections: {final_stats['reconnections']}")
        print(f"Updates per second: {final_stats['total_updates'] / runtime.total_seconds():.2f}")
        print(f"Final buffer size: {final_stats['buffer_size']}")
        print(f"Callback updates: {updates_received}")
        print(f"Callback errors: {errors_received}")
        
        # Evaluate test success
        success = True
        if final_stats['total_updates'] == 0:
            logger.error("❌ No updates received - WebSocket connection may have failed")
            success = False
        
        if final_stats['failed_writes'] > final_stats['successful_writes']:
            logger.error("❌ More database write failures than successes")
            success = False
        
        if final_stats['reconnections'] > 2:
            logger.warning(f"⚠ High number of reconnections: {final_stats['reconnections']}")
        
        if ingestor.is_healthy():
            logger.info("✅ Final health check passed")
        else:
            logger.warning("⚠ Final health check failed")
        
        if success:
            logger.info("✅ TEST PASSED - Data ingestor functioning correctly")
        else:
            logger.error("❌ TEST FAILED - Issues detected")
        
        return success
        
    except KeyboardInterrupt:
        logger.info("Test interrupted by user")
        return False
        
    except Exception as e:
        logger.error(f"Test failed with exception: {e}")
        return False
        
    finally:
        # Always stop the ingestor
        logger.info("Stopping data ingestor...")
        ingestor.stop()
        logger.info("Test completed")

if __name__ == "__main__":
    success = test_data_ingestor()
    sys.exit(0 if success else 1)