#!/usr/bin/env python3
"""
test_live_data_ingestion.py - Test Live Data Ingestion

Quick test script to demonstrate the DataIngestor working with live Bybit data.

Sprint 2 - Priority 2: Implement Live WebSocket Data Ingestion
"""

import os
import sys
import time
import logging
import signal
from datetime import datetime
from data_ingestor import create_data_ingestor, L2Update

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_live_ingestion(duration_seconds: int = 30):
    """Test live data ingestion for a specified duration."""
    
    print("🚀 BTC Live Data Ingestion Test")
    print("=" * 50)
    print(f"Duration: {duration_seconds} seconds")
    print(f"Exchange: Bybit (Demo Trading)")
    print(f"Symbol: BTC/USDT:USDT")
    print("=" * 50)
    
    # Configuration for testing
    config_dict = {
        'exchange': 'bybit',
        'symbol': 'BTC/USDT:USDT',
        'sandbox': False,  # Use Demo Trading (mainnet)
        'db_path': './trading_bot_live.db',
        'table_name': 'l2_training_data_practical',
        'log_updates': False,  # Don't log every update (too verbose)
        'buffer_size': 20,  # Write to DB every 20 updates
        'write_interval': 2.0,  # Write every 2 seconds
        'orderbook_depth': 5,  # Only use top 5 levels for testing
        'data_retention_hours': 1  # Keep only 1 hour of test data
    }
    
    # Statistics tracking
    stats = {
        'updates_received': 0,
        'last_price': None,
        'price_changes': 0,
        'start_time': datetime.now()
    }
    
    def on_update(update: L2Update):
        """Callback for each L2 update."""
        stats['updates_received'] += 1
        
        # Track price changes
        if update.bids and update.asks:
            bid_price = update.bids[0][0]
            ask_price = update.asks[0][0]
            mid_price = (bid_price + ask_price) / 2
            
            if stats['last_price'] and abs(mid_price - stats['last_price']) > 0.01:
                stats['price_changes'] += 1
            
            stats['last_price'] = mid_price
            
            # Print periodic updates
            if stats['updates_received'] % 50 == 0:
                spread = ask_price - bid_price
                print(f"📊 Update #{stats['updates_received']}: "
                      f"Mid={mid_price:.2f}, Spread={spread:.2f}, "
                      f"Time={update.timestamp.strftime('%H:%M:%S')}")
    
    def on_error(error: Exception):
        """Callback for errors."""
        print(f"❌ Error: {error}")
    
    # Create and configure ingestor
    ingestor = create_data_ingestor(config_dict)
    ingestor.set_update_callback(on_update)
    ingestor.set_error_callback(on_error)
    
    # Handle shutdown
    shutdown_requested = False
    
    def signal_handler(signum, frame):
        nonlocal shutdown_requested
        print(f"\\n⏹ Shutdown requested...")
        shutdown_requested = True
    
    signal.signal(signal.SIGINT, signal_handler)
    
    try:
        # Start ingestion
        print("🔌 Starting live data ingestion...")
        if not ingestor.start():
            print("❌ Failed to start data ingestion")
            return False
        
        print(f"✅ Data ingestion started. Running for {duration_seconds} seconds...")
        
        # Run for specified duration
        start_time = time.time()
        while (time.time() - start_time) < duration_seconds and not shutdown_requested:
            time.sleep(1)
            
            # Check health
            if not ingestor.is_healthy():
                print("⚠ Data ingestor is not healthy")
                break
        
        # Print final statistics
        runtime = time.time() - start_time
        ingestion_stats = ingestor.get_stats()
        
        print("\\n" + "=" * 50)
        print("📈 LIVE DATA INGESTION TEST RESULTS")
        print("=" * 50)
        print(f"Runtime: {runtime:.1f} seconds")
        print(f"Updates received: {stats['updates_received']}")
        print(f"Price changes: {stats['price_changes']}")
        print(f"Final price: ${stats['last_price']:.2f}" if stats['last_price'] else "No price data")
        print(f"Update rate: {stats['updates_received']/runtime:.1f} updates/sec")
        
        print("\\nIngestor Statistics:")
        print(f"  Total updates: {ingestion_stats['total_updates']}")
        print(f"  Successful writes: {ingestion_stats['successful_writes']}")
        print(f"  Failed writes: {ingestion_stats['failed_writes']}")
        print(f"  Reconnections: {ingestion_stats['reconnections']}")
        print(f"  Buffer size: {ingestion_stats['buffer_size']}")
        print(f"  Is healthy: {ingestor.is_healthy()}")
        
        # Success criteria
        success = (
            stats['updates_received'] > 0 and
            ingestion_stats['successful_writes'] > 0 and
            ingestion_stats['failed_writes'] == 0 and
            ingestor.is_healthy()
        )
        
        if success:
            print("\\n🎉 LIVE DATA INGESTION TEST PASSED!")
            print("✅ WebSocket connection established")
            print("✅ L2 updates received and processed")
            print("✅ Data successfully written to database")
            print("✅ System remained healthy throughout test")
        else:
            print("\\n❌ LIVE DATA INGESTION TEST FAILED!")
            if stats['updates_received'] == 0:
                print("  - No updates received")
            if ingestion_stats['successful_writes'] == 0:
                print("  - No successful database writes")
            if ingestion_stats['failed_writes'] > 0:
                print("  - Database write failures occurred")
            if not ingestor.is_healthy():
                print("  - System became unhealthy")
        
        return success
        
    except Exception as e:
        print(f"❌ Test failed with exception: {e}")
        return False
        
    finally:
        print("\\n🔌 Stopping data ingestion...")
        ingestor.stop()
        print("✅ Data ingestion stopped")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test live data ingestion")
    parser.add_argument('--duration', type=int, default=30, 
                       help="Test duration in seconds (default: 30)")
    args = parser.parse_args()
    
    success = test_live_ingestion(args.duration)
    sys.exit(0 if success else 1)