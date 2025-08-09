#!/usr/bin/env python3
"""
WebSocket Live Data Collection - High Performance Real-Time Data

Uses the existing data_ingestor.py WebSocket infrastructure for sub-second data collection.
This approach can handle much higher frequencies than REST API polling.
"""
import asyncio
import sqlite3
import time
import threading
import signal
import argparse
from datetime import datetime
from typing import Dict, Any, Optional
import logging

# Import the existing high-performance WebSocket data ingestor
from data_ingestor import DataIngestor, DataIngestorConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class HighFrequencyDataCollector:
    """
    High-frequency data collector using WebSocket streams.
    Can achieve sub-second data collection rates.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.running = False
        self.stop_event = threading.Event()
        self.data_ingestor: Optional[DataIngestor] = None
        self.stats = {
            'start_time': datetime.now(),
            'total_updates': 0,
            'successful_writes': 0,
            'failed_writes': 0,
            'updates_per_second': 0,
            'last_update_time': None
        }
        
        # Create data ingestor configuration
        ingestor_config_dict = {
            'exchange': 'bybit',
            'symbol': 'BTC/USDT:USDT',
            'sandbox': False,  # Use mainnet for live data
            'db_path': './trading_bot_live.db',
            'table_name': 'l2_training_data_practical',
            'buffer_size': config.get('buffer_size', 200),  # Larger buffer for high frequency
            'write_interval': config.get('write_interval', 0.5),  # Write every 500ms
            'orderbook_depth': 10,
            'max_reconnect_attempts': 10,
            'reconnect_delay': 2.0,
            'log_updates': False  # Disable verbose logging for performance
        }
        
        self.ingestor_config = DataIngestorConfig(ingestor_config_dict)
        
        # Statistics tracking
        self.last_stats_time = time.time()
        self.updates_since_last_stats = 0
        
    def setup_callbacks(self):
        """Setup callbacks for the data ingestor."""
        def on_update(l2_update):
            """Called on each L2 update."""
            self.stats['total_updates'] += 1
            self.stats['last_update_time'] = datetime.now()
            self.updates_since_last_stats += 1
            
            # Update stats every 10 seconds
            now = time.time()
            if now - self.last_stats_time >= 10:
                duration = now - self.last_stats_time
                self.stats['updates_per_second'] = self.updates_since_last_stats / duration
                
                if not self.config.get('silent', False):
                    logger.info(f"Updates/sec: {self.stats['updates_per_second']:.1f}, "
                              f"Total: {self.stats['total_updates']}, "
                              f"Buffer: {len(self.data_ingestor.normal_priority_buffer) if self.data_ingestor else 0}")
                
                self.last_stats_time = now
                self.updates_since_last_stats = 0
        
        def on_error(error):
            """Called on errors."""
            self.stats['failed_writes'] += 1
            if not self.config.get('silent', False):
                logger.error(f"Data ingestor error: {error}")
        
        return on_update, on_error
    
    def start(self) -> bool:
        """Start high-frequency data collection."""
        try:
            logger.info("Starting high-frequency WebSocket data collection...")
            logger.info(f"Target frequency: Real-time WebSocket updates")
            logger.info(f"Write interval: {self.ingestor_config.write_interval}s")
            logger.info(f"Buffer size: {self.ingestor_config.buffer_size}")
            
            # Create and configure data ingestor
            self.data_ingestor = DataIngestor(self.ingestor_config)
            
            # Setup callbacks
            on_update, on_error = self.setup_callbacks()
            self.data_ingestor.set_update_callback(on_update)
            self.data_ingestor.set_error_callback(on_error)
            
            # Start the ingestor
            if not self.data_ingestor.start():
                logger.error("Failed to start data ingestor")
                return False
            
            self.running = True
            logger.info("✓ High-frequency data collection started")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start data collection: {e}")
            return False
    
    def run(self, duration_seconds: int):
        """Run data collection for specified duration."""
        if not self.running:
            if not self.start():
                return False
        
        logger.info(f"Running for {duration_seconds} seconds...")
        
        try:
            # Wait for specified duration
            self.stop_event.wait(duration_seconds)
            
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        
        finally:
            self.stop()
        
        return True
    
    def stop(self):
        """Stop data collection."""
        logger.info("Stopping high-frequency data collection...")
        self.running = False
        self.stop_event.set()
        
        if self.data_ingestor:
            self.data_ingestor.stop()
        
        # Print final statistics
        self.print_final_stats()
        logger.info("✓ Data collection stopped")
    
    def print_final_stats(self):
        """Print final collection statistics."""
        if not self.config.get('silent', False):
            duration = (datetime.now() - self.stats['start_time']).total_seconds()
            
            print("\n" + "="*50)
            print("HIGH-FREQUENCY DATA COLLECTION COMPLETE")
            print("="*50)
            print(f"Duration: {duration:.1f} seconds")
            print(f"Total updates: {self.stats['total_updates']}")
            print(f"Average rate: {self.stats['total_updates']/duration:.1f} updates/sec")
            print(f"Last update: {self.stats['last_update_time']}")
            
            if self.data_ingestor:
                ingestor_stats = self.data_ingestor.get_stats()
                print(f"Successful writes: {ingestor_stats.get('successful_writes', 0)}")
                print(f"Failed writes: {ingestor_stats.get('failed_writes', 0)}")
                print(f"Write rate: {ingestor_stats.get('successful_writes_per_second', 0):.1f} writes/sec")
            print("="*50)
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get current health status."""
        health = {
            'running': self.running,
            'total_updates': self.stats['total_updates'],
            'updates_per_second': self.stats['updates_per_second'],
            'last_update_time': self.stats['last_update_time']
        }
        
        if self.data_ingestor:
            ingestor_health = self.data_ingestor.get_execution_health_report()
            health.update({
                'data_quality_score': ingestor_health.get('overall_health_score', 0),
                'execution_ready': ingestor_health.get('execution_readiness', False),
                'buffer_utilization': ingestor_health.get('detailed_stats', {}).get('total_buffer_size', 0)
            })
        
        return health


def check_recent_data():
    """Check most recent data in database."""
    try:
        conn = sqlite3.connect('./trading_bot_live.db')
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT COUNT(*), data_source, MAX(timestamp) as latest
            FROM l2_training_data_practical 
            WHERE timestamp > datetime('now', '-5 minutes')
            GROUP BY data_source
            ORDER BY latest DESC
        """)
        
        print("\nRecent data (last 5 minutes):")
        for row in cursor.fetchall():
            print(f"  {row[1]}: {row[0]} rows, latest: {row[2]}")
        
        # Show latest few records
        cursor.execute("""
            SELECT timestamp, data_source, mid_price, spread 
            FROM l2_training_data_practical 
            ORDER BY timestamp DESC LIMIT 5
        """)
        
        print("\nLatest records:")
        for row in cursor.fetchall():
            print(f"  {row[0]} | {row[1]} | ${row[2]:.2f} | ${row[3]:.2f}")
        
        conn.close()
        
    except Exception as e:
        print(f"Error checking recent data: {e}")


def main():
    """Main function for high-frequency data collection."""
    parser = argparse.ArgumentParser(description='High-frequency WebSocket data collection')
    parser.add_argument('--duration', type=int, default=60, 
                       help='Collection duration in seconds (default: 60)')
    parser.add_argument('--buffer-size', type=int, default=200,
                       help='Buffer size for updates (default: 200)')
    parser.add_argument('--write-interval', type=float, default=0.5,
                       help='Database write interval in seconds (default: 0.5)')
    parser.add_argument('--silent', action='store_true',
                       help='Reduce output for high-frequency collection')
    parser.add_argument('--check-data', action='store_true',
                       help='Check recent data and exit')
    
    args = parser.parse_args()
    
    if args.check_data:
        check_recent_data()
        return
    
    # Configuration
    config = {
        'buffer_size': args.buffer_size,
        'write_interval': args.write_interval,
        'silent': args.silent
    }
    
    # Create collector
    collector = HighFrequencyDataCollector(config)
    
    # Setup signal handlers
    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}")
        collector.stop()
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Run collection
    print("HIGH-FREQUENCY WEBSOCKET DATA COLLECTION")
    print("=" * 50)
    print(f"Duration: {args.duration} seconds")
    print(f"Method: WebSocket real-time updates")
    print(f"Buffer size: {args.buffer_size}")
    print(f"Write interval: {args.write_interval}s")
    print("Press Ctrl+C to stop early")
    print()
    
    success = collector.run(args.duration)
    
    if success:
        print("\n✓ Collection completed successfully")
        check_recent_data()
    else:
        print("\n✗ Collection failed")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())