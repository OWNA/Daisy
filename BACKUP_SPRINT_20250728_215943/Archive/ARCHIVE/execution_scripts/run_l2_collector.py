#!/usr/bin/env python3
"""
Run L2 Data Collector
Collects Level 2 order book data via WebSocket for later use in training/backtesting
"""

import sys
import yaml
import time
import signal
from datetime import datetime
from l2_data_collector import L2DataCollector


class L2CollectorRunner:
    """
    Manages the L2 data collection process
    """
    
    def __init__(self, config_path='config.yaml'):
        """Initialize the L2 collector runner"""
        self.config_path = config_path
        self.collector = None
        self.is_running = False
        
        # Set up signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        print(f"\n\nReceived signal {signum}. Shutting down gracefully...")
        self.stop_collection()
        sys.exit(0)
    
    def load_config(self):
        """Load configuration from file"""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config
        except Exception as e:
            print(f"Error loading config from {self.config_path}: {e}")
            return None
    
    def start_collection(self):
        """Start L2 data collection"""
        print("="*60)
        print("L2 DATA COLLECTOR")
        print("="*60)
        print(f"Started at: {datetime.now()}")
        
        # Load config
        config = self.load_config()
        if not config:
            print("Failed to load configuration. Exiting.")
            return
        
        # Check if L2 collection is enabled
        if not config.get('use_l2_features', False):
            print("\nWARNING: L2 features are disabled in config!")
            print("Set 'use_l2_features: true' to enable L2 data collection")
            response = input("\nDo you want to continue anyway? (y/n): ")
            if response.lower() != 'y':
                print("Exiting.")
                return
        
        # Display configuration
        print("\nConfiguration:")
        print(f"   Symbol: {config.get('symbol', 'BTC/USDT:USDT')}")
        print(f"   Exchange: {config.get('exchange_name', 'bybit')}")
        print(f"   Market Type: {config.get('market_type', 'linear')}")
        print(f"   WebSocket Depth: {config.get('l2_websocket_depth', 50)}")
        print(f"   Collection Duration: {config.get('l2_collection_duration_seconds', 120)}s")
        print(f"   Max File Size: {config.get('l2_max_file_size_mb', 20)}MB")
        print(f"   Data Directory: {config.get('l2_data_folder', 'l2_data')}")
        
        # Initialize collector
        try:
            self.collector = L2DataCollector(config, './')
            self.is_running = True
        except Exception as e:
            print(f"\nError initializing L2 collector: {e}")
            return
        
        # Start collection
        print("\nStarting L2 data collection...")
        print("Press Ctrl+C to stop collection gracefully\n")
        
        try:
            self.collector.start_collection_websocket()
        except Exception as e:
            print(f"\nError during collection: {e}")
        finally:
            self.is_running = False
            print("\nL2 data collection completed")
    
    def stop_collection(self):
        """Stop L2 data collection"""
        if self.collector and self.is_running:
            print("\nStopping L2 data collection...")
            try:
                self.collector.stop_collection_websocket()
            except Exception as e:
                print(f"Error stopping collector: {e}")
            self.is_running = False
    
    def run_continuous(self, interval_seconds=3600):
        """
        Run L2 collection continuously with periodic restarts
        
        Args:
            interval_seconds: How often to restart collection (default: 1 hour)
        """
        print(f"Running continuous L2 collection (restart every {interval_seconds}s)")
        
        while True:
            try:
                # Start collection
                self.start_collection()
                
                # Wait for interval or until stopped
                time.sleep(interval_seconds)
                
                # Stop current collection
                self.stop_collection()
                
                # Brief pause before restart
                print("\nRestarting collection in 10 seconds...")
                time.sleep(10)
                
            except KeyboardInterrupt:
                print("\nContinuous collection interrupted by user")
                break
            except Exception as e:
                print(f"\nError in continuous collection: {e}")
                print("Retrying in 30 seconds...")
                time.sleep(30)


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run L2 Data Collector')
    parser.add_argument(
        '--config', 
        type=str, 
        default='config.yaml',
        help='Path to configuration file (default: config.yaml)'
    )
    parser.add_argument(
        '--continuous',
        action='store_true',
        help='Run continuously with periodic restarts'
    )
    parser.add_argument(
        '--interval',
        type=int,
        default=3600,
        help='Restart interval in seconds for continuous mode (default: 3600)'
    )
    
    args = parser.parse_args()
    
    # Create runner
    runner = L2CollectorRunner(args.config)
    
    # Run collection
    if args.continuous:
        runner.run_continuous(args.interval)
    else:
        runner.start_collection()


if __name__ == "__main__":
    main()