#!/usr/bin/env python3
"""
Direct WebSocket Data Collection - Simplified High-Performance Approach

Uses direct WebSocket connection to Bybit for maximum reliability and speed.
"""
import json
import sqlite3
import time
import threading
import websocket
import argparse
from datetime import datetime
from typing import Dict, Any, Optional, List
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DirectWebSocketCollector:
    """Direct WebSocket data collector for Bybit BTC/USDT orderbook."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.running = False
        self.ws: Optional[websocket.WebSocketApp] = None
        self.ws_thread: Optional[threading.Thread] = None
        
        # Statistics
        self.stats = {
            'start_time': datetime.now(),
            'total_updates': 0,
            'successful_writes': 0,
            'failed_writes': 0,
            'last_update_time': None,
            'connection_attempts': 0,
            'reconnections': 0
        }
        
        # Buffer for batch writes
        self.update_buffer: List[Dict[str, Any]] = []
        self.buffer_lock = threading.Lock()
        self.max_buffer_size = config.get('buffer_size', 50)
        
        # Writer thread
        self.writer_thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()
        
        # Database setup
        self.setup_database()
        
    def setup_database(self):
        """Ensure database table exists."""
        try:
            conn = sqlite3.connect('./trading_bot_live.db')
            cursor = conn.cursor()
            
            # Check if table exists
            cursor.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name='l2_training_data_practical'
            """)
            
            if not cursor.fetchone():
                logger.info("Creating l2_training_data_practical table...")
                cursor.execute("""
                    CREATE TABLE l2_training_data_practical (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        symbol TEXT NOT NULL,
                        exchange TEXT DEFAULT 'bybit',
                        data_source TEXT DEFAULT 'websocket_live',
                        mid_price REAL,
                        spread REAL,
                        bid_price_1 REAL, bid_size_1 REAL,
                        ask_price_1 REAL, ask_size_1 REAL,
                        bid_price_2 REAL, bid_size_2 REAL,
                        ask_price_2 REAL, ask_size_2 REAL,
                        bid_price_3 REAL, bid_size_3 REAL,
                        ask_price_3 REAL, ask_size_3 REAL,
                        bid_price_4 REAL, bid_size_4 REAL,
                        ask_price_4 REAL, ask_size_4 REAL,
                        bid_price_5 REAL, bid_size_5 REAL,
                        ask_price_5 REAL, ask_size_5 REAL,
                        microprice REAL,
                        weighted_bid_price REAL,
                        weighted_ask_price REAL,
                        order_book_imbalance REAL,
                        data_quality_score REAL DEFAULT 1.0,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Create index
                cursor.execute("""
                    CREATE INDEX idx_timestamp_symbol 
                    ON l2_training_data_practical(timestamp, symbol)
                """)
                
                conn.commit()
                logger.info("✓ Database table created")
            
            conn.close()
            
        except Exception as e:
            logger.error(f"Database setup error: {e}")
            raise
    
    def on_message(self, ws, message):
        """Handle incoming WebSocket messages."""
        try:
            data = json.loads(message)
            
            # Check if this is orderbook data
            if 'topic' in data and 'orderbook' in data.get('topic', ''):
                if 'data' in data and data['data']:
                    self.process_orderbook_update(data['data'], data.get('ts'))
            elif 'success' in data and data.get('success'):
                logger.info(f"Subscription successful: {data}")
            elif 'op' in data:
                logger.debug(f"Operation response: {data}")
                
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse JSON: {message[:100]}...")
        except Exception as e:
            logger.error(f"Message processing error: {e}")
    
    def process_orderbook_update(self, orderbook_data, exchange_timestamp=None):
        """Process orderbook update and add to buffer."""
        try:
            bids = orderbook_data.get('b', [])
            asks = orderbook_data.get('a', [])
            
            if not bids or not asks:
                return
            
            # Convert to floats and ensure we have at least 5 levels
            bid_levels = []
            ask_levels = []
            
            for i, (price, size) in enumerate(bids[:5]):
                bid_levels.append([float(price), float(size)])
            
            for i, (price, size) in enumerate(asks[:5]):
                ask_levels.append([float(price), float(size)])
            
            # Pad to 5 levels if needed
            while len(bid_levels) < 5:
                if bid_levels:
                    last_price = bid_levels[-1][0] * 0.999  # Slightly lower
                    bid_levels.append([last_price, 0.0])
                else:
                    bid_levels.append([0.0, 0.0])
            
            while len(ask_levels) < 5:
                if ask_levels:
                    last_price = ask_levels[-1][0] * 1.001  # Slightly higher
                    ask_levels.append([last_price, 0.0])
                else:
                    ask_levels.append([0.0, 0.0])
            
            # Calculate features
            best_bid_price, best_bid_size = bid_levels[0]
            best_ask_price, best_ask_size = ask_levels[0]
            
            mid_price = (best_bid_price + best_ask_price) / 2
            spread = best_ask_price - best_bid_price
            
            # Microprice
            total_size = best_bid_size + best_ask_size
            if total_size > 0:
                microprice = (best_bid_price * best_ask_size + best_ask_price * best_bid_size) / total_size
            else:
                microprice = mid_price
            
            # Weighted prices (top 3 levels)
            bid_value = sum(price * size for price, size in bid_levels[:3])
            bid_volume = sum(size for price, size in bid_levels[:3])
            ask_value = sum(price * size for price, size in ask_levels[:3])
            ask_volume = sum(size for price, size in ask_levels[:3])
            
            weighted_bid_price = bid_value / bid_volume if bid_volume > 0 else best_bid_price
            weighted_ask_price = ask_value / ask_volume if ask_volume > 0 else best_ask_price
            
            # Order book imbalance
            total_volume = bid_volume + ask_volume
            order_book_imbalance = (bid_volume - ask_volume) / total_volume if total_volume > 0 else 0
            
            # Create update record
            update_record = {
                'timestamp': datetime.now().isoformat(),
                'symbol': 'BTCUSDT',
                'exchange': 'bybit',
                'data_source': 'websocket_live',
                'mid_price': mid_price,
                'spread': spread,
                'microprice': microprice,
                'weighted_bid_price': weighted_bid_price,
                'weighted_ask_price': weighted_ask_price,
                'order_book_imbalance': order_book_imbalance,
                'data_quality_score': 1.0
            }
            
            # Add price/size levels
            for i in range(5):
                update_record[f'bid_price_{i+1}'] = bid_levels[i][0]
                update_record[f'bid_size_{i+1}'] = bid_levels[i][1]
                update_record[f'ask_price_{i+1}'] = ask_levels[i][0]
                update_record[f'ask_size_{i+1}'] = ask_levels[i][1]
            
            # Add to buffer
            with self.buffer_lock:
                self.update_buffer.append(update_record)
                
                # Trigger immediate write if buffer is full
                if len(self.update_buffer) >= self.max_buffer_size:
                    self.flush_buffer()
            
            # Update stats
            self.stats['total_updates'] += 1
            self.stats['last_update_time'] = datetime.now()
            
            if not self.config.get('silent', False):
                if self.stats['total_updates'] % 10 == 0:  # Log every 10 updates
                    logger.info(f"Updates: {self.stats['total_updates']}, "
                              f"BTC: ${mid_price:.2f}, "
                              f"Spread: ${spread:.2f}, "
                              f"Buffer: {len(self.update_buffer)}")
            
        except Exception as e:
            logger.error(f"Error processing orderbook update: {e}")
    
    def flush_buffer(self):
        """Flush buffered updates to database."""
        if not self.update_buffer:
            return
        
        try:
            updates_to_write = self.update_buffer.copy()
            self.update_buffer.clear()
            
            conn = sqlite3.connect('./trading_bot_live.db')
            cursor = conn.cursor()
            
            # Prepare insert statement
            columns = list(updates_to_write[0].keys())
            placeholders = ', '.join(['?' for _ in columns])
            column_names = ', '.join(columns)
            
            insert_sql = f"""
                INSERT INTO l2_training_data_practical ({column_names})
                VALUES ({placeholders})
            """
            
            # Execute batch insert
            for update in updates_to_write:
                values = [update[col] for col in columns]
                cursor.execute(insert_sql, values)
            
            conn.commit()
            conn.close()
            
            self.stats['successful_writes'] += len(updates_to_write)
            
            if not self.config.get('silent', False):
                logger.debug(f"Wrote {len(updates_to_write)} updates to database")
            
        except Exception as e:
            logger.error(f"Database write error: {e}")
            self.stats['failed_writes'] += len(updates_to_write) if 'updates_to_write' in locals() else 0
    
    def writer_loop(self):
        """Background writer thread loop."""
        logger.info("Writer thread started")
        
        while not self.stop_event.wait(self.config.get('write_interval', 1.0)):
            try:
                with self.buffer_lock:
                    if self.update_buffer:
                        self.flush_buffer()
            except Exception as e:
                logger.error(f"Writer loop error: {e}")
        
        # Final flush
        try:
            with self.buffer_lock:
                if self.update_buffer:
                    logger.info(f"Final flush: {len(self.update_buffer)} updates")
                    self.flush_buffer()
        except Exception as e:
            logger.error(f"Final flush error: {e}")
        
        logger.info("Writer thread stopped")
    
    def on_error(self, ws, error):
        """Handle WebSocket errors."""
        logger.error(f"WebSocket error: {error}")
        self.stats['reconnections'] += 1
    
    def on_close(self, ws, close_status_code, close_msg):
        """Handle WebSocket close."""
        logger.info(f"WebSocket closed: {close_status_code} - {close_msg}")
    
    def on_open(self, ws):
        """Handle WebSocket open and subscribe to orderbook."""
        logger.info("WebSocket connected - subscribing to orderbook")
        
        # Subscribe to BTCUSDT orderbook with 50 levels depth
        subscribe_msg = {
            "op": "subscribe",
            "args": ["orderbook.50.BTCUSDT"]
        }
        
        ws.send(json.dumps(subscribe_msg))
        logger.info("Subscription sent for orderbook.50.BTCUSDT")
    
    def start(self) -> bool:
        """Start WebSocket data collection."""
        try:
            logger.info("Starting direct WebSocket data collection...")
            
            # Start writer thread
            self.writer_thread = threading.Thread(target=self.writer_loop, daemon=True)
            self.writer_thread.start()
            
            # Create WebSocket connection
            websocket.enableTrace(False)  # Disable verbose logging
            
            ws_url = "wss://stream.bybit.com/v5/public/linear"
            logger.info(f"Connecting to: {ws_url}")
            
            self.ws = websocket.WebSocketApp(
                ws_url,
                on_open=self.on_open,
                on_message=self.on_message,
                on_error=self.on_error,
                on_close=self.on_close
            )
            
            # Start WebSocket in separate thread
            self.ws_thread = threading.Thread(
                target=lambda: self.ws.run_forever(
                    ping_interval=20,
                    ping_timeout=10
                ),
                daemon=True
            )
            
            self.running = True
            self.ws_thread.start()
            
            # Wait a moment for connection
            time.sleep(2)
            
            logger.info("✓ WebSocket data collection started")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start WebSocket collection: {e}")
            return False
    
    def run(self, duration_seconds: int):
        """Run data collection for specified duration."""
        if not self.running:
            if not self.start():
                return False
        
        logger.info(f"Running for {duration_seconds} seconds...")
        
        try:
            # Monitor collection
            start_time = time.time()
            last_stats_time = start_time
            
            while (time.time() - start_time) < duration_seconds:
                time.sleep(5)  # Check every 5 seconds
                
                # Print periodic stats
                now = time.time()
                if now - last_stats_time >= 30:  # Every 30 seconds
                    elapsed = now - start_time
                    rate = self.stats['total_updates'] / elapsed if elapsed > 0 else 0
                    
                    if not self.config.get('silent', False):
                        logger.info(f"Running: {elapsed:.0f}s, "
                                  f"Updates: {self.stats['total_updates']}, "
                                  f"Rate: {rate:.1f}/sec, "
                                  f"Buffer: {len(self.update_buffer)}")
                    
                    last_stats_time = now
        
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        
        finally:
            self.stop()
        
        return True
    
    def stop(self):
        """Stop data collection."""
        logger.info("Stopping WebSocket data collection...")
        
        self.running = False
        self.stop_event.set()
        
        # Close WebSocket
        if self.ws:
            self.ws.close()
        
        # Wait for threads
        if self.ws_thread and self.ws_thread.is_alive():
            self.ws_thread.join(timeout=5)
        
        if self.writer_thread and self.writer_thread.is_alive():
            self.writer_thread.join(timeout=5)
        
        # Print final stats
        self.print_final_stats()
        logger.info("✓ WebSocket data collection stopped")
    
    def print_final_stats(self):
        """Print final statistics."""
        duration = (datetime.now() - self.stats['start_time']).total_seconds()
        rate = self.stats['total_updates'] / duration if duration > 0 else 0
        
        print("\n" + "="*60)
        print("DIRECT WEBSOCKET DATA COLLECTION COMPLETE")
        print("="*60)
        print(f"Duration: {duration:.1f} seconds")
        print(f"Total updates: {self.stats['total_updates']}")
        print(f"Average rate: {rate:.1f} updates/sec")
        print(f"Successful writes: {self.stats['successful_writes']}")
        print(f"Failed writes: {self.stats['failed_writes']}")
        print(f"Last update: {self.stats['last_update_time']}")
        print("="*60)


def check_recent_data():
    """Check most recent data in database."""
    try:
        conn = sqlite3.connect('./trading_bot_live.db')
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT COUNT(*), data_source, MAX(timestamp) as latest
            FROM l2_training_data_practical 
            WHERE timestamp > datetime('now', '-10 minutes')
            GROUP BY data_source
            ORDER BY latest DESC
        """)
        
        print("\nRecent data (last 10 minutes):")
        for row in cursor.fetchall():
            print(f"  {row[1]}: {row[0]} rows, latest: {row[2]}")
        
        # Show latest records
        cursor.execute("""
            SELECT timestamp, data_source, mid_price, spread 
            FROM l2_training_data_practical 
            ORDER BY timestamp DESC LIMIT 8
        """)
        
        print("\nLatest records:")
        for row in cursor.fetchall():
            print(f"  {row[0][-8:]} | {row[1]:<15} | ${row[2]:>8.2f} | ${row[3]:>5.2f}")
        
        conn.close()
        
    except Exception as e:
        print(f"Error checking recent data: {e}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Direct WebSocket data collection')
    parser.add_argument('--duration', type=int, default=60,
                       help='Collection duration in seconds (default: 60)')
    parser.add_argument('--buffer-size', type=int, default=25,
                       help='Buffer size for batching (default: 25)')
    parser.add_argument('--write-interval', type=float, default=2.0,
                       help='Write interval in seconds (default: 2.0)')
    parser.add_argument('--silent', action='store_true',
                       help='Reduce output')
    parser.add_argument('--check-data', action='store_true',
                       help='Check recent data and exit')
    
    args = parser.parse_args()
    
    if args.check_data:
        check_recent_data()
        return 0
    
    config = {
        'buffer_size': args.buffer_size,
        'write_interval': args.write_interval,
        'silent': args.silent
    }
    
    print("DIRECT WEBSOCKET DATA COLLECTION")
    print("="*50)
    print(f"Duration: {args.duration} seconds")
    print(f"Buffer size: {args.buffer_size}")
    print(f"Write interval: {args.write_interval}s")
    print("Press Ctrl+C to stop early\n")
    
    collector = DirectWebSocketCollector(config)
    
    try:
        success = collector.run(args.duration)
        if success:
            print("\n✓ Collection completed successfully")
            check_recent_data()
        else:
            print("\n✗ Collection failed")
            return 1
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        collector.stop()
    
    return 0


if __name__ == "__main__":
    exit(main())