#!/usr/bin/env python3
"""
Fast live data collection - 1-5 second intervals
"""
import ccxt
import sqlite3
import time
import json
from datetime import datetime
import numpy as np
import argparse

def get_btc_orderbook_data():
    """Get live BTC/USDT orderbook data from Bybit"""
    try:
        # Initialize exchange with minimal config
        exchange = ccxt.bybit({
            'sandbox': False,  # Use mainnet for real data
            'options': {
                'defaultType': 'linear'  # Perpetual futures
            }
        })
        
        # Get orderbook for BTC perpetual
        symbol = 'BTC/USDT:USDT'
        orderbook = exchange.fetch_order_book(symbol, limit=20)
        ticker = exchange.fetch_ticker(symbol)
        
        # Calculate basic features
        bids = orderbook['bids'][:10]  # Top 10 bids
        asks = orderbook['asks'][:10]  # Top 10 asks
        
        # Ensure we have at least 10 levels by padding with reasonable defaults
        while len(bids) < 10:
            last_bid = bids[-1] if bids else [0, 0]
            bids.append([last_bid[0] * 0.999, 0])  # Slightly lower price, 0 volume
        
        while len(asks) < 10:
            last_ask = asks[-1] if asks else [999999, 0]
            asks.append([last_ask[0] * 1.001, 0])  # Slightly higher price, 0 volume
        
        if not bids or not asks:
            print("No orderbook data available")
            return None
            
        # Basic calculations
        mid_price = (bids[0][0] + asks[0][0]) / 2
        spread = asks[0][0] - bids[0][0]
        
        # Basic microstructure features
        total_bid_volume = sum([bid[1] for bid in bids])
        total_ask_volume = sum([ask[1] for ask in asks])
        
        order_imbalance = (total_bid_volume - total_ask_volume) / (total_bid_volume + total_ask_volume)
        
        # Create data row
        timestamp = datetime.now().isoformat()
        
        # Create complete L2 data structure
        data = {
            'timestamp': timestamp,
            'symbol': 'BTCUSDT',
            'exchange': 'bybit',
            'data_source': 'live_trading',
            'mid_price': mid_price,
            'spread': spread,
        }
        
        # Add all 10 levels of bids and asks
        for i in range(10):
            data[f'bid_price_{i+1}'] = bids[i][0]
            data[f'bid_size_{i+1}'] = bids[i][1]
            data[f'ask_price_{i+1}'] = asks[i][0]
            data[f'ask_size_{i+1}'] = asks[i][1]
        
        # Calculate additional required fields
        data['microprice'] = mid_price  # Simplified microprice
        data['weighted_bid_price'] = sum(bids[i][0] * bids[i][1] for i in range(5)) / max(sum(bids[i][1] for i in range(5)), 1)
        data['weighted_ask_price'] = sum(asks[i][0] * asks[i][1] for i in range(5)) / max(sum(asks[i][1] for i in range(5)), 1)
        data['order_book_imbalance'] = order_imbalance
        data['data_quality_score'] = 1.0  # Mark as high quality
        
        return data
        
    except Exception as e:
        print(f"Error fetching live data: {e}")
        return None

def insert_to_database(data):
    """Insert data into the trading database"""
    if not data:
        return False
        
    try:
        conn = sqlite3.connect('./trading_bot_live.db')
        cursor = conn.cursor()
        
        # Build dynamic insert with all available columns
        columns = ['timestamp', 'symbol', 'exchange', 'data_source', 'mid_price', 'spread',
                  'microprice', 'weighted_bid_price', 'weighted_ask_price', 'order_book_imbalance', 'data_quality_score']
        values = [data['timestamp'], data['symbol'], data['exchange'], data['data_source'], 
                 data['mid_price'], data['spread'], data['microprice'], data['weighted_bid_price'],
                 data['weighted_ask_price'], data['order_book_imbalance'], data['data_quality_score']]
        
        # Add all 10 levels of L2 data
        for i in range(1, 11):
            if f'bid_price_{i}' in data:
                columns.extend([f'bid_price_{i}', f'bid_size_{i}', f'ask_price_{i}', f'ask_size_{i}'])
                values.extend([data[f'bid_price_{i}'], data[f'bid_size_{i}'], 
                              data[f'ask_price_{i}'], data[f'ask_size_{i}']])
        
        placeholders = ', '.join(['?' for _ in values])
        column_names = ', '.join(columns)
        
        sql = f"""
        INSERT INTO l2_training_data_practical ({column_names})
        VALUES ({placeholders})
        """
        
        cursor.execute(sql, values)
        
        conn.commit()
        conn.close()
        return True
        
    except Exception as e:
        print(f"Database error: {e}")
        return False

def main():
    """Run fast live data collection"""
    parser = argparse.ArgumentParser(description='Fast live BTC data collection')
    parser.add_argument('--interval', type=float, default=1.0, help='Collection interval in seconds (default: 1.0)')
    parser.add_argument('--duration', type=int, default=60, help='Total duration in seconds (default: 60)')
    parser.add_argument('--silent', action='store_true', help='Reduce output for high-frequency collection')
    
    args = parser.parse_args()
    
    if not args.silent:
        print("FAST LIVE DATA COLLECTION")
        print("=" * 50)
        print(f"Interval: {args.interval}s")
        print(f"Duration: {args.duration}s")
        print(f"Expected samples: {int(args.duration / args.interval)}")
        print("Getting live BTC/USDT data from Bybit...")
    
    start_time = time.time()
    count = 0
    errors = 0
    
    try:
        # Collect data at specified interval
        while (time.time() - start_time) < args.duration:
            cycle_start = time.time()
            count += 1
            
            if not args.silent:
                print(f"\nCollection #{count}")
            
            data = get_btc_orderbook_data()
            if data:
                if insert_to_database(data):
                    if not args.silent:
                        mid_price = data['mid_price']
                        spread = data['spread']
                        imbalance = data['order_book_imbalance']
                        print(f"BTC ${mid_price:.2f}, spread ${spread:.2f}, imb {imbalance:.3f}")
                else:
                    errors += 1
            else:
                errors += 1
                if not args.silent:
                    print("Failed to get data")
            
            # Sleep for remaining time in interval
            elapsed = time.time() - cycle_start
            sleep_time = max(0, args.interval - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)
            elif not args.silent:
                print(f"Warning: Collection took {elapsed:.2f}s (longer than {args.interval}s interval)")
                
    except KeyboardInterrupt:
        print(f"\nStopped by user after {count} collections")
    
    if not args.silent:
        print(f"\nFast data collection complete!")
        print(f"Collections: {count}")
        print(f"Errors: {errors}")
        print(f"Success rate: {((count-errors)/count*100):.1f}%")
        
        # Show recent data
        conn = sqlite3.connect('./trading_bot_live.db')
        cursor = conn.cursor()
        cursor.execute("""
            SELECT timestamp, mid_price, bid_price_1, ask_price_1
            FROM l2_training_data_practical 
            WHERE data_source = 'live_trading' 
            ORDER BY timestamp DESC LIMIT 10
        """)
        
        print("\nMost recent data:")
        for row in cursor.fetchall():
            spread = row[3] - row[2]
            print(f"  {row[0][-8:]} | ${row[1]:.2f} | spread ${spread:.2f}")
        
        conn.close()

if __name__ == "__main__":
    main()