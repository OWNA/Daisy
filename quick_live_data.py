#!/usr/bin/env python3
"""
Quick live data ingestion for BTC/USDT without loading all market instruments
"""
import ccxt
import sqlite3
import time
import json
from datetime import datetime
import numpy as np

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
        
        print("Fetching live BTC/USDT orderbook...")
        
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
        
        # Volume calculations (use correct column names)
        bid_size_1 = bids[0][1] if len(bids) > 0 else 0
        ask_size_1 = asks[0][1] if len(asks) > 0 else 0
        
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
        
        print(f"Live data: BTC ${mid_price:.2f}, spread ${spread:.2f}, imbalance {order_imbalance:.3f}")
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
        print(f"Inserted live data at {data['timestamp']}")
        return True
        
    except Exception as e:
        print(f"Database error: {e}")
        return False

def main():
    """Run live data collection"""
    print("QUICK LIVE DATA COLLECTION")
    print("=" * 40)
    print("Getting live BTC/USDT data from Bybit...")
    
    # Collect data every 10 seconds for 1 minute
    for i in range(6):
        print(f"\nCollection #{i+1}/6")
        data = get_btc_orderbook_data()
        if data:
            insert_to_database(data)
        time.sleep(10)
    
    print("\nLive data collection complete!")
    
    # Show what we collected
    conn = sqlite3.connect('./trading_bot_live.db')
    cursor = conn.cursor()
    cursor.execute("""
        SELECT timestamp, mid_price, bid_price_1, ask_price_1
        FROM l2_training_data_practical 
        WHERE data_source = 'live_trading' 
        ORDER BY timestamp DESC LIMIT 10
    """)
    
    print("\nRecent live data:")
    for row in cursor.fetchall():
        spread = row[3] - row[2]
        print(f"  {row[0][:19]} | ${row[1]:.2f} | bid ${row[2]:.2f} | ask ${row[3]:.2f} | spread ${spread:.2f}")
    
    conn.close()

if __name__ == "__main__":
    main()