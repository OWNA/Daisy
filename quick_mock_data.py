#!/usr/bin/env python3
"""
Quick Mock Data Generator for Paper Trading Test
"""

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def create_mock_data():
    """Generate 30 seconds worth of mock L2 data for testing"""
    
    # Connect to database
    conn = sqlite3.connect('./trading_bot_live.db')
    
    # Check if table exists, create if not
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS l2_training_data_practical (
            timestamp TEXT,
            symbol TEXT DEFAULT 'BTCUSDT',
            exchange TEXT DEFAULT 'bybit',
            bid_price_1 REAL,
            bid_size_1 REAL,
            ask_price_1 REAL,
            ask_size_1 REAL,
            mid_price REAL,
            spread REAL,
            spread_bps REAL,
            microprice REAL,
            order_book_imbalance REAL,
            data_source TEXT DEFAULT 'mock_data'
        )
    """)
    
    # Generate mock data for last 5 minutes
    end_time = datetime.now()
    start_time = end_time - timedelta(minutes=5)
    
    # Base BTC price around $95,000
    base_price = 95000.0
    price_volatility = 50.0  # $50 volatility
    
    data = []
    current_time = start_time
    current_price = base_price
    
    # Generate data every 5 seconds
    while current_time <= end_time:
        # Random walk for price
        price_change = np.random.normal(0, price_volatility)
        current_price += price_change
        
        # Generate realistic bid/ask spread (0.5-2.0 USD typical for BTC)
        spread = np.random.uniform(0.5, 2.0)
        bid_price = current_price - spread/2
        ask_price = current_price + spread/2
        mid_price = (bid_price + ask_price) / 2
        
        # Generate sizes (typical BTC perp sizes)
        bid_size = np.random.uniform(0.1, 5.0)
        ask_size = np.random.uniform(0.1, 5.0)
        
        # Calculate features
        spread_bps = (spread / mid_price) * 10000
        total_size = bid_size + ask_size
        microprice = (bid_price * ask_size + ask_price * bid_size) / total_size if total_size > 0 else mid_price
        order_book_imbalance = (bid_size - ask_size) / total_size if total_size > 0 else 0
        
        data.append({
            'timestamp': current_time.isoformat(),
            'symbol': 'BTCUSDT',
            'exchange': 'bybit',
            'bid_price_1': bid_price,
            'bid_size_1': bid_size,
            'ask_price_1': ask_price,
            'ask_size_1': ask_size,
            'mid_price': mid_price,
            'spread': spread,
            'spread_bps': spread_bps,
            'microprice': microprice,
            'order_book_imbalance': order_book_imbalance,
            'data_source': 'mock_data'
        })
        
        current_time += timedelta(seconds=5)
    
    # Insert data
    df = pd.DataFrame(data)
    df.to_sql('l2_training_data_practical', conn, if_exists='append', index=False)
    
    print(f"Created {len(data)} mock data rows")
    print(f"Price range: ${min(row['mid_price'] for row in data):.2f} - ${max(row['mid_price'] for row in data):.2f}")
    print(f"Time range: {start_time} to {end_time}")
    
    conn.close()

if __name__ == "__main__":
    create_mock_data()