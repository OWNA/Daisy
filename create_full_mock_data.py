#!/usr/bin/env python3
"""
Create full mock L2 data matching the 63-column schema
"""

import sqlite3
import numpy as np
from datetime import datetime, timedelta

def create_full_mock_data():
    """Generate complete mock L2 data for the full schema"""
    
    conn = sqlite3.connect('./trading_bot_live.db')
    cursor = conn.cursor()
    
    # Clear existing mock data
    cursor.execute("DELETE FROM l2_training_data_practical WHERE data_source = 'mock_data'")
    
    # Generate data for last 30 seconds (very recent for paper trading)
    end_time = datetime.now()
    start_time = end_time - timedelta(seconds=30)
    
    base_price = 95000.0
    
    data_rows = []
    current_time = start_time
    current_price = base_price
    
    # Generate data every 5 seconds for paper trading
    while current_time <= end_time:
        # Price movement
        price_change = np.random.normal(0, 25.0)  # $25 volatility
        current_price += price_change
        
        # Generate 10 levels of order book
        bid_prices = []
        bid_sizes = []
        ask_prices = []
        ask_sizes = []
        
        # Generate realistic order book levels
        spread = np.random.uniform(0.5, 3.0)  # BTC spread
        best_bid = current_price - spread/2
        best_ask = current_price + spread/2
        
        # Generate 10 levels each side
        for level in range(10):
            # Bids (decreasing prices)
            bid_price = best_bid - (level * np.random.uniform(0.5, 2.0))
            bid_size = np.random.uniform(0.1, 2.0) * (1 + level * 0.2)  # Larger sizes at worse prices
            bid_prices.append(bid_price)
            bid_sizes.append(bid_size)
            
            # Asks (increasing prices)  
            ask_price = best_ask + (level * np.random.uniform(0.5, 2.0))
            ask_size = np.random.uniform(0.1, 2.0) * (1 + level * 0.2)
            ask_prices.append(ask_price)
            ask_sizes.append(ask_size)
        
        # Calculate derived features
        mid_price = (bid_prices[0] + ask_prices[0]) / 2
        spread_val = ask_prices[0] - bid_prices[0]
        spread_bps = (spread_val / mid_price) * 10000
        
        total_bid_volume = sum(bid_sizes)
        total_ask_volume = sum(ask_sizes)
        
        # Weighted prices
        weighted_bid_price = sum(p * s for p, s in zip(bid_prices, bid_sizes)) / total_bid_volume
        weighted_ask_price = sum(p * s for p, s in zip(ask_prices, ask_sizes)) / total_ask_volume
        
        # Order book imbalance
        order_book_imbalance = (bid_sizes[0] - ask_sizes[0]) / (bid_sizes[0] + ask_sizes[0])
        
        # Microprice
        microprice = (bid_prices[0] * ask_sizes[0] + ask_prices[0] * bid_sizes[0]) / (bid_sizes[0] + ask_sizes[0])
        
        # Price impacts
        price_impact_bid = ((weighted_bid_price - bid_prices[0]) / bid_prices[0]) * 10000
        price_impact_ask = ((ask_prices[0] - weighted_ask_price) / ask_prices[0]) * 10000
        
        # Create row data
        row_data = [
            current_time.isoformat(),  # timestamp
            'BTCUSDT',  # symbol
            'bybit',  # exchange
        ]
        
        # Add 10 bid levels
        for i in range(10):
            row_data.extend([bid_prices[i], bid_sizes[i]])
        
        # Add 10 ask levels
        for i in range(10):
            row_data.extend([ask_prices[i], ask_sizes[i]])
        
        # Add calculated features  
        row_data.extend([
            current_price,  # Use current_price as mid_price
            spread_val,
            spread_bps,
            total_bid_volume,
            total_ask_volume,
            weighted_bid_price,
            weighted_ask_price,
            order_book_imbalance,
            microprice,
            price_impact_bid,
            price_impact_ask,
            None,  # target_return_1min
            None,  # target_return_5min
            None,  # target_volatility
            None,  # target_direction
            None,  # update_id
            None,  # sequence_id
            0.95,  # data_quality_score
            'mock_data',  # data_source
            None,  # exchange_timestamp
            None   # sequence
        ])
        
        data_rows.append(row_data)
        current_time += timedelta(seconds=5)
    
    # Insert all data
    insert_sql = """
        INSERT INTO l2_training_data_practical (
            timestamp, symbol, exchange,
            bid_price_1, bid_size_1, bid_price_2, bid_size_2, bid_price_3, bid_size_3,
            bid_price_4, bid_size_4, bid_price_5, bid_size_5, bid_price_6, bid_size_6,
            bid_price_7, bid_size_7, bid_price_8, bid_size_8, bid_price_9, bid_size_9,
            bid_price_10, bid_size_10,
            ask_price_1, ask_size_1, ask_price_2, ask_size_2, ask_price_3, ask_size_3,
            ask_price_4, ask_size_4, ask_price_5, ask_size_5, ask_price_6, ask_size_6,
            ask_price_7, ask_size_7, ask_price_8, ask_size_8, ask_price_9, ask_size_9,
            ask_price_10, ask_size_10,
            mid_price, spread, spread_bps, total_bid_volume_10, total_ask_volume_10,
            weighted_bid_price, weighted_ask_price, order_book_imbalance, microprice,
            price_impact_bid, price_impact_ask, target_return_1min, target_return_5min,
            target_volatility, target_direction, update_id, sequence_id, data_quality_score,
            data_source, exchange_timestamp, sequence
        ) VALUES (""" + ",".join(["?"] * 64) + ")"
    
    cursor.executemany(insert_sql, data_rows)
    conn.commit()
    
    print(f"Created {len(data_rows)} complete mock L2 data rows")
    print(f"Time range: {start_time} to {end_time}")
    # Price is at index 43 (timestamp, symbol, exchange + 40 bid/ask levels = 43)
    prices = [row[43] for row in data_rows]  
    print(f"Price range: ${min(prices):.2f} - ${max(prices):.2f}")
    
    conn.close()

if __name__ == "__main__":
    create_full_mock_data()