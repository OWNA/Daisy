#!/usr/bin/env python3
"""
Check what data we're actually getting from Bybit via CCXT
"""

import ccxt
import os
import yaml
import json
from datetime import datetime

# Load config
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

print("="*60)
print("BYBIT DATA CHECK")
print("="*60)

# Initialize exchange
exchange_config = config.get('exchange', {})
exchange = ccxt.bybit({
    'apiKey': os.getenv('BYBIT_API_KEY', ''),
    'secret': os.getenv('BYBIT_API_SECRET', ''),
    'enableRateLimit': True,
    'options': {
        'defaultType': exchange_config.get('market_type', 'linear')
    }
})

if exchange_config.get('testnet', True):
    exchange.set_sandbox_mode(True)
    print("Using Bybit TESTNET")
else:
    print("Using Bybit MAINNET")

symbol = config['symbol']
print(f"\nChecking symbol: {symbol}")

try:
    # 1. Check ticker for current price
    print("\n1. Fetching ticker...")
    ticker = exchange.fetch_ticker(symbol)
    print(f"   Last price: ${ticker['last']:,.2f}")
    print(f"   Bid: ${ticker['bid']:,.2f}")
    print(f"   Ask: ${ticker['ask']:,.2f}")
    print(f"   24h Volume: {ticker['baseVolume']:,.4f} BTC")
    
    # 2. Check orderbook
    print("\n2. Fetching order book...")
    orderbook = exchange.fetch_l2_order_book(symbol, limit=10)
    
    print(f"   Timestamp: {datetime.fromtimestamp(orderbook['timestamp']/1000)}")
    print(f"   Bids: {len(orderbook['bids'])} levels")
    print(f"   Asks: {len(orderbook['asks'])} levels")
    
    if orderbook['bids']:
        print(f"\n   Top 5 Bids:")
        for i, (price, size) in enumerate(orderbook['bids'][:5]):
            print(f"     Level {i+1}: ${price:,.2f} x {size:.4f} BTC")
    
    if orderbook['asks']:
        print(f"\n   Top 5 Asks:")
        for i, (price, size) in enumerate(orderbook['asks'][:5]):
            print(f"     Level {i+1}: ${price:,.2f} x {size:.4f} BTC")
    
    # Calculate mid price
    if orderbook['bids'] and orderbook['asks']:
        best_bid = orderbook['bids'][0][0]
        best_ask = orderbook['asks'][0][0]
        mid_price = (best_bid + best_ask) / 2
        spread = best_ask - best_bid
        spread_bps = (spread / mid_price) * 10000
        
        print(f"\n   Calculated metrics:")
        print(f"     Mid price: ${mid_price:,.2f}")
        print(f"     Spread: ${spread:.2f}")
        print(f"     Spread (bps): {spread_bps:.2f}")
    
    # 3. Save sample orderbook for analysis
    sample_file = "sample_orderbook.json"
    with open(sample_file, 'w') as f:
        json.dump(orderbook, f, indent=2)
    print(f"\n3. Sample orderbook saved to {sample_file}")
    
    # 4. Check WebSocket L2 data format
    print("\n4. Checking L2 data files...")
    l2_files = []
    if os.path.exists('./l2_data'):
        for f in os.listdir('./l2_data'):
            if f.endswith('.parquet'):
                l2_files.append(f)
        
        if l2_files:
            print(f"   Found {len(l2_files)} L2 data files")
            latest_file = sorted(l2_files)[-1]
            print(f"   Latest file: {latest_file}")
            
            # Try to read a sample
            try:
                import pandas as pd
                df = pd.read_parquet(f'./l2_data/{latest_file}')
                print(f"   L2 data shape: {df.shape}")
                print(f"   L2 columns: {list(df.columns)[:10]}...")
                
                # Check data format
                if not df.empty:
                    row = df.iloc[0]
                    print(f"\n   Sample L2 record:")
                    if 'b' in df.columns and row['b'] is not None:
                        bids = row['b']
                        if isinstance(bids, str):
                            import ast
                            bids = ast.literal_eval(bids)
                        if bids and len(bids) > 0:
                            print(f"     First bid: {bids[0]}")
                    if 'a' in df.columns and row['a'] is not None:
                        asks = row['a']
                        if isinstance(asks, str):
                            import ast
                            asks = ast.literal_eval(asks)
                        if asks and len(asks) > 0:
                            print(f"     First ask: {asks[0]}")
            except Exception as e:
                print(f"   Error reading L2 file: {e}")
        else:
            print("   No L2 data files found")
    
except Exception as e:
    print(f"\nERROR: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*60)
print("DIAGNOSIS:")
print("="*60)

if 'ticker' in locals() and ticker['last'] < 10000:
    print("❌ Price data looks suspicious - might be using wrong symbol or testnet")
elif 'ticker' in locals() and ticker['last'] > 200000:
    print("❌ Price too high - check if using correct market type")
elif 'ticker' in locals():
    print("✅ Price data looks reasonable")

print("\nNEXT STEPS:")
print("1. Check if Bybit testnet has realistic BTC prices")
print("2. Verify symbol format matches between WebSocket and CCXT")
print("3. Consider switching to mainnet for realistic prices (paper trading only)")