#!/usr/bin/env python3
"""
Diagnose the price discrepancy between training data and live trading
"""

import os
import json
import gzip
import yaml
import ccxt
import pandas as pd
from datetime import datetime

print("="*70)
print("PRICE DISCREPANCY DIAGNOSIS")
print("="*70)

# Load config
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# 1. Check L2 WebSocket data files
print("\n1. Checking L2 WebSocket collected data...")
l2_data_path = './l2_data'
if os.path.exists(l2_data_path):
    files = [f for f in os.listdir(l2_data_path) if f.endswith('.gz') or f.endswith('.jsonl')]
    if files:
        latest_file = sorted(files)[-1]
        print(f"   Latest L2 file: {latest_file}")
        
        # Read a few records
        file_path = os.path.join(l2_data_path, latest_file)
        is_gzipped = latest_file.endswith('.gz')
        open_func = gzip.open if is_gzipped else open
        read_mode = 'rt' if is_gzipped else 'r'
        
        with open_func(file_path, read_mode, encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= 3:  # Just read 3 records
                    break
                record = json.loads(line)
                
                print(f"\n   Record {i+1}:")
                print(f"     Symbol: {record.get('symbol', 'N/A')}")
                print(f"     Timestamp: {datetime.fromtimestamp(record.get('timestamp_ms', 0)/1000)}")
                
                bids = record.get('bids', record.get('b', []))
                asks = record.get('asks', record.get('a', []))
                
                if bids and asks:
                    best_bid = bids[0][0]
                    best_ask = asks[0][0]
                    mid_price = (best_bid + best_ask) / 2
                    print(f"     Best bid: ${best_bid:,.2f}")
                    print(f"     Best ask: ${best_ask:,.2f}")
                    print(f"     Mid price: ${mid_price:,.2f}")
    else:
        print("   No L2 data files found")
else:
    print("   L2 data directory not found")

# 2. Check CCXT live connection
print("\n\n2. Checking CCXT live connection...")
try:
    exchange_config = config.get('exchange', {})
    exchange = ccxt.bybit({
        'apiKey': os.getenv('BYBIT_API_KEY', ''),
        'secret': os.getenv('BYBIT_API_SECRET', ''),
        'enableRateLimit': True,
        'options': {
            'defaultType': exchange_config.get('market_type', 'linear')
        }
    })
    
    # Check testnet setting
    is_testnet = exchange_config.get('testnet', True)
    if is_testnet:
        exchange.set_sandbox_mode(True)
        print("   ⚠️  Using TESTNET mode")
    else:
        print("   Using MAINNET mode")
    
    # Get ticker
    symbol = config['symbol']
    ticker = exchange.fetch_ticker(symbol)
    print(f"\n   CCXT Ticker for {symbol}:")
    print(f"     Last price: ${ticker['last']:,.2f}")
    print(f"     Bid: ${ticker['bid']:,.2f}")
    print(f"     Ask: ${ticker['ask']:,.2f}")
    
    # Get orderbook
    orderbook = exchange.fetch_l2_order_book(symbol, limit=5)
    if orderbook['bids'] and orderbook['asks']:
        best_bid = orderbook['bids'][0][0]
        best_ask = orderbook['asks'][0][0]
        mid_price = (best_bid + best_ask) / 2
        print(f"\n   CCXT Orderbook:")
        print(f"     Best bid: ${best_bid:,.2f}")
        print(f"     Best ask: ${best_ask:,.2f}")
        print(f"     Mid price: ${mid_price:,.2f}")
        
except Exception as e:
    print(f"   Error: {e}")

# 3. Check training data features
print("\n\n3. Checking processed training data...")
features_file = "trading_bot_data/model_features_BTC_USDTUSDT_l2_only.json"
if os.path.exists(features_file):
    with open(features_file, 'r') as f:
        features_data = json.load(f)
    print(f"   Model trained with {len(features_data['trained_features'])} features")
    
    # Check if model expects price normalization
    price_features = [f for f in features_data['trained_features'] if 'price' in f and 'impact' not in f]
    print(f"   Price-related features: {len(price_features)}")
    print(f"   First few: {price_features[:5]}")

# 4. Diagnosis
print("\n\n" + "="*70)
print("DIAGNOSIS:")
print("="*70)

print("\n🔍 Likely issues:")
print("\n1. TESTNET vs MAINNET mismatch:")
print("   - WebSocket data might be from mainnet (realistic prices)")
print("   - CCXT might be using testnet (unrealistic prices)")
print("   - Solution: Set 'testnet: false' in config.yaml for paper trading")

print("\n2. Symbol format mismatch:")
print("   - WebSocket might use 'BTCUSDT'")
print("   - CCXT might need 'BTC/USDT:USDT'")
print("   - Check if symbols match between collection and trading")

print("\n3. Data processing issue:")
print("   - Feature engineering might be calculating weighted prices incorrectly")
print("   - Check if all price levels are being filled properly")

print("\n\n📋 RECOMMENDED FIXES:")
print("\n1. For paper trading with realistic prices:")
print("   - Set 'testnet: false' in config.yaml")
print("   - Use read-only API keys or no keys for safety")

print("\n2. Verify symbol consistency:")
print("   - Check L2 collector symbol vs CCXT symbol")
print("   - Ensure format matches (e.g., 'BTC/USDT:USDT' for perpetual)")

print("\n3. Add debug logging:")
print("   - Log raw orderbook data before processing")
print("   - Log calculated mid_price at each step")