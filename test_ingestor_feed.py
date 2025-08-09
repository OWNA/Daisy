#!/usr/bin/env python3
"""
test_ingestor_feed.py - Raw Bybit WebSocket Feed Debug Tool

This script connects directly to the Bybit WebSocket and prints the raw L2 data
exactly as received from the exchange. No processing, no database writes.
Pure raw data inspection for debugging data integrity issues.

DEBUG_PLAN.md - Task 1.2: Raw Feed Inspection
"""

import asyncio
import sys

if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import json
import signal
import time
import os
from datetime import datetime
from typing import Dict, Any
from dotenv import load_dotenv

# Use ccxt.pro for WebSocket
import ccxt.pro as ccxtpro

# Global flag for clean shutdown
running = True


def print_separator():
    """Print a visual separator."""
    print("=" * 80)


def format_timestamp(ts):
    """Format timestamp for display."""
    if ts:
        try:
            return datetime.fromtimestamp(ts / 1000).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        except:
            return f"Invalid timestamp: {ts}"
    return "No timestamp"


async def websocket_client(exchange, symbol):
    """The main websocket client loop."""
    update_count = 0
    while True:
        try:
            orderbook = await exchange.watch_order_book(symbol)
            update_count += 1
            
            # Extract the most important, undeniable data points
            timestamp = orderbook.get('timestamp')
            dt_object = datetime.fromtimestamp(timestamp / 1000)
            
            bids = orderbook.get('bids', [])
            asks = orderbook.get('asks', [])

            if not bids or not asks:
                print(f"[{dt_object.isoformat()}] Incomplete order book received.")
                continue

            best_bid_price = bids[0][0]
            best_ask_price = asks[0][0]
            spread = best_ask_price - best_bid_price

            # Print the clean, essential data
            print(f"[{dt_object.isoformat()}] Update #{update_count} | Best Bid: {best_bid_price:.2f} | Best Ask: {best_ask_price:.2f} | Spread: {spread:.2f}")

        except Exception as e:
            print(f"[{datetime.now().isoformat()}] An error occurred: {e}")
            await asyncio.sleep(5) # Wait before retrying

async def main():
    """Main function to run the feed test."""
    print("Starting raw Bybit WebSocket feed inspection...")
    print("This will show CLEANED data to isolate the core price feed.")
    print("Press Ctrl+C to stop.")
    print("\n" + "="*80)
    
    # Get config from environment (assuming .env file is loaded or vars are set)
    load_dotenv()
    api_key = os.getenv('BYBIT_API_KEY_MAIN_TEST')
    api_secret = os.getenv('BYBIT_API_SECRET_MAIN_TEST')
    symbol = 'BTC/USDT:USDT'

    if not api_key or not api_secret:
        print("ERROR: API key and secret not found in .env file.")
        return

    exchange = ccxtpro.bybit({
        'apiKey': api_key,
        'secret': api_secret,
        'options': {'defaultType': 'linear'},
        'enableRateLimit': True,
        'sandbox': True
    })
    
    task = asyncio.create_task(websocket_client(exchange, symbol))

    # Handle graceful shutdown
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, task.cancel)

    try:
        await task
    except asyncio.CancelledError:
        print("\n" + "="*80)
        print("Feed inspection stopped by user.")
    finally:
        await exchange.close()
        print("Exchange connection closed.")


if __name__ == "__main__":
    asyncio.run(main())