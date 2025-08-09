#!/usr/bin/env python3
"""
Fix paper trading to use mainnet prices
"""

import yaml
import shutil
from datetime import datetime

print("="*60)
print("FIXING PAPER TRADING CONFIGURATION")
print("="*60)

# Backup current config
backup_name = f"config_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.yaml"
shutil.copy('config.yaml', backup_name)
print(f"\n1. Backed up current config to: {backup_name}")

# Load config
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Check current settings
print("\n2. Current settings:")
print(f"   Symbol: {config.get('symbol', 'N/A')}")
print(f"   Exchange testnet: {config.get('exchange', {}).get('testnet', 'N/A')}")

# Make changes for paper trading
changes_made = []

# 1. Set testnet to false for realistic prices
if config.get('exchange', {}).get('testnet', True):
    config['exchange']['testnet'] = False
    changes_made.append("Set exchange.testnet to False for mainnet prices")

# 2. Ensure symbol format is correct
current_symbol = config.get('symbol', '')
if current_symbol == 'BTCUSDT' or current_symbol == 'BTC/USDT':
    # For perpetual futures on Bybit via CCXT
    config['symbol'] = 'BTC/USDT:USDT'
    changes_made.append(f"Changed symbol from '{current_symbol}' to 'BTC/USDT:USDT' for CCXT compatibility")

# 3. Add paper trading safety settings
if 'safety' not in config:
    config['safety'] = {}

config['safety']['paper_trading_only'] = True
config['safety']['max_position_size_btc'] = 0.1
config['safety']['require_confirmation'] = False
changes_made.append("Added safety settings for paper trading")

# Save updated config
if changes_made:
    with open('config.yaml', 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    print("\n3. Changes made:")
    for change in changes_made:
        print(f"   ✓ {change}")
    
    print("\n4. Config updated successfully!")
else:
    print("\n3. No changes needed - config already correct")

print("\n" + "="*60)
print("IMPORTANT NOTES:")
print("="*60)
print("\n⚠️  WARNING: Now using MAINNET prices for paper trading!")
print("   - This gives realistic prices but uses mainnet data")
print("   - Paper trading is still simulated (no real trades)")
print("   - Do NOT use real API keys with trading permissions")
print("\n✅ SAFE FOR PAPER TRADING:")
print("   - No actual orders will be placed")
print("   - Using mainnet prices for realistic simulation")
print("   - Position tracking is local only")

print("\n📋 NEXT STEPS:")
print("1. Run paper trading: python main.py trade --paper")
print("2. You should now see realistic BTC prices (~$67,000)")
print("3. Position sizes should be reasonable (0.001-0.1 BTC)")

print("\n🔄 TO REVERT:")
print(f"   cp {backup_name} config.yaml")