#!/usr/bin/env python3
"""
Quick setup script for the Advanced Trading CLI
"""

import subprocess
import sys
import os

print("🚀 Advanced Trading CLI - Quick Setup")
print("="*60)

# Check Python version
print(f"\nPython version: {sys.version}")

# Install required packages
print("\n📦 Installing required packages...")
packages = [
    "pandas>=1.5.0",
    "numpy>=1.21.0",
    "scikit-learn>=1.1.0",
    "lightgbm>=3.3.0",
    "matplotlib>=3.5.0",
    "PyYAML>=6.0",
    "ccxt>=3.0.0",
    "scipy>=1.9.0",
    "optuna>=3.0.0",
    "shap>=0.41.0",
    "websocket-client>=1.4.0",
    "python-dotenv>=0.19.0",
    "dill>=0.3.5",
    "pandas-ta>=0.3.14b",
    "PyEMD>=0.3.3",
    "rich>=12.0.0",
    "click>=8.0.0",
    "plotly>=5.0.0",
    "seaborn>=0.11.0",
    "flask>=2.0.0",
    "kaleido>=0.2.0"
]

for package in packages:
    print(f"  Installing {package}...")
    subprocess.run([sys.executable, "-m", "pip", "install", package], 
                   capture_output=True, text=True)

print("\n✅ Dependencies installed!")

# Check if config exists
if not os.path.exists("config.yaml"):
    print("\n⚠️  config.yaml not found!")
    print("Please ensure config.yaml exists before running the CLI.")
else:
    print("\n✅ config.yaml found!")

# Check if model exists
if os.path.exists("lgbm_model_BTC_USDTUSDT_optuna_l2_hht.txt"):
    print("✅ Trained model found!")
else:
    print("⚠️  No trained model found. You'll need to train one first.")

# Create L2 data directory if needed
if not os.path.exists("l2_data"):
    os.makedirs("l2_data")
    print("✅ Created l2_data directory")

# Create results directories
for dir_name in ["backtest_results", "paper_trading_results", "shap_plots"]:
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
        print(f"✅ Created {dir_name} directory")

print("\n🎉 Setup complete!")
print("\nYou can now use the Advanced Trading CLI:")
print("  PowerShell: ./trade <command>")
print("  Direct: python trade_cli_advanced.py <command>")
print("\nTry: ./trade --help")
print("="*60)
