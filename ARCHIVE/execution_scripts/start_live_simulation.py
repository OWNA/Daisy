#!/usr/bin/env python3
"""
Start L2-Only Live Simulation
Simple script to run L2-only live simulation (paper trading) with your trained L2 model
"""

import os
import sys
import yaml
import warnings
import argparse

# Suppress warnings
warnings.filterwarnings('ignore')

# Set up environment
BOT_BASE_DIR = './'
os.environ['BOT_BASE_DIR'] = BOT_BASE_DIR
sys.path.append(BOT_BASE_DIR)


def check_l2_prerequisites(config_path='config.yaml'):
    """Check if L2-only model and data files exist"""
    print("🔍 Checking L2-Only Prerequisites")
    print("="*60)
    
    # Load config to get symbol
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        symbol = config.get('symbol', 'BTC/USDT:USDT')
    except:
        symbol = 'BTC/USDT:USDT'
    
    # Generate L2-only model file names
    safe_symbol = symbol.replace('/', '_').replace(':', '')
    l2_model_path = f'lgbm_model_{safe_symbol}_l2_only.txt'
    l2_features_path = f'model_features_{safe_symbol}_l2_only.json'
    
    # Fallback to original model names
    fallback_model_path = f'lgbm_model_{safe_symbol}_1m.txt'
    fallback_features_path = f'model_features_{safe_symbol}_1m.json'
    
    required_files = {
        config_path: 'Configuration file',
        'trading_bot.db': 'L2 training database (optional)'
    }
    
    # Check for L2-specific model first
    if os.path.exists(l2_model_path) and os.path.exists(l2_features_path):
        required_files[l2_model_path] = 'L2-only trained model'
        required_files[l2_features_path] = 'L2-only model features'
        print(f"✅ Found L2-only model: {l2_model_path}")
    elif os.path.exists(fallback_model_path) and os.path.exists(fallback_features_path):
        required_files[fallback_model_path] = 'Fallback trained model'
        required_files[fallback_features_path] = 'Fallback model features'
        print(f"✅ Found fallback model: {fallback_model_path}")
        print("⚠️  Using fallback model - consider training L2-only model")
    else:
        print(f"❌ No suitable model found!")
        print(f"   Expected L2-only: {l2_model_path}")
        print(f"   Expected fallback: {fallback_model_path}")
        return False
    
    missing_files = []
    for file_path, description in required_files.items():
        if os.path.exists(file_path):
            print(f"✅ {description}: {file_path}")
        else:
            if 'optional' not in description.lower():
                print(f"❌ Missing {description}: {file_path}")
                missing_files.append(file_path)
            else:
                print(f"⚠️  Optional {description}: {file_path}")
    
    if missing_files:
        print("\n⚠️  Missing required files. Please ensure you have:")
        for file_path in missing_files:
            print(f"   - {file_path}")
        print("\nRun L2-only training first: python run_training_simple.py --l2-only")
        return False
    
    print("✅ All L2-only prerequisites met!")
    return True


def load_and_update_l2_config(config_path='config.yaml'):
    """Load configuration and enable L2-only simulation"""
    print("\n⚙️  Loading L2-Only Configuration")
    print("="*60)
    
    # Try to use the live simulation config first, fallback to main config
    config_files = [config_path, 'config_live_sim.yaml', 'config.yaml']
    config = None
    used_config_file = None
    
    for config_file in config_files:
        if os.path.exists(config_file):
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
            used_config_file = config_file
            print(f"✅ Loaded configuration from {config_file}")
            break
    
    if config is None:
        print("❌ No configuration file found!")
        return None
    
    # Force L2-only mode for live simulation
    config['l2_only_mode'] = True
    config['use_l2_features'] = True
    config['use_l2_features_for_training'] = True
    
    # Enable simulation
    config['run_simulation_flag'] = True
    
    # L2-specific simulation settings
    config['l2_sampling_frequency_ms'] = config.get('l2_sampling_frequency_ms', 100)
    config['l2_buffer_size'] = config.get('l2_buffer_size', 10000)
    
    # Show current L2-only simulation settings
    print("📋 L2-Only Simulation Settings:")
    print(f"   L2-Only Mode: {config.get('l2_only_mode', False)}")
    print(f"   Symbol: {config.get('symbol', 'BTC/USDT:USDT')}")
    print(f"   Exchange: {config.get('exchange_name', 'bybit')}")
    print(f"   Testnet: {config.get('exchange_testnet', True)}")
    balance = config.get('initial_balance', 10000)
    print(f"   Initial Balance: ${balance:,.2f}")
    print(f"   Leverage: {config.get('leverage', 2)}x")
    duration = config.get('simulation_duration_seconds', 1800)
    print(f"   Duration: {duration} seconds")
    print(f"   Threshold: {config.get('simulation_threshold', 0.15)}")
    print(f"   L2 Sampling: {config.get('l2_sampling_frequency_ms', 100)}ms")
    print(f"   L2 Buffer Size: {config.get('l2_buffer_size', 10000)}")
    print(f"   L2 Features: {config.get('use_l2_features', False)} (enabled for L2-only)")
    
    return config


def start_l2_simulation(config):
    """Start the L2-only live simulation"""
    print("\n🚀 Starting L2-Only Live Simulation")
    print("="*60)
    
    try:
        # Import required modules
        print("📦 Importing L2-only modules...")
        from tradingbotorchestrator import TradingBotOrchestrator
        
        # Import optional libraries with fallbacks
        modules = {}
        
        # pandas_ta
        try:
            import pandas_ta as ta
            modules['ta'] = ta
            modules['HAS_PANDAS_TA'] = True
            print("✅ pandas_ta imported")
        except ImportError:
            modules['ta'] = None
            modules['HAS_PANDAS_TA'] = False
            print("⚠️  pandas_ta not available")
        
        # PyEMD
        try:
            from PyEMD import EMD
            modules['EMD'] = EMD
            modules['HAS_PYEMD'] = True
            print("✅ PyEMD imported")
        except ImportError:
            class EMD:
                def __init__(self):
                    pass
                def emd(self, signal, max_imf=3):
                    import numpy as np
                    return [signal, np.zeros_like(signal)]
            modules['EMD'] = EMD
            modules['HAS_PYEMD'] = False
            print("⚠️  PyEMD not available - using fallback")
        
        # scipy.signal.hilbert
        try:
            from scipy.signal import hilbert
            modules['hilbert'] = hilbert
            modules['HAS_SCIPY_HILBERT'] = True
            print("✅ scipy.signal.hilbert imported")
        except ImportError:
            def hilbert(signal, N=None, axis=-1):
                import numpy as np
                return signal + 1j * np.zeros_like(signal)
            modules['hilbert'] = hilbert
            modules['HAS_SCIPY_HILBERT'] = False
            print("⚠️  scipy.signal.hilbert not available - using fallback")
        
        # optuna
        try:
            import optuna
            modules['optuna'] = optuna
            modules['HAS_OPTUNA'] = True
            print("✅ Optuna imported")
        except ImportError:
            modules['optuna'] = None
            modules['HAS_OPTUNA'] = False
            print("⚠️  Optuna not available")
        
        # shap
        try:
            import shap
            modules['shap'] = shap
            modules['HAS_SHAP'] = True
            print("✅ SHAP imported")
        except ImportError:
            modules['shap'] = None
            modules['HAS_SHAP'] = False
            print("⚠️  SHAP not available")
        
        print("✅ L2-only modules imported successfully")
        
        # Create L2-only orchestrator
        print("🤖 Initializing L2-only orchestrator...")
        orchestrator = TradingBotOrchestrator(
            config=config,
            global_library_flags={
                'HAS_PANDAS_TA': modules['HAS_PANDAS_TA'],
                'HAS_PYEMD': modules['HAS_PYEMD'],
                'HAS_SCIPY_HILBERT': modules['HAS_SCIPY_HILBERT'],
                'HAS_OPTUNA': modules['HAS_OPTUNA'],
                'HAS_SHAP': modules['HAS_SHAP']
            },
            global_library_modules={
                'ta': modules['ta'],
                'EMD': modules['EMD'],
                'hilbert': modules['hilbert'],
                'optuna': modules['optuna'],
                'shap': modules['shap']
            }
        )
        print("✅ L2-only orchestrator initialized")
        
        # Validate L2-only mode
        if not orchestrator.config.get('l2_only_mode', False):
            print("❌ ERROR: Orchestrator not in L2-only mode!")
            return False
        
        # Start L2-only live simulation
        print("\n🎯 Starting L2-Only Live Simulation...")
        print("="*60)
        print("📊 The L2-only simulation will:")
        print("   • Stream live L2 order book data")
        print("   • Generate L2 microstructure features in real-time")
        print("   • Make predictions using your trained L2-only model")
        print("   • Execute paper trades based on L2 signals")
        print("   • Track performance with L2-specific metrics")
        print("   • Monitor spread, liquidity, and order book dynamics")
        print("\n⚠️  This is PAPER TRADING - no real money at risk")
        print("⏱️  Press Ctrl+C to stop the simulation early")
        print("="*60)
        
        # Run the L2-only simulation
        success = orchestrator.run_l2_live_simulation()
        
        if success:
            print("\n✅ L2-only live simulation completed successfully!")
            print("\n📊 Check the following files for L2-only results:")
            print("   • paper_trading_results/l2_simulation_trades.csv")
            print("   • paper_trading_results/l2_simulation_equity.csv")
            print("   • paper_trading_results/l2_simulation_metrics.csv")
            print("   • L2-specific performance plots (if generated)")
        else:
            print("\n❌ L2-only live simulation encountered errors")
            
        return success
            
    except KeyboardInterrupt:
        print("\n⚠️  L2-only simulation stopped by user")
        return False
    except Exception as e:
        print(f"\n❌ Error during L2-only simulation: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main function for L2-only live simulation"""
    # Parse arguments
    parser = argparse.ArgumentParser(description='L2-Only Live Simulation Starter')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to config file')
    parser.add_argument('--l2-only', action='store_true', default=True,
                        help='Force L2-only mode (default: True)')
    args = parser.parse_args()
    
    print("🚀 L2-Only Live Trading Simulation Starter")
    print("="*60)
    print("This script will start an L2-only live simulation (paper trading)")
    print("using your trained L2-only model to make real-time predictions")
    print("based on Level 2 order book data.")
    print("="*60)
    
    # Check L2-only prerequisites
    if not check_l2_prerequisites(args.config):
        return
    
    # Load L2-only configuration
    config = load_and_update_l2_config(args.config)
    if config is None:
        print("❌ Failed to load L2-only configuration")
        return
    
    # Validate L2-only mode
    if not config.get('l2_only_mode', False) and not args.l2_only:
        print("❌ ERROR: This script requires L2-only mode!")
        print("   Set 'l2_only_mode: true' in config or use --l2-only flag")
        return
    
    # Force L2-only mode
    config['l2_only_mode'] = True
    
    # Confirm with user
    print("\n❓ Ready to start L2-only live simulation?")
    duration = config.get('simulation_duration_seconds', 1800)
    print(f"   Duration: {duration} seconds ({duration//60} minutes)")
    print("   Mode: L2-Only Paper Trading (no real money)")
    print(f"   L2 Sampling: {config.get('l2_sampling_frequency_ms', 100)}ms")
    print(f"   Symbol: {config.get('symbol', 'BTC/USDT:USDT')}")
    
    response = input("\nStart L2-only simulation? (y/N): ").strip().lower()
    if response not in ['y', 'yes']:
        print("❌ L2-only simulation cancelled by user")
        return
    
    # Start L2-only simulation
    success = start_l2_simulation(config)
    
    print("\n" + "="*60)
    if success:
        print("✅ L2-Only Live Simulation Finished Successfully")
        print("📋 Next steps:")
        print("   1. Analyze results: python analyze_predictions.py")
        print("   2. Start live trading: python run_trading_bot.py --config config.yaml")
    else:
        print("❌ L2-Only Live Simulation Failed")
        print("📋 Troubleshooting:")
        print("   1. Check L2 model exists: python run_training_simple.py --l2-only")
        print("   2. Verify config: check l2_only_mode: true in config")
        print("   3. Check network connection for L2 data streaming")
    print("="*60)


if __name__ == "__main__":
    main() 