#!/usr/bin/env python3
"""
L2-Only Trading Bot
Main script to run the L2-only Trading Bot locally
Adapted for pure L2 order book strategy
"""

import os
import sys
import io
import yaml
import warnings
import traceback
import argparse
from datetime import datetime
from pathlib import Path

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Suppress warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

# Suppress Git errors from matplotlib
stderr_backup = sys.stderr
try:
    sys.stderr = io.StringIO()
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
finally:
    sys.stderr = stderr_backup


class L2TradingBotRunner:
    """Main class to run the L2-only trading bot workflow"""
    
    def __init__(self, config_path='config.yaml', force_l2_only=True):
        """Initialize the L2-only trading bot runner"""
        self.bot_base_dir = os.environ.get('BOT_BASE_DIR', './')
        self.config_path = config_path
        self.force_l2_only = force_l2_only
        self.config = {}
        self.bot_orchestrator = None
        self.setup_environment()
    
    def setup_environment(self):
        """Setup the environment and paths"""
        print("🚀 Setting up L2-Only Trading Bot Environment")
        print("="*60)
        
        # Ensure bot base directory exists
        os.makedirs(self.bot_base_dir, exist_ok=True)
        
        # Add bot base directory to Python path
        if self.bot_base_dir not in sys.path:
            sys.path.append(self.bot_base_dir)
            print(f"✅ Added {self.bot_base_dir} to Python path")
        
        # Set environment variable
        os.environ['BOT_BASE_DIR'] = self.bot_base_dir
        print(f"✅ BOT_BASE_DIR set to: {self.bot_base_dir}")
    
    def check_dependencies(self):
        """Check if all required dependencies are installed"""
        print("\n📦 Checking L2-Only Dependencies")
        print("="*60)
        
        required_modules = {
            'ccxt': 'CCXT',
            'lightgbm': 'LightGBM',
            'pandas': 'Pandas',
            'numpy': 'NumPy',
            'optuna': 'Optuna',
            'shap': 'SHAP',
            'PyEMD': 'EMD-signal',
            'matplotlib': 'Matplotlib',
            'scipy': 'SciPy',
            'yaml': 'PyYAML',
            'websocket': 'websocket-client',
            'dill': 'Dill',
            'sklearn': 'scikit-learn',
            'pandas_ta': 'pandas_ta'
        }
        
        missing_modules = []
        
        for module_name, display_name in required_modules.items():
            try:
                __import__(module_name)
                print(f"✅ {display_name} is installed")
            except ImportError:
                print(f"❌ {display_name} is NOT installed")
                missing_modules.append(display_name)
        
        if missing_modules:
            print(f"\n⚠️  Missing modules: {', '.join(missing_modules)}")
            print("Please run: pip install -r requirements.txt")
            return False
        
        print("\n✅ All L2-only dependencies are installed!")
        return True
    
    def load_configuration(self):
        """Load or create L2-only configuration"""
        print("\n⚙️  Loading L2-Only Configuration")
        print("="*60)
        
        config_path = os.path.join(self.bot_base_dir, self.config_path)
        
        # Check if config exists
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    self.config = yaml.safe_load(f)
                print(f"✅ Loaded existing configuration from {config_path}")
            except Exception as e:
                print(f"❌ Error loading config: {e}")
                self.create_l2_only_config()
        else:
            print("ℹ️  No existing config found, creating L2-only configuration")
            self.create_l2_only_config()
        
        # Validate and enforce L2-only mode
        if not self.config.get('l2_only_mode', False) and not self.force_l2_only:
            print("❌ ERROR: This bot requires L2-only mode!")
            print("   Set 'l2_only_mode: true' in config or use --l2-only flag")
            sys.exit(1)
        
        # Force L2-only mode
        self.config['l2_only_mode'] = True
        self.config['use_l2_features'] = True
        self.config['use_l2_features_for_training'] = True
        
        # Update paths for local environment
        self.config['base_dir'] = self.bot_base_dir
        
        # Show key L2-only configuration items
        print("\n📋 L2-Only Configuration:")
        print(f"   L2-Only Mode: {self.config.get('l2_only_mode', False)}")
        print(f"   Symbol: {self.config.get('symbol', 'BTC/USDT:USDT')}")
        print(f"   Exchange: {self.config.get('exchange_name', 'bybit')}")
        print(f"   Testnet: {self.config.get('exchange_testnet', True)}")
        print(f"   L2 Sampling: {self.config.get('l2_sampling_frequency_ms', 100)}ms")
        print(f"   L2 Buffer Size: {self.config.get('l2_buffer_size', 10000)}")
    
    def create_l2_only_config(self):
        """Create default L2-only configuration"""
        self.config = {
            # L2-Only Mode Configuration
            'l2_only_mode': True,
            'use_l2_features': True,
            'use_l2_features_for_training': True,
            
            # Exchange & Symbol
            'exchange_name': 'bybit',
            'exchange_testnet': True,
            'symbol': 'BTC/USDT:USDT',
            'market_type': 'linear',
            'timeframe': None,  # Not used in L2-only mode
            
            # L2 Data Configuration
            'l2_websocket_depth': 50,
            'l2_collection_duration_seconds': 21600,
            'l2_max_file_size_mb': 50,
            'l2_data_folder': 'l2_data',
            'l2_log_file': 'l2_data_collector.log',
            
            # L2 Processing Settings
            'l2_sampling_frequency_ms': 100,
            'l2_buffer_size': 10000,
            'l2_alignment_method': 'nearest',
            'l2_max_time_diff_ms': 1000,
            
            # Data Fetching & Paths
            'base_dir': self.bot_base_dir,
            'database_path': './trading_bot.db',
            
            # Feature Engineering (L2-only)
            'feature_window': 100,  # Increased for L2 data
            'use_hht_features': True,
            'hht_emd_noise_width': 0.01,  # Reduced for high-frequency data
            'ohlcv_base_features': [],  # Disabled for L2-only mode
            'ta_features': [],          # Disabled for L2-only mode
            
            # L2 Feature Configuration
            'l2_features': [
                'bid_ask_spread', 'bid_ask_spread_pct', 'weighted_mid_price',
                'microprice', 'order_book_imbalance_2', 'order_book_imbalance_3',
                'order_book_imbalance_5', 'total_bid_volume_2', 'total_ask_volume_2',
                'total_bid_volume_3', 'total_ask_volume_3', 'price_impact_buy',
                'price_impact_sell', 'price_impact_1', 'price_impact_5',
                'price_impact_10', 'bid_slope', 'ask_slope', 'l2_volatility_1min',
                'l2_volatility_5min', 'realized_volatility', 'order_flow_imbalance',
                'trade_intensity', 'effective_spread'
            ],
            
            # Label Generation (L2-based)
            'model_type': 'regression',
            'labeling_method': 'l2_volatility_normalized_return',
            'label_volatility_window': 50,  # Increased for L2 data
            'label_clip_quantiles': [0.005, 0.995],  # Tighter clipping
            'label_shift': -1,
            
            # Model Training
            'random_state': 42,
            'test_size': 0.2,
            'min_training_samples': 1000,  # Increased for L2 complexity
            'optuna_trials': 200,  # Increased for L2 complexity
            'lgbm_n_jobs': -1,
            
            # Risk Management (L2-adapted)
            'risk_management': {
                'max_drawdown': 0.15,
                'volatility_lookback': 100,  # L2-based lookback
                'position_sizing_mode': 'l2_volatility_target',
                'volatility_target_pct': 0.015,
                'max_equity_risk_pct': 0.03,
                'fixed_fraction_pct': 0.01,
                'sl_atr_multiplier': 1.0,  # Tighter for L2
                'tp_atr_multiplier': 2.0
            },
            
            # Execution (L2-optimized)
            'execution': {
                'slippage_model_pct': 0.0002,  # Lower for L2 precision
                'max_order_book_levels': 50,
                'default_entry_order_type': 'limit',  # More precise for L2
                'default_exit_order_type': 'limit',
                'l2_execution_mode': True
            },
            
            # Trading Parameters
            'initial_balance': 10000,
            'commission_pct': 0.0006,
            'leverage': 2,  # Reduced for L2 strategy
            'prediction_threshold': 0.15,  # Adjusted for L2 sensitivity
            
            # Live Simulation (L2-mode)
            'run_simulation_flag': True,
            'simulation_threshold': 0.15,
            'min_simulation_interval_seconds': 5,  # Faster for L2
            'simulation_duration_seconds': 1800,
            
            # Visualization
            'show_plots': True,
            'plot_style': 'seaborn-v0_8-darkgrid',
            'use_shap_for_importance': True,
            'shap_max_samples': 2000,  # Increased for L2 complexity
            
            # L2 Monitoring & Alerting
            'l2_monitoring': {
                'log_stats_interval': 60,
                'check_connection_interval': 30,
                'alert_on_disconnect': True,
                'alert_on_data_gap': True,
                'max_data_gap_seconds': 10
            },
            
            # Performance optimization
            'optimization_metric': 'sharpe_ratio',
            'optimization_trials': 200,
            
            # Fallback parameters (L2-adapted)
            'fallback_atr_pct_for_backtest': 0.01,  # Tighter for L2
            'fallback_volatility_pct_for_sizing': 0.01
        }
        
        # Save L2-only config
        config_path = os.path.join(self.bot_base_dir, self.config_path)
        with open(config_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False, indent=2)
        print(f"✅ Created L2-only configuration at {config_path}")
    
    def load_api_keys(self):
        """Load API keys from environment or .env file"""
        print("\n🔑 Loading API Keys")
        print("="*60)
        
        # Check for API keys (prioritize testnet for safety)
        api_key = os.getenv('BYBIT_API_KEY_MAIN_TEST') or os.getenv('BYBIT_API_KEY')
        api_secret = os.getenv('BYBIT_API_SECRET_MAIN_TEST') or os.getenv('BYBIT_API_SECRET')
        
        if api_key and api_secret:
            key_type = "TESTNET" if os.getenv('BYBIT_API_KEY_MAIN_TEST') else "MAINNET"
            print(f"✅ API keys found in environment ({key_type})")
            self.config['api_key'] = api_key
            self.config['api_secret'] = api_secret
        else:
            print("⚠️  No API keys found - will run in data-only mode")
            print("   Set BYBIT_API_KEY_MAIN_TEST and BYBIT_API_SECRET_MAIN_TEST for testnet")
            print("   Or BYBIT_API_KEY and BYBIT_API_SECRET for mainnet")
    
    def import_modules(self):
        """Import required modules with fallbacks"""
        print("\n📥 Importing L2-Only Modules")
        print("="*60)
        
        # Core modules
        try:
            global TradingBotOrchestrator
            from tradingbotorchestrator import TradingBotOrchestrator
            print("✅ TradingBotOrchestrator imported")
        except ImportError as e:
            print(f"❌ Failed to import TradingBotOrchestrator: {e}")
            return False
        
        # Optional modules with fallbacks
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
            print("⚠️  pandas_ta not available - some L2 features may be limited")
        
        # PyEMD
        try:
            from PyEMD import EMD
            modules['EMD'] = EMD
            modules['HAS_PYEMD'] = True
            print("✅ PyEMD imported")
        except ImportError:
            # Create dummy EMD class for L2-only mode
            class EMD:
                def __init__(self):
                    pass
                def emd(self, signal, max_imf=3):
                    import numpy as np
                    return [signal, np.zeros_like(signal)]
            
            modules['EMD'] = EMD
            modules['HAS_PYEMD'] = False
            print("⚠️  PyEMD not available - using fallback for L2-only mode")
        
        # scipy.signal.hilbert
        try:
            from scipy.signal import hilbert
            modules['hilbert'] = hilbert
            modules['HAS_SCIPY_HILBERT'] = True
            print("✅ scipy.signal.hilbert imported")
        except ImportError:
            # Create dummy hilbert function for L2-only mode
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
            print("⚠️  Optuna not available - optimization disabled")
        
        # shap
        try:
            import shap
            modules['shap'] = shap
            modules['HAS_SHAP'] = True
            print("✅ SHAP imported")
        except ImportError:
            modules['shap'] = None
            modules['HAS_SHAP'] = False
            print("⚠️  SHAP not available - feature importance plots disabled")
        
        self.modules = modules
        return True
    
    def initialize_orchestrator(self):
        """Initialize the L2-only trading bot orchestrator"""
        print("\n🤖 Initializing L2-Only Trading Bot Orchestrator")
        print("="*60)
        
        try:
            # Initialize with L2-only configuration
            self.bot_orchestrator = TradingBotOrchestrator(
                config=self.config,
                api_key=self.config.get('api_key'),
                api_secret=self.config.get('api_secret'),
                global_library_flags={
                    'HAS_PANDAS_TA': self.modules['HAS_PANDAS_TA'],
                    'HAS_PYEMD': self.modules['HAS_PYEMD'],
                    'HAS_SCIPY_HILBERT': self.modules['HAS_SCIPY_HILBERT'],
                    'HAS_OPTUNA': self.modules['HAS_OPTUNA'],
                    'HAS_SHAP': self.modules['HAS_SHAP']
                },
                global_library_modules={
                    'ta': self.modules['ta'],
                    'EMD': self.modules['EMD'],
                    'hilbert': self.modules['hilbert'],
                    'optuna': self.modules['optuna'],
                    'shap': self.modules['shap']
                }
            )
            
            print("✅ L2-Only TradingBotOrchestrator initialized successfully")
            
            # Validate L2-only mode
            if not self.bot_orchestrator.config.get('l2_only_mode', False):
                print("❌ ERROR: Orchestrator not in L2-only mode!")
                return False
                
            print("✅ L2-only mode validated in orchestrator")
            return True
            
        except Exception as e:
            print(f"❌ Failed to initialize L2-only orchestrator: {e}")
            traceback.print_exc()
            return False
    
    def run_workflow(self, workflow_type='l2_live_trading'):
        """Run the L2-only trading workflow"""
        print(f"\n🚀 Running L2-Only Workflow: {workflow_type}")
        print("="*60)
        
        try:
            if workflow_type == 'l2_live_trading':
                # Run L2-only live trading
                result = self.bot_orchestrator.run_l2_live_simulation()
                
            elif workflow_type == 'l2_training':
                # Run L2-only model training
                result = self.bot_orchestrator.train_l2_model()
                
            elif workflow_type == 'l2_backtest':
                # Run L2-only backtesting
                result = self.bot_orchestrator.run_l2_backtest()
                
            else:
                print(f"❌ Unknown L2 workflow type: {workflow_type}")
                return False
            
            if result:
                print(f"✅ L2-only workflow '{workflow_type}' completed successfully")
                return True
            else:
                print(f"❌ L2-only workflow '{workflow_type}' failed")
                return False
                
        except Exception as e:
            print(f"❌ Error in L2-only workflow '{workflow_type}': {e}")
            traceback.print_exc()
            return False
    
    def run(self, workflow_type='l2_live_trading'):
        """Run the complete L2-only trading bot"""
        print("🚀 Starting L2-Only Trading Bot")
        print("="*60)
        print(f"Timestamp: {datetime.now()}")
        print(f"Config: {self.config_path}")
        print(f"Workflow: {workflow_type}")
        print("="*60)
        
        # Step 1: Check dependencies
        if not self.check_dependencies():
            print("❌ Dependency check failed")
            return False
        
        # Step 2: Load configuration
        self.load_configuration()
        
        # Step 3: Load API keys
        self.load_api_keys()
        
        # Step 4: Import modules
        if not self.import_modules():
            print("❌ Module import failed")
            return False
        
        # Step 5: Initialize orchestrator
        if not self.initialize_orchestrator():
            print("❌ Orchestrator initialization failed")
            return False
        
        # Step 6: Run workflow
        if not self.run_workflow(workflow_type):
            print("❌ Workflow execution failed")
            return False
        
        print("\n🎉 L2-Only Trading Bot completed successfully!")
        return True


def main():
    """Main entry point for L2-only trading bot"""
    # Parse arguments
    parser = argparse.ArgumentParser(description='L2-Only Trading Bot')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to config file')
    parser.add_argument('--workflow', type=str, default='l2_live_trading',
                        choices=['l2_live_trading', 'l2_training', 'l2_backtest'],
                        help='Workflow type to run')
    parser.add_argument('--l2-only', action='store_true', default=True,
                        help='Force L2-only mode (default: True)')
    args = parser.parse_args()
    
    try:
        # Initialize and run L2-only bot
        bot_runner = L2TradingBotRunner(
            config_path=args.config,
            force_l2_only=args.l2_only
        )
        
        success = bot_runner.run(workflow_type=args.workflow)
        
        if success:
            print("\n✅ L2-Only Trading Bot execution completed successfully")
            sys.exit(0)
        else:
            print("\n❌ L2-Only Trading Bot execution failed")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n⚠️  L2-Only Trading Bot interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error in L2-only trading bot: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main() 