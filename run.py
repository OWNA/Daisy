#!/usr/bin/env python3
"""
run.py - Main Paper Trading Deployment Script

This script deploys the enhanced BTC trading system to paper trading environment.
It uses the validated pipeline to execute mock trades with real market data.

Priority 7: Paper Trading Deployment - Day 5
"""

import os
import sys
import time
import logging
import signal
import threading
import traceback
import sqlite3
import pandas as pd
import numpy as np
import ccxt
import yaml
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from contextlib import contextmanager
from dotenv import load_dotenv

# Import component factory and integration test
try:
    from component_factory import ComponentFactory, create_factory_from_config
    from production_model_predictor import ProductionModelPredictor
    
    # Production model predictor is initialized in PaperTradingEngine
    pass
        
except ImportError as e:
    print(f"Critical Error: Could not import system components: {e}")
    print("Please ensure all required modules are available.")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f'paper_trading_{datetime.now().strftime("%Y%m%d")}.log')
    ]
)
logger = logging.getLogger(__name__)


class PaperTradingConfig:
    """Configuration management for paper trading deployment."""
    
    def __init__(self, config_file: str = 'config.yaml', env_file: str = '.env'):
        """Initialize configuration from files."""
        self.config_file = config_file
        self.env_file = env_file
        
        # Load environment variables
        load_dotenv(env_file)
        
        # Load YAML configuration
        with open(config_file, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Override with paper trading specific settings
        self._setup_paper_trading_config()
        
        logger.info("PaperTradingConfig initialized")
        logger.info(f"Base directory: {self.config['base_dir']}")
        logger.info(f"Database: {self.config['database_path']}")
        logger.info(f"Exchange: {self.config['exchange']['name']} (testnet: {self.config['exchange']['testnet']})")
    
    def _setup_paper_trading_config(self):
        """Configure settings specifically for paper trading."""
        
        # Use Demo Trading (mainnet) for paper trading
        self.config['exchange']['testnet'] = False
        
        # Paper trading specific settings
        self.config.update({
            'paper_trading': {
                'enabled': True,
                'initial_balance': 10000,  # $10k mock balance
                'max_position_size': 0.1,  # Max 0.1 BTC position
                'max_daily_trades': 50,    # Limit trades per day
                'risk_limit_pct': 2.0,    # Max 2% risk per trade
                'profit_target_pct': 1.0, # Take profit at 1%
                'stop_loss_pct': 0.5      # Stop loss at 0.5%
            },
            'monitoring': {
                'dashboard_enabled': True,
                'metrics_interval_s': 60,  # Update metrics every minute
                'performance_tracking': True,
                'alert_threshold_loss_pct': 5.0
            }
        })
        
        # Enhanced ML model settings
        self.config.update({
            'ml_model': {
                'ensemble_enabled': True,
                'confidence_threshold': 0.6,
                'prediction_horizons': [10, 50, 100, 300],
                'feature_count': 12,  # Basic L2 features from integration test
                'retrain_enabled': False,  # Use pre-trained models only
                'fallback_to_mock': True   # Use mock predictions if models unavailable
            }
        })
    
    def get_exchange_config(self) -> Dict[str, Any]:
        """Get exchange configuration for CCXT with comprehensive timestamp fix."""
        return {
            'apiKey': os.getenv('BYBIT_API_KEY_MAIN'),
            'secret': os.getenv('BYBIT_API_SECRET_MAIN'),
            'sandbox': False,  # Use Demo Trading (mainnet)
            'enableRateLimit': True,
            'adjustForTimeDifference': True,  # Fix for retCode 10002 timestamp error
            'options': {
                'defaultType': 'linear',  # USDT perpetual futures
                'recvWindow': 10000,      # Increase receive window to 10 seconds
                'adjustForTimeDifference': True,  # Also set in options
                'timeDifference': 0       # Let CCXT auto-calculate
            }
        }
    
    def get_symbol_config(self) -> Dict[str, Any]:
        """Get trading symbol configuration."""
        return {
            'symbol': 'BTC/USDT:USDT',  # Bybit perpetual futures format
            'base_currency': 'BTC',
            'quote_currency': 'USDT',
            'min_order_size': 0.001,    # Minimum order size
            'price_precision': 1,       # Price decimal places
            'amount_precision': 3       # Amount decimal places
        }


class PaperTradingDashboard:
    """Simple performance dashboard for paper trading."""
    
    def __init__(self, config: PaperTradingConfig):
        self.config = config
        self.metrics = {
            'start_time': datetime.now(),
            'total_trades': 0,
            'profitable_trades': 0,
            'total_pnl': 0.0,
            'max_drawdown': 0.0,
            'current_balance': config.config['paper_trading']['initial_balance'],
            'current_position': 0.0,
            'last_signal': None,
            'last_prediction': None,
            'system_health': 'GOOD'
        }
        
        self.trade_history = []
        self.performance_history = []
        
    def update_trade(self, trade_details: Dict[str, Any]):
        """Update dashboard with new trade information."""
        self.trade_history.append(trade_details)
        self.metrics['total_trades'] += 1
        
        if trade_details.get('pnl', 0) > 0:
            self.metrics['profitable_trades'] += 1
        
        self.metrics['last_signal'] = trade_details.get('signal')
        
    def update_performance(self, pnl: float, balance: float, position: float):
        """Update performance metrics."""
        self.metrics['total_pnl'] = pnl
        self.metrics['current_balance'] = balance
        self.metrics['current_position'] = position
        
        # Calculate drawdown
        if balance < self.config.config['paper_trading']['initial_balance']:
            drawdown_pct = (1 - balance / self.config.config['paper_trading']['initial_balance']) * 100
            self.metrics['max_drawdown'] = max(self.metrics['max_drawdown'], drawdown_pct)
        
        # Store performance snapshot
        self.performance_history.append({
            'timestamp': datetime.now(),
            'balance': balance,
            'pnl': pnl,
            'position': position
        })
        
        # Keep only recent history (last 24 hours)
        cutoff_time = datetime.now() - timedelta(hours=24)
        self.performance_history = [
            p for p in self.performance_history 
            if p['timestamp'] > cutoff_time
        ]
    
    def print_dashboard(self):
        """Print dashboard to console."""
        print("\n" + "="*80)
        print("PAPER TRADING DASHBOARD")
        print("="*80)
        
        runtime = datetime.now() - self.metrics['start_time']
        win_rate = (self.metrics['profitable_trades'] / max(self.metrics['total_trades'], 1)) * 100
        
        print(f"Runtime: {runtime}")
        print(f"Balance: ${self.metrics['current_balance']:.2f}")
        print(f"Total P&L: ${self.metrics['total_pnl']:.2f}")
        print(f"Trades: {self.metrics['total_trades']} (Win Rate: {win_rate:.1f}%)")
        print(f"Max Drawdown: {self.metrics['max_drawdown']:.2f}%")
        print(f"Position: {self.metrics['current_position']:.4f} BTC")
        print(f"System Health: {self.metrics['system_health']}")
        
        if self.metrics['last_signal']:
            print(f"Last Signal: {self.metrics['last_signal']}")
        
        print("="*80)
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get metrics summary for logging."""
        return self.metrics.copy()


class PaperTradingEngine:
    """Main paper trading engine that orchestrates the validated pipeline."""
    
    def __init__(self, config: PaperTradingConfig):
        """Initialize the paper trading engine."""
        self.config = config
        self.running = False
        self.shutdown_event = threading.Event()
        
        # Initialize dashboard
        self.dashboard = PaperTradingDashboard(config)
        
        # Initialize component factory
        self.component_factory = ComponentFactory(config.config)
        
        # Component references (will be set during initialization)
        self.exchange = None
        self.feature_registry = None
        self.feature_engineer = None
        self.model_predictor = None
        self.model_trainer = None
        self.order_executor = None
        self.production_model_predictor = None
        
        # Trading state
        self.current_position = 0.0
        self.current_balance = config.config['paper_trading']['initial_balance']
        self.trade_count = 0
        self.last_prediction_time = None
        
        logger.info("PaperTradingEngine initialized with ComponentFactory")
    
    def initialize_components(self) -> bool:
        """Initialize all trading system components using ComponentFactory."""
        try:
            logger.info("Initializing trading system components with ComponentFactory...")
            
            # Initialize all components through factory
            initialization_results = self.component_factory.initialize_all_components()
            
            # Check for critical component failures
            critical_components = ['database', 'feature_registry', 'feature_engineer']
            critical_success = all(initialization_results.get(comp, False) for comp in critical_components)
            
            if not critical_success:
                logger.error("Critical components failed to initialize")
                return False
            
            # Get component references
            self.exchange = self.component_factory.get_component('exchange')
            self.feature_registry = self.component_factory.get_component('feature_registry')
            self.feature_engineer = self.component_factory.get_component('feature_engineer')
            self.model_predictor = self.component_factory.get_component('model_predictor')
            self.model_trainer = self.component_factory.get_component('model_trainer')
            self.order_executor = self.component_factory.get_component('order_executor')
            
            # Validate exchange connection if available
            if self.exchange:
                try:
                    logger.info("Testing exchange connection...")
                    # Synchronize time with Bybit server
                    server_time = self.exchange.fetch_time()
                    local_time = int(time.time() * 1000)
                    time_diff = server_time - local_time
                    
                    logger.info(f"Server time: {datetime.fromtimestamp(server_time/1000)}")
                    logger.info(f"Local time: {datetime.fromtimestamp(local_time/1000)}")
                    logger.info(f"Time difference: {time_diff}ms")
                    
                    # Test connection
                    balance = self.exchange.fetch_balance()
                    logger.info("[OK] Exchange connection successful")
                    
                except Exception as e:
                    logger.warning(f"Exchange connection failed: {e}")
                    logger.info("Continuing without live exchange (paper trading mode)")
                    self.exchange = None
            
            # Initialize production model predictor
            try:
                production_config = {
                    'base_dir': self.config.config.get('base_dir', './trading_bot_data'),
                    'symbol': self.config.config.get('symbol', 'BTCUSDT'),
                    'confidence_threshold': self.config.config.get('ml_model', {}).get('confidence_threshold', 0.6)
                }
                
                self.production_model_predictor = ProductionModelPredictor(production_config)
                
                if self.production_model_predictor.is_model_available():
                    logger.info("✓ Production model predictor initialized successfully")
                    model_info = self.production_model_predictor.get_model_info()
                    logger.info(f"  Model horizons: {model_info['horizons']}")
                    logger.info(f"  Features: {len(model_info['features'])} features")
                else:
                    logger.warning("⚠ Production model not available - will use fallback predictions")
                    self.production_model_predictor = None
                    
            except Exception as e:
                logger.warning(f"Failed to initialize production model predictor: {e}")
                self.production_model_predictor = None
            
            # Log component status
            success_count = sum(initialization_results.values())
            total_count = len(initialization_results)
            
            logger.info(f"ComponentFactory initialization: {success_count}/{total_count} components successful")
            
            # List successful and failed components
            successful_components = [name for name, success in initialization_results.items() if success]
            failed_components = [name for name, success in initialization_results.items() if not success]
            
            if successful_components:
                logger.info(f"Successful components: {', '.join(successful_components)}")
            
            if failed_components:
                logger.warning(f"Failed components: {', '.join(failed_components)}")
            
            return critical_success
            
        except Exception as e:
            logger.error(f"[ERROR] ComponentFactory initialization failed: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return False
    
    def get_latest_market_data(self) -> Optional[pd.DataFrame]:
        """Get latest L2 market data for feature generation."""
        try:
            # Try to get live market data using public API (no auth required)
            live_data = self._fetch_live_market_data()
            if live_data is not None:
                logger.info(f"Using live market data: BTC ${live_data['mid_price'].iloc[-1]:.2f}")
                return live_data
            
            # Fallback to database data (but warn user)
            logger.warning("Using historical database data instead of live market data")
            if self.model_trainer is None:
                logger.error("Model trainer not available")
                return None
            
            # Get latest 100 rows for feature generation
            raw_data = self.model_trainer.load_raw_l2_data_from_db('BTCUSDT', limit=100)
            
            if raw_data.empty:
                logger.warning("No recent market data available")
                return None
            
            return raw_data
            
        except Exception as e:
            logger.error(f"Error loading market data: {e}")
            return None
    
    def _fetch_live_market_data(self) -> Optional[pd.DataFrame]:
        """Fetch live market data using Bybit public API."""
        try:
            import requests
            
            # Use Bybit public API (no auth required)
            url = "https://api.bybit.com/v5/market/orderbook"
            params = {"category": "linear", "symbol": "BTCUSDT", "limit": 10}
            
            response = requests.get(url, params=params, timeout=5)
            if response.status_code != 200:
                return None
            
            data = response.json()
            result = data.get('result', {})
            bids = result.get('b', [])  # Bybit uses 'b' for bids
            asks = result.get('a', [])  # Bybit uses 'a' for asks
            
            if not bids or not asks:
                return None
            
            # Create L2 data format
            l2_data = {
                'timestamp': datetime.now().isoformat(),
                'symbol': 'BTCUSDT',
                'exchange': 'bybit_public',
                'data_source': 'live_api',
            }
            
            # Add bid/ask levels (up to 5 levels)
            for i in range(min(5, len(bids))):
                l2_data[f'bid_price_{i+1}'] = float(bids[i][0])
                l2_data[f'bid_size_{i+1}'] = float(bids[i][1])
            
            for i in range(min(5, len(asks))):
                l2_data[f'ask_price_{i+1}'] = float(asks[i][0])
                l2_data[f'ask_size_{i+1}'] = float(asks[i][1])
            
            # Fill missing levels
            for i in range(len(bids), 5):
                l2_data[f'bid_price_{i+1}'] = 0.0
                l2_data[f'bid_size_{i+1}'] = 0.0
            
            for i in range(len(asks), 5):
                l2_data[f'ask_price_{i+1}'] = 0.0
                l2_data[f'ask_size_{i+1}'] = 0.0
            
            # Calculate basic features
            best_bid = l2_data['bid_price_1']
            best_ask = l2_data['ask_price_1']
            l2_data['mid_price'] = (best_bid + best_ask) / 2
            l2_data['spread'] = best_ask - best_bid
            
            # Create DataFrame with repeated data for windowing
            df = pd.DataFrame([l2_data] * 100)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            return df
            
        except Exception as e:
            logger.debug(f"Failed to fetch live market data: {e}")
            return None
    
    def _convert_orderbook_to_l2_format(self, orderbook: Dict) -> Optional[pd.DataFrame]:
        """Convert CCXT orderbook to our L2 data format."""
        try:
            bids = orderbook.get('bids', [])
            asks = orderbook.get('asks', [])
            
            if not bids or not asks:
                return None
            
            # Create basic L2 row
            data = {
                'timestamp': datetime.now().isoformat(),
                'symbol': 'BTCUSDT',
                'exchange': 'bybit',
                'data_source': 'live_api',
            }
            
            # Add bid/ask levels (up to 5 levels)
            for i in range(min(5, len(bids))):
                data[f'bid_price_{i+1}'] = float(bids[i][0])
                data[f'bid_size_{i+1}'] = float(bids[i][1])
            
            for i in range(min(5, len(asks))):
                data[f'ask_price_{i+1}'] = float(asks[i][0])
                data[f'ask_size_{i+1}'] = float(asks[i][1])
            
            # Fill missing levels with zeros
            for i in range(len(bids), 5):
                data[f'bid_price_{i+1}'] = 0.0
                data[f'bid_size_{i+1}'] = 0.0
            
            for i in range(len(asks), 5):
                data[f'ask_price_{i+1}'] = 0.0
                data[f'ask_size_{i+1}'] = 0.0
            
            # Calculate basic features
            best_bid = data['bid_price_1']
            best_ask = data['ask_price_1']
            data['mid_price'] = (best_bid + best_ask) / 2
            data['spread'] = best_ask - best_bid
            
            # Create DataFrame with current timestamp repeated for windowing
            df = pd.DataFrame([data] * 100)  # Repeat for windowing functions
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            return df
            
        except Exception as e:
            logger.error(f"Error converting orderbook to L2 format: {e}")
            return None
    
    def generate_features(self, raw_data: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Generate L2 features using Feature Registry."""
        try:
            if self.feature_registry is None:
                logger.error("Feature registry not available - falling back to basic calculation")
                return self._generate_features_fallback(raw_data)
            
            # Use Feature Registry to compute all features
            features_df = self.feature_registry.compute_all_features(raw_data)
            
            # Combine with original data
            result_df = raw_data.copy()
            for col in features_df.columns:
                result_df[col] = features_df[col]
            
            logger.debug(f"Generated {len(features_df.columns)} features using Feature Registry")
            
            return result_df
            
        except Exception as e:
            logger.error(f"Error generating features with registry: {e}")
            # Fallback to basic calculation
            return self._generate_features_fallback(raw_data)
    
    def _generate_features_fallback(self, raw_data: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Fallback feature generation method."""
        try:
            # Use same feature generation logic as integration test
            df_basic = raw_data.copy()
            
            # Basic spread features
            df_basic['spread_bps'] = (df_basic['spread'] / df_basic['mid_price']) * 10000
            
            # Basic volume features
            df_basic['total_bid_volume_5'] = (
                df_basic['bid_size_1'] + df_basic['bid_size_2'] + df_basic['bid_size_3'] + 
                df_basic['bid_size_4'] + df_basic['bid_size_5']
            )
            df_basic['total_ask_volume_5'] = (
                df_basic['ask_size_1'] + df_basic['ask_size_2'] + df_basic['ask_size_3'] + 
                df_basic['ask_size_4'] + df_basic['ask_size_5']
            )
            
            # Basic volatility
            df_basic['mid_price_return'] = df_basic['mid_price'].pct_change()
            df_basic['l2_volatility_10'] = df_basic['mid_price_return'].rolling(10, min_periods=2).std()
            df_basic['l2_volatility_50'] = df_basic['mid_price_return'].rolling(50, min_periods=5).std()
            
            # Basic imbalance features
            df_basic['order_book_imbalance_2'] = (
                (df_basic['bid_size_1'] + df_basic['bid_size_2'] - df_basic['ask_size_1'] - df_basic['ask_size_2']) /
                (df_basic['bid_size_1'] + df_basic['bid_size_2'] + df_basic['ask_size_1'] + df_basic['ask_size_2'] + 1e-8)
            )
            
            # Basic pressure features
            df_basic['bid_pressure'] = df_basic['total_bid_volume_5'] / (df_basic['total_bid_volume_5'] + df_basic['total_ask_volume_5'] + 1e-8)
            df_basic['ask_pressure'] = df_basic['total_ask_volume_5'] / (df_basic['total_bid_volume_5'] + df_basic['total_ask_volume_5'] + 1e-8)
            
            # Fill NaN values
            df_basic = df_basic.ffill().bfill().fillna(0)
            
            return df_basic
            
        except Exception as e:
            logger.error(f"Error in fallback feature generation: {e}")
            return None
    
    def get_prediction(self, features_df: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Get model prediction from features."""
        try:
            # Try production model first
            if self.production_model_predictor and self.production_model_predictor.is_model_available():
                logger.debug("Using production model predictor")
                prediction = self.production_model_predictor.predict(features_df)
                
                if prediction:
                    logger.info(f"Production model prediction: Signal={prediction['signal']}, "
                              f"Confidence={prediction['confidence']:.3f}, Source={prediction['source']}")
                    return prediction
                else:
                    logger.warning("Production model prediction failed, falling back to legacy predictor")
            
            # Try component factory model predictor
            if self.model_predictor and hasattr(self.model_predictor, 'predict'):
                logger.debug("Using component factory model predictor")
                recent_features = features_df.tail(1)
                prediction = self.model_predictor.predict(recent_features)
                
                if prediction:
                    return prediction
            
            # Fallback to mock prediction for paper trading
            logger.info("Using mock prediction as final fallback")
            
            # Get current price and calculate basic momentum signal
            current_price = features_df['mid_price'].iloc[-1]
            price_change = features_df['mid_price'].pct_change(10).iloc[-1]
            volatility = features_df['l2_volatility_10'].iloc[-1]
            
            # Simple momentum-based mock signal - make it more active for testing
            if abs(price_change) > 0.0001:  # 0.01% price move (more sensitive)
                signal = 1 if price_change > 0 else -1
                confidence = min(abs(price_change) * 2000, 0.9)  # Higher confidence scaling
            else:
                signal = 1 if np.random.random() > 0.7 else 0  # Random trades for testing
                confidence = 0.65  # Above threshold to trigger trades
            
            mock_prediction = {
                'signal': signal,
                'confidence': confidence,
                'prediction_value': price_change,
                'volatility_regime': 'normal' if volatility < 0.02 else 'high',
                'timestamp': datetime.now(),
                'horizons': [10, 50, 100],
                'source': 'mock_fallback',
                'current_price': current_price
            }
            
            return mock_prediction
            
        except Exception as e:
            logger.error(f"Error getting prediction: {e}")
            return None
    
    def execute_paper_trade(self, prediction: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Execute paper trade based on prediction."""
        try:
            signal = prediction.get('signal', 0)
            confidence = prediction.get('confidence', 0)
            current_price = prediction.get('current_price', 50000)
            
            # Check if signal meets confidence threshold
            min_confidence = self.config.config['ml_model']['confidence_threshold']
            if confidence < min_confidence:
                logger.info(f"Signal confidence {confidence:.3f} below threshold {min_confidence}")
                return None
            
            # Check daily trade limit
            if self.trade_count >= self.config.config['paper_trading']['max_daily_trades']:
                logger.warning("Daily trade limit reached")
                return None
            
            # Calculate position size based on confidence and risk limits
            max_position = self.config.config['paper_trading']['max_position_size']
            risk_pct = self.config.config['paper_trading']['risk_limit_pct']
            
            position_size = min(
                max_position * confidence,  # Scale by confidence
                (self.current_balance * risk_pct / 100) / current_price  # Risk-based sizing
            )
            
            if signal == 0 or position_size < self.config.get_symbol_config()['min_order_size']:
                return None
            
            # Create paper trade
            side = 'buy' if signal > 0 else 'sell'
            
            trade_details = {
                'timestamp': datetime.now(),
                'trade_id': f"paper_{int(time.time())}",
                'side': side,
                'symbol': 'BTC/USDT:USDT',
                'amount': position_size,
                'price': current_price,
                'type': 'market',  # Paper trade - assume immediate fill
                'confidence': confidence,
                'signal': signal,
                'prediction_value': prediction.get('prediction_value', 0),
                'status': 'filled',
                'pnl': 0.0,  # Will be calculated on close
                'source': 'paper_trading'
            }
            
            # Update position
            if side == 'buy':
                self.current_position += position_size
            else:
                self.current_position -= position_size
            
            # Update balance (subtract fees)
            fee = position_size * current_price * 0.0006  # 0.06% maker fee
            self.current_balance -= fee
            
            self.trade_count += 1
            
            logger.info(f"Paper trade executed: {side} {position_size:.4f} BTC at ${current_price:.2f}")
            logger.info(f"   Confidence: {confidence:.3f}, Position: {self.current_position:.4f} BTC")
            
            return trade_details
            
        except Exception as e:
            logger.error(f"Error executing paper trade: {e}")
            return None
    
    def run_trading_cycle(self):
        """Run one complete trading cycle."""
        try:
            logger.info("Starting trading cycle...")
            
            # 1. Get latest market data
            market_data = self.get_latest_market_data()
            if market_data is None:
                logger.warning("No market data available - skipping cycle")
                return
            
            # 2. Generate features
            features = self.generate_features(market_data)
            if features is None:
                logger.warning("Feature generation failed - skipping cycle")
                return
            
            # 3. Get prediction
            prediction = self.get_prediction(features)
            if prediction is None:
                logger.warning("Prediction failed - skipping cycle")
                return
            
            logger.info(f"Prediction: Signal={prediction['signal']}, Confidence={prediction['confidence']:.3f}")
            
            # 4. Execute trade if signal is strong enough
            trade = self.execute_paper_trade(prediction)
            if trade:
                self.dashboard.update_trade(trade)
            
            # 5. Update dashboard
            self.dashboard.update_performance(0, self.current_balance, self.current_position)
            self.dashboard.print_dashboard()
            
            self.last_prediction_time = datetime.now()
            
        except Exception as e:
            logger.error(f"Error in trading cycle: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
    
    def run(self):
        """Run the paper trading engine."""
        logger.info("Starting Paper Trading Engine")
        
        # Initialize components
        if not self.initialize_components():
            logger.error("[ERROR] Failed to initialize components - exiting")
            return False
        
        # Start main trading loop
        self.running = True
        cycle_interval = 30  # 30 seconds between cycles
        
        logger.info(f"Paper trading started - cycle interval: {cycle_interval}s")
        
        try:
            # Basic validation that production model is working before loop
            if self.production_model_predictor and self.production_model_predictor.is_model_available():
                logger.info("[OK] Production model is available and loaded")
            else:
                logger.warning("[WARNING] Production model not available - will use fallback predictions")

            while self.running and not self.shutdown_event.is_set():
                cycle_start = time.time()
                
                # Run trading cycle
                self.run_trading_cycle()
                
                # Wait for next cycle
                cycle_duration = time.time() - cycle_start
                sleep_time = max(0, cycle_interval - cycle_duration)
                
                if sleep_time > 0:
                    self.shutdown_event.wait(sleep_time)
        
        except KeyboardInterrupt:
            logger.info("Shutdown requested by user")
        except Exception as e:
            logger.error(f"[ERROR] Critical error in trading loop: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
        finally:
            self.shutdown()
        
        return True
    
    def shutdown(self):
        """Shutdown the trading engine gracefully."""
        logger.info("Shutting down Paper Trading Engine")
        self.running = False
        self.shutdown_event.set()
        
        # Shutdown ComponentFactory and all components
        if hasattr(self, 'component_factory'):
            self.component_factory.shutdown_all_components()
        
        # Print final dashboard
        self.dashboard.print_dashboard()
        
        # Save final metrics
        final_metrics = self.dashboard.get_metrics_summary()
        logger.info(f"Final Performance: Balance=${final_metrics['current_balance']:.2f}, "
                   f"P&L=${final_metrics['total_pnl']:.2f}, Trades={final_metrics['total_trades']}")


def setup_signal_handlers(engine: PaperTradingEngine):
    """Setup signal handlers for graceful shutdown."""
    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum} - initiating shutdown")
        engine.shutdown()
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)


def main():
    """Main function to run paper trading deployment."""
    
    print("BTC Paper Trading System - Day 5 Deployment")
    print("=" * 60)
    
    try:
        # Initialize configuration
        config = PaperTradingConfig()
        
        # Initialize trading engine
        engine = PaperTradingEngine(config)
        
        # Setup signal handlers for graceful shutdown
        setup_signal_handlers(engine)
        
        # Start paper trading
        success = engine.run()
        
        if success:
            logger.info("[OK] Paper trading deployment completed successfully")
            sys.exit(0)
        else:
            logger.error("[ERROR] Paper trading deployment failed")
            sys.exit(1)
            
    except Exception as e:
        logger.critical(f"Critical error in paper trading deployment: {e}")
        logger.critical(f"Traceback: {traceback.format_exc()}")
        sys.exit(2)


if __name__ == "__main__":
    main()