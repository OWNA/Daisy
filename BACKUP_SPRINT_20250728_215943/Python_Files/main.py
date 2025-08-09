#!/usr/bin/env python3
"""
Main entry point for L2-only Bitcoin trading system
Single file to orchestrate all operations
"""

import sys
import os
import yaml
import logging
import argparse
from datetime import datetime
import pandas as pd
import ccxt

# Core components
from database import TradingDatabase
from l2_data_collector import L2DataCollector
from datahandler import DataHandler
from featureengineer import FeatureEngineer
from labelgenerator import LabelGenerator
from modeltrainer import ModelTrainer
from modelpredictor import ModelPredictor
from advancedriskmanager import AdvancedRiskManager
from smartorderexecutor import SmartOrderExecutor

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TradingSystem:
    """Simple trading system orchestrator"""
    
    def __init__(self, config_path='config.yaml'):
        """Initialize trading system with config"""
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize database
        self.db = TradingDatabase(self.config.get('database_path', './trading_bot.db'))
        
        # Initialize exchange (if needed)
        self.exchange = None
        if self.config.get('exchange', {}).get('name'):
            self._init_exchange()
    
    def _init_exchange(self):
        """Initialize exchange connection"""
        exchange_config = self.config.get('exchange', {})
        exchange_name = exchange_config.get('name', 'bybit')
        
        try:
            exchange_class = getattr(ccxt, exchange_name)
            self.exchange = exchange_class({
                'apiKey': os.getenv('BYBIT_API_KEY', ''),
                'secret': os.getenv('BYBIT_API_SECRET', ''),
                'enableRateLimit': True,
                'options': {
                    'defaultType': exchange_config.get('market_type', 'linear')
                }
            })
            
            if exchange_config.get('testnet', True):
                self.exchange.set_sandbox_mode(True)
                
            logger.info(f"Exchange {exchange_name} initialized")
        except Exception as e:
            logger.error(f"Failed to initialize exchange: {e}")
            self.exchange = None
    
    def collect_data(self, duration_minutes=5):
        """Collect L2 data from Bybit"""
        logger.info(f"Starting L2 data collection for {duration_minutes} minutes")
        
        collector = L2DataCollector(
            self.config,
            bot_base_dir='./'
        )
        
        try:
            # Run collection
            success = collector.start_collection(
                duration_minutes=duration_minutes,
                unit='minutes'
            )
            
            if success:
                logger.info("L2 data collection completed successfully")
            else:
                logger.error("L2 data collection failed")
                
        except Exception as e:
            logger.error(f"Error during data collection: {e}")
            return False
        
        return True
    
    def train_model(self):
        """Train model on collected L2 data"""
        logger.info("Starting model training pipeline")
        
        try:
            # Initialize components
            data_handler = DataHandler(self.config, self.exchange)
            feature_engineer = FeatureEngineer(self.config)
            label_generator = LabelGenerator(self.config)
            model_trainer = ModelTrainer(self.config)
            
            # Load L2 data
            logger.info("Loading L2 data...")
            df = data_handler.load_and_prepare_historical_data(
                fetch_ohlcv_limit=50000,  # This becomes the limit for L2 data
                use_historical_l2=True
            )
            
            if df is None or df.empty:
                logger.error("No L2 data available for training")
                return False
            
            logger.info(f"Loaded {len(df)} L2 records")
            
            # Generate features
            logger.info("Generating L2 features...")
            df_features = feature_engineer.generate_features(df)
            
            if 'close' not in df_features.columns and 'mid_price' in df_features.columns:
                # Add close column for label generation (uses mid_price in L2-only mode)
                df_features['close'] = df_features['mid_price']
            
            if df_features.empty:
                logger.error("Feature generation failed")
                return False
            
            # Save features to database
            logger.info("Saving features to database...")
            # TODO: Implement feature persistence
            
            # Generate labels
            logger.info("Generating labels...")
            df_labeled, target_mean, target_std = label_generator.generate_labels(df_features)
            
            if df_labeled.empty:
                logger.error("Label generation failed")
                return False
            
            # Train model
            logger.info("Training LightGBM model...")
            booster, features = model_trainer.train_model(df_labeled)
            
            if booster is None:
                logger.error("Model training failed")
                return False
            
            # Model is already saved by ModelTrainer, just save scaling params
            # Match ModelTrainer's filename pattern and directory
            safe_symbol = self.config['symbol'].replace('/', '_').replace(':', '')
            base_dir = self.config.get('base_dir', './')
            model_path = os.path.join(
                base_dir,
                f"lgbm_model_{safe_symbol}_l2_only.txt"
            )
            
            # Save scaling parameters
            scaling_params = {
                'target_mean': float(target_mean),
                'target_std': float(target_std),
                'features': features
            }
            
            import json
            scaling_path = model_path.replace('.txt', '_scaling.json')
            with open(scaling_path, 'w') as f:
                json.dump(scaling_params, f)
            
            logger.info(f"Model saved by ModelTrainer to {model_path}")
            logger.info(f"Scaling parameters saved to {scaling_path}")
            
            logger.info("Model training completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error during model training: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def trade(self, paper_trading=True):
        """Run live trading"""
        logger.info(f"Starting {'paper' if paper_trading else 'live'} trading")
        
        if not self.exchange:
            logger.error("Exchange not initialized")
            return False
        
        try:
            # Initialize components
            data_handler = DataHandler(self.config, self.exchange)
            feature_engineer = FeatureEngineer(self.config)
            model_predictor = ModelPredictor(self.config, data_handler)
            risk_manager = AdvancedRiskManager(self.config)
            order_executor = SmartOrderExecutor(self.exchange, self.config)
            
            # Load model
            # Match ModelTrainer's filename pattern and directory
            safe_symbol = self.config['symbol'].replace('/', '_').replace(':', '')
            base_dir = self.config.get('base_dir', './')
            model_path = os.path.join(
                base_dir,
                f"lgbm_model_{safe_symbol}_l2_only.txt"
            )
            
            if not os.path.exists(model_path):
                logger.error(f"Model not found at {model_path}. Run 'train' first.")
                return False
            
            # Load scaling parameters
            import json
            scaling_path = model_path.replace('.txt', '_scaling.json')
            with open(scaling_path, 'r') as f:
                scaling_params = json.load(f)
            
            # Load model and features
            features_path = model_path.replace('lgbm_model_', 'model_features_').replace('.txt', '.json')
            success = model_predictor.load_model_and_features(
                model_file_path=model_path,
                features_file_path=features_path
            )
            
            if not success:
                logger.error("Failed to load model")
                return False
                
            model_predictor.set_scaling_params(
                scaling_params['target_mean'],
                scaling_params['target_std']
            )
            
            logger.info("Model loaded successfully")
            
            # Trading loop
            position = 0
            balance = self.config.get('initial_balance', 10000)
            
            while True:
                try:
                    # Fetch latest L2 data
                    orderbook = self.exchange.fetch_l2_order_book(
                        self.config['symbol'],
                        limit=self.config.get('l2_websocket_depth', 50)
                    )
                    
                    # Convert to DataFrame format
                    l2_data = data_handler._process_l2_snapshot(orderbook)
                    
                    # Generate features
                    features = feature_engineer.generate_features(
                        pd.DataFrame([l2_data])
                    )
                    
                    # Get prediction
                    signals = model_predictor.predict_signals(features)
                    
                    if signals is None or signals.empty:
                        logger.warning("No signals generated")
                        continue
                    
                    # Extract prediction values for debugging
                    pred_scaled = signals['pred_scaled'].iloc[-1] if 'pred_scaled' in signals.columns else None
                    pred_unscaled = signals['pred_unscaled_target'].iloc[-1] if 'pred_unscaled_target' in signals.columns else None
                    
                    # Get the latest signal value
                    if 'signal' in signals.columns:
                        prediction = signals['signal'].iloc[-1]
                    else:
                        # If no signal column, use the first numeric column
                        prediction = signals.iloc[-1, 0]
                    
                    # Log prediction details every 10 iterations
                    if not hasattr(self, '_trade_iter'):
                        self._trade_iter = 0
                    self._trade_iter += 1
                    
                    if self._trade_iter % 10 == 0:
                        logger.info(f"Prediction details - Scaled: {pred_scaled:.6f}, Unscaled: {pred_unscaled:.6f}, Signal: {prediction}, Price: ${l2_data['mid_price']:.2f}")
                    
                    # Risk management
                    if prediction != 0:
                        # Sanity check for Bitcoin price
                        if l2_data['mid_price'] < 10000 or l2_data['mid_price'] > 200000:
                            logger.warning(f"Unrealistic BTC price detected: ${l2_data['mid_price']:.2f}, skipping trade")
                            continue
                            
                        # Calculate volatility estimate (simplified for now)
                        volatility_pct = 0.02  # 2% daily volatility estimate for BTC
                        
                        size = risk_manager.calculate_position_size(
                            balance,
                            volatility_pct,
                            entry_price=l2_data['mid_price']
                        )
                        
                        if size > 0:
                            # Convert USD size to BTC
                            size_btc = size / l2_data['mid_price']
                            
                            # Execute order
                            if paper_trading:
                                logger.info(f"PAPER TRADE: {'BUY' if prediction > 0 else 'SELL'} signal, size: {size_btc:.6f} BTC (${size:.2f}), price: ${l2_data['mid_price']:.2f}")
                                # Update balance for paper trading
                                if prediction > 0:  # Buy
                                    position += size_btc
                                    balance -= size
                                else:  # Sell
                                    position -= size_btc
                                    balance += size
                                logger.info(f"Position: {position:.6f} BTC, Balance: ${balance:.2f}")
                            else:
                                order = order_executor.execute_order(
                                    self.config['symbol'],
                                    'buy' if prediction > 0 else 'sell',
                                    size,
                                    l2_data['mid_price']
                                )
                                if order:
                                    logger.info(f"Order executed: {order}")
                    
                    # Sleep before next iteration
                    import time
                    time.sleep(1)
                    
                except KeyboardInterrupt:
                    logger.info("Trading stopped by user")
                    break
                except Exception as e:
                    logger.error(f"Trading error: {e}")
                    import time
                    time.sleep(5)
            
        except Exception as e:
            logger.error(f"Failed to start trading: {e}")
            return False
        
        return True
    
    def backtest(self):
        """Run backtest on historical data"""
        logger.info("Backtesting not yet implemented in minimal version")
        return False


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='L2 Bitcoin Trading System')
    parser.add_argument('command', 
                       choices=['collect', 'train', 'trade', 'backtest'],
                       help='Command to execute')
    parser.add_argument('--duration', type=int, default=5,
                       help='Data collection duration in minutes')
    parser.add_argument('--paper', action='store_true',
                       help='Run in paper trading mode')
    parser.add_argument('--config', default='config.yaml',
                       help='Path to config file')
    
    args = parser.parse_args()
    
    # Initialize system
    system = TradingSystem(args.config)
    
    # Execute command
    if args.command == 'collect':
        success = system.collect_data(args.duration)
    elif args.command == 'train':
        success = system.train_model()
    elif args.command == 'trade':
        success = system.trade(paper_trading=args.paper)
    elif args.command == 'backtest':
        success = system.backtest()
    else:
        logger.error(f"Unknown command: {args.command}")
        success = False
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()