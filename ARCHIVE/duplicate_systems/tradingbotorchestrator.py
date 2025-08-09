# tradingbotorchestrator.py
# L2-Only Trading Bot Orchestrator - Phase 3 Implementation
# Converted from OHLCV+L2 to pure L2 order book orchestration

import os
import time
import pandas as pd
import numpy as np
import ccxt
import traceback
import logging
from datetime import datetime, timezone

# Import L2-only components
from advancedriskmanager import AdvancedRiskManager
from smartorderexecutor import SmartOrderExecutor
from datahandler import DataHandler
from featureengineer import FeatureEngineer
from labelgenerator import LabelGenerator
from modeltrainer import ModelTrainer
from modelpredictor import ModelPredictor
from strategybacktester import StrategyBacktester
from livesimulator import LiveSimulator
from visualizer import Visualizer

# L2-specific imports
from l2_price_reconstructor import L2PriceReconstructor
from l2_volatility_estimator import L2VolatilityEstimator


class TradingBotOrchestrator:
    """
    L2-Only Trading Bot Orchestrator for pure Level 2 order book strategies.
    
    This orchestrator manages the complete L2-only workflow:
    - L2 data streaming and processing
    - L2-derived feature engineering
    - L2-only model training and prediction
    - L2-based live simulation and trading
    """

    def __init__(self, config, api_key=None, api_secret=None,
                 global_library_flags=None, global_library_modules=None):
        """
        Initializes the L2-Only TradingBotOrchestrator.

        Args:
            config (dict): L2-only configuration dictionary
            api_key (str): Exchange API key
            api_secret (str): Exchange API secret
            global_library_flags (dict): Library availability flags
            global_library_modules (dict): Library modules
        """
        self.config = config
        self.api_key = api_key
        self.api_secret = api_secret
        self.exchange = None

        # Validate L2-only mode
        self.l2_only_mode = config.get('l2_only_mode', True)
        if not self.l2_only_mode:
            raise ValueError("TradingBotOrchestrator requires l2_only_mode=True for Phase 3 implementation")

        self.lib_flags = global_library_flags or {}
        self.lib_modules = global_library_modules or {}

        # L2-only data structures (no OHLCV)
        self.df_l2_features = pd.DataFrame()
        self.df_labeled_l2_features = pd.DataFrame()
        self.l2_data_buffer = []
        self.trained_model_booster = None
        self.trained_ensemble_models = None
        self.trained_features_list = []
        self.target_mean_for_prediction = None
        self.target_std_for_prediction = None
        self.backtest_results = pd.DataFrame()
        self.backtest_trades_log = pd.DataFrame()
        self.l2_performance_metrics = {}

        # L2 components
        self.l2_price_reconstructor = None
        self.l2_volatility_estimator = None

        # Initialize logging
        self.logger = logging.getLogger(__name__)

        self._initialize_exchange()
        self._initialize_l2_components()
        self._initialize_components()

        self.logger.info("L2-Only TradingBotOrchestrator initialized successfully")

    def _initialize_exchange(self):
        """Initialize exchange with L2 order book support validation."""
        try:
            import ccxt
            ccxt_module = ccxt
        except ImportError:
            ccxt_module = self.lib_modules.get('ccxt')
        if not ccxt_module:
            self.logger.error("CCXT library module not available. Cannot initialize exchange.")
            return

        try:
            exchange_config = {
                'enableRateLimit': True, 
                'options': {'adjustForTimeDifference': True}
            }
            if self.api_key and self.api_secret:
                exchange_config.update({'apiKey': self.api_key, 'secret': self.api_secret})

            exchange_name = self.config.get('exchange_name', 'bybit')
            if not hasattr(ccxt_module, exchange_name):
                self.logger.error(f"CCXT does not support exchange '{exchange_name}'.")
                return

            self.exchange = getattr(ccxt_module, exchange_name)(exchange_config)
            
            # Validate L2 support
            if not self.exchange.has.get('fetchL2OrderBook'):
                raise ValueError(f"Exchange '{exchange_name}' does not support L2 order book fetching required for L2-only mode")

            if self.config.get('exchange_testnet', False):
                self.exchange.set_sandbox_mode(True)
                self.logger.info(f"Exchange '{exchange_name}' initialized in SANDBOX/TESTNET mode with L2 support.")
            else:
                self.logger.info(f"Exchange '{exchange_name}' initialized in LIVE mode with L2 support.")

            self.exchange.load_markets()
            self.logger.info(f"Markets loaded for '{exchange_name}' with L2 order book support confirmed.")
            
        except Exception as e:
            self.logger.error(f"Exchange initialization failed for '{self.config.get('exchange_name', 'bybit')}': {e}")
            traceback.print_exc()
            self.exchange = None

    def _initialize_l2_components(self):
        """Initialize L2-specific components."""
        try:
            self.l2_price_reconstructor = L2PriceReconstructor(self.config)
            self.l2_volatility_estimator = L2VolatilityEstimator(self.config)
            self.logger.info("L2-specific components initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize L2 components: {e}")
            raise

    def _initialize_components(self):
        """Initialize all L2-compatible trading components."""
        if not self.exchange and not self.config.get('allow_no_exchange_init', False):
            self.logger.warning("Exchange not initialized. Some components may not function.")

        # Risk manager (L2-compatible)
        self.risk_manager = AdvancedRiskManager(
            self.config.get('risk_management', {})
        ) if AdvancedRiskManager else None

        # Order executor (L2-compatible)
        if SmartOrderExecutor and self.exchange:
            self.order_executor = SmartOrderExecutor(
                self.exchange, 
                self.config.get('execution', {})
            )
        else:
            self.order_executor = None
            if not self.config.get('allow_no_exchange_init', False):
                self.logger.warning("SmartOrderExecutor not initialized")

        # Data handler (L2-only mode)
        if DataHandler:
            # Allow DataHandler to initialize without exchange for data-only mode
            self.data_handler = DataHandler(self.config, self.exchange)
        else:
            self.data_handler = None

        # Feature engineer (L2-only mode)
        self.feature_engineer = FeatureEngineer(
            config=self.config,
            has_pandas_ta=self.lib_flags.get('HAS_PANDAS_TA', False),
            has_pyemd=self.lib_flags.get('HAS_PYEMD', False),
            has_scipy_hilbert=self.lib_flags.get('HAS_SCIPY_HILBERT', False),
            ta_module=self.lib_modules.get('ta'),
            emd_class=self.lib_modules.get('EMD'),
            hilbert_func=self.lib_modules.get('hilbert')
        ) if FeatureEngineer else None

        # Label generator (L2-compatible)
        self.label_generator = LabelGenerator(self.config) if LabelGenerator else None

        # Model trainer (L2-only mode)
        self.model_trainer = ModelTrainer(
            config=self.config,
            feature_list_all_defined=self.feature_engineer.all_defined_feature_columns if self.feature_engineer else [],
            has_optuna=self.lib_flags.get('HAS_OPTUNA', False),
            optuna_module=self.lib_modules.get('optuna')
        ) if ModelTrainer else None

        # Model predictor (L2-only mode)
        self.model_predictor = ModelPredictor(
            config=self.config,
            data_handler=self.data_handler,
            label_generator=self.label_generator
        ) if ModelPredictor else None

        # Strategy backtester (L2-compatible)
        self.backtester = StrategyBacktester(
            config=self.config,
            risk_manager=self.risk_manager,
            data_handler=self.data_handler,
            feature_engineer=self.feature_engineer,
            has_pandas_ta=self.lib_flags.get('HAS_PANDAS_TA', False),
            ta_module=self.lib_modules.get('ta')
        ) if StrategyBacktester and self.risk_manager else None

        # Live simulator (L2-only mode)
        if (LiveSimulator and self.exchange and self.data_handler and 
            self.feature_engineer and self.model_predictor and 
            self.risk_manager and self.order_executor):
            self.live_simulator = LiveSimulator(
                config=self.config, 
                exchange_api=self.exchange,
                data_handler=self.data_handler, 
                feature_engineer=self.feature_engineer,
                model_predictor=self.model_predictor, 
                risk_manager=self.risk_manager,
                order_executor=self.order_executor
            )
        else:
            self.live_simulator = None
            self.logger.warning("LiveSimulator not initialized - missing required components")

        # Visualizer (L2-compatible)
        self.visualizer = Visualizer(
            config=self.config,
            has_matplotlib=self.lib_flags.get('HAS_MATPLOTLIB', False), 
            plt_module=self.lib_modules.get('plt'),
            has_shap=self.lib_flags.get('HAS_SHAP', False), 
            shap_module=self.lib_flags.get('shap'),
            has_pyemd=self.lib_flags.get('HAS_PYEMD', False), 
            emd_class=self.lib_modules.get('EMD'),
            has_scipy_hilbert=self.lib_flags.get('HAS_SCIPY_HILBERT', False), 
            hilbert_func=self.lib_modules.get('hilbert')
        ) if Visualizer else None

        # Component validation
        missing_components = []
        if not self.data_handler:
            missing_components.append("DataHandler")
        if not self.feature_engineer:
            missing_components.append("FeatureEngineer")
        if not self.model_trainer:
            missing_components.append("ModelTrainer")
        if not self.model_predictor:
            missing_components.append("ModelPredictor")
        if not self.live_simulator:
            missing_components.append("LiveSimulator")

        if missing_components:
            self.logger.warning(f"Missing L2 components: {', '.join(missing_components)}")

    def prepare_l2_data_for_training(self, l2_data_input=None, save_features=True):
        """
        Prepares L2 data for training by processing order book snapshots,
        generating L2 features, and creating labels.
        
        Args:
            l2_data_input (pd.DataFrame): Optional L2 data input
            save_features (bool): Whether to save generated features
            
        Returns:
            bool: Success status
        """
        if not self.data_handler or not self.feature_engineer or not self.label_generator:
            self.logger.error("L2 data processing components not initialized.")
            return False

        self.logger.info("--- Starting L2 Data Preparation ---")

        try:
            # Load L2 data
            if l2_data_input is not None and not l2_data_input.empty:
                self.logger.info("Using provided L2 DataFrame as input for data preparation.")
                current_l2_data = l2_data_input
            else:
                self.logger.info("Loading L2 data from database/historical sources.")
                # Load L2 data from database or historical sources
                current_l2_data = self.data_handler.load_l2_historical_data()
                
            if current_l2_data.empty:
                self.logger.error("Failed to load L2 data for training preparation.")
                return False

            # Generate L2-only features
            self.logger.info("Generating L2-only features...")
            current_df_features = self.feature_engineer.generate_l2_only_features(
                current_l2_data, 
                save=save_features
            )
            
            if current_df_features.empty:
                self.logger.error("Failed to engineer L2 features.")
                self.df_l2_features = pd.DataFrame()
                return False
                
            self.df_l2_features = current_df_features

            # Generate labels from L2-derived price series
            self.logger.info("Generating labels from L2-derived price series...")
            current_df_labeled, mean_val, std_val = self.label_generator.generate_labels(
                self.df_l2_features
            )
            
            if current_df_labeled.empty:
                self.logger.error("Failed to generate labels from L2 features.")
                self.df_labeled_l2_features = pd.DataFrame()
                return False
                
            self.df_labeled_l2_features = current_df_labeled

            # Store L2-derived scaling parameters
            self.target_mean_for_prediction = mean_val
            self.target_std_for_prediction = std_val
            
            if self.model_predictor:
                self.model_predictor.set_scaling_params(mean_val, std_val)

            self.logger.info(f"L2 data preparation complete. Features: {len(self.df_l2_features)}, "
                           f"Labeled: {len(self.df_labeled_l2_features)}")
            return True

        except Exception as e:
            self.logger.error(f"Error in L2 data preparation: {e}")
            traceback.print_exc()
            return False

    def train_l2_model(self, df_training_data=None, save_model=True):
        """
        Trains L2-only model on L2-derived features.
        
        Args:
            df_training_data (pd.DataFrame): Optional training data
            save_model (bool): Whether to save the trained model
            
        Returns:
            bool: Success status
        """
        data_to_train_on = (df_training_data if df_training_data is not None and not df_training_data.empty 
                           else self.df_labeled_l2_features)

        if not self.model_trainer or data_to_train_on is None or data_to_train_on.empty:
            self.logger.error("ModelTrainer not initialized or no L2 labeled data available for training.")
            return False

        self.logger.info("--- Starting L2-Only Model Training ---")

        try:
            # Validate L2-only features
            l2_feature_count = sum(1 for col in data_to_train_on.columns 
                                 if any(l2_prefix in col for l2_prefix in 
                                       ['bid_ask_spread', 'weighted_mid_price', 'microprice', 
                                        'order_book_imbalance', 'price_impact', 'l2_volatility']))
            
            self.logger.info(f"Training with {l2_feature_count} L2-derived features")

            train_ensemble = self.config.get('train_ensemble', False)

            # Pass L2-derived scaling parameters
            self.config['target_mean_for_prediction'] = self.target_mean_for_prediction
            self.config['target_std_for_prediction'] = self.target_std_for_prediction

            if train_ensemble:
                self.logger.info("Training L2-only ensemble model...")
                ensemble_models, features_used = self.model_trainer.train_ensemble_model(
                    data_to_train_on, save=save_model
                )
                
                if ensemble_models:
                    self.trained_ensemble_models = ensemble_models
                    self.trained_features_list = features_used
                    
                    # Load into predictor
                    if self.model_predictor:
                        self.model_predictor.model_object = ensemble_models
                        self.model_predictor.trained_features = features_used
                        self.model_predictor.is_ensemble = True
                        
                    self.logger.info("L2-only ensemble model training successful.")
                    return True
            else:
                self.logger.info("Training L2-only standard model...")
                booster, features_used = self.model_trainer.train_standard_model(
                    data_to_train_on, save=save_model
                )
                
                if booster:
                    self.trained_model_booster = booster
                    self.trained_features_list = features_used
                    
                    # Load into predictor
                    if self.model_predictor:
                        self.model_predictor.model_object = booster
                        self.model_predictor.trained_features = features_used
                        self.model_predictor.is_ensemble = False
                        
                    self.logger.info("L2-only standard model training successful.")
                    return True

            self.logger.error("L2-only model training failed.")
            return False

        except Exception as e:
            self.logger.error(f"Error in L2 model training: {e}")
            traceback.print_exc()
            return False

    def run_l2_backtest(self, df_backtest_data=None, load_latest_model=True):
        """
        Runs L2-only backtesting process on L2-derived features.
        
        Args:
            df_backtest_data (pd.DataFrame): Optional L2 backtest data
            load_latest_model (bool): Whether to load the latest trained model
            
        Returns:
            tuple: (backtest_results, trades_log)
        """
        if not self.backtester or not self.model_predictor:
            self.logger.error("Backtester or ModelPredictor not initialized for L2 backtesting.")
            return pd.DataFrame(), pd.DataFrame()

        # Use L2 features for backtesting
        data_for_backtest = (df_backtest_data if df_backtest_data is not None and not df_backtest_data.empty 
                           else self.df_l2_features)

        if data_for_backtest.empty:
            self.logger.error("No L2 feature data available for backtesting.")
            # Try to load from L2 prepared data
            try:
                if hasattr(self.feature_engineer, 'l2_prepared_data_path'):
                    l2_data_path = self.feature_engineer.l2_prepared_data_path
                    if os.path.exists(l2_data_path):
                        self.logger.info("Loading L2 prepared data for backtest.")
                        data_for_backtest = pd.read_csv(l2_data_path, parse_dates=['timestamp'])
                        if 'timestamp' in data_for_backtest.columns:
                            if data_for_backtest['timestamp'].dt.tz is None:
                                data_for_backtest['timestamp'] = data_for_backtest['timestamp'].dt.tz_localize('utc')
                            else:
                                data_for_backtest['timestamp'] = data_for_backtest['timestamp'].dt.tz_convert('utc')
            except Exception as e:
                self.logger.error(f"Error loading L2 prepared data for backtest: {e}")
                return pd.DataFrame(), pd.DataFrame()
                
            if data_for_backtest.empty:
                return pd.DataFrame(), pd.DataFrame()

        self.logger.info("--- Starting L2-Only Backtest ---")
        
        try:
            if load_latest_model:
                use_ensemble_for_pred = self.config.get('use_ensemble_for_backtest', 
                                                       self.config.get('train_ensemble', False))
                if not self.model_predictor.load_model_and_features(load_ensemble=use_ensemble_for_pred):
                    self.logger.error("Failed to load L2 model for backtest predictions.")
                    return pd.DataFrame(), pd.DataFrame()

            # Ensure L2-derived scaling params are set
            if self.model_predictor.target_mean is None or self.model_predictor.target_std is None:
                self.logger.warning("L2 scaling params not set in ModelPredictor. Attempting to use orchestrator values.")
                if self.target_mean_for_prediction is not None and self.target_std_for_prediction is not None:
                    self.model_predictor.set_scaling_params(self.target_mean_for_prediction, 
                                                           self.target_std_for_prediction)
                elif not self.model_predictor._ensure_scaling_info():
                    self.logger.warning("Could not ensure L2 scaling info for backtest. Predictions may be affected.")

            # Generate L2-based signals
            df_with_signals = self.model_predictor.predict_signals(
                data_for_backtest,
                use_ensemble=self.config.get('use_ensemble_for_backtest', 
                                           self.config.get('train_ensemble', False))
            )
            
            if df_with_signals is None or df_with_signals.empty:
                self.logger.error("Failed to generate L2-based signals for backtest.")
                return pd.DataFrame(), pd.DataFrame()

            # Run L2-enhanced backtest
            self.backtest_results, self.backtest_trades_log = self.backtester.run_backtest(df_with_signals)

            if self.backtest_results is not None and not self.backtest_results.empty:
                self.logger.info("L2-only backtest complete.")
                
                # L2-enhanced visualization
                if self.visualizer:
                    initial_bal = self.backtester.initial_balance
                    final_bal = self.backtest_results['equity'].iloc[-1] if not self.backtest_results['equity'].empty else initial_bal
                    return_pct = ((final_bal - initial_bal) / initial_bal * 100) if initial_bal != 0 else 0
                    self.visualizer.plot_equity_curve(self.backtest_results, initial_bal, final_bal, 
                                                     return_pct, "L2-Only Backtest")
                    
                return self.backtest_results, self.backtest_trades_log
            else:
                self.logger.error("L2 backtest failed to produce results.")
                return pd.DataFrame(), pd.DataFrame()
                
        except Exception as e:
            self.logger.error(f"Error in L2 backtest: {e}")
            traceback.print_exc()
            return pd.DataFrame(), pd.DataFrame()

    def run_l2_live_simulation(self):
        """
        Runs L2-only live simulation with real-time order book processing.
        
        Returns:
            bool: Success status
        """
        if not self.live_simulator:
            self.logger.error("L2 LiveSimulator not initialized.")
            return False

        self.logger.info("--- Starting L2-Only Live Simulation ---")
        
        try:
            # Validate L2-only model is loaded
            use_ensemble_for_sim = self.config.get('use_ensemble_for_simulation', False)
            if not self.model_predictor.model_object:
                if not self.model_predictor.load_model_and_features(load_ensemble=use_ensemble_for_sim):
                    self.logger.error("Failed to load L2 model for live simulation.")
                    return False

            # Start L2 live simulation
            initial_equity = self.config.get('initial_balance', 10000)
            threshold = self.config.get('simulation_threshold', 
                                      self.config.get('prediction_threshold', 0.15))
            commission_pct = self.config.get('commission_pct', 0.0006)
            leverage = self.config.get('leverage', 2)

            self.live_simulator.start_live_simulation(
                initial_equity=initial_equity,
                threshold=threshold,
                commission_pct=commission_pct,
                leverage=leverage
            )

            self.logger.info("L2-only live simulation started successfully.")
            return True
            
        except Exception as e:
            self.logger.error(f"Error starting L2 live simulation: {e}")
            traceback.print_exc()
            return False

    def stop_l2_live_simulation(self):
        """
        Stops the L2-only live simulation.
        
        Returns:
            dict: L2 performance metrics
        """
        if not self.live_simulator:
            self.logger.warning("L2 LiveSimulator not initialized.")
            return {}

        try:
            self.live_simulator.stop_live_simulation()
            
            # Get L2 performance metrics
            l2_metrics = self.live_simulator.get_l2_performance_metrics()
            self.l2_performance_metrics = l2_metrics
            
            self.logger.info("L2-only live simulation stopped.")
            self.logger.info(f"L2 Performance Summary: {l2_metrics}")
            
            return l2_metrics
            
        except Exception as e:
            self.logger.error(f"Error stopping L2 live simulation: {e}")
            return {}

    def visualize_l2_results(self):
        """
        Visualizes L2-only strategy results and performance metrics.
        """
        if not self.visualizer:
            self.logger.warning("Visualizer not initialized for L2 results.")
            return

        try:
            self.logger.info("--- Generating L2-Only Visualizations ---")

            # Visualize L2 backtest results
            if not self.backtest_results.empty:
                self.logger.info("Plotting L2 backtest equity curve...")
                initial_bal = self.config.get('initial_balance', 10000)
                final_bal = self.backtest_results['equity'].iloc[-1]
                return_pct = ((final_bal - initial_bal) / initial_bal * 100) if initial_bal != 0 else 0
                
                self.visualizer.plot_equity_curve(
                    self.backtest_results, initial_bal, final_bal, 
                    return_pct, "L2-Only Strategy Backtest"
                )

            # Visualize L2 live simulation results
            if self.live_simulator:
                sim_equity_data = self.live_simulator.get_simulation_equity_data()
                if not sim_equity_data.empty:
                    self.logger.info("Plotting L2 live simulation equity curve...")
                    initial_sim_bal = sim_equity_data['equity'].iloc[0] if len(sim_equity_data) > 0 else 10000
                    final_sim_bal = sim_equity_data['equity'].iloc[-1] if len(sim_equity_data) > 0 else initial_sim_bal
                    sim_return_pct = ((final_sim_bal - initial_sim_bal) / initial_sim_bal * 100) if initial_sim_bal != 0 else 0
                    
                    self.visualizer.plot_equity_curve(
                        sim_equity_data, initial_sim_bal, final_sim_bal,
                        sim_return_pct, "L2-Only Live Simulation"
                    )

            # Visualize L2 feature importance
            if self.trained_model_booster and self.trained_features_list:
                self.logger.info("Plotting L2 feature importance...")
                self.visualizer.plot_feature_importance(
                    self.trained_model_booster, self.trained_features_list,
                    title="L2-Only Feature Importance"
                )

            # Visualize L2 performance metrics
            if self.l2_performance_metrics:
                self.logger.info("Plotting L2 performance metrics...")
                self._plot_l2_performance_metrics()

            self.logger.info("L2-only visualizations complete.")
            
        except Exception as e:
            self.logger.error(f"Error in L2 visualization: {e}")
            traceback.print_exc()

    def _plot_l2_performance_metrics(self):
        """Plot L2-specific performance metrics."""
        try:
            import matplotlib.pyplot as plt
            
            metrics = self.l2_performance_metrics
            
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('L2-Only Strategy Performance Metrics', fontsize=16)
            
            # Latency metrics
            ax1.bar(['Avg Processing', 'P99 Processing', 'Avg Feature Gen', 'Avg Prediction'], 
                   [metrics.get('avg_processing_latency_ms', 0),
                    metrics.get('p99_processing_latency_ms', 0),
                    metrics.get('avg_feature_generation_ms', 0),
                    metrics.get('avg_prediction_time_ms', 0)])
            ax1.set_title('L2 Processing Latency (ms)')
            ax1.set_ylabel('Milliseconds')
            
            # L2 updates and buffer
            ax2.bar(['Total L2 Updates', 'Buffer Size'], 
                   [metrics.get('total_l2_updates', 0),
                    metrics.get('l2_buffer_size', 0)])
            ax2.set_title('L2 Data Processing Volume')
            ax2.set_ylabel('Count')
            
            # Microstructure alpha
            ax3.bar(['Microstructure Alpha'], [metrics.get('microstructure_alpha', 0)])
            ax3.set_title('L2 Microstructure Alpha')
            ax3.set_ylabel('Alpha Value')
            
            # Summary metrics
            summary_text = f"""
L2 Performance Summary:
• Total L2 Updates: {metrics.get('total_l2_updates', 0):,}
• Avg Latency: {metrics.get('avg_processing_latency_ms', 0):.2f}ms
• Microstructure Alpha: {metrics.get('microstructure_alpha', 0):.4f}
• Buffer Utilization: {metrics.get('l2_buffer_size', 0)}
            """
            ax4.text(0.1, 0.5, summary_text, fontsize=12, verticalalignment='center')
            ax4.set_xlim(0, 1)
            ax4.set_ylim(0, 1)
            ax4.axis('off')
            ax4.set_title('L2 Strategy Summary')
            
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            self.logger.error(f"Error plotting L2 performance metrics: {e}")

    def run_l2_full_workflow(self):
        """
        Runs the complete L2-only trading workflow.
        
        Returns:
            bool: Success status
        """
        self.logger.info("=== Starting L2-Only Full Workflow ===")
        
        try:
            # Step 1: Prepare L2 data for training
            self.logger.info("Step 1: Preparing L2 data for training...")
            if not self.prepare_l2_data_for_training():
                self.logger.error("L2 data preparation failed.")
                return False

            # Step 2: Train L2-only model
            self.logger.info("Step 2: Training L2-only model...")
            if not self.train_l2_model():
                self.logger.error("L2 model training failed.")
                return False

            # Step 3: Run L2 backtest
            self.logger.info("Step 3: Running L2-only backtest...")
            backtest_results, trades_log = self.run_l2_backtest()
            if backtest_results.empty:
                self.logger.error("L2 backtest failed.")
                return False

            # Step 4: Visualize L2 results
            self.logger.info("Step 4: Generating L2 visualizations...")
            self.visualize_l2_results()

            # Step 5: Optional live simulation
            run_live_sim = self.config.get('run_simulation_flag', False)
            if run_live_sim:
                self.logger.info("Step 5: Starting L2 live simulation...")
                if self.run_l2_live_simulation():
                    self.logger.info("L2 live simulation started. Use stop_l2_live_simulation() to stop.")
                else:
                    self.logger.error("L2 live simulation failed to start.")

            self.logger.info("=== L2-Only Full Workflow Complete ===")
            
            # Summary
            self._log_l2_workflow_summary()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error in L2 full workflow: {e}")
            traceback.print_exc()
            return False

    def _log_l2_workflow_summary(self):
        """Log L2 workflow summary."""
        try:
            summary = {
                'l2_features_count': len(self.df_l2_features),
                'l2_labeled_count': len(self.df_labeled_l2_features),
                'trained_features_count': len(self.trained_features_list),
                'backtest_trades': len(self.backtest_trades_log) if not self.backtest_trades_log.empty else 0,
                'model_type': 'ensemble' if self.config.get('train_ensemble', False) else 'standard',
                'l2_only_mode': self.l2_only_mode
            }
            
            self.logger.info("L2 Workflow Summary:")
            for key, value in summary.items():
                self.logger.info(f"  {key}: {value}")
                
        except Exception as e:
            self.logger.error(f"Error logging L2 workflow summary: {e}")

    def get_l2_strategy_status(self):
        """
        Get comprehensive L2 strategy status.
        
        Returns:
            dict: L2 strategy status and metrics
        """
        try:
            status = {
                'l2_only_mode': self.l2_only_mode,
                'exchange_connected': self.exchange is not None,
                'l2_support_confirmed': self.exchange.has.get('fetchL2OrderBook') if self.exchange else False,
                'components_initialized': {
                    'data_handler': self.data_handler is not None,
                    'feature_engineer': self.feature_engineer is not None,
                    'model_trainer': self.model_trainer is not None,
                    'model_predictor': self.model_predictor is not None,
                    'live_simulator': self.live_simulator is not None,
                    'l2_price_reconstructor': self.l2_price_reconstructor is not None,
                    'l2_volatility_estimator': self.l2_volatility_estimator is not None
                },
                'data_status': {
                    'l2_features_available': not self.df_l2_features.empty,
                    'l2_features_count': len(self.df_l2_features),
                    'labeled_data_available': not self.df_labeled_l2_features.empty,
                    'labeled_data_count': len(self.df_labeled_l2_features)
                },
                'model_status': {
                    'model_trained': self.trained_model_booster is not None or self.trained_ensemble_models is not None,
                    'model_type': 'ensemble' if self.trained_ensemble_models else 'standard' if self.trained_model_booster else 'none',
                    'trained_features_count': len(self.trained_features_list),
                    'scaling_params_set': self.target_mean_for_prediction is not None and self.target_std_for_prediction is not None
                },
                'simulation_status': {
                    'live_simulator_running': self.live_simulator.simulation_running if self.live_simulator else False,
                    'backtest_completed': not self.backtest_results.empty,
                    'trades_logged': len(self.backtest_trades_log) if not self.backtest_trades_log.empty else 0
                },
                'l2_performance_metrics': self.l2_performance_metrics
            }
            
            return status
            
        except Exception as e:
            self.logger.error(f"Error getting L2 strategy status: {e}")
            return {'error': str(e)}