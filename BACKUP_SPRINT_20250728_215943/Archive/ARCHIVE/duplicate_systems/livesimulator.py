# livesimulator.py
# L2-Only Live Simulation System - Phase 3 Implementation
# Converted from OHLCV-based to pure L2 order book simulation

import os
import threading
import traceback
import pandas as pd
import ccxt
import time
import json
import numpy as np
from datetime import datetime, timezone
import logging

# L2-only imports
from l2_price_reconstructor import L2PriceReconstructor
from l2_volatility_estimator import L2VolatilityEstimator

class LiveSimulator:
    """
    L2-Only Live Simulation System for real-time paper trading.
    
    This system operates entirely on Level 2 order book data streams,
    providing microsecond-level precision for alpha generation.
    """

    def __init__(self, config, exchange_api, data_handler, feature_engineer,
                 model_predictor, risk_manager, order_executor):
        """
        Initializes the L2-Only LiveSimulator.

        Args:
            config (dict): L2-only configuration dictionary.
            exchange_api: Initialized CCXT exchange object with L2 support.
            data_handler (DataHandler): L2-capable DataHandler instance.
            feature_engineer (FeatureEngineer): L2-only FeatureEngineer instance.
            model_predictor (ModelPredictor): L2-only ModelPredictor instance.
            risk_manager: L2-compatible risk manager.
            order_executor: Smart order executor.
        """
        self.config = config
        self.exchange = exchange_api
        self.data_handler = data_handler
        self.feature_engineer = feature_engineer
        self.model_predictor = model_predictor
        self.risk_manager = risk_manager
        self.order_executor = order_executor

        # L2-only configuration
        self.symbol = config.get('symbol', 'BTC/USDT')
        self.base_dir = config.get('base_dir', './trading_bot_data')
        self.l2_only_mode = config.get('l2_only_mode', True)
        self.l2_sampling_frequency_ms = config.get('l2_sampling_frequency_ms', 100)
        self.l2_buffer_size = config.get('l2_buffer_size', 10000)
        
        # Validate L2-only mode
        if not self.l2_only_mode:
            raise ValueError("LiveSimulator requires l2_only_mode=True for Phase 3 implementation")
        
        # Validate L2 support
        if not (self.exchange and self.exchange.has.get('fetchL2OrderBook')):
            raise ValueError("Exchange must support L2 order book for L2-only simulation")

        # Initialize L2 components
        self.l2_price_reconstructor = L2PriceReconstructor(config)
        self.l2_volatility_estimator = L2VolatilityEstimator(config)
        
        # Simulation state
        self.live_equity_history = []
        self.simulation_running = False
        self.simulation_stop_event = threading.Event()
        self.simulation_thread = None
        self.l2_data_buffer = []
        self.last_l2_timestamp = None
        
        # L2-based position tracking
        self.current_live_position = {
            "side": None, 
            "entry_price": 0.0, 
            "size": 0.0, 
            "timestamp": None,
            "sl_price": None, 
            "tp_price": None, 
            "entry_commission": 0.0,
            "l2_entry_spread": 0.0,  # L2-specific: spread at entry
            "l2_entry_depth": 0.0,   # L2-specific: liquidity depth at entry
            "l2_price_impact": 0.0   # L2-specific: estimated price impact
        }

        # Logging setup
        safe_symbol = self.symbol.replace('/', '_').replace(':', '')
        self.simulation_log_path = os.path.join(
            self.base_dir, 
            f"l2_simulation_log_{safe_symbol}.jsonl"
        )
        
        # L2 performance metrics
        self.l2_metrics = {
            'total_l2_updates': 0,
            'l2_processing_latency_ms': [],
            'l2_feature_generation_time_ms': [],
            'l2_prediction_time_ms': [],
            'l2_data_quality_score': [],
            'microstructure_alpha': 0.0
        }

        self.logger = logging.getLogger(__name__)
        self.logger.info("L2-Only LiveSimulator initialized successfully")

    def _log_simulation_action(self, timestamp, action_type, details):
        """Logs L2 simulation actions to a JSONL file with enhanced L2 metrics."""
        try:
            ts_iso = pd.Timestamp(timestamp).tz_convert('UTC').isoformat() if pd.Timestamp(timestamp).tzinfo else \
                     pd.Timestamp(timestamp).tz_localize('UTC').isoformat()

            def convert_numpy_types(obj):
                if isinstance(obj, np.generic): return obj.item()
                if isinstance(obj, pd.Timestamp): return obj.isoformat()
                if isinstance(obj, (datetime, pd.Timestamp)): return obj.isoformat()
                return obj

            details_serializable = {k: convert_numpy_types(v) for k, v in details.items()}
            
            # Add L2-specific metrics
            details_serializable.update({
                'l2_mode': True,
                'l2_total_updates': self.l2_metrics['total_l2_updates'],
                'l2_buffer_size': len(self.l2_data_buffer),
                'l2_avg_latency_ms': np.mean(self.l2_metrics['l2_processing_latency_ms'][-100:]) if self.l2_metrics['l2_processing_latency_ms'] else 0
            })
            
            log_entry = {"timestamp": ts_iso, "action": action_type, **details_serializable}

            os.makedirs(os.path.dirname(self.simulation_log_path), exist_ok=True)
            with open(self.simulation_log_path, "a") as f:
                json.dump(log_entry, f)
                f.write("\n")
                
        except Exception as e:
            self.logger.error(f"Error logging L2 simulation action: {e}. Details: {details}")
            traceback.print_exc(limit=1)

    def _fetch_l2_data_stream(self):
        """
        Fetches real-time L2 order book data and maintains buffer.
        
        Returns:
            dict: Latest L2 snapshot with enhanced metadata
        """
        try:
            start_time = time.perf_counter()
            
            # Fetch L2 order book snapshot
            l2_snapshot = self.data_handler.fetch_l2_order_book_snapshot()
            
            if not l2_snapshot or 'bids' not in l2_snapshot or 'asks' not in l2_snapshot:
                return None
            
            # Calculate processing latency
            processing_latency_ms = (time.perf_counter() - start_time) * 1000
            self.l2_metrics['l2_processing_latency_ms'].append(processing_latency_ms)
            
            # Keep only last 1000 latency measurements
            if len(self.l2_metrics['l2_processing_latency_ms']) > 1000:
                self.l2_metrics['l2_processing_latency_ms'] = self.l2_metrics['l2_processing_latency_ms'][-1000:]
            
            # Enhance L2 snapshot with metadata
            enhanced_snapshot = {
                **l2_snapshot,
                'timestamp_ns': time.time_ns(),
                'processing_latency_ms': processing_latency_ms,
                'buffer_position': len(self.l2_data_buffer)
            }
            
            # Add to buffer
            self.l2_data_buffer.append(enhanced_snapshot)
            
            # Maintain buffer size
            if len(self.l2_data_buffer) > self.l2_buffer_size:
                self.l2_data_buffer = self.l2_data_buffer[-self.l2_buffer_size:]
            
            self.l2_metrics['total_l2_updates'] += 1
            self.last_l2_timestamp = enhanced_snapshot['timestamp_ns']
            
            return enhanced_snapshot
            
        except Exception as e:
            self.logger.error(f"Error fetching L2 data stream: {e}")
            return None

    def _generate_l2_features(self, l2_snapshot):
        """
        Generates L2-only features from order book snapshot.
        
        Args:
            l2_snapshot (dict): L2 order book snapshot
            
        Returns:
            pd.DataFrame: L2 features for prediction
        """
        try:
            start_time = time.perf_counter()
            
            if not l2_snapshot or 'bids' not in l2_snapshot or 'asks' not in l2_snapshot:
                return None
            
            # Generate L2 price series
            l2_prices = self.l2_price_reconstructor.reconstruct_price_series([l2_snapshot])
            
            if l2_prices is None or l2_prices.empty:
                return None
            
            # Calculate L2 microstructure features
            l2_features = self.feature_engineer.calculate_l2_features_from_snapshot(
                l2_snapshot['bids'], 
                l2_snapshot['asks']
            )
            
            # Add L2 volatility features
            if len(self.l2_data_buffer) >= 50:  # Need sufficient history
                recent_snapshots = self.l2_data_buffer[-50:]
                l2_volatility = self.l2_volatility_estimator.estimate_volatility(recent_snapshots)
                l2_features.update(l2_volatility)
            
            # Create feature DataFrame
            feature_row = {
                'timestamp': pd.Timestamp.now(tz='UTC')
            }
            
            # Add L2 price data if available
            if l2_prices is not None and not l2_prices.empty:
                feature_row.update(l2_prices.iloc[-1].to_dict())
            
            # Add L2 features
            feature_row.update(l2_features)
            
            features_df = pd.DataFrame([feature_row])
            
            # Calculate feature generation time
            feature_time_ms = (time.perf_counter() - start_time) * 1000
            self.l2_metrics['l2_feature_generation_time_ms'].append(feature_time_ms)
            
            # Keep only last 1000 measurements
            if len(self.l2_metrics['l2_feature_generation_time_ms']) > 1000:
                self.l2_metrics['l2_feature_generation_time_ms'] = self.l2_metrics['l2_feature_generation_time_ms'][-1000:]
            
            return features_df
            
        except Exception as e:
            self.logger.error(f"Error generating L2 features: {e}")
            return None

    def _calculate_l2_position_metrics(self, l2_snapshot, current_price):
        """
        Calculates L2-specific position metrics.
        
        Args:
            l2_snapshot (dict): Current L2 snapshot
            current_price (float): Current price
            
        Returns:
            dict: L2 position metrics
        """
        try:
            if not l2_snapshot or 'bids' not in l2_snapshot or 'asks' not in l2_snapshot:
                return {}
            
            bids = l2_snapshot['bids']
            asks = l2_snapshot['asks']
            
            if not bids or not asks:
                return {}
            
            # Calculate spread
            best_bid = bids[0][0] if bids else 0
            best_ask = asks[0][0] if asks else 0
            spread = best_ask - best_bid if best_bid > 0 and best_ask > 0 else 0
            
            # Calculate available liquidity depth
            total_bid_volume = sum(bid[1] for bid in bids[:5])  # Top 5 levels
            total_ask_volume = sum(ask[1] for ask in asks[:5])  # Top 5 levels
            
            # Estimate price impact for typical trade size
            trade_size_usd = 1000  # $1000 trade
            trade_size_base = trade_size_usd / current_price if current_price > 0 else 0
            
            price_impact = self._estimate_price_impact(bids, asks, trade_size_base)
            
            return {
                'l2_spread': spread,
                'l2_spread_bps': (spread / current_price * 10000) if current_price > 0 else 0,
                'l2_bid_depth': total_bid_volume,
                'l2_ask_depth': total_ask_volume,
                'l2_price_impact': price_impact,
                'l2_liquidity_score': min(total_bid_volume, total_ask_volume)
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating L2 position metrics: {e}")
            return {}

    def _estimate_price_impact(self, bids, asks, trade_size):
        """
        Estimates price impact for a given trade size using L2 data.
        
        Args:
            bids (list): Bid levels [(price, size), ...]
            asks (list): Ask levels [(price, size), ...]
            trade_size (float): Trade size in base currency
            
        Returns:
            float: Estimated price impact in basis points
        """
        try:
            if not bids or not asks or trade_size <= 0:
                return 0.0
            
            # For buy order, walk through asks
            remaining_size = trade_size
            total_cost = 0.0
            
            for price, size in asks:
                if remaining_size <= 0:
                    break
                
                fill_size = min(remaining_size, size)
                total_cost += fill_size * price
                remaining_size -= fill_size
            
            if remaining_size > 0:
                # Not enough liquidity
                return 1000.0  # 10% impact as penalty
            
            # Calculate average fill price
            avg_fill_price = total_cost / trade_size if trade_size > 0 else 0
            best_ask = asks[0][0]
            
            # Price impact in basis points
            price_impact_bps = ((avg_fill_price - best_ask) / best_ask * 10000) if best_ask > 0 else 0
            
            return max(0, price_impact_bps)
            
        except Exception as e:
            self.logger.error(f"Error estimating price impact: {e}")
            return 0.0

    def _simulation_loop(self, initial_equity, threshold, loop_interval_seconds, 
                        commission_pct, leverage):
        """
        Core L2-only simulation loop with real-time order book processing.
        
        Args:
            initial_equity (float): Starting equity
            threshold (float): Signal threshold for trading
            loop_interval_seconds (float): Loop interval (should be small for L2)
            commission_pct (float): Commission percentage
            leverage (float): Leverage multiplier
        """
        self.logger.info("--- L2-Only Live Simulation Thread Started ---")
        self.logger.info(f"Params: InitEq={initial_equity}, Thresh={threshold}, "
                        f"Interval={loop_interval_seconds}s, Comm={commission_pct*100:.4f}%, "
                        f"Lev={leverage}x, L2_Freq={self.l2_sampling_frequency_ms}ms")

        # Initialize simulation state
        self.live_equity_history = [{'timestamp': datetime.now(timezone.utc), 'equity': initial_equity}]
        self.current_live_position = {
            "side": None, "entry_price": 0.0, "size": 0.0, "timestamp": None,
            "sl_price": None, "tp_price": None, "entry_commission": 0.0,
            "l2_entry_spread": 0.0, "l2_entry_depth": 0.0, "l2_price_impact": 0.0
        }
        balance = initial_equity

        # Validate L2-only model
        use_ensemble_for_sim = self.config.get('use_ensemble_for_simulation', False)
        if not self.model_predictor.model_object:
            if not self.model_predictor.load_model_and_features(load_ensemble=use_ensemble_for_sim):
                self.logger.error("FATAL: Failed to load L2-only model for simulation")
                self._log_simulation_action(datetime.now(timezone.utc), "ERROR_L2_MODEL_LOAD", 
                                          {"message": "Failed to load L2-only model"})
                self.simulation_running = False
                return

        # Ensure L2-only scaling info
        if not self.model_predictor._ensure_scaling_info():
            self.logger.warning("L2 scaling info could not be confirmed. Using L2-derived defaults.")

        # Main L2 simulation loop
        while self.simulation_running and not self.simulation_stop_event.is_set():
            loop_start_time = time.perf_counter()
            now_utc = datetime.now(timezone.utc)
            
            try:
                # Fetch real-time L2 data
                l2_snapshot = self._fetch_l2_data_stream()
                
                if not l2_snapshot:
                    self.logger.warning(f"[{now_utc.strftime('%H:%M:%S')}] No L2 data available. Waiting...")
                    self._log_simulation_action(now_utc, "WAIT_L2_DATA", {})
                    time.sleep(max(0.01, loop_interval_seconds - (time.perf_counter() - loop_start_time)))
                    continue

                # Generate L2-only features
                features_df = self._generate_l2_features(l2_snapshot)
                
                if features_df is None or features_df.empty:
                    self.logger.warning(f"[{now_utc.strftime('%H:%M:%S')}] Failed to generate L2 features. Waiting...")
                    self._log_simulation_action(now_utc, "WAIT_L2_FEATURES", {})
                    time.sleep(max(0.01, loop_interval_seconds - (time.perf_counter() - loop_start_time)))
                    continue

                # Get L2-derived price
                current_price = features_df.get('weighted_mid_price', features_df.get('microprice', 0)).iloc[0]
                if current_price <= 0:
                    # Fallback to best bid/ask midpoint
                    if l2_snapshot['bids'] and l2_snapshot['asks']:
                        current_price = (l2_snapshot['bids'][0][0] + l2_snapshot['asks'][0][0]) / 2
                    else:
                        continue

                # Calculate L2 volatility for risk management
                l2_volatility = features_df.get('l2_volatility_1min', 0.01).iloc[0]
                if pd.isna(l2_volatility) or l2_volatility <= 0:
                    l2_volatility = 0.01 * current_price  # 1% default

                # Generate L2-only prediction
                prediction_start = time.perf_counter()
                predicted_df = self.model_predictor.predict_signals(
                    features_df,
                    use_ensemble=use_ensemble_for_sim
                )
                prediction_time_ms = (time.perf_counter() - prediction_start) * 1000
                self.l2_metrics['l2_prediction_time_ms'].append(prediction_time_ms)
                
                if len(self.l2_metrics['l2_prediction_time_ms']) > 1000:
                    self.l2_metrics['l2_prediction_time_ms'] = self.l2_metrics['l2_prediction_time_ms'][-1000:]

                if predicted_df is None or predicted_df.empty:
                    self._log_simulation_action(now_utc, "WAIT_L2_PREDICTION", {})
                    time.sleep(max(0.01, loop_interval_seconds - (time.perf_counter() - loop_start_time)))
                    continue

                # Extract prediction results
                current_signal = predicted_df["signal"].iloc[0]
                current_timestamp = pd.Timestamp(predicted_df["timestamp"].iloc[0])
                pred_scaled = predicted_df["pred_scaled"].iloc[0] if "pred_scaled" in predicted_df.columns else np.nan
                pred_unscaled = predicted_df.get("pred_unscaled_target", pd.Series([np.nan])).iloc[0]

                # Calculate L2 position metrics
                l2_metrics = self._calculate_l2_position_metrics(l2_snapshot, current_price)
                
                # Update position tracking with L2 metrics
                pos = self.current_live_position
                unrealized_pnl = 0
                
                if pos["side"] is not None and pos["size"] > 0:
                    if pos["side"] == "long":
                        unrealized_pnl = (current_price - pos["entry_price"]) * pos["size"] * leverage
                    elif pos["side"] == "short":
                        unrealized_pnl = (pos["entry_price"] - current_price) * pos["size"] * leverage

                current_equity = balance + unrealized_pnl
                
                # L2-enhanced logging
                detailed_log = {
                    "signal": current_signal,
                    "price": current_price,
                    "l2_price_type": "weighted_mid_price",
                    "pred_scaled": pred_scaled,
                    "pred_unscaled": pred_unscaled,
                    "equity": current_equity,
                    "unrealized_pnl": unrealized_pnl,
                    "position": dict(pos),
                    "l2_volatility": l2_volatility,
                    "processing_latency_ms": l2_snapshot.get('processing_latency_ms', 0),
                    "feature_generation_ms": self.l2_metrics['l2_feature_generation_time_ms'][-1] if self.l2_metrics['l2_feature_generation_time_ms'] else 0,
                    "prediction_time_ms": prediction_time_ms,
                    **l2_metrics
                }

                # Position management logic (enhanced for L2)
                if abs(current_signal) >= threshold:
                    if pos["side"] is None:  # No position, consider entry
                        # L2-enhanced entry logic
                        if l2_metrics.get('l2_liquidity_score', 0) > 0.1:  # Sufficient liquidity
                            # Calculate position size based on L2 volatility
                            risk_per_trade = 0.02  # 2% risk per trade
                            position_size = (current_equity * risk_per_trade) / (l2_volatility * leverage)
                            
                            new_side = "long" if current_signal > 0 else "short"
                            
                            # Enhanced position entry with L2 metrics
                            pos.update({
                                "side": new_side,
                                "entry_price": current_price,
                                "size": position_size,
                                "timestamp": current_timestamp,
                                "entry_commission": position_size * current_price * commission_pct,
                                "l2_entry_spread": l2_metrics.get('l2_spread', 0),
                                "l2_entry_depth": l2_metrics.get('l2_liquidity_score', 0),
                                "l2_price_impact": l2_metrics.get('l2_price_impact', 0)
                            })
                            
                            balance -= pos["entry_commission"]
                            
                            # Set L2-based stop loss and take profit
                            sl_distance = l2_volatility * 1.5  # 1.5x L2 volatility
                            tp_distance = l2_volatility * 3.0  # 3x L2 volatility
                            
                            if new_side == "long":
                                pos["sl_price"] = current_price - sl_distance
                                pos["tp_price"] = current_price + tp_distance
                            else:
                                pos["sl_price"] = current_price + sl_distance
                                pos["tp_price"] = current_price - tp_distance

                            detailed_log["action"] = f"ENTER_{new_side.upper()}"
                            detailed_log["l2_entry_analysis"] = {
                                "spread_bps": l2_metrics.get('l2_spread_bps', 0),
                                "price_impact_bps": l2_metrics.get('l2_price_impact', 0),
                                "liquidity_score": l2_metrics.get('l2_liquidity_score', 0)
                            }
                            
                            self.logger.info(f"L2 ENTRY: {new_side.upper()} at {current_price:.2f}, "
                                           f"Size: {position_size:.6f}, Spread: {l2_metrics.get('l2_spread_bps', 0):.1f}bps")

                # Check exit conditions with L2 enhancements
                if pos["side"] is not None:
                    should_exit = False
                    exit_reason = ""
                    
                    # L2-based stop loss/take profit
                    if pos["side"] == "long":
                        if current_price <= pos["sl_price"]:
                            should_exit, exit_reason = True, "SL_HIT"
                        elif current_price >= pos["tp_price"]:
                            should_exit, exit_reason = True, "TP_HIT"
                    elif pos["side"] == "short":
                        if current_price >= pos["sl_price"]:
                            should_exit, exit_reason = True, "SL_HIT"
                        elif current_price <= pos["tp_price"]:
                            should_exit, exit_reason = True, "TP_HIT"
                    
                    # L2-based signal reversal
                    if not should_exit and abs(current_signal) >= threshold:
                        if (pos["side"] == "long" and current_signal < -threshold) or \
                           (pos["side"] == "short" and current_signal > threshold):
                            should_exit, exit_reason = True, "SIGNAL_REVERSAL"
                    
                    # L2 liquidity-based exit (if liquidity dries up)
                    if not should_exit and l2_metrics.get('l2_liquidity_score', 1) < 0.05:
                        should_exit, exit_reason = True, "LOW_LIQUIDITY"

                    if should_exit:
                        # Calculate P&L with L2 metrics
                        exit_commission = pos["size"] * current_price * commission_pct
                        
                        if pos["side"] == "long":
                            pnl = (current_price - pos["entry_price"]) * pos["size"] * leverage
                        else:
                            pnl = (pos["entry_price"] - current_price) * pos["size"] * leverage
                        
                        net_pnl = pnl - pos["entry_commission"] - exit_commission
                        balance += pnl - exit_commission
                        
                        # Update microstructure alpha tracking
                        if exit_reason in ["TP_HIT", "SIGNAL_REVERSAL"]:
                            self.l2_metrics['microstructure_alpha'] += net_pnl

                        detailed_log.update({
                            "action": f"EXIT_{pos['side'].upper()}",
                            "exit_reason": exit_reason,
                            "pnl": pnl,
                            "net_pnl": net_pnl,
                            "exit_commission": exit_commission,
                            "l2_exit_analysis": {
                                "exit_spread_bps": l2_metrics.get('l2_spread_bps', 0),
                                "exit_price_impact_bps": l2_metrics.get('l2_price_impact', 0),
                                "holding_period_ms": (time.time_ns() - pos.get('entry_timestamp_ns', time.time_ns())) / 1_000_000
                            }
                        })
                        
                        self.logger.info(f"L2 EXIT: {pos['side'].upper()} at {current_price:.2f}, "
                                       f"PnL: {net_pnl:.2f}, Reason: {exit_reason}")

                        # Reset position
                        self.current_live_position = {
                            "side": None, "entry_price": 0.0, "size": 0.0, "timestamp": None,
                            "sl_price": None, "tp_price": None, "entry_commission": 0.0,
                            "l2_entry_spread": 0.0, "l2_entry_depth": 0.0, "l2_price_impact": 0.0
                        }

                # Update equity history
                current_equity = balance + unrealized_pnl
                self.live_equity_history.append({
                    'timestamp': now_utc,
                    'equity': current_equity,
                    'l2_updates': self.l2_metrics['total_l2_updates'],
                    'avg_latency_ms': np.mean(self.l2_metrics['l2_processing_latency_ms'][-100:]) if self.l2_metrics['l2_processing_latency_ms'] else 0
                })

                # Log simulation state
                self._log_simulation_action(now_utc, "L2_SIMULATION_TICK", detailed_log)

                # Adaptive sleep based on L2 frequency
                elapsed_time = time.perf_counter() - loop_start_time
                target_interval = max(self.l2_sampling_frequency_ms / 1000, loop_interval_seconds)
                sleep_time = max(0.001, target_interval - elapsed_time)  # Minimum 1ms sleep
                time.sleep(sleep_time)

            except Exception as e:
                self.logger.error(f"Error in L2 simulation loop: {e}")
                traceback.print_exc()
                self._log_simulation_action(now_utc, "ERROR_L2_LOOP", {"error": str(e)})
                time.sleep(1)  # Longer sleep on error

        self.logger.info("L2-Only Live Simulation Thread Stopped")
        
        # Log final L2 performance metrics
        final_metrics = {
            "total_l2_updates": self.l2_metrics['total_l2_updates'],
            "avg_processing_latency_ms": np.mean(self.l2_metrics['l2_processing_latency_ms']) if self.l2_metrics['l2_processing_latency_ms'] else 0,
            "avg_feature_generation_ms": np.mean(self.l2_metrics['l2_feature_generation_time_ms']) if self.l2_metrics['l2_feature_generation_time_ms'] else 0,
            "avg_prediction_time_ms": np.mean(self.l2_metrics['l2_prediction_time_ms']) if self.l2_metrics['l2_prediction_time_ms'] else 0,
            "microstructure_alpha": self.l2_metrics['microstructure_alpha'],
            "final_equity": current_equity
        }
        
        self._log_simulation_action(datetime.now(timezone.utc), "L2_SIMULATION_COMPLETE", final_metrics)
        self.logger.info(f"L2 Simulation Final Metrics: {final_metrics}")

    def start_live_simulation(self, initial_equity=None, threshold=None, 
                              commission_pct=None, leverage=None):
        """Starts the L2-only live simulation thread."""
        if self.simulation_running:
            self.logger.info("L2 simulation already running.")
            return
        if not self.exchange:
            self.logger.error("Exchange not initialized. Cannot start L2 simulation.")
            return
        if not all([self.data_handler, self.feature_engineer, self.model_predictor, self.risk_manager, self.order_executor]):
            self.logger.error("One or more core components not initialized for L2 simulation.")
            return

        # L2-only configuration
        _initial_equity = initial_equity if initial_equity is not None else self.config.get('initial_balance', 10000)
        _threshold = threshold if threshold is not None else self.config.get('simulation_threshold',
                                                                           self.config.get('prediction_threshold', 0.15))
        _commission_pct = commission_pct if commission_pct is not None else self.config.get('commission_pct', 0.0006)
        _leverage = leverage if leverage is not None else self.config.get('leverage', 2)

        # L2-optimized loop interval (much faster than OHLCV)
        min_interval = self.config.get('min_simulation_interval_seconds', 5)  # 5 seconds for L2
        loop_interval = max(min_interval, self.l2_sampling_frequency_ms / 1000)  # Based on L2 frequency

        self.simulation_running = True
        self.simulation_stop_event.clear()
        self.live_equity_history = []

        self.simulation_thread = threading.Thread(
            target=self._simulation_loop,
            kwargs={
                'initial_equity': _initial_equity, 
                'threshold': _threshold,
                'loop_interval_seconds': loop_interval,
                'commission_pct': _commission_pct, 
                'leverage': _leverage
            },
            daemon=True
        )
        self.simulation_thread.start()
        self.logger.info(f"L2-only live simulation thread started (Interval: {loop_interval}s, "
                        f"L2_Freq: {self.l2_sampling_frequency_ms}ms). Log: {self.simulation_log_path}")

    def stop_live_simulation(self, wait_time=15):
        """Stops the running L2 live simulation thread."""
        if not self.simulation_running or self.simulation_thread is None:
            self.logger.info("L2 simulation not running or thread not found.")
            return

        self.logger.info("Attempting to stop L2 live simulation gracefully...")
        self.simulation_stop_event.set()
        if self.simulation_thread.is_alive():
            self.simulation_thread.join(timeout=wait_time)

        if self.simulation_thread.is_alive():
            self.logger.warning(f"L2 simulation thread did not stop within {wait_time}s.")
        else:
            self.logger.info("L2 simulation thread stopped successfully.")

        self.simulation_running = False
        self.simulation_thread = None

        self.logger.info("L2-only live simulation stopped.")

    def get_simulation_equity_data(self):
        """Returns the collected L2 simulation equity history for analysis."""
        return pd.DataFrame(self.live_equity_history) if self.live_equity_history else pd.DataFrame()
    
    def get_l2_performance_metrics(self):
        """
        Returns L2-specific performance metrics for analysis.
        
        Returns:
            dict: L2 performance metrics
        """
        return {
            'total_l2_updates': self.l2_metrics['total_l2_updates'],
            'avg_processing_latency_ms': np.mean(self.l2_metrics['l2_processing_latency_ms']) if self.l2_metrics['l2_processing_latency_ms'] else 0,
            'p99_processing_latency_ms': np.percentile(self.l2_metrics['l2_processing_latency_ms'], 99) if self.l2_metrics['l2_processing_latency_ms'] else 0,
            'avg_feature_generation_ms': np.mean(self.l2_metrics['l2_feature_generation_time_ms']) if self.l2_metrics['l2_feature_generation_time_ms'] else 0,
            'avg_prediction_time_ms': np.mean(self.l2_metrics['l2_prediction_time_ms']) if self.l2_metrics['l2_prediction_time_ms'] else 0,
            'microstructure_alpha': self.l2_metrics['microstructure_alpha'],
            'l2_buffer_size': len(self.l2_data_buffer),
            'last_l2_timestamp': self.last_l2_timestamp
        }