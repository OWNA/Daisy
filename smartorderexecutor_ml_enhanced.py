"""
ML-Enhanced Smart Order Executor
Integrates ML model features for intelligent execution decisions
Optimized for Bybit perpetual futures with advanced order types
"""

import ccxt
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import time
from datetime import datetime
import asyncio


@dataclass
class ExecutionDecision:
    """Encapsulates execution decision based on ML features"""
    strategy: str  # passive, balanced, aggressive, urgent
    order_type: str  # limit, market, iceberg, twap
    price_adjustment: float  # Price adjustment factor
    size_allocation: List[float]  # Size split for multiple orders
    time_horizon: float  # Expected execution time in seconds
    confidence: float  # Confidence in execution decision


class MLEnhancedOrderExecutor:
    """
    Advanced order executor that uses ML features for execution optimization
    """
    
    def __init__(self, exchange_api, exec_config: dict, ml_config: dict):
        """
        Initialize ML-enhanced executor
        
        Args:
            exchange_api: CCXT exchange instance
            exec_config: Execution configuration
            ml_config: ML feature thresholds and weights
        """
        self.exchange = exchange_api
        
        # Base execution parameters
        self.base_slippage_pct = exec_config.get('slippage_model_pct', 0.0005)
        self.max_levels = exec_config.get('max_order_book_levels', 50)
        self.min_order_size = exec_config.get('min_order_size', 0.001)
        
        # ML feature thresholds
        self.ml_thresholds = ml_config.get('thresholds', {
            'spread_stability_passive': 0.2,     # Below this -> passive
            'spread_stability_aggressive': 0.6,  # Above this -> aggressive
            'ofi_urgent': 0.7,                  # Above this -> urgent
            'pressure_imbalance_signal': 0.5,   # Significant imbalance
            'book_resilience_min': 0.3,         # Minimum acceptable resilience
            'volume_concentration_liquid': 0.6   # Good liquidity concentration
        })
        
        # Feature weights for decision scoring
        self.feature_weights = ml_config.get('weights', {
            'spread_stability_norm_100': 1.0,
            'ofi_normalized_1m': 0.8,
            'pressure_imbalance_weighted': 0.7,
            'book_resilience': 0.6,
            'volume_concentration': 0.5,
            'quote_lifetime': 0.4
        })
        
        # Execution strategy parameters
        self.strategy_params = {
            'passive': {
                'price_offset_bps': -5,      # Place 5 bps inside best quote
                'max_wait_time': 30,         # Max 30 seconds wait
                'cancel_on_move_bps': 10,    # Cancel if market moves 10 bps
                'use_post_only': True
            },
            'balanced': {
                'price_offset_bps': 2,       # Place 2 bps through best quote  
                'max_wait_time': 10,
                'split_orders': 2,           # Split into 2 orders
                'time_between_orders': 0.5
            },
            'aggressive': {
                'price_offset_bps': 10,      # Cross spread aggressively
                'max_wait_time': 5,
                'use_iceberg': True,         # Use iceberg for large orders
                'show_ratio': 0.2            # Show 20% of total size
            },
            'urgent': {
                'use_market_order': True,    # Direct market order
                'backup_limit_bps': 20,      # Backup limit 20 bps through
                'max_slippage_bps': 50       # Max acceptable slippage
            }
        }
        
        print("ML-Enhanced SmartOrderExecutor initialized")
        print(f"Feature weights: {list(self.feature_weights.keys())}")
        
    def analyze_ml_features(self, features: Dict[str, float]) -> ExecutionDecision:
        """
        Analyze ML features to determine optimal execution strategy
        
        Args:
            features: Dictionary of ML feature values
            
        Returns:
            ExecutionDecision object with strategy details
        """
        # Extract key features
        spread_stability = features.get('spread_stability_norm_100', 0.5)
        ofi = abs(features.get('ofi_normalized_1m', 0))
        pressure_imbalance = abs(features.get('pressure_imbalance_weighted', 0))
        book_resilience = features.get('book_resilience', 0.5)
        volume_concentration = features.get('volume_concentration', 0.5)
        quote_lifetime = features.get('quote_lifetime', 5)
        
        # Calculate composite urgency score
        urgency_score = self._calculate_urgency_score(features)
        
        # Determine execution strategy
        if spread_stability < self.ml_thresholds['spread_stability_passive'] and \
           ofi < 0.3 and book_resilience > 0.6:
            # Stable market, good for passive execution
            strategy = 'passive'
            order_type = 'limit'
            price_adj = 1.0 - self.strategy_params['passive']['price_offset_bps'] / 10000
            size_allocation = [1.0]  # Single order
            time_horizon = 20.0
            
        elif ofi > self.ml_thresholds['ofi_urgent'] or \
             pressure_imbalance > 0.8:
            # High urgency - immediate execution needed
            strategy = 'urgent'
            order_type = 'market' if ofi > 0.85 else 'limit'
            price_adj = 1.0 + self.strategy_params['urgent']['backup_limit_bps'] / 10000
            size_allocation = [1.0]
            time_horizon = 1.0
            
        elif spread_stability > self.ml_thresholds['spread_stability_aggressive'] or \
             book_resilience < self.ml_thresholds['book_resilience_min']:
            # Unstable or thin book - aggressive but careful
            strategy = 'aggressive'
            order_type = 'iceberg' if volume_concentration < 0.5 else 'limit'
            price_adj = 1.0 + self.strategy_params['aggressive']['price_offset_bps'] / 10000
            
            # Split large orders in thin books
            if book_resilience < 0.3:
                size_allocation = [0.3, 0.3, 0.4]  # 3 chunks
            else:
                size_allocation = [0.5, 0.5]  # 2 chunks
            time_horizon = 5.0
            
        else:
            # Balanced conditions
            strategy = 'balanced'
            order_type = 'limit'
            price_adj = 1.0 + self.strategy_params['balanced']['price_offset_bps'] / 10000
            size_allocation = [0.6, 0.4]  # Slight front-loading
            time_horizon = 10.0
            
        # Calculate confidence based on feature quality
        confidence = self._calculate_confidence(features, urgency_score)
        
        return ExecutionDecision(
            strategy=strategy,
            order_type=order_type,
            price_adjustment=price_adj,
            size_allocation=size_allocation,
            time_horizon=time_horizon,
            confidence=confidence
        )
        
    def _calculate_urgency_score(self, features: Dict[str, float]) -> float:
        """Calculate weighted urgency score from features"""
        score = 0.0
        weight_sum = 0.0
        
        # Normalized scores for each feature
        feature_scores = {
            'spread_stability_norm_100': 1 - min(features.get('spread_stability_norm_100', 0.5), 1.0),
            'ofi_normalized_1m': min(abs(features.get('ofi_normalized_1m', 0)), 1.0),
            'pressure_imbalance_weighted': min(abs(features.get('pressure_imbalance_weighted', 0)), 1.0),
            'book_resilience': min(features.get('book_resilience', 0.5), 1.0),
            'volume_concentration': features.get('volume_concentration', 0.5)
        }
        
        for feature, value in feature_scores.items():
            if feature in self.feature_weights:
                score += value * self.feature_weights[feature]
                weight_sum += self.feature_weights[feature]
                
        return score / weight_sum if weight_sum > 0 else 0.5
        
    def _calculate_confidence(self, features: Dict[str, float], urgency_score: float) -> float:
        """Calculate confidence in execution decision"""
        # Base confidence on feature availability
        available_features = sum(1 for f in self.feature_weights if f in features)
        feature_coverage = available_features / len(self.feature_weights)
        
        # Adjust based on market conditions
        spread_stability = features.get('spread_stability_norm_100', 0.5)
        book_resilience = features.get('book_resilience', 0.5)
        
        # Lower confidence in unstable markets
        stability_factor = 1.0 - (spread_stability * 0.3)
        
        # Higher confidence with better book resilience
        resilience_factor = 0.7 + (book_resilience * 0.3)
        
        confidence = feature_coverage * stability_factor * resilience_factor
        
        # Extreme urgency reduces confidence slightly (more risk)
        if urgency_score > 0.9 or urgency_score < 0.1:
            confidence *= 0.9
            
        return min(confidence, 1.0)
        
    async def execute_with_ml_decision(
        self,
        symbol: str,
        side: str,
        amount: float,
        features: Dict[str, float],
        signal_strength: float = None
    ) -> Dict[str, any]:
        """
        Execute order using ML-driven decision making
        
        Args:
            symbol: Trading symbol
            side: 'buy' or 'sell'
            amount: Total amount to execute
            features: ML feature dictionary
            signal_strength: Optional signal strength override
            
        Returns:
            Execution result dictionary
        """
        start_time = time.time()
        
        # Get execution decision from ML features
        decision = self.analyze_ml_features(features)
        
        print(f"ML Execution Decision: {decision.strategy} strategy, {decision.order_type} order")
        print(f"Confidence: {decision.confidence:.2f}, Time horizon: {decision.time_horizon}s")
        
        # Fetch current market state
        order_book = self.exchange.fetch_order_book(symbol, limit=self.max_levels)
        ticker = self.exchange.fetch_ticker(symbol)
        
        best_bid = order_book['bids'][0][0] if order_book['bids'] else 0
        best_ask = order_book['asks'][0][0] if order_book['asks'] else 0
        mid_price = (best_bid + best_ask) / 2
        
        # Execute based on strategy
        if decision.strategy == 'passive':
            result = await self._execute_passive(
                symbol, side, amount, order_book, decision
            )
            
        elif decision.strategy == 'urgent':
            result = await self._execute_urgent(
                symbol, side, amount, order_book, decision
            )
            
        elif decision.strategy == 'aggressive':
            result = await self._execute_aggressive(
                symbol, side, amount, order_book, decision
            )
            
        else:  # balanced
            result = await self._execute_balanced(
                symbol, side, amount, order_book, decision
            )
            
        # Calculate execution metrics
        execution_time = time.time() - start_time
        
        # Add ML context to result
        result['ml_decision'] = {
            'strategy': decision.strategy,
            'confidence': decision.confidence,
            'features_used': list(features.keys()),
            'execution_time': execution_time,
            'signal_strength': signal_strength
        }
        
        return result
        
    async def _execute_passive(
        self, 
        symbol: str, 
        side: str, 
        amount: float,
        order_book: dict,
        decision: ExecutionDecision
    ) -> dict:
        """Execute passive order strategy with dynamic best bid/ask following"""
        params = self.strategy_params['passive']
        
        # Initialize tracking variables
        current_order_id = None
        follow_count = 0
        max_follows = 10  # Prevent infinite following
        total_executed = 0
        total_cost = 0
        start_time = time.time()
        
        # Tick size estimation (assume 0.5 for most crypto pairs)
        tick_size = 0.5
        tolerance_ticks = 2  # As specified in PROJECT_PLAN.md
        tolerance_amount = tick_size * tolerance_ticks
        
        print(f"[PASSIVE] Starting best bid/ask follower strategy for {symbol}")
        print(f"[PASSIVE] Tolerance: {tolerance_ticks} ticks ({tolerance_amount} price units)")
        
        try:
            while follow_count < max_follows and time.time() - start_time < params['max_wait_time']:
                # Get current best bid/ask
                current_book = self.exchange.fetch_order_book(symbol, limit=10)
                
                if side == 'buy':
                    # For buy orders, follow best bid
                    current_best = current_book['bids'][0][0] if current_book['bids'] else 0
                    our_price = current_best  # Place at best bid
                else:
                    # For sell orders, follow best ask  
                    current_best = current_book['asks'][0][0] if current_book['asks'] else 0
                    our_price = current_best  # Place at best ask
                
                if current_best == 0:
                    print(f"[PASSIVE] ERROR: No {side} side liquidity available")
                    break
                
                # Check if we need to place/replace order
                need_new_order = False
                if current_order_id is None:
                    need_new_order = True
                    print(f"[PASSIVE] Placing initial order at {our_price}")
                else:
                    # Check existing order status
                    try:
                        existing_order = self.exchange.fetch_order(current_order_id, symbol)
                        
                        if existing_order['status'] == 'closed':
                            # Order filled!
                            total_executed += existing_order['filled']
                            if existing_order['average']:
                                total_cost += existing_order['filled'] * existing_order['average']
                            
                            print(f"[PASSIVE] Order {current_order_id} filled: {existing_order['filled']} @ {existing_order['average']}")
                            
                            # Check if we have remaining amount to execute
                            remaining_amount = amount - total_executed
                            if remaining_amount > self.min_order_size:
                                current_order_id = None
                                amount = remaining_amount  # Update amount for next order
                                need_new_order = True
                                print(f"[PASSIVE] Remaining amount: {remaining_amount}, placing new order")
                            else:
                                # Fully executed
                                avg_price = total_cost / total_executed if total_executed > 0 else our_price
                                return {
                                    'success': True,
                                    'order_id': existing_order['id'],
                                    'executed_amount': total_executed,
                                    'average_price': avg_price,
                                    'strategy': 'passive_following',
                                    'follow_count': follow_count
                                }
                        
                        elif existing_order['status'] == 'open':
                            # Check if market moved beyond tolerance
                            current_order_price = existing_order['price']
                            price_diff = abs(our_price - current_order_price)
                            
                            if price_diff > tolerance_amount:
                                # Market moved beyond tolerance, need to replace
                                print(f"[PASSIVE] Market moved {price_diff:.2f} (tolerance: {tolerance_amount})")
                                print(f"[PASSIVE] Cancelling order {current_order_id} and replacing")
                                
                                # Cancel existing order
                                try:
                                    self.exchange.cancel_order(current_order_id, symbol)
                                    print(f"[PASSIVE] Successfully cancelled order {current_order_id}")
                                except Exception as cancel_e:
                                    print(f"[PASSIVE] WARNING: Failed to cancel order {current_order_id}: {cancel_e}")
                                
                                current_order_id = None
                                need_new_order = True
                                follow_count += 1
                                
                                # Brief pause to avoid rate limits
                                await asyncio.sleep(0.1)
                            else:
                                # Order is still good, wait and check again
                                await asyncio.sleep(0.5)
                                continue
                        else:
                            # Order was cancelled or failed
                            print(f"[PASSIVE] Order {current_order_id} status: {existing_order['status']}")
                            current_order_id = None
                            need_new_order = True
                            
                    except Exception as fetch_e:
                        print(f"[PASSIVE] ERROR fetching order {current_order_id}: {fetch_e}")
                        current_order_id = None
                        need_new_order = True
                
                # Place new order if needed
                if need_new_order:
                    order_params = {'postOnly': True} if params['use_post_only'] else {}
                    
                    try:
                        new_order = self.exchange.create_limit_order(
                            symbol=symbol,
                            side=side,
                            amount=amount,
                            price=our_price,
                            params=order_params
                        )
                        
                        current_order_id = new_order['id']
                        print(f"[PASSIVE] Placed order {current_order_id} for {amount} @ {our_price}")
                        
                        # Brief pause after placing order
                        await asyncio.sleep(0.2)
                        
                    except Exception as place_e:
                        print(f"[PASSIVE] ERROR placing order: {place_e}")
                        break
                
                # Small delay before next iteration
                await asyncio.sleep(0.5)
            
            # Execution loop ended - clean up any remaining order
            if current_order_id:
                try:
                    # Check final order status
                    final_order = self.exchange.fetch_order(current_order_id, symbol)
                    if final_order['status'] == 'closed':
                        total_executed += final_order['filled']
                        if final_order['average']:
                            total_cost += final_order['filled'] * final_order['average']
                        print(f"[PASSIVE] Final order filled: {final_order['filled']} @ {final_order['average']}")
                    else:
                        # Cancel unfilled order
                        self.exchange.cancel_order(current_order_id, symbol)
                        print(f"[PASSIVE] Cancelled unfilled order {current_order_id}")
                        
                        # Include any partial fills
                        if final_order['filled'] > 0:
                            total_executed += final_order['filled']
                            if final_order['average']:
                                total_cost += final_order['filled'] * final_order['average']
                            print(f"[PASSIVE] Partial fill: {final_order['filled']} @ {final_order['average']}")
                            
                except Exception as cleanup_e:
                    print(f"[PASSIVE] ERROR during cleanup: {cleanup_e}")
            
            # Calculate results
            avg_price = total_cost / total_executed if total_executed > 0 else 0
            success = total_executed >= amount * 0.8  # 80% fill considered success
            
            # Determine reason if not successful
            reason = None
            if not success:
                if time.time() - start_time >= params['max_wait_time']:
                    reason = 'timeout'
                elif follow_count >= max_follows:
                    reason = 'max_follows_exceeded'
                else:
                    reason = 'insufficient_fill'
            
            print(f"[PASSIVE] Execution completed:")
            print(f"[PASSIVE] - Executed: {total_executed}/{amount} ({(total_executed/amount)*100:.1f}%)")
            print(f"[PASSIVE] - Average price: {avg_price}")
            print(f"[PASSIVE] - Follow count: {follow_count}")
            print(f"[PASSIVE] - Success: {success}")
            
            return {
                'success': success,
                'order_id': current_order_id,
                'executed_amount': total_executed,
                'average_price': avg_price,
                'strategy': 'passive_following',
                'follow_count': follow_count,
                'reason': reason,
                'execution_time': time.time() - start_time
            }
            
        except Exception as e:
            print(f"[PASSIVE] CRITICAL ERROR: {e}")
            
            # Emergency cleanup
            if current_order_id:
                try:
                    self.exchange.cancel_order(current_order_id, symbol)
                    print(f"[PASSIVE] Emergency cancelled order {current_order_id}")
                except:
                    pass
            
            return {
                'success': False,
                'error': str(e),
                'strategy': 'passive_following',
                'follow_count': follow_count,
                'executed_amount': total_executed
            }
            
    async def _execute_urgent(
        self,
        symbol: str,
        side: str,
        amount: float,
        order_book: dict,
        decision: ExecutionDecision
    ) -> dict:
        """Execute urgent order strategy"""
        params = self.strategy_params['urgent']
        
        if params['use_market_order'] and decision.confidence > 0.7:
            # Direct market order for highest urgency
            try:
                order = self.exchange.create_market_order(
                    symbol=symbol,
                    side=side,
                    amount=amount
                )
                
                # Fetch filled order details
                filled_order = self.exchange.fetch_order(order['id'], symbol)
                
                return {
                    'success': True,
                    'order_id': order['id'],
                    'executed_amount': filled_order['filled'],
                    'average_price': filled_order['average'],
                    'strategy': 'urgent_market'
                }
                
            except Exception as e:
                print(f"Market order failed: {e}, falling back to aggressive limit")
                
        # Fallback to aggressive limit order
        if side == 'buy':
            limit_price = order_book['asks'][0][0] * (1 + params['backup_limit_bps'] / 10000)
        else:
            limit_price = order_book['bids'][0][0] * (1 - params['backup_limit_bps'] / 10000)
            
        try:
            order = self.exchange.create_limit_order(
                symbol=symbol,
                side=side,
                amount=amount,
                price=limit_price
            )
            
            # Quick fill check
            await asyncio.sleep(0.5)
            filled_order = self.exchange.fetch_order(order['id'], symbol)
            
            return {
                'success': filled_order['status'] == 'closed',
                'order_id': order['id'],
                'executed_amount': filled_order['filled'],
                'average_price': filled_order['average'] if filled_order['average'] else limit_price,
                'strategy': 'urgent_limit'
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'strategy': 'urgent'
            }
            
    async def _execute_aggressive(
        self,
        symbol: str,
        side: str,
        amount: float,
        order_book: dict,
        decision: ExecutionDecision
    ) -> dict:
        """Execute aggressive order strategy with iceberg option"""
        params = self.strategy_params['aggressive']
        
        # Calculate aggressive price
        if side == 'buy':
            base_price = order_book['asks'][0][0]
            limit_price = base_price * (1 + params['price_offset_bps'] / 10000)
        else:
            base_price = order_book['bids'][0][0]
            limit_price = base_price * (1 - params['price_offset_bps'] / 10000)
            
        executed_orders = []
        total_filled = 0
        total_cost = 0
        
        # Execute size allocation
        for i, allocation in enumerate(decision.size_allocation):
            chunk_size = amount * allocation
            
            if params['use_iceberg'] and i == 0:  # First chunk as iceberg
                # Bybit iceberg order
                order_params = {
                    'displayQty': chunk_size * params['show_ratio']  # Show only portion
                }
            else:
                order_params = {}
                
            try:
                order = self.exchange.create_limit_order(
                    symbol=symbol,
                    side=side,
                    amount=chunk_size,
                    price=limit_price,
                    params=order_params
                )
                
                executed_orders.append(order)
                
                # Brief pause between chunks
                if i < len(decision.size_allocation) - 1:
                    await asyncio.sleep(0.2)
                    
            except Exception as e:
                print(f"Chunk {i} failed: {e}")
                
        # Collect results
        await asyncio.sleep(1.0)  # Allow orders to fill
        
        for order in executed_orders:
            try:
                filled = self.exchange.fetch_order(order['id'], symbol)
                total_filled += filled['filled']
                if filled['average']:
                    total_cost += filled['filled'] * filled['average']
            except:
                pass
                
        avg_price = total_cost / total_filled if total_filled > 0 else 0
        
        return {
            'success': total_filled > 0,
            'executed_amount': total_filled,
            'average_price': avg_price,
            'strategy': 'aggressive_iceberg' if params['use_iceberg'] else 'aggressive',
            'chunks_executed': len(executed_orders)
        }
        
    async def _execute_balanced(
        self,
        symbol: str,
        side: str,
        amount: float,
        order_book: dict,
        decision: ExecutionDecision
    ) -> dict:
        """Execute balanced order strategy with smart splitting"""
        params = self.strategy_params['balanced']
        
        executed_orders = []
        total_filled = 0
        total_cost = 0
        
        for i, allocation in enumerate(decision.size_allocation):
            chunk_size = amount * allocation
            
            # Adjust price for each chunk
            price_adjust = 1 + (i * 2) / 10000  # Increment by 2 bps each chunk
            
            if side == 'buy':
                base_price = order_book['asks'][0][0]
                limit_price = base_price * price_adjust * decision.price_adjustment
            else:
                base_price = order_book['bids'][0][0]
                limit_price = base_price * (2 - price_adjust) * decision.price_adjustment
                
            try:
                order = self.exchange.create_limit_order(
                    symbol=symbol,
                    side=side,
                    amount=chunk_size,
                    price=limit_price
                )
                
                executed_orders.append(order)
                
                # Time between orders
                if i < len(decision.size_allocation) - 1:
                    await asyncio.sleep(params['time_between_orders'])
                    
            except Exception as e:
                print(f"Balanced chunk {i} failed: {e}")
                
        # Collect results
        await asyncio.sleep(2.0)
        
        for order in executed_orders:
            try:
                filled = self.exchange.fetch_order(order['id'], symbol)
                total_filled += filled['filled']
                if filled['average']:
                    total_cost += filled['filled'] * filled['average']
                    
                # Cancel unfilled portions
                if filled['status'] == 'open':
                    self.exchange.cancel_order(order['id'], symbol)
                    
            except:
                pass
                
        avg_price = total_cost / total_filled if total_filled > 0 else 0
        
        return {
            'success': total_filled >= amount * 0.8,  # 80% fill considered success
            'executed_amount': total_filled,
            'average_price': avg_price,
            'strategy': 'balanced',
            'fill_rate': total_filled / amount
        }
        
    async def _monitor_passive_order(
        self,
        order_id: str,
        symbol: str,
        reference_price: float,
        cancel_threshold_bps: float,
        max_wait_seconds: float
    ) -> Optional[dict]:
        """Monitor passive order for fill or cancellation conditions"""
        start_time = time.time()
        
        while time.time() - start_time < max_wait_seconds:
            try:
                # Check order status
                order = self.exchange.fetch_order(order_id, symbol)
                
                if order['status'] == 'closed':
                    return order
                    
                # Check if market has moved too far
                ticker = self.exchange.fetch_ticker(symbol)
                current_price = ticker['last']
                
                price_move_bps = abs(current_price - reference_price) / reference_price * 10000
                
                if price_move_bps > cancel_threshold_bps:
                    print(f"Market moved {price_move_bps:.1f} bps, cancelling passive order")
                    return None
                    
                await asyncio.sleep(0.5)  # Check every 500ms
                
            except Exception as e:
                print(f"Error monitoring order: {e}")
                
        return None  # Timeout
        
    def get_execution_analytics(self, executions: List[dict]) -> dict:
        """Analyze execution performance"""
        if not executions:
            return {}
            
        df = pd.DataFrame(executions)
        
        # Group by strategy
        strategy_stats = {}
        
        for strategy in df['strategy'].unique():
            strategy_df = df[df['strategy'] == strategy]
            
            stats = {
                'count': len(strategy_df),
                'success_rate': (strategy_df['success'].sum() / len(strategy_df)) * 100,
                'avg_fill_amount': strategy_df['executed_amount'].mean(),
                'avg_price': strategy_df['average_price'].mean()
            }
            
            # Calculate slippage if reference price available
            if 'reference_price' in strategy_df.columns:
                slippage = (strategy_df['average_price'] - strategy_df['reference_price']) / \
                          strategy_df['reference_price'] * 10000
                stats['avg_slippage_bps'] = slippage.mean()
                stats['max_slippage_bps'] = slippage.max()
                
            strategy_stats[strategy] = stats
            
        return {
            'total_executions': len(df),
            'overall_success_rate': (df['success'].sum() / len(df)) * 100,
            'strategy_breakdown': strategy_stats
        }


# Testing function
async def test_ml_executor():
    """Test ML-enhanced executor with sample features"""
    
    # Initialize with test config
    exec_config = {
        'slippage_model_pct': 0.0005,
        'max_order_book_levels': 50,
        'min_order_size': 0.001
    }
    
    ml_config = {
        'thresholds': {
            'spread_stability_passive': 0.2,
            'spread_stability_aggressive': 0.6,
            'ofi_urgent': 0.7,
            'pressure_imbalance_signal': 0.5,
            'book_resilience_min': 0.3,
            'volume_concentration_liquid': 0.6
        },
        'weights': {
            'spread_stability_norm_100': 1.0,
            'ofi_normalized_1m': 0.8,
            'pressure_imbalance_weighted': 0.7,
            'book_resilience': 0.6,
            'volume_concentration': 0.5
        }
    }
    
    # Mock exchange for testing
    class MockExchange:
        def fetch_order_book(self, symbol, limit=50):
            return {
                'bids': [[100000, 1.0], [99995, 2.0], [99990, 3.0]],
                'asks': [[100010, 1.0], [100015, 2.0], [100020, 3.0]]
            }
            
        def fetch_ticker(self, symbol):
            return {'last': 100005, 'bid': 100000, 'ask': 100010}
            
    executor = MLEnhancedOrderExecutor(MockExchange(), exec_config, ml_config)
    
    # Test scenarios
    test_features = [
        {
            'name': 'Stable market',
            'features': {
                'spread_stability_norm_100': 0.1,
                'ofi_normalized_1m': 0.1,
                'pressure_imbalance_weighted': 0.2,
                'book_resilience': 0.8,
                'volume_concentration': 0.7
            }
        },
        {
            'name': 'High OFI urgent',
            'features': {
                'spread_stability_norm_100': 0.5,
                'ofi_normalized_1m': 0.85,
                'pressure_imbalance_weighted': 0.7,
                'book_resilience': 0.5,
                'volume_concentration': 0.6
            }
        }
    ]
    
    for test in test_features:
        print(f"\nTesting: {test['name']}")
        decision = executor.analyze_ml_features(test['features'])
        print(f"Decision: {decision.strategy} ({decision.confidence:.2f} confidence)")
        print(f"Order type: {decision.order_type}")
        print(f"Size allocation: {decision.size_allocation}")


if __name__ == "__main__":
    asyncio.run(test_ml_executor())