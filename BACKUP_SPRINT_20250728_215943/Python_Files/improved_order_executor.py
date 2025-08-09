"""
Improved Smart Order Executor with Advanced Execution Algorithms
Optimized for Bybit perpetual futures with focus on reducing slippage
"""

import ccxt
import numpy as np
from typing import Dict, List, Tuple, Optional
import time
import logging

logger = logging.getLogger(__name__)


class ImprovedOrderExecutor:
    """
    Advanced order execution system with:
    - Dynamic liquidity-based order placement
    - Smart order routing and splitting
    - Post-only order optimization
    - Adaptive urgency-based execution
    """
    
    def __init__(self, exchange_api: ccxt.Exchange, exec_config: dict):
        """
        Initialize the improved executor
        
        Args:
            exchange_api: CCXT exchange instance
            exec_config: Execution configuration
        """
        self.exchange = exchange_api
        
        # Liquidity analysis parameters
        self.min_order_book_depth = exec_config.get('min_order_book_depth', 100)
        self.liquidity_impact_threshold = exec_config.get('liquidity_impact_threshold', 0.001)  # 0.1%
        
        # Order splitting parameters
        self.max_single_order_pct = exec_config.get('max_single_order_pct', 0.2)  # Max 20% of available liquidity
        self.min_order_size_usd = exec_config.get('min_order_size_usd', 10)
        
        # Execution strategy parameters
        self.passive_spread_bps = exec_config.get('passive_spread_bps', 1)  # 1 basis point inside spread
        self.aggressive_spread_bps = exec_config.get('aggressive_spread_bps', 5)  # 5 bps through spread
        self.post_only_retry_limit = exec_config.get('post_only_retry_limit', 3)
        
        # Time-based parameters
        self.order_timeout_seconds = exec_config.get('order_timeout_seconds', 30)
        self.between_order_delay_ms = exec_config.get('between_order_delay_ms', 100)
        
        # Fee optimization
        self.maker_fee = exec_config.get('maker_fee', -0.00025)  # Bybit maker rebate
        self.taker_fee = exec_config.get('taker_fee', 0.00075)   # Bybit taker fee
        
        logger.info(f"ImprovedOrderExecutor initialized with maker fee: {self.maker_fee}, taker fee: {self.taker_fee}")

    def analyze_order_book_liquidity(self, symbol: str, side: str, amount_usd: float) -> Dict:
        """
        Analyze order book to determine liquidity and optimal execution strategy
        
        Args:
            symbol: Trading symbol
            side: 'buy' or 'sell'
            amount_usd: Order size in USD
            
        Returns:
            Dictionary with liquidity analysis
        """
        try:
            # Fetch deep order book
            order_book = self.exchange.fetch_l2_order_book(symbol, limit=self.min_order_book_depth)
            
            # Determine which side to analyze
            book_side = order_book['asks'] if side == 'buy' else order_book['bids']
            
            # Calculate cumulative liquidity and average prices at different levels
            cumulative_volume = 0
            cumulative_cost = 0
            liquidity_levels = []
            
            for i, (price, volume) in enumerate(book_side):
                cumulative_volume += volume
                cumulative_cost += price * volume
                volume_usd = volume * price
                
                liquidity_levels.append({
                    'level': i,
                    'price': price,
                    'volume': volume,
                    'volume_usd': volume_usd,
                    'cumulative_volume': cumulative_volume,
                    'cumulative_volume_usd': cumulative_cost,
                    'avg_price': cumulative_cost / cumulative_volume if cumulative_volume > 0 else price
                })
                
                # Stop when we have enough liquidity for our order
                if cumulative_cost >= amount_usd * 2:  # 2x buffer
                    break
            
            # Calculate key metrics
            best_price = book_side[0][0] if book_side else None
            
            # Find the level where our order would be filled
            fill_level = None
            expected_avg_price = None
            for level in liquidity_levels:
                if level['cumulative_volume_usd'] >= amount_usd:
                    fill_level = level['level']
                    expected_avg_price = level['avg_price']
                    break
            
            # Calculate price impact
            if best_price and expected_avg_price:
                if side == 'buy':
                    price_impact = (expected_avg_price - best_price) / best_price
                else:
                    price_impact = (best_price - expected_avg_price) / best_price
            else:
                price_impact = 0
            
            # Determine execution strategy based on liquidity
            if fill_level is None:
                strategy = 'insufficient_liquidity'
            elif fill_level <= 2 and price_impact < self.liquidity_impact_threshold:
                strategy = 'aggressive'  # Good liquidity, can cross spread
            elif fill_level <= 5:
                strategy = 'passive'     # Medium liquidity, use limit orders
            else:
                strategy = 'patient'     # Thin liquidity, split orders
            
            return {
                'best_price': best_price,
                'expected_avg_price': expected_avg_price,
                'price_impact': price_impact,
                'fill_level': fill_level,
                'liquidity_levels': liquidity_levels[:10],  # Top 10 levels
                'strategy': strategy,
                'total_liquidity_usd': cumulative_cost
            }
            
        except Exception as e:
            logger.error(f"Error analyzing order book: {e}")
            return {
                'strategy': 'fallback',
                'error': str(e)
            }

    def calculate_order_slices(self, amount_usd: float, liquidity_analysis: Dict) -> List[Dict]:
        """
        Split large orders into optimal slices based on liquidity
        
        Args:
            amount_usd: Total order size in USD
            liquidity_analysis: Results from analyze_order_book_liquidity
            
        Returns:
            List of order slices with prices and sizes
        """
        slices = []
        remaining_amount = amount_usd
        
        if liquidity_analysis.get('strategy') == 'insufficient_liquidity':
            # Single small probe order
            return [{
                'size_usd': min(amount_usd, self.min_order_size_usd * 10),
                'price_strategy': 'market',
                'urgency': 'high'
            }]
        
        liquidity_levels = liquidity_analysis.get('liquidity_levels', [])
        
        for i, level in enumerate(liquidity_levels):
            if remaining_amount <= 0:
                break
                
            # Don't take more than X% of any price level
            max_take = level['volume_usd'] * self.max_single_order_pct
            slice_size = min(remaining_amount, max_take)
            
            # Skip if slice is too small
            if slice_size < self.min_order_size_usd:
                continue
            
            # Determine price strategy for this slice
            if i == 0:
                price_strategy = 'at_touch'  # Best bid/ask
            elif i <= 2:
                price_strategy = 'near_touch'  # Within few ticks
            else:
                price_strategy = 'in_book'  # Deeper in book
            
            slices.append({
                'size_usd': slice_size,
                'price_level': i,
                'price': level['price'],
                'price_strategy': price_strategy,
                'urgency': 'low' if i > 3 else 'medium'
            })
            
            remaining_amount -= slice_size
        
        # If we still have remaining amount, add a patient slice
        if remaining_amount > self.min_order_size_usd:
            slices.append({
                'size_usd': remaining_amount,
                'price_strategy': 'patient',
                'urgency': 'low'
            })
        
        return slices

    def execute_order_slice(self, symbol: str, side: str, slice_config: Dict, 
                          current_price: float, spread: float) -> Optional[Dict]:
        """
        Execute a single order slice with the appropriate strategy
        
        Args:
            symbol: Trading symbol
            side: 'buy' or 'sell'
            slice_config: Slice configuration from calculate_order_slices
            current_price: Current mid price
            spread: Current bid-ask spread
            
        Returns:
            Order result or None if failed
        """
        try:
            # Calculate order size in base currency
            amount = slice_config['size_usd'] / current_price
            
            # Determine order price based on strategy
            if slice_config['price_strategy'] == 'market':
                # Use market order for urgent fills
                return self._place_market_order(symbol, side, amount)
                
            elif slice_config['price_strategy'] == 'at_touch':
                # Place at best bid/ask
                if side == 'buy':
                    limit_price = current_price - spread/2
                else:
                    limit_price = current_price + spread/2
                    
            elif slice_config['price_strategy'] == 'near_touch':
                # Place slightly inside the spread
                offset_bps = self.passive_spread_bps if slice_config['urgency'] == 'low' else 0
                if side == 'buy':
                    limit_price = current_price - spread/2 + (current_price * offset_bps / 10000)
                else:
                    limit_price = current_price + spread/2 - (current_price * offset_bps / 10000)
                    
            elif slice_config['price_strategy'] == 'in_book':
                # Use the price from liquidity analysis
                limit_price = slice_config.get('price', current_price)
                
            else:  # patient
                # Place well inside the spread for passive fill
                offset = spread * 0.25  # 25% inside spread
                if side == 'buy':
                    limit_price = current_price - offset
                else:
                    limit_price = current_price + offset
            
            # Try post-only first if not urgent
            if slice_config['urgency'] != 'high':
                order = self._place_post_only_order(symbol, side, amount, limit_price)
                if order:
                    return order
            
            # Fall back to regular limit order
            return self._place_limit_order(symbol, side, amount, limit_price)
            
        except Exception as e:
            logger.error(f"Error executing order slice: {e}")
            return None

    def _place_post_only_order(self, symbol: str, side: str, amount: float, 
                             price: float, retry_count: int = 0) -> Optional[Dict]:
        """
        Place a post-only order with retry logic
        
        Args:
            symbol: Trading symbol
            side: 'buy' or 'sell'
            amount: Order amount
            price: Limit price
            retry_count: Current retry attempt
            
        Returns:
            Order result or None if failed
        """
        try:
            params = {
                'postOnly': True,
                'timeInForce': 'PO'  # Bybit post-only
            }
            
            order = self.exchange.create_order(
                symbol=symbol,
                type='limit',
                side=side,
                amount=amount,
                price=price,
                params=params
            )
            
            if order and order.get('id'):
                logger.info(f"Post-only order placed: {order['id']} - {side} {amount} @ {price}")
                return order
                
        except ccxt.InvalidOrder as e:
            # Order would cross the spread
            if retry_count < self.post_only_retry_limit:
                # Adjust price and retry
                spread_adjustment = 0.0001  # 1 basis point
                if side == 'buy':
                    new_price = price * (1 - spread_adjustment)
                else:
                    new_price = price * (1 + spread_adjustment)
                    
                time.sleep(0.1)  # Brief delay
                return self._place_post_only_order(symbol, side, amount, new_price, retry_count + 1)
                
        except Exception as e:
            logger.error(f"Error placing post-only order: {e}")
            
        return None

    def _place_limit_order(self, symbol: str, side: str, amount: float, price: float) -> Optional[Dict]:
        """Place a regular limit order"""
        try:
            order = self.exchange.create_order(
                symbol=symbol,
                type='limit',
                side=side,
                amount=amount,
                price=price
            )
            
            if order and order.get('id'):
                logger.info(f"Limit order placed: {order['id']} - {side} {amount} @ {price}")
                return order
                
        except Exception as e:
            logger.error(f"Error placing limit order: {e}")
            
        return None

    def _place_market_order(self, symbol: str, side: str, amount: float) -> Optional[Dict]:
        """Place a market order for urgent execution"""
        try:
            order = self.exchange.create_order(
                symbol=symbol,
                type='market',
                side=side,
                amount=amount
            )
            
            if order and order.get('id'):
                logger.info(f"Market order placed: {order['id']} - {side} {amount}")
                return order
                
        except Exception as e:
            logger.error(f"Error placing market order: {e}")
            
        return None

    def execute_smart_order(self, symbol: str, side: str, amount_usd: float, 
                          urgency: str = 'medium', signal_strength: float = None) -> Dict:
        """
        Main entry point for smart order execution
        
        Args:
            symbol: Trading symbol (e.g., 'BTC/USDT:USDT')
            side: 'buy' or 'sell'
            amount_usd: Order size in USD
            urgency: 'low', 'medium', or 'high'
            signal_strength: Optional signal strength for additional context
            
        Returns:
            Execution result dictionary
        """
        start_time = time.time()
        
        # Get current market state
        ticker = self.exchange.fetch_ticker(symbol)
        current_price = ticker['last']
        spread = ticker['ask'] - ticker['bid'] if ticker['ask'] and ticker['bid'] else current_price * 0.0001
        
        # Analyze liquidity
        liquidity = self.analyze_order_book_liquidity(symbol, side, amount_usd)
        
        # Adjust strategy based on urgency
        if urgency == 'high':
            liquidity['strategy'] = 'aggressive'
        elif urgency == 'low' and liquidity['strategy'] != 'insufficient_liquidity':
            liquidity['strategy'] = 'patient'
        
        # Calculate order slices
        slices = self.calculate_order_slices(amount_usd, liquidity)
        
        # Execute slices
        executed_orders = []
        total_filled_usd = 0
        total_fees_usd = 0
        
        for i, slice_config in enumerate(slices):
            # Update current price for each slice
            ticker = self.exchange.fetch_ticker(symbol)
            current_price = ticker['last']
            spread = ticker['ask'] - ticker['bid'] if ticker['ask'] and ticker['bid'] else current_price * 0.0001
            
            # Execute the slice
            order = self.execute_order_slice(symbol, side, slice_config, current_price, spread)
            
            if order:
                executed_orders.append(order)
                
                # Estimate filled amount and fees
                filled_usd = slice_config['size_usd']
                if order.get('type') == 'market' or slice_config['urgency'] == 'high':
                    fee = filled_usd * abs(self.taker_fee)
                else:
                    fee = filled_usd * abs(self.maker_fee)  # Could be negative (rebate)
                
                total_filled_usd += filled_usd
                total_fees_usd += fee
                
                # Delay between orders to avoid overwhelming the exchange
                if i < len(slices) - 1:
                    time.sleep(self.between_order_delay_ms / 1000)
            
            # Check timeout
            if time.time() - start_time > self.order_timeout_seconds:
                logger.warning("Order execution timeout reached")
                break
        
        # Calculate execution metrics
        execution_time = time.time() - start_time
        fill_rate = total_filled_usd / amount_usd if amount_usd > 0 else 0
        
        # Estimate average fill price
        if executed_orders:
            avg_fill_price = sum(o.get('price', current_price) for o in executed_orders) / len(executed_orders)
        else:
            avg_fill_price = current_price
        
        # Calculate realized slippage
        if side == 'buy':
            slippage_pct = (avg_fill_price - liquidity.get('best_price', current_price)) / liquidity.get('best_price', current_price)
        else:
            slippage_pct = (liquidity.get('best_price', current_price) - avg_fill_price) / liquidity.get('best_price', current_price)
        
        result = {
            'success': len(executed_orders) > 0,
            'executed_orders': executed_orders,
            'total_orders': len(executed_orders),
            'amount_requested_usd': amount_usd,
            'amount_filled_usd': total_filled_usd,
            'fill_rate': fill_rate,
            'avg_fill_price': avg_fill_price,
            'total_fees_usd': total_fees_usd,
            'net_amount_usd': total_filled_usd - total_fees_usd,
            'slippage_pct': slippage_pct,
            'execution_time_seconds': execution_time,
            'liquidity_analysis': liquidity,
            'slices_executed': len([s for s in slices if any(o for o in executed_orders)]),
            'strategy_used': liquidity.get('strategy', 'unknown')
        }
        
        # Log execution summary
        logger.info(f"Smart order execution completed: {side} ${amount_usd:.2f} @ avg price ${avg_fill_price:.2f}, "
                   f"slippage: {slippage_pct*100:.3f}%, fees: ${total_fees_usd:.2f}, "
                   f"fill rate: {fill_rate*100:.1f}%, time: {execution_time:.1f}s")
        
        return result

    def cancel_all_orders(self, symbol: str) -> bool:
        """Cancel all open orders for a symbol"""
        try:
            self.exchange.cancel_all_orders(symbol)
            return True
        except Exception as e:
            logger.error(f"Error canceling orders: {e}")
            return False