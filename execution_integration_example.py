"""
Integration example showing how to use the improved execution components
This demonstrates the complete execution flow with all optimizations
"""

import ccxt
import pandas as pd
from datetime import datetime
import logging
import os
from typing import Dict, Optional

# Import the new execution components
from improved_order_executor import ImprovedOrderExecutor
from dynamic_position_sizer import DynamicPositionSizer
from execution_analytics import ExecutionAnalytics

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class OptimizedTradingExecutor:
    """
    Complete trading executor integrating all optimization components
    """
    
    def __init__(self, config: dict):
        """
        Initialize the optimized executor
        
        Args:
            config: Trading system configuration
        """
        self.config = config
        
        # Initialize exchange
        self.exchange = self._init_exchange()
        
        # Initialize execution components
        self.order_executor = ImprovedOrderExecutor(
            self.exchange,
            config.get('execution', {})
        )
        
        self.position_sizer = DynamicPositionSizer(
            config.get('position_sizing', {})
        )
        
        self.execution_analytics = ExecutionAnalytics(
            config.get('analytics', {})
        )
        
        # Track positions
        self.positions = {}
        self.account_balance = config.get('initial_balance', 10000)
        
    def _init_exchange(self) -> ccxt.Exchange:
        """Initialize exchange connection"""
        exchange_config = self.config.get('exchange', {})
        exchange_name = exchange_config.get('name', 'bybit')
        
        try:
            exchange_class = getattr(ccxt, exchange_name)
            exchange = exchange_class({
                'apiKey': os.getenv('BYBIT_API_KEY', ''),
                'secret': os.getenv('BYBIT_API_SECRET', ''),
                'enableRateLimit': True,
                'options': {
                    'defaultType': exchange_config.get('market_type', 'linear'),
                    'adjustForTimeDifference': True
                }
            })
            
            if exchange_config.get('testnet', True):
                exchange.set_sandbox_mode(True)
                
            logger.info(f"Exchange {exchange_name} initialized successfully")
            return exchange
            
        except Exception as e:
            logger.error(f"Failed to initialize exchange: {e}")
            raise
    
    def execute_signal(self,
                      symbol: str,
                      signal: Dict,
                      l2_snapshots: Optional[pd.DataFrame] = None,
                      price_history: Optional[pd.DataFrame] = None) -> Dict:
        """
        Execute a trading signal with full optimization
        
        Args:
            symbol: Trading symbol (e.g., 'BTC/USDT:USDT')
            signal: Signal dictionary with 'side', 'strength', 'confidence'
            l2_snapshots: Recent L2 order book data
            price_history: Recent price history
            
        Returns:
            Execution result
        """
        try:
            # Extract signal components
            side = signal.get('side')  # 'buy' or 'sell'
            signal_strength = signal.get('strength', 0.5)
            signal_confidence = signal.get('confidence', 0.7)
            urgency = signal.get('urgency', 'medium')
            
            if not side:
                logger.error("No side specified in signal")
                return {'success': False, 'error': 'No side specified'}
            
            # Get current market data
            ticker = self.exchange.fetch_ticker(symbol)
            current_price = ticker['last']
            
            # Get funding rate for perpetuals
            funding_rate = None
            if ':' in symbol:  # Perpetual futures
                try:
                    funding = self.exchange.fetch_funding_rate(symbol)
                    funding_rate = funding['fundingRate']
                except:
                    pass
            
            # 1. Calculate dynamic position size
            logger.info(f"Calculating position size for {side} signal (strength: {signal_strength:.2f})")
            
            position_result = self.position_sizer.calculate_dynamic_position_size(
                account_balance=self.account_balance,
                current_price=current_price,
                side=side,
                signal_strength=signal_strength * signal_confidence,
                l2_snapshots=l2_snapshots,
                price_history=price_history,
                funding_rate=funding_rate,
                existing_positions=self.positions
            )
            
            position_value_usd = position_result['position_value']
            
            logger.info(f"Position size calculated: ${position_value_usd:.2f} "
                       f"({position_result['position_size']:.6f} BTC), "
                       f"risk: {position_result['risk_pct']*100:.2f}%")
            
            # 2. Get adaptive execution parameters
            adaptive_params = self.execution_analytics.get_adaptive_parameters()
            logger.info(f"Using adaptive parameters: {adaptive_params}")
            
            # 3. Execute the order with smart routing
            execution_result = self.order_executor.execute_smart_order(
                symbol=symbol,
                side=side,
                amount_usd=position_value_usd,
                urgency=urgency,
                signal_strength=signal_strength
            )
            
            # 4. Record execution for analytics
            execution_result['signal'] = signal
            execution_result['position_sizing'] = position_result
            self.execution_analytics.record_execution(execution_result)
            
            # 5. Update position tracking
            if execution_result['success']:
                position_key = f"{symbol}_{side}"
                self.positions[position_key] = {
                    'symbol': symbol,
                    'side': side,
                    'size': position_result['position_size'],
                    'value': execution_result['amount_filled_usd'],
                    'entry_price': execution_result['avg_fill_price'],
                    'entry_time': datetime.now(),
                    'signal': signal
                }
                
                # Update account balance (simplified)
                fees = execution_result['total_fees_usd']
                self.account_balance -= fees
            
            # 6. Generate execution summary
            summary = {
                'success': execution_result['success'],
                'symbol': symbol,
                'side': side,
                'signal_strength': signal_strength,
                'position_value_usd': position_value_usd,
                'filled_value_usd': execution_result['amount_filled_usd'],
                'fill_rate': execution_result['fill_rate'],
                'avg_price': execution_result['avg_fill_price'],
                'slippage_pct': execution_result['slippage_pct'],
                'total_fees_usd': execution_result['total_fees_usd'],
                'execution_time': execution_result['execution_time_seconds'],
                'strategy_used': execution_result['strategy_used'],
                'orders_placed': execution_result['total_orders']
            }
            
            # Log summary
            logger.info(f"Execution complete: {side} ${summary['filled_value_usd']:.2f} @ "
                       f"${summary['avg_price']:.2f}, slippage: {summary['slippage_pct']*100:.3f}%, "
                       f"fees: ${summary['total_fees_usd']:.2f}")
            
            return summary
            
        except Exception as e:
            logger.error(f"Error executing signal: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def get_execution_report(self) -> Dict:
        """Get comprehensive execution performance report"""
        return self.execution_analytics.analyze_execution_performance()


# Example usage configuration
EXAMPLE_CONFIG = {
    'exchange': {
        'name': 'bybit',
        'market_type': 'linear',
        'testnet': False
    },
    'initial_balance': 20000,
    'execution': {
        # Liquidity analysis
        'min_order_book_depth': 100,
        'liquidity_impact_threshold': 0.001,  # 10 bps
        
        # Order splitting
        'max_single_order_pct': 0.2,  # Max 20% of level
        'min_order_size_usd': 10,
        
        # Execution strategy
        'passive_spread_bps': 1,
        'aggressive_spread_bps': 5,
        'post_only_retry_limit': 3,
        
        # Timing
        'order_timeout_seconds': 30,
        'between_order_delay_ms': 100,
        
        # Fees (Bybit perpetual)
        'maker_fee': -0.00025,  # 2.5 bps rebate
        'taker_fee': 0.00075    # 7.5 bps fee
    },
    'position_sizing': {
        'base_risk_pct': 0.01,      # 1% base risk
        'max_risk_pct': 0.03,       # 3% max risk
        'target_volatility': 0.02,   # 2% daily target
        'signal_scale_min': 0.5,
        'signal_scale_max': 2.0,
        'trend_boost_factor': 1.2,
        'range_reduce_factor': 0.8,
        'funding_threshold': 0.0001,
        'max_position_value': 50000,
        'min_position_value': 100
    },
    'analytics': {
        'max_history_size': 1000,
        'analysis_window_hours': 24,
        'target_fill_rate': 0.95,
        'max_acceptable_slippage': 0.002,  # 20 bps
        'cost_warning_threshold': 0.001     # 10 bps
    }
}


def main():
    """Example of using the optimized executor"""
    # Initialize executor
    executor = OptimizedTradingExecutor(EXAMPLE_CONFIG)
    
    # Example signal
    signal = {
        'side': 'buy',
        'strength': 0.8,      # Strong signal
        'confidence': 0.9,    # High confidence
        'urgency': 'medium'   # Normal urgency
    }
    
    # Execute with mock data (in production, use real L2 data)
    result = executor.execute_signal(
        symbol='BTC/USDT:USDT',
        signal=signal
    )
    
    print(f"\nExecution Result: {result}")
    
    # Get performance report
    report = executor.get_execution_report()
    print(f"\nPerformance Report: {report}")


if __name__ == "__main__":
    main()