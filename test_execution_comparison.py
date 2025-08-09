"""
Test Harness for Improved Order Execution System
Safely compares old vs new execution performance with paper trading
"""

import ccxt
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import time
import logging
from datetime import datetime
import json
import os
from collections import defaultdict
import asyncio
import threading
from queue import Queue

# Import both execution systems
from smartorderexecutor import SmartOrderExecutor
from improved_order_executor import ImprovedOrderExecutor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PaperTradingExchange:
    """
    Mock exchange wrapper for paper trading that simulates real market conditions
    """
    
    def __init__(self, real_exchange: ccxt.Exchange, initial_balance_usd: float = 100000):
        self.real_exchange = real_exchange
        self.balance_usd = initial_balance_usd
        self.positions = {}
        self.orders = {}
        self.order_counter = 1000
        self.fill_simulator = OrderFillSimulator()
        
        # Track all orders for analysis
        self.order_history = []
        self.fill_history = []
        
        # Copy exchange properties
        self.id = real_exchange.id
        self.has = real_exchange.has
        
    def fetch_ticker(self, symbol: str):
        """Fetch real ticker data"""
        return self.real_exchange.fetch_ticker(symbol)
    
    def fetch_l2_order_book(self, symbol: str, limit: int = None):
        """Fetch real order book data"""
        return self.real_exchange.fetch_l2_order_book(symbol, limit)
    
    def create_order(self, symbol: str, type: str, side: str, amount: float, 
                    price: float = None, params: dict = None):
        """Simulate order creation with realistic fill behavior"""
        try:
            # Generate order ID
            order_id = f"PAPER_{self.order_counter}"
            self.order_counter += 1
            
            # Get current market data
            ticker = self.fetch_ticker(symbol)
            current_price = ticker['last']
            
            # Create order object
            order = {
                'id': order_id,
                'symbol': symbol,
                'type': type,
                'side': side,
                'amount': amount,
                'price': price,
                'status': 'open',
                'filled': 0,
                'remaining': amount,
                'timestamp': int(time.time() * 1000),
                'datetime': datetime.utcnow().isoformat(),
                'params': params or {},
                'fills': []
            }
            
            # Store order
            self.orders[order_id] = order
            self.order_history.append(order.copy())
            
            # Simulate immediate fill for market orders
            if type == 'market':
                fill_result = self.fill_simulator.simulate_market_fill(
                    symbol, side, amount, self.fetch_l2_order_book(symbol)
                )
                order['status'] = 'closed'
                order['filled'] = amount
                order['remaining'] = 0
                order['average'] = fill_result['avg_price']
                order['fills'] = fill_result['fills']
                
                # Record fills
                self.fill_history.extend(fill_result['fills'])
                
            # For limit orders, check if immediate fill is possible
            elif type == 'limit':
                if params and params.get('postOnly'):
                    # Post-only orders must not cross the spread
                    if self._would_cross_spread(symbol, side, price):
                        raise ccxt.InvalidOrder("Post-only order would cross the spread")
                
                # Check for immediate fill
                fill_result = self.fill_simulator.simulate_limit_fill(
                    symbol, side, amount, price, self.fetch_l2_order_book(symbol)
                )
                
                if fill_result['filled_amount'] > 0:
                    order['filled'] = fill_result['filled_amount']
                    order['remaining'] = amount - fill_result['filled_amount']
                    order['average'] = fill_result['avg_price']
                    order['fills'] = fill_result['fills']
                    
                    if order['remaining'] == 0:
                        order['status'] = 'closed'
                    
                    # Record fills
                    self.fill_history.extend(fill_result['fills'])
            
            logger.info(f"Paper order created: {order_id} - {side} {amount} {symbol} @ {price or 'market'}")
            return order
            
        except Exception as e:
            logger.error(f"Error creating paper order: {e}")
            raise
    
    def cancel_all_orders(self, symbol: str):
        """Cancel all open orders for a symbol"""
        for order_id, order in list(self.orders.items()):
            if order['symbol'] == symbol and order['status'] == 'open':
                order['status'] = 'canceled'
                logger.info(f"Paper order canceled: {order_id}")
    
    def _would_cross_spread(self, symbol: str, side: str, price: float) -> bool:
        """Check if limit price would cross the spread"""
        ticker = self.fetch_ticker(symbol)
        if side == 'buy':
            return price >= ticker['ask']
        else:
            return price <= ticker['bid']


class OrderFillSimulator:
    """
    Simulates realistic order fills based on order book data
    """
    
    def simulate_market_fill(self, symbol: str, side: str, amount: float, 
                           order_book: dict) -> dict:
        """Simulate market order fill with realistic slippage"""
        fills = []
        remaining = amount
        total_cost = 0
        
        # Walk through the order book
        book_side = order_book['asks'] if side == 'buy' else order_book['bids']
        
        for price, volume in book_side:
            if remaining <= 0:
                break
                
            fill_amount = min(remaining, volume)
            fills.append({
                'price': price,
                'amount': fill_amount,
                'timestamp': int(time.time() * 1000)
            })
            
            total_cost += fill_amount * price
            remaining -= fill_amount
        
        avg_price = total_cost / (amount - remaining) if amount > remaining else 0
        
        return {
            'fills': fills,
            'avg_price': avg_price,
            'filled_amount': amount - remaining,
            'slippage': self._calculate_slippage(side, avg_price, book_side[0][0] if book_side else 0)
        }
    
    def simulate_limit_fill(self, symbol: str, side: str, amount: float, 
                          limit_price: float, order_book: dict) -> dict:
        """Simulate limit order fill based on order book"""
        fills = []
        
        # Check if order can be filled immediately
        book_side = order_book['bids'] if side == 'buy' else order_book['asks']
        
        if not book_side:
            return {'fills': [], 'avg_price': 0, 'filled_amount': 0}
        
        best_price = book_side[0][0]
        
        # For buy orders, fill if our price >= ask
        # For sell orders, fill if our price <= bid
        can_fill = (side == 'buy' and limit_price >= best_price) or \
                   (side == 'sell' and limit_price <= best_price)
        
        if can_fill:
            # Simulate partial fill based on available liquidity
            fill_amount = min(amount, book_side[0][1] * 0.5)  # Assume we get 50% of top level
            fills.append({
                'price': limit_price,
                'amount': fill_amount,
                'timestamp': int(time.time() * 1000)
            })
            
            return {
                'fills': fills,
                'avg_price': limit_price,
                'filled_amount': fill_amount
            }
        
        return {'fills': [], 'avg_price': 0, 'filled_amount': 0}
    
    def _calculate_slippage(self, side: str, fill_price: float, best_price: float) -> float:
        """Calculate slippage percentage"""
        if best_price == 0:
            return 0
            
        if side == 'buy':
            return (fill_price - best_price) / best_price
        else:
            return (best_price - fill_price) / best_price


class ExecutionTestHarness:
    """
    Main test harness for comparing execution systems
    """
    
    def __init__(self, exchange_config: dict, test_config: dict):
        """
        Initialize test harness
        
        Args:
            exchange_config: Bybit exchange configuration
            test_config: Test parameters (symbols, order sizes, etc.)
        """
        # Initialize real exchange for market data
        self.real_exchange = ccxt.bybit({
            'apiKey': exchange_config.get('api_key'),
            'secret': exchange_config.get('api_secret'),
            'enableRateLimit': True,
            'options': {
                'defaultType': 'swap',  # For perpetual futures
                'adjustForTimeDifference': True
            }
        })
        
        # Create paper trading wrappers
        self.paper_exchange_old = PaperTradingExchange(self.real_exchange)
        self.paper_exchange_new = PaperTradingExchange(self.real_exchange)
        
        # Initialize execution systems
        old_config = {
            'slippage_model_pct': 0.0005,
            'max_order_book_levels': 20
        }
        
        new_config = {
            'min_order_book_depth': 100,
            'liquidity_impact_threshold': 0.001,
            'max_single_order_pct': 0.2,
            'min_order_size_usd': 10,
            'passive_spread_bps': 1,
            'aggressive_spread_bps': 5,
            'post_only_retry_limit': 3,
            'order_timeout_seconds': 30,
            'between_order_delay_ms': 100,
            'maker_fee': -0.00025,
            'taker_fee': 0.00075
        }
        
        self.old_executor = SmartOrderExecutor(self.paper_exchange_old, old_config)
        self.new_executor = ImprovedOrderExecutor(self.paper_exchange_new, new_config)
        
        # Test configuration
        self.test_config = test_config
        self.test_results = []
        self.real_time_metrics = Queue()
        
        # Safety checks
        self.safety_limits = {
            'max_position_size_usd': test_config.get('max_position_size_usd', 10000),
            'max_orders_per_minute': test_config.get('max_orders_per_minute', 60),
            'emergency_stop': False
        }
        
        # Metrics tracking
        self.metrics = {
            'old_system': defaultdict(list),
            'new_system': defaultdict(list)
        }
        
    def run_comparison_test(self, test_scenarios: List[dict]) -> dict:
        """
        Run comparison tests between old and new execution systems
        
        Args:
            test_scenarios: List of test scenarios with order parameters
            
        Returns:
            Comparison results and metrics
        """
        logger.info("Starting execution system comparison test")
        
        results = {
            'scenarios': [],
            'summary': {},
            'timestamp': datetime.utcnow().isoformat()
        }
        
        for i, scenario in enumerate(test_scenarios):
            if self.safety_limits['emergency_stop']:
                logger.warning("Emergency stop activated, halting tests")
                break
                
            logger.info(f"\n--- Running Test Scenario {i+1}/{len(test_scenarios)} ---")
            logger.info(f"Scenario: {scenario}")
            
            # Safety check
            if not self._safety_check(scenario):
                logger.warning(f"Scenario {i+1} failed safety check, skipping")
                continue
            
            # Run test for both systems
            scenario_result = self._run_single_scenario(scenario)
            results['scenarios'].append(scenario_result)
            
            # Real-time metrics update
            self._update_real_time_metrics(scenario_result)
            
            # Brief pause between scenarios
            time.sleep(2)
        
        # Generate summary statistics
        results['summary'] = self._generate_summary_statistics(results['scenarios'])
        
        # Save results
        self._save_test_results(results)
        
        return results
    
    def _run_single_scenario(self, scenario: dict) -> dict:
        """Run a single test scenario on both systems"""
        symbol = scenario['symbol']
        side = scenario['side']
        amount_usd = scenario['amount_usd']
        urgency = scenario.get('urgency', 'medium')
        
        # Get current market price for old system
        ticker = self.real_exchange.fetch_ticker(symbol)
        current_price = ticker['last']
        amount = amount_usd / current_price
        
        # Test old system
        logger.info("Testing OLD execution system...")
        old_start = time.time()
        
        try:
            old_result = self.old_executor.execute_order(
                symbol=symbol,
                side=side,
                amount=amount,
                desired_price=current_price,
                order_type='limit'
            )
            old_execution_time = time.time() - old_start
            
            # Extract metrics from old system
            old_metrics = self._extract_old_system_metrics(
                old_result, amount_usd, current_price, old_execution_time
            )
        except Exception as e:
            logger.error(f"Old system error: {e}")
            old_metrics = {'error': str(e), 'success': False}
        
        # Test new system
        logger.info("Testing NEW execution system...")
        new_start = time.time()
        
        try:
            new_result = self.new_executor.execute_smart_order(
                symbol=symbol,
                side=side,
                amount_usd=amount_usd,
                urgency=urgency,
                signal_strength=scenario.get('signal_strength')
            )
            new_execution_time = time.time() - new_start
            
            # Metrics are already well-structured in new system
            new_metrics = new_result
        except Exception as e:
            logger.error(f"New system error: {e}")
            new_metrics = {'error': str(e), 'success': False}
        
        # Compare results
        comparison = self._compare_execution_results(old_metrics, new_metrics)
        
        return {
            'scenario': scenario,
            'old_system': old_metrics,
            'new_system': new_metrics,
            'comparison': comparison,
            'timestamp': datetime.utcnow().isoformat()
        }
    
    def _extract_old_system_metrics(self, order_result: dict, amount_usd: float, 
                                   base_price: float, execution_time: float) -> dict:
        """Extract metrics from old system order result"""
        if not order_result:
            return {
                'success': False,
                'execution_time_seconds': execution_time,
                'error': 'No order result'
            }
        
        # Get order details from paper trading
        order_id = order_result.get('id')
        if not order_id:
            return {
                'success': False,
                'execution_time_seconds': execution_time,
                'error': 'No order ID'
            }
        
        paper_order = self.paper_exchange_old.orders.get(order_id, {})
        
        # Calculate metrics
        filled_amount = paper_order.get('filled', 0)
        avg_price = paper_order.get('average', base_price)
        filled_usd = filled_amount * avg_price
        
        # Estimate fees (old system doesn't track this)
        if paper_order.get('type') == 'market':
            fees_usd = filled_usd * 0.00075  # Taker fee
        else:
            fees_usd = filled_usd * 0.00075  # Conservative estimate
        
        # Calculate slippage
        if paper_order.get('side') == 'buy':
            slippage_pct = (avg_price - base_price) / base_price
        else:
            slippage_pct = (base_price - avg_price) / base_price
        
        return {
            'success': paper_order.get('status') in ['closed', 'open'],
            'executed_orders': [paper_order],
            'total_orders': 1,
            'amount_requested_usd': amount_usd,
            'amount_filled_usd': filled_usd,
            'fill_rate': filled_usd / amount_usd if amount_usd > 0 else 0,
            'avg_fill_price': avg_price,
            'total_fees_usd': fees_usd,
            'net_amount_usd': filled_usd - fees_usd,
            'slippage_pct': slippage_pct,
            'execution_time_seconds': execution_time,
            'strategy_used': 'simple_limit'
        }
    
    def _compare_execution_results(self, old_metrics: dict, new_metrics: dict) -> dict:
        """Compare execution results between old and new systems"""
        comparison = {
            'improvement_metrics': {},
            'winner': None,
            'key_differences': []
        }
        
        # Both systems must succeed for valid comparison
        if not (old_metrics.get('success') and new_metrics.get('success')):
            comparison['winner'] = 'new' if new_metrics.get('success') else 'old'
            comparison['key_differences'].append("One system failed to execute")
            return comparison
        
        # Calculate improvements
        metrics_to_compare = [
            ('slippage_pct', 'lower_better'),
            ('total_fees_usd', 'lower_better'),
            ('fill_rate', 'higher_better'),
            ('execution_time_seconds', 'lower_better')
        ]
        
        score_old = 0
        score_new = 0
        
        for metric, direction in metrics_to_compare:
            old_val = old_metrics.get(metric, 0)
            new_val = new_metrics.get(metric, 0)
            
            if old_val == 0 and new_val == 0:
                continue
                
            # Calculate improvement percentage
            if direction == 'lower_better':
                improvement = (old_val - new_val) / abs(old_val) * 100 if old_val != 0 else 0
                if new_val < old_val:
                    score_new += 1
                else:
                    score_old += 1
            else:
                improvement = (new_val - old_val) / abs(old_val) * 100 if old_val != 0 else 0
                if new_val > old_val:
                    score_new += 1
                else:
                    score_old += 1
            
            comparison['improvement_metrics'][metric] = {
                'old_value': old_val,
                'new_value': new_val,
                'improvement_pct': improvement,
                'better_system': 'new' if improvement > 0 else 'old'
            }
        
        # Determine winner
        comparison['winner'] = 'new' if score_new > score_old else 'old'
        
        # Key differences
        if new_metrics.get('total_orders', 0) > 1:
            comparison['key_differences'].append(
                f"New system used {new_metrics['total_orders']} orders vs 1 in old system"
            )
        
        if new_metrics.get('strategy_used') != 'simple_limit':
            comparison['key_differences'].append(
                f"New system used {new_metrics.get('strategy_used')} strategy"
            )
        
        # Calculate cost savings
        fee_savings = old_metrics.get('total_fees_usd', 0) - new_metrics.get('total_fees_usd', 0)
        if fee_savings > 0:
            comparison['key_differences'].append(
                f"New system saved ${fee_savings:.2f} in fees"
            )
        
        return comparison
    
    def _safety_check(self, scenario: dict) -> bool:
        """Perform safety checks before executing scenario"""
        # Check position size
        if scenario['amount_usd'] > self.safety_limits['max_position_size_usd']:
            logger.warning(f"Order size ${scenario['amount_usd']} exceeds safety limit")
            return False
        
        # Check order rate (simple check - could be more sophisticated)
        # This is a simplified check - in production would track actual rate
        return True
    
    def _update_real_time_metrics(self, scenario_result: dict):
        """Update real-time metrics for monitoring"""
        metrics = {
            'timestamp': datetime.utcnow().isoformat(),
            'scenario': scenario_result['scenario'],
            'old_slippage': scenario_result['old_system'].get('slippage_pct', 0),
            'new_slippage': scenario_result['new_system'].get('slippage_pct', 0),
            'old_fees': scenario_result['old_system'].get('total_fees_usd', 0),
            'new_fees': scenario_result['new_system'].get('total_fees_usd', 0),
            'winner': scenario_result['comparison'].get('winner')
        }
        
        self.real_time_metrics.put(metrics)
        
        # Log summary
        logger.info(f"\n=== Scenario Result ===")
        logger.info(f"Winner: {metrics['winner']}")
        logger.info(f"Old System - Slippage: {metrics['old_slippage']*100:.3f}%, Fees: ${metrics['old_fees']:.2f}")
        logger.info(f"New System - Slippage: {metrics['new_slippage']*100:.3f}%, Fees: ${metrics['new_fees']:.2f}")
    
    def _generate_summary_statistics(self, scenario_results: List[dict]) -> dict:
        """Generate summary statistics from all test scenarios"""
        summary = {
            'total_scenarios': len(scenario_results),
            'new_system_wins': 0,
            'old_system_wins': 0,
            'average_improvements': {},
            'total_cost_savings': 0,
            'execution_reliability': {}
        }
        
        # Aggregate metrics
        metrics_aggregator = defaultdict(lambda: {'old': [], 'new': []})
        
        for result in scenario_results:
            # Count wins
            winner = result['comparison'].get('winner')
            if winner == 'new':
                summary['new_system_wins'] += 1
            elif winner == 'old':
                summary['old_system_wins'] += 1
            
            # Collect metrics
            for metric in ['slippage_pct', 'total_fees_usd', 'fill_rate', 'execution_time_seconds']:
                old_val = result['old_system'].get(metric)
                new_val = result['new_system'].get(metric)
                
                if old_val is not None:
                    metrics_aggregator[metric]['old'].append(old_val)
                if new_val is not None:
                    metrics_aggregator[metric]['new'].append(new_val)
            
            # Track cost savings
            old_fees = result['old_system'].get('total_fees_usd', 0)
            new_fees = result['new_system'].get('total_fees_usd', 0)
            summary['total_cost_savings'] += (old_fees - new_fees)
        
        # Calculate averages
        for metric, values in metrics_aggregator.items():
            if values['old'] and values['new']:
                avg_old = np.mean(values['old'])
                avg_new = np.mean(values['new'])
                improvement = ((avg_old - avg_new) / abs(avg_old) * 100) if metric in ['slippage_pct', 'total_fees_usd', 'execution_time_seconds'] else ((avg_new - avg_old) / abs(avg_old) * 100)
                
                summary['average_improvements'][metric] = {
                    'old_average': avg_old,
                    'new_average': avg_new,
                    'improvement_pct': improvement
                }
        
        # Win rate
        total_valid = summary['new_system_wins'] + summary['old_system_wins']
        if total_valid > 0:
            summary['new_system_win_rate'] = summary['new_system_wins'] / total_valid
        
        return summary
    
    def _save_test_results(self, results: dict):
        """Save test results to file"""
        timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        filename = f"execution_test_results_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Test results saved to {filename}")
    
    def run_real_time_monitor(self):
        """Run real-time monitoring dashboard (runs in separate thread)"""
        def monitor_loop():
            while not self.safety_limits['emergency_stop']:
                try:
                    # Get latest metrics
                    if not self.real_time_metrics.empty():
                        metric = self.real_time_metrics.get()
                        
                        # Simple console output - could be enhanced with a GUI
                        print(f"\n[{metric['timestamp']}] Real-time Update:")
                        print(f"  Scenario: {metric['scenario']['symbol']} {metric['scenario']['side']} ${metric['scenario']['amount_usd']}")
                        print(f"  Winner: {metric['winner']}")
                        print(f"  Slippage - Old: {metric['old_slippage']*100:.3f}%, New: {metric['new_slippage']*100:.3f}%")
                        print(f"  Fees - Old: ${metric['old_fees']:.2f}, New: ${metric['new_fees']:.2f}")
                        
                except Exception as e:
                    logger.error(f"Monitor error: {e}")
                
                time.sleep(0.1)
        
        monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        monitor_thread.start()
    
    def emergency_stop(self):
        """Emergency stop for all testing"""
        logger.warning("EMERGENCY STOP ACTIVATED")
        self.safety_limits['emergency_stop'] = True


def create_test_scenarios(symbols: List[str], amounts: List[float], 
                         urgencies: List[str]) -> List[dict]:
    """Create a comprehensive set of test scenarios"""
    scenarios = []
    
    for symbol in symbols:
        for amount in amounts:
            for urgency in urgencies:
                # Test both buy and sell
                for side in ['buy', 'sell']:
                    scenarios.append({
                        'symbol': symbol,
                        'side': side,
                        'amount_usd': amount,
                        'urgency': urgency,
                        'signal_strength': np.random.uniform(0.5, 1.0)
                    })
    
    return scenarios


def main():
    """Main test execution"""
    print("\n=== Bybit Order Execution Comparison Test ===\n")
    
    # Configuration
    exchange_config = {
        'api_key': os.getenv('BYBIT_API_KEY'),
        'api_secret': os.getenv('BYBIT_API_SECRET')
    }
    
    test_config = {
        'max_position_size_usd': 5000,
        'max_orders_per_minute': 60
    }
    
    # Initialize test harness
    harness = ExecutionTestHarness(exchange_config, test_config)
    
    # Start real-time monitor
    harness.run_real_time_monitor()
    
    # Create test scenarios
    test_scenarios = create_test_scenarios(
        symbols=['BTC/USDT:USDT'],
        amounts=[100, 500, 1000, 2000],
        urgencies=['low', 'medium', 'high']
    )
    
    print(f"Running {len(test_scenarios)} test scenarios...")
    print("Press Ctrl+C for emergency stop\n")
    
    try:
        # Run comparison tests
        results = harness.run_comparison_test(test_scenarios)
        
        # Display summary
        print("\n=== TEST SUMMARY ===")
        summary = results['summary']
        print(f"Total Scenarios: {summary['total_scenarios']}")
        print(f"New System Wins: {summary['new_system_wins']}")
        print(f"Old System Wins: {summary['old_system_wins']}")
        print(f"New System Win Rate: {summary.get('new_system_win_rate', 0)*100:.1f}%")
        print(f"Total Fee Savings: ${summary['total_cost_savings']:.2f}")
        
        print("\nAverage Improvements:")
        for metric, improvement in summary['average_improvements'].items():
            print(f"  {metric}: {improvement['improvement_pct']:.2f}% improvement")
            print(f"    Old: {improvement['old_average']:.4f}, New: {improvement['new_average']:.4f}")
        
    except KeyboardInterrupt:
        harness.emergency_stop()
        print("\nTest interrupted by user")
    except Exception as e:
        logger.error(f"Test failed: {e}")
        harness.emergency_stop()
    
    print("\nTest completed. Results saved to file.")


if __name__ == "__main__":
    main()