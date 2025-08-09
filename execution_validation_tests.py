"""
Execution Validation Tests for ML-Enhanced Trading System
Designed to validate execution improvements with new ML model features
Focus on spread_stability_norm_100, OFI, and stability indicators
"""

import ccxt
import pandas as pd
import numpy as np
import time
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
import asyncio
import aiohttp


@dataclass
class ExecutionTestResult:
    """Store results from execution tests"""
    test_id: str
    timestamp: datetime
    signal_strength: float
    feature_values: Dict[str, float]
    execution_strategy: str
    order_type: str
    intended_price: float
    executed_price: float
    slippage_bps: float
    fill_time_ms: float
    market_conditions: Dict[str, float]
    success: bool
    notes: str


class ExecutionValidator:
    """
    Validates execution performance with ML-enhanced signals
    Maps signal quality to execution urgency based on feature importance
    """
    
    def __init__(self, exchange_config: dict):
        """Initialize with exchange configuration"""
        self.exchange = ccxt.bybit({
            'apiKey': exchange_config.get('api_key'),
            'secret': exchange_config.get('api_secret'),
            'enableRateLimit': True,
            'options': {
                'defaultType': 'linear',  # USDT perpetual
            }
        })
        
        if exchange_config.get('testnet', False):
            self.exchange.set_sandbox_mode(True)
            
        self.symbol = 'BTC/USDT:USDT'
        self.test_results = []
        
        # Signal strength to urgency mapping
        self.urgency_thresholds = {
            'passive': 0.3,      # Below 30% strength -> passive orders
            'balanced': 0.6,     # 30-60% -> balanced approach  
            'aggressive': 0.85,  # 60-85% -> aggressive fills
            'urgent': 1.0        # Above 85% -> immediate execution
        }
        
        # Feature importance weights (from ML model)
        self.feature_weights = {
            'spread_stability_norm_100': 1.0,  # Top feature
            'ofi_normalized_1m': 0.8,
            'pressure_imbalance_weighted': 0.7,
            'book_resilience': 0.6,
            'volume_concentration': 0.5
        }
        
    async def fetch_market_conditions(self) -> Dict[str, float]:
        """Fetch current market conditions from Bybit"""
        try:
            # Fetch order book
            order_book = self.exchange.fetch_order_book(self.symbol, limit=50)
            
            # Fetch recent trades
            trades = self.exchange.fetch_trades(self.symbol, limit=100)
            
            # Fetch ticker
            ticker = self.exchange.fetch_ticker(self.symbol)
            
            # Calculate market metrics
            best_bid = order_book['bids'][0][0]
            best_ask = order_book['asks'][0][0]
            mid_price = (best_bid + best_ask) / 2
            spread_bps = ((best_ask - best_bid) / mid_price) * 10000
            
            # Calculate order book imbalance
            bid_volume_5 = sum([level[1] for level in order_book['bids'][:5]])
            ask_volume_5 = sum([level[1] for level in order_book['asks'][:5]])
            book_imbalance = (bid_volume_5 - ask_volume_5) / (bid_volume_5 + ask_volume_5)
            
            # Calculate recent volatility
            recent_prices = [trade['price'] for trade in trades]
            if len(recent_prices) > 1:
                returns = np.diff(recent_prices) / recent_prices[:-1]
                volatility_1min = np.std(returns) * np.sqrt(60)  # Annualized
            else:
                volatility_1min = 0.0
                
            # Book depth metrics
            bid_depth_10bps = self._calculate_depth_at_distance(order_book['bids'], mid_price, 10)
            ask_depth_10bps = self._calculate_depth_at_distance(order_book['asks'], mid_price, 10)
            
            return {
                'timestamp': datetime.now(),
                'mid_price': mid_price,
                'best_bid': best_bid,
                'best_ask': best_ask,
                'spread_bps': spread_bps,
                'book_imbalance': book_imbalance,
                'volatility_1min': volatility_1min,
                'bid_depth_10bps': bid_depth_10bps,
                'ask_depth_10bps': ask_depth_10bps,
                'daily_volume': ticker.get('quoteVolume', 0),
                'funding_rate': ticker.get('info', {}).get('fundingRate', 0)
            }
            
        except Exception as e:
            print(f"Error fetching market conditions: {e}")
            return {}
            
    def _calculate_depth_at_distance(self, book_side: list, mid_price: float, distance_bps: float) -> float:
        """Calculate cumulative depth within X basis points of mid price"""
        if not book_side:
            return 0.0
            
        threshold_price = mid_price * (1 + distance_bps / 10000)
        cumulative_depth = 0.0
        
        for price, volume in book_side:
            if (book_side == 'bids' and price < mid_price * (1 - distance_bps / 10000)) or \
               (book_side == 'asks' and price > threshold_price):
                break
            cumulative_depth += volume
            
        return cumulative_depth
        
    def calculate_execution_urgency(self, feature_values: Dict[str, float]) -> Tuple[float, str]:
        """
        Calculate execution urgency based on ML features
        Returns (urgency_score, strategy)
        """
        # Normalize spread stability (lower is better, so invert)
        spread_stability = feature_values.get('spread_stability_norm_100', 0.5)
        stability_score = 1 - min(spread_stability, 1.0)  # Invert: stable = high score
        
        # OFI indicates directional pressure
        ofi_score = abs(feature_values.get('ofi_normalized_1m', 0))
        
        # Pressure imbalance
        pressure_score = abs(feature_values.get('pressure_imbalance_weighted', 0))
        
        # Book resilience (higher is better)
        resilience_score = min(feature_values.get('book_resilience', 0.5), 1.0)
        
        # Volume concentration (higher means more liquid at top)
        concentration_score = feature_values.get('volume_concentration', 0.5)
        
        # Weighted urgency calculation
        urgency = (
            self.feature_weights['spread_stability_norm_100'] * stability_score +
            self.feature_weights['ofi_normalized_1m'] * ofi_score +
            self.feature_weights['pressure_imbalance_weighted'] * pressure_score +
            self.feature_weights['book_resilience'] * resilience_score +
            self.feature_weights['volume_concentration'] * concentration_score
        ) / sum(self.feature_weights.values())
        
        # Determine strategy
        if urgency < self.urgency_thresholds['passive']:
            strategy = 'passive'
        elif urgency < self.urgency_thresholds['balanced']:
            strategy = 'balanced'
        elif urgency < self.urgency_thresholds['aggressive']:
            strategy = 'aggressive'
        else:
            strategy = 'urgent'
            
        return urgency, strategy
        
    async def test_execution_strategy(
        self, 
        signal_strength: float,
        feature_values: Dict[str, float],
        test_size: float = 0.001,  # 0.001 BTC for testing
        dry_run: bool = True
    ) -> ExecutionTestResult:
        """
        Test execution with given signal and features
        """
        test_id = f"test_{int(time.time() * 1000)}"
        start_time = time.time()
        
        try:
            # Get current market conditions
            market_conditions = await self.fetch_market_conditions()
            
            # Calculate execution urgency
            urgency_score, strategy = self.calculate_execution_urgency(feature_values)
            
            # Determine order parameters based on strategy
            if strategy == 'passive':
                # Place limit order at favorable price
                order_type = 'limit'
                if signal_strength > 0:  # Buy signal
                    # Place at or below best bid for maker rebate
                    limit_price = market_conditions['best_bid']
                else:  # Sell signal
                    limit_price = market_conditions['best_ask']
                    
            elif strategy == 'balanced':
                # Place limit order near touch
                order_type = 'limit'
                if signal_strength > 0:
                    # Slightly above best bid
                    limit_price = market_conditions['best_bid'] * 1.0001
                else:
                    limit_price = market_conditions['best_ask'] * 0.9999
                    
            elif strategy == 'aggressive':
                # Cross spread partially
                order_type = 'limit'
                spread = market_conditions['best_ask'] - market_conditions['best_bid']
                if signal_strength > 0:
                    # Cross 75% of spread
                    limit_price = market_conditions['best_bid'] + spread * 0.75
                else:
                    limit_price = market_conditions['best_ask'] - spread * 0.75
                    
            else:  # urgent
                # Market order or aggressive limit
                order_type = 'market'
                limit_price = market_conditions['mid_price']
                
            # Execute order (or simulate)
            if dry_run:
                # Simulate execution
                if order_type == 'market':
                    executed_price = market_conditions['best_ask'] if signal_strength > 0 else market_conditions['best_bid']
                    fill_time_ms = 50  # Assume 50ms for market order
                else:
                    # Simulate based on book conditions
                    if strategy == 'passive':
                        # May take time to fill
                        executed_price = limit_price
                        fill_time_ms = 5000  # Assume 5 seconds
                    else:
                        # Should fill quickly
                        executed_price = limit_price
                        fill_time_ms = 500
                        
                success = True
                notes = f"Simulated {strategy} execution"
                
            else:
                # Real execution
                side = 'buy' if signal_strength > 0 else 'sell'
                
                if order_type == 'market':
                    order = self.exchange.create_market_order(
                        symbol=self.symbol,
                        side=side,
                        amount=test_size
                    )
                else:
                    order = self.exchange.create_limit_order(
                        symbol=self.symbol,
                        side=side,
                        amount=test_size,
                        price=limit_price
                    )
                    
                # Wait for fill (with timeout)
                fill_timeout = 30 if strategy == 'passive' else 5
                filled = await self._wait_for_fill(order['id'], timeout=fill_timeout)
                
                if filled:
                    executed_price = filled['average']
                    success = True
                    notes = f"Order {order['id']} filled"
                else:
                    # Cancel unfilled order
                    self.exchange.cancel_order(order['id'], self.symbol)
                    executed_price = 0
                    success = False
                    notes = f"Order {order['id']} cancelled - no fill"
                    
                fill_time_ms = (time.time() - start_time) * 1000
                
            # Calculate slippage
            intended_price = market_conditions['mid_price']
            if executed_price > 0:
                if signal_strength > 0:  # Buy
                    slippage_bps = ((executed_price - intended_price) / intended_price) * 10000
                else:  # Sell
                    slippage_bps = ((intended_price - executed_price) / intended_price) * 10000
            else:
                slippage_bps = 0
                
            # Create result
            result = ExecutionTestResult(
                test_id=test_id,
                timestamp=datetime.now(),
                signal_strength=signal_strength,
                feature_values=feature_values,
                execution_strategy=strategy,
                order_type=order_type,
                intended_price=intended_price,
                executed_price=executed_price,
                slippage_bps=slippage_bps,
                fill_time_ms=fill_time_ms,
                market_conditions=market_conditions,
                success=success,
                notes=notes
            )
            
            self.test_results.append(result)
            return result
            
        except Exception as e:
            print(f"Error in execution test: {e}")
            return ExecutionTestResult(
                test_id=test_id,
                timestamp=datetime.now(),
                signal_strength=signal_strength,
                feature_values=feature_values,
                execution_strategy='error',
                order_type='none',
                intended_price=0,
                executed_price=0,
                slippage_bps=0,
                fill_time_ms=0,
                market_conditions=market_conditions,
                success=False,
                notes=f"Error: {str(e)}"
            )
            
    async def _wait_for_fill(self, order_id: str, timeout: float = 30) -> Optional[dict]:
        """Wait for order to fill with timeout"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                order = self.exchange.fetch_order(order_id, self.symbol)
                if order['status'] == 'closed':
                    return order
                    
                await asyncio.sleep(0.1)  # Check every 100ms
                
            except Exception as e:
                print(f"Error checking order status: {e}")
                
        return None
        
    def generate_test_scenarios(self) -> List[Dict[str, any]]:
        """Generate comprehensive test scenarios"""
        scenarios = []
        
        # Scenario 1: High stability, low OFI -> Passive execution preferred
        scenarios.append({
            'name': 'stable_market_passive',
            'signal_strength': 0.7,
            'features': {
                'spread_stability_norm_100': 0.1,  # Very stable
                'ofi_normalized_1m': 0.05,         # Low order flow
                'pressure_imbalance_weighted': 0.1,
                'book_resilience': 0.8,
                'volume_concentration': 0.7
            },
            'expected_strategy': 'passive'
        })
        
        # Scenario 2: High OFI, pressure imbalance -> Urgent execution
        scenarios.append({
            'name': 'high_ofi_urgent',
            'signal_strength': 0.8,
            'features': {
                'spread_stability_norm_100': 0.5,
                'ofi_normalized_1m': 0.9,          # High order flow
                'pressure_imbalance_weighted': 0.8, # Strong imbalance
                'book_resilience': 0.5,
                'volume_concentration': 0.6
            },
            'expected_strategy': 'urgent'
        })
        
        # Scenario 3: Unstable spread, low resilience -> Aggressive but careful
        scenarios.append({
            'name': 'unstable_aggressive',
            'signal_strength': 0.6,
            'features': {
                'spread_stability_norm_100': 0.8,  # Unstable
                'ofi_normalized_1m': 0.4,
                'pressure_imbalance_weighted': 0.5,
                'book_resilience': 0.3,            # Low resilience
                'volume_concentration': 0.5
            },
            'expected_strategy': 'aggressive'
        })
        
        # Scenario 4: Balanced conditions
        scenarios.append({
            'name': 'balanced_conditions',
            'signal_strength': 0.5,
            'features': {
                'spread_stability_norm_100': 0.4,
                'ofi_normalized_1m': 0.3,
                'pressure_imbalance_weighted': 0.3,
                'book_resilience': 0.6,
                'volume_concentration': 0.6
            },
            'expected_strategy': 'balanced'
        })
        
        # Scenario 5: Weak signal with good conditions -> Skip or minimal
        scenarios.append({
            'name': 'weak_signal_good_conditions',
            'signal_strength': 0.2,
            'features': {
                'spread_stability_norm_100': 0.1,
                'ofi_normalized_1m': 0.1,
                'pressure_imbalance_weighted': 0.1,
                'book_resilience': 0.9,
                'volume_concentration': 0.8
            },
            'expected_strategy': 'passive'
        })
        
        return scenarios
        
    async def run_validation_suite(self, dry_run: bool = True):
        """Run complete validation test suite"""
        print(f"Starting execution validation suite (dry_run={dry_run})")
        print(f"Testing with symbol: {self.symbol}")
        print("-" * 80)
        
        # Get current market snapshot
        market = await self.fetch_market_conditions()
        print(f"Current market conditions:")
        print(f"  BTC Price: ${market.get('mid_price', 0):,.2f}")
        print(f"  Spread: {market.get('spread_bps', 0):.2f} bps")
        print(f"  Book Imbalance: {market.get('book_imbalance', 0):.3f}")
        print(f"  1min Volatility: {market.get('volatility_1min', 0):.4f}")
        print("-" * 80)
        
        # Run test scenarios
        scenarios = self.generate_test_scenarios()
        
        for scenario in scenarios:
            print(f"\nTesting scenario: {scenario['name']}")
            print(f"  Signal strength: {scenario['signal_strength']}")
            print(f"  Expected strategy: {scenario['expected_strategy']}")
            
            result = await self.test_execution_strategy(
                signal_strength=scenario['signal_strength'],
                feature_values=scenario['features'],
                dry_run=dry_run
            )
            
            print(f"  Result: {result.execution_strategy} execution")
            print(f"  Slippage: {result.slippage_bps:.2f} bps")
            print(f"  Fill time: {result.fill_time_ms:.0f} ms")
            print(f"  Success: {result.success}")
            
            # Validate against expectations
            if result.execution_strategy == scenario['expected_strategy']:
                print(f"  ✓ Strategy matches expectation")
            else:
                print(f"  ✗ Strategy mismatch! Got {result.execution_strategy}")
                
        # Generate summary report
        self.generate_summary_report()
        
    def generate_summary_report(self):
        """Generate summary report of test results"""
        if not self.test_results:
            print("\nNo test results to summarize")
            return
            
        print("\n" + "=" * 80)
        print("EXECUTION VALIDATION SUMMARY")
        print("=" * 80)
        
        # Group by strategy
        strategy_results = {}
        for result in self.test_results:
            strategy = result.execution_strategy
            if strategy not in strategy_results:
                strategy_results[strategy] = []
            strategy_results[strategy].append(result)
            
        # Analyze each strategy
        for strategy, results in strategy_results.items():
            print(f"\n{strategy.upper()} Strategy:")
            print(f"  Tests run: {len(results)}")
            
            success_rate = sum(1 for r in results if r.success) / len(results) * 100
            print(f"  Success rate: {success_rate:.1f}%")
            
            slippages = [r.slippage_bps for r in results if r.success]
            if slippages:
                print(f"  Avg slippage: {np.mean(slippages):.2f} bps")
                print(f"  Max slippage: {np.max(slippages):.2f} bps")
                
            fill_times = [r.fill_time_ms for r in results if r.success]
            if fill_times:
                print(f"  Avg fill time: {np.mean(fill_times):.0f} ms")
                
        # Overall recommendations
        print("\n" + "-" * 80)
        print("RECOMMENDATIONS:")
        print("-" * 80)
        
        print("\n1. Spread Stability Mapping:")
        print("   - spread_stability_norm_100 < 0.2 → Passive orders (maker rebates)")
        print("   - spread_stability_norm_100 > 0.6 → Avoid passive, use aggressive limits")
        
        print("\n2. OFI Signal Urgency:")
        print("   - |ofi_normalized_1m| > 0.7 → Immediate execution required")
        print("   - |ofi_normalized_1m| < 0.2 → Patient execution acceptable")
        
        print("\n3. Optimal Testing Windows:")
        print("   - Asian session (low volatility): Test passive strategies")
        print("   - US session (high volatility): Test aggressive/urgent strategies")
        print("   - Weekend (thin liquidity): Test adaptive strategies")
        
        # Save results to file
        results_df = pd.DataFrame([asdict(r) for r in self.test_results])
        filename = f"execution_validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        results_df.to_csv(filename, index=False)
        print(f"\nResults saved to: {filename}")


async def main():
    """Main execution for testing"""
    # Configuration
    config = {
        'api_key': None,  # Set your API key
        'api_secret': None,  # Set your API secret
        'testnet': True  # Use testnet for safety
    }
    
    # Initialize validator
    validator = ExecutionValidator(config)
    
    # Run validation suite
    await validator.run_validation_suite(dry_run=True)
    

if __name__ == "__main__":
    asyncio.run(main())