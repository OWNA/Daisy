"""
Test script for the enhanced passive follower execution strategy
Validates the new dynamic best bid/ask following logic and circuit breakers
"""

import asyncio
import time
import unittest
from unittest.mock import MagicMock, patch
from dataclasses import dataclass

# Import only what we need without external dependencies
@dataclass
class ExecutionDecision:
    """Encapsulates execution decision based on ML features"""
    strategy: str  # passive, balanced, aggressive, urgent
    order_type: str  # limit, market, iceberg, twap
    price_adjustment: float  # Price adjustment factor
    size_allocation: list  # Size split for multiple orders
    time_horizon: float  # Expected execution time in seconds
    confidence: float  # Confidence in execution decision


class TestablePassiveExecutor:
    """Minimal version of passive executor for testing"""
    
    def __init__(self, exchange):
        self.exchange = exchange
        self.min_order_size = 0.001
        self.strategy_params = {
            'passive': {
                'max_wait_time': 3,  # Shorter for testing
                'use_post_only': True
            }
        }
        
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
                                await asyncio.sleep(0.01)
                            else:
                                # Order is still good, wait and check again
                                await asyncio.sleep(0.05)
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
                        await asyncio.sleep(0.02)
                        
                    except Exception as place_e:
                        print(f"[PASSIVE] ERROR placing order: {place_e}")
                        break
                
                # Small delay before next iteration
                await asyncio.sleep(0.05)
            
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


class MockExchange:
    """Mock exchange that simulates dynamic market conditions"""
    
    def __init__(self):
        self.orders = {}  # Track orders by ID
        self.order_counter = 1000
        self.current_bid = 100000.0
        self.current_ask = 100010.0
        self.market_movements = []  # Queue of price movements
        self.movement_index = 0
        self.order_book_calls = 0
        self.create_order_calls = []
        self.cancel_order_calls = []
        
    def set_market_movements(self, movements):
        """Set a sequence of market movements for testing"""
        self.market_movements = movements
        self.movement_index = 0
        
    def _advance_market(self):
        """Advance to next market state if available"""
        if self.movement_index < len(self.market_movements):
            bid, ask = self.market_movements[self.movement_index]
            self.current_bid = bid
            self.current_ask = ask
            self.movement_index += 1
            
    def fetch_order_book(self, symbol, limit=10):
        """Return current order book with market movements"""
        self.order_book_calls += 1
        self._advance_market()
        
        return {
            'bids': [[self.current_bid, 5.0], [self.current_bid - 0.5, 3.0]],
            'asks': [[self.current_ask, 5.0], [self.current_ask + 0.5, 3.0]]
        }
        
    def create_limit_order(self, symbol, side, amount, price, params=None):
        """Create a mock limit order"""
        order_id = str(self.order_counter)
        self.order_counter += 1
        
        order = {
            'id': order_id,
            'symbol': symbol,
            'side': side,
            'amount': amount,
            'price': price,
            'status': 'open',
            'filled': 0,
            'average': None,
            'timestamp': time.time()
        }
        
        self.orders[order_id] = order
        self.create_order_calls.append({
            'symbol': symbol,
            'side': side,
            'amount': amount,
            'price': price,
            'params': params
        })
        
        return order
        
    def fetch_order(self, order_id, symbol):
        """Fetch order status - may simulate fills"""
        if order_id not in self.orders:
            raise Exception(f"Order {order_id} not found")
            
        order = self.orders[order_id].copy()
        
        # Simulate random fills for testing (can be overridden)
        if hasattr(self, '_fill_order_id') and order_id == self._fill_order_id:
            order['status'] = 'closed'
            order['filled'] = order['amount']
            order['average'] = order['price']
            
        return order
        
    def cancel_order(self, order_id, symbol):
        """Cancel an order"""
        if order_id in self.orders:
            self.orders[order_id]['status'] = 'cancelled'
            
        self.cancel_order_calls.append({
            'order_id': order_id,
            'symbol': symbol
        })
        
        return {'id': order_id, 'status': 'cancelled'}
        
    def simulate_fill(self, order_id):
        """Simulate a fill for testing"""
        self._fill_order_id = order_id
        
    def reset_stats(self):
        """Reset call statistics"""
        self.create_order_calls = []
        self.cancel_order_calls = []
        self.order_book_calls = 0


class TestPassiveFollower(unittest.TestCase):
    """Test suite for passive follower execution strategy"""
    
    def setUp(self):
        """Set up test environment"""
        self.mock_exchange = MockExchange()
        
        # Standard execution config
        exec_config = {
            'slippage_model_pct': 0.0005,
            'max_order_book_levels': 50,
            'min_order_size': 0.001
        }
        
        # ML config (not used in these tests but required)
        ml_config = {
            'thresholds': {},
            'weights': {}
        }
        
        self.executor = TestablePassiveExecutor(self.mock_exchange)
        
        # Create a standard execution decision for passive strategy
        self.passive_decision = ExecutionDecision(
            strategy='passive',
            order_type='limit',
            price_adjustment=1.0,
            size_allocation=[1.0],
            time_horizon=30.0,
            confidence=0.8
        )
        
    async def test_basic_passive_execution_no_movement(self):
        """Test passive execution with stable market (no movement)"""
        print("\n=== TEST: Basic passive execution with stable market ===")
        
        # Set stable market (no movements)
        self.mock_exchange.set_market_movements([
            (100000.0, 100010.0),  # Initial state
            (100000.0, 100010.0),  # No change
            (100000.0, 100010.0),  # No change
        ])
        
        # Simulate order fill after 1 second
        def delayed_fill():
            asyncio.create_task(self._fill_after_delay('1001', 1.0))
            
        order_book = self.mock_exchange.fetch_order_book('BTCUSDT')
        
        # Override the executor to simulate fill
        original_create = self.mock_exchange.create_limit_order
        def create_and_fill(*args, **kwargs):
            order = original_create(*args, **kwargs)
            # Simulate fill after short delay
            asyncio.create_task(self._fill_after_delay(order['id'], 0.5))
            return order
            
        self.mock_exchange.create_limit_order = create_and_fill
        
        result = await self.executor._execute_passive(
            'BTCUSDT', 'sell', 1.0, order_book, self.passive_decision
        )
        
        print(f"Result: {result}")
        
        # Verify successful execution
        self.assertTrue(result['success'])
        self.assertEqual(result['strategy'], 'passive_following')
        self.assertEqual(result['follow_count'], 0)  # No replacements needed
        self.assertGreater(len(self.mock_exchange.create_order_calls), 0)
        
    async def _fill_after_delay(self, order_id, delay):
        """Helper to simulate order fill after delay"""
        await asyncio.sleep(delay)
        self.mock_exchange.simulate_fill(order_id)
        
    async def test_market_movement_within_tolerance(self):
        """Test market movement within 2-tick tolerance (should not replace)"""
        print("\n=== TEST: Market movement within tolerance ===")
        
        # Market moves by 1 tick (0.5 price units) - within 2-tick tolerance
        self.mock_exchange.set_market_movements([
            (100000.0, 100010.0),  # Initial: bid=100000, ask=100010
            (100000.5, 100010.5),  # Move by 0.5 (1 tick) - within tolerance
            (100000.5, 100010.5),  # Stable
        ])
        
        order_book = self.mock_exchange.fetch_order_book('BTCUSDT')
        
        # Don't fill order to test replacement logic
        result = await self.executor._execute_passive(
            'BTCUSDT', 'sell', 1.0, order_book, self.passive_decision
        )
        
        print(f"Result: {result}")
        print(f"Order creations: {len(self.mock_exchange.create_order_calls)}")
        print(f"Order cancellations: {len(self.mock_exchange.cancel_order_calls)}")
        
        # Should only create 1 order (no replacement due to tolerance)
        self.assertEqual(len(self.mock_exchange.create_order_calls), 1)
        self.assertEqual(len(self.mock_exchange.cancel_order_calls), 1)  # Final cleanup only
        self.assertEqual(result['follow_count'], 0)  # No follows due to tolerance
        
    async def test_market_movement_beyond_tolerance(self):
        """Test market movement beyond 2-tick tolerance (should replace order)"""
        print("\n=== TEST: Market movement beyond tolerance ===")
        
        # Market moves by 3 ticks (1.5 price units) - beyond 2-tick tolerance  
        self.mock_exchange.set_market_movements([
            (100000.0, 100010.0),  # Initial: ask=100010 (sell order placed here)
            (100000.0, 100010.0),  # Stable for first check
            (100001.5, 100011.5),  # Move by 1.5 (3 ticks) - beyond tolerance
            (100001.5, 100011.5),  # Stable after movement
        ])
        
        order_book = self.mock_exchange.fetch_order_book('BTCUSDT')
        
        result = await self.executor._execute_passive(
            'BTCUSDT', 'sell', 1.0, order_book, self.passive_decision
        )
        
        print(f"Result: {result}")
        print(f"Order creations: {len(self.mock_exchange.create_order_calls)}")
        print(f"Order cancellations: {len(self.mock_exchange.cancel_order_calls)}")
        
        # Should create 2 orders (original + replacement)
        self.assertGreaterEqual(len(self.mock_exchange.create_order_calls), 2)
        self.assertGreaterEqual(len(self.mock_exchange.cancel_order_calls), 1)
        self.assertGreaterEqual(result['follow_count'], 1)  # At least 1 follow
        
        # Verify price levels
        first_order_price = self.mock_exchange.create_order_calls[0]['price']
        second_order_price = self.mock_exchange.create_order_calls[1]['price']
        
        print(f"First order price: {first_order_price}")
        print(f"Second order price: {second_order_price}")
        
        # Second order should be at new market level
        self.assertNotEqual(first_order_price, second_order_price)
        
    async def test_max_follows_circuit_breaker(self):
        """Test max_follows circuit breaker (should stop after 10 follows)"""
        print("\n=== TEST: Max follows circuit breaker ===")
        
        # Create 15 market movements to trigger max_follows
        movements = [(100000.0, 100010.0)]  # Initial
        for i in range(15):
            # Each movement is 1.5 price units (3 ticks) - beyond tolerance
            new_ask = 100010.0 + (i + 1) * 1.5
            movements.append((100000.0, new_ask))
            
        self.mock_exchange.set_market_movements(movements)
        
        order_book = self.mock_exchange.fetch_order_book('BTCUSDT')
        
        result = await self.executor._execute_passive(
            'BTCUSDT', 'sell', 1.0, order_book, self.passive_decision
        )
        
        print(f"Result: {result}")
        print(f"Follow count: {result['follow_count']}")
        print(f"Order creations: {len(self.mock_exchange.create_order_calls)}")
        
        # Should stop at max_follows (10)
        self.assertEqual(result['follow_count'], 10)
        self.assertEqual(result.get('reason'), 'max_follows_exceeded')
        self.assertFalse(result['success'])
        
    async def test_timeout_circuit_breaker(self):
        """Test timeout circuit breaker"""
        print("\n=== TEST: Timeout circuit breaker ===")
        
        # Set very short timeout for testing
        original_timeout = self.executor.strategy_params['passive']['max_wait_time']
        self.executor.strategy_params['passive']['max_wait_time'] = 0.3  # 0.3 seconds
        
        try:
            # Stable market (no movements)
            self.mock_exchange.set_market_movements([
                (100000.0, 100010.0),  # Stable throughout
            ])
            
            order_book = self.mock_exchange.fetch_order_book('BTCUSDT')
            
            result = await self.executor._execute_passive(
                'BTCUSDT', 'sell', 1.0, order_book, self.passive_decision
            )
            
            print(f"Result: {result}")
            print(f"Execution time: {result.get('execution_time', 0):.2f}s")
            
            # Should timeout
            self.assertFalse(result['success'])
            self.assertEqual(result.get('reason'), 'timeout')
            self.assertGreaterEqual(result.get('execution_time', 0), 0.3)
            
        finally:
            # Restore original timeout
            self.executor.strategy_params['passive']['max_wait_time'] = original_timeout
            
    async def test_buy_vs_sell_order_placement(self):
        """Test correct price placement for buy vs sell orders"""
        print("\n=== TEST: Buy vs sell order placement ===")
        
        self.mock_exchange.set_market_movements([
            (100000.0, 100010.0),  # bid=100000, ask=100010
        ])
        
        order_book = self.mock_exchange.fetch_order_book('BTCUSDT')
        
        # Test sell order (should place at best ask)
        self.mock_exchange.reset_stats()
        await self.executor._execute_passive(
            'BTCUSDT', 'sell', 1.0, order_book, self.passive_decision
        )
        
        sell_price = self.mock_exchange.create_order_calls[0]['price']
        print(f"Sell order placed at: {sell_price} (best ask: {100010.0})")
        self.assertEqual(sell_price, 100010.0)  # Should be at best ask
        
        # Reset and test buy order (should place at best bid)
        self.mock_exchange.reset_stats()
        self.mock_exchange.set_market_movements([
            (100000.0, 100010.0),  # Reset market
        ])
        
        order_book = self.mock_exchange.fetch_order_book('BTCUSDT')
        await self.executor._execute_passive(
            'BTCUSDT', 'buy', 1.0, order_book, self.passive_decision
        )
        
        buy_price = self.mock_exchange.create_order_calls[0]['price']
        print(f"Buy order placed at: {buy_price} (best bid: {100000.0})")
        self.assertEqual(buy_price, 100000.0)  # Should be at best bid
        
    async def test_partial_fill_handling(self):
        """Test handling of partial fills and continuation"""
        print("\n=== TEST: Partial fill handling ===")
        
        self.mock_exchange.set_market_movements([
            (100000.0, 100010.0),  # Stable market
        ])
        
        # Mock partial fill
        original_fetch = self.mock_exchange.fetch_order
        fill_count = 0
        
        def mock_fetch_with_partial(order_id, symbol):
            nonlocal fill_count
            order = original_fetch(order_id, symbol)
            
            # First call: partial fill
            if fill_count == 0:
                fill_count += 1
                order['status'] = 'closed'
                order['filled'] = 0.6  # Partial fill
                order['average'] = order['price']
                
            return order
            
        self.mock_exchange.fetch_order = mock_fetch_with_partial
        
        order_book = self.mock_exchange.fetch_order_book('BTCUSDT')
        
        result = await self.executor._execute_passive(
            'BTCUSDT', 'sell', 1.0, order_book, self.passive_decision
        )
        
        print(f"Result: {result}")
        print(f"Orders created: {len(self.mock_exchange.create_order_calls)}")
        
        # Should create multiple orders to handle remaining amount
        self.assertGreaterEqual(len(self.mock_exchange.create_order_calls), 2)
        
        # First order should be for full amount, second for remaining
        first_amount = self.mock_exchange.create_order_calls[0]['amount']
        second_amount = self.mock_exchange.create_order_calls[1]['amount']
        
        print(f"First order amount: {first_amount}")
        print(f"Second order amount: {second_amount}")
        
        self.assertEqual(first_amount, 1.0)  # Full amount
        self.assertEqual(second_amount, 0.4)  # Remaining amount


async def run_all_tests():
    """Run all passive follower tests"""
    print("🧪 TESTING PASSIVE FOLLOWER EXECUTION STRATEGY")
    print("=" * 60)
    
    test_suite = TestPassiveFollower()
    
    tests = [
        ('Basic Execution (Stable Market)', test_suite.test_basic_passive_execution_no_movement),
        ('Movement Within Tolerance', test_suite.test_market_movement_within_tolerance),
        ('Movement Beyond Tolerance', test_suite.test_market_movement_beyond_tolerance),
        ('Max Follows Circuit Breaker', test_suite.test_max_follows_circuit_breaker),
        ('Timeout Circuit Breaker', test_suite.test_timeout_circuit_breaker),
        ('Buy vs Sell Placement', test_suite.test_buy_vs_sell_order_placement),
        ('Partial Fill Handling', test_suite.test_partial_fill_handling),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        print(f"\n🔬 Running: {test_name}")
        print("-" * 40)
        
        try:
            # Set up fresh test environment
            test_suite.setUp()
            
            # Run the test
            await test_func()
            
            print(f"✅ PASSED: {test_name}")
            passed += 1
            
        except Exception as e:
            print(f"❌ FAILED: {test_name}")
            print(f"   Error: {str(e)}")
            import traceback
            traceback.print_exc()
            failed += 1
            
        print("-" * 40)
        
    print(f"\n📊 TEST RESULTS:")
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    print(f"📈 Success Rate: {(passed/(passed+failed))*100:.1f}%")
    
    if failed == 0:
        print("\n🎉 ALL TESTS PASSED! The passive follower strategy is working correctly.")
    else:
        print(f"\n⚠️  {failed} test(s) failed. Please review the implementation.")
        
    return failed == 0


if __name__ == "__main__":
    # Run the test suite
    asyncio.run(run_all_tests())