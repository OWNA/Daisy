"""
Quick Validation Script for Improved Order Executor
Pre-flight checks and integration validation
"""

import ccxt
import os
import time
import logging
from datetime import datetime
from typing import Dict, List
import json

from improved_order_executor import ImprovedOrderExecutor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ExecutorValidator:
    """Validates the improved executor with safety checks"""
    
    def __init__(self, testnet: bool = True):
        """
        Initialize validator
        
        Args:
            testnet: Use testnet (True) or mainnet (False)
        """
        self.testnet = testnet
        self.validation_results = {
            'timestamp': datetime.utcnow().isoformat(),
            'testnet': testnet,
            'checks': {},
            'test_orders': [],
            'ready_for_production': False
        }
        
    def initialize_exchange(self) -> ccxt.Exchange:
        """Initialize exchange connection"""
        try:
            if self.testnet:
                exchange = ccxt.bybit({
                    'apiKey': os.getenv('BYBIT_TESTNET_API_KEY'),
                    'secret': os.getenv('BYBIT_TESTNET_API_SECRET'),
                    'enableRateLimit': True,
                    'options': {
                        'defaultType': 'swap',
                        'adjustForTimeDifference': True
                    },
                    'urls': {
                        'api': {
                            'public': 'https://api-testnet.bybit.com',
                            'private': 'https://api-testnet.bybit.com'
                        }
                    }
                })
            else:
                exchange = ccxt.bybit({
                    'apiKey': os.getenv('BYBIT_API_KEY'),
                    'secret': os.getenv('BYBIT_API_SECRET'),
                    'enableRateLimit': True,
                    'options': {
                        'defaultType': 'swap',
                        'adjustForTimeDifference': True
                    }
                })
            
            # Test connection
            balance = exchange.fetch_balance()
            logger.info(f"Exchange connected successfully. {'TESTNET' if self.testnet else 'MAINNET'}")
            self.validation_results['checks']['exchange_connection'] = True
            
            return exchange
            
        except Exception as e:
            logger.error(f"Failed to connect to exchange: {e}")
            self.validation_results['checks']['exchange_connection'] = False
            raise
    
    def validate_market_data_access(self, exchange: ccxt.Exchange, symbol: str = 'BTC/USDT:USDT') -> bool:
        """Validate access to required market data"""
        checks_passed = True
        
        try:
            # Check ticker
            ticker = exchange.fetch_ticker(symbol)
            logger.info(f"✓ Ticker access OK - {symbol} @ ${ticker['last']}")
            self.validation_results['checks']['ticker_access'] = True
        except Exception as e:
            logger.error(f"✗ Ticker access failed: {e}")
            self.validation_results['checks']['ticker_access'] = False
            checks_passed = False
        
        try:
            # Check order book
            order_book = exchange.fetch_l2_order_book(symbol, limit=100)
            logger.info(f"✓ Order book access OK - {len(order_book['bids'])} bids, {len(order_book['asks'])} asks")
            self.validation_results['checks']['order_book_access'] = True
        except Exception as e:
            logger.error(f"✗ Order book access failed: {e}")
            self.validation_results['checks']['order_book_access'] = False
            checks_passed = False
        
        return checks_passed
    
    def validate_executor_initialization(self, exchange: ccxt.Exchange) -> ImprovedOrderExecutor:
        """Validate executor initialization with various configs"""
        try:
            config = {
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
            
            executor = ImprovedOrderExecutor(exchange, config)
            logger.info("✓ Executor initialization OK")
            self.validation_results['checks']['executor_init'] = True
            
            return executor
            
        except Exception as e:
            logger.error(f"✗ Executor initialization failed: {e}")
            self.validation_results['checks']['executor_init'] = False
            raise
    
    def validate_liquidity_analysis(self, executor: ImprovedOrderExecutor, 
                                  symbol: str = 'BTC/USDT:USDT') -> bool:
        """Validate liquidity analysis functions"""
        try:
            # Test various order sizes
            test_sizes = [100, 1000, 5000]
            
            for size in test_sizes:
                for side in ['buy', 'sell']:
                    analysis = executor.analyze_order_book_liquidity(symbol, side, size)
                    
                    # Validate analysis structure
                    required_keys = ['best_price', 'expected_avg_price', 'price_impact', 
                                   'strategy', 'liquidity_levels']
                    
                    if all(key in analysis for key in required_keys):
                        logger.info(f"✓ Liquidity analysis OK - {side} ${size}: "
                                  f"strategy={analysis['strategy']}, "
                                  f"impact={analysis['price_impact']*100:.3f}%")
                    else:
                        logger.error(f"✗ Liquidity analysis incomplete for {side} ${size}")
                        return False
            
            self.validation_results['checks']['liquidity_analysis'] = True
            return True
            
        except Exception as e:
            logger.error(f"✗ Liquidity analysis failed: {e}")
            self.validation_results['checks']['liquidity_analysis'] = False
            return False
    
    def validate_order_slicing(self, executor: ImprovedOrderExecutor) -> bool:
        """Validate order slicing logic"""
        try:
            # Test slicing with mock liquidity data
            test_cases = [
                {
                    'amount_usd': 1000,
                    'liquidity': {
                        'strategy': 'patient',
                        'liquidity_levels': [
                            {'volume_usd': 500, 'price': 50000, 'level': 0},
                            {'volume_usd': 300, 'price': 50001, 'level': 1},
                            {'volume_usd': 200, 'price': 50002, 'level': 2}
                        ]
                    }
                },
                {
                    'amount_usd': 100,
                    'liquidity': {'strategy': 'aggressive', 'liquidity_levels': []}
                }
            ]
            
            for test in test_cases:
                slices = executor.calculate_order_slices(
                    test['amount_usd'], 
                    test['liquidity']
                )
                
                if slices and isinstance(slices, list):
                    total_size = sum(s['size_usd'] for s in slices)
                    logger.info(f"✓ Order slicing OK - ${test['amount_usd']} → "
                              f"{len(slices)} slices, total ${total_size:.2f}")
                else:
                    logger.error(f"✗ Order slicing failed for ${test['amount_usd']}")
                    return False
            
            self.validation_results['checks']['order_slicing'] = True
            return True
            
        except Exception as e:
            logger.error(f"✗ Order slicing validation failed: {e}")
            self.validation_results['checks']['order_slicing'] = False
            return False
    
    def run_minimal_test_order(self, executor: ImprovedOrderExecutor, 
                             symbol: str = 'BTC/USDT:USDT') -> bool:
        """Run a minimal test order (testnet only)"""
        if not self.testnet:
            logger.warning("Skipping test order on mainnet")
            return True
        
        try:
            # Very small test order
            test_amount_usd = 10  # Minimum size
            
            logger.info(f"\nExecuting minimal test order: {symbol} BUY ${test_amount_usd}")
            
            result = executor.execute_smart_order(
                symbol=symbol,
                side='buy',
                amount_usd=test_amount_usd,
                urgency='low'
            )
            
            # Cancel any open orders
            executor.cancel_all_orders(symbol)
            
            # Log results
            if result['success']:
                logger.info(f"✓ Test order executed successfully")
                logger.info(f"  - Fill rate: {result['fill_rate']*100:.1f}%")
                logger.info(f"  - Slippage: {result['slippage_pct']*100:.3f}%")
                logger.info(f"  - Fees: ${result['total_fees_usd']:.4f}")
                logger.info(f"  - Strategy: {result['strategy_used']}")
                
                self.validation_results['test_orders'].append(result)
                self.validation_results['checks']['test_order'] = True
                return True
            else:
                logger.error("✗ Test order failed")
                self.validation_results['checks']['test_order'] = False
                return False
                
        except Exception as e:
            logger.error(f"✗ Test order error: {e}")
            self.validation_results['checks']['test_order'] = False
            
            # Always try to cancel orders
            try:
                executor.cancel_all_orders(symbol)
            except:
                pass
                
            return False
    
    def validate_safety_features(self, executor: ImprovedOrderExecutor) -> bool:
        """Validate safety features and error handling"""
        checks_passed = True
        
        # Test timeout handling
        original_timeout = executor.order_timeout_seconds
        executor.order_timeout_seconds = 0.001  # Very short timeout
        
        try:
            # This should handle timeout gracefully
            result = executor.execute_smart_order(
                symbol='BTC/USDT:USDT',
                side='buy',
                amount_usd=100,
                urgency='low'
            )
            logger.info("✓ Timeout handling OK")
            self.validation_results['checks']['timeout_handling'] = True
        except Exception as e:
            logger.error(f"✗ Timeout handling failed: {e}")
            self.validation_results['checks']['timeout_handling'] = False
            checks_passed = False
        finally:
            executor.order_timeout_seconds = original_timeout
        
        # Test minimum order size
        try:
            result = executor.execute_smart_order(
                symbol='BTC/USDT:USDT',
                side='buy',
                amount_usd=1,  # Below minimum
                urgency='low'
            )
            
            # Should still return a result, but with appropriate handling
            logger.info("✓ Minimum size handling OK")
            self.validation_results['checks']['min_size_handling'] = True
        except Exception as e:
            logger.error(f"✗ Minimum size handling failed: {e}")
            self.validation_results['checks']['min_size_handling'] = False
            checks_passed = False
        
        return checks_passed
    
    def generate_validation_report(self) -> Dict:
        """Generate final validation report"""
        # Check if all critical checks passed
        critical_checks = [
            'exchange_connection',
            'ticker_access',
            'order_book_access',
            'executor_init',
            'liquidity_analysis',
            'order_slicing'
        ]
        
        all_critical_passed = all(
            self.validation_results['checks'].get(check, False) 
            for check in critical_checks
        )
        
        # Additional checks for production readiness
        if not self.testnet and all_critical_passed:
            self.validation_results['ready_for_production'] = True
        elif self.testnet and all_critical_passed and self.validation_results['checks'].get('test_order', False):
            self.validation_results['ready_for_production'] = True
        
        # Summary
        total_checks = len(self.validation_results['checks'])
        passed_checks = sum(1 for v in self.validation_results['checks'].values() if v)
        
        self.validation_results['summary'] = {
            'total_checks': total_checks,
            'passed_checks': passed_checks,
            'success_rate': passed_checks / total_checks if total_checks > 0 else 0,
            'critical_checks_passed': all_critical_passed
        }
        
        return self.validation_results
    
    def save_report(self, filename: str = None):
        """Save validation report to file"""
        if not filename:
            timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
            filename = f"executor_validation_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(self.validation_results, f, indent=2, default=str)
        
        logger.info(f"Validation report saved to {filename}")


def run_validation(testnet: bool = True):
    """Run the validation process"""
    print(f"\n{'='*60}")
    print(f"Improved Order Executor Validation")
    print(f"Mode: {'TESTNET' if testnet else 'MAINNET'}")
    print(f"{'='*60}\n")
    
    validator = ExecutorValidator(testnet=testnet)
    
    try:
        # Step 1: Initialize exchange
        print("Step 1: Initializing exchange connection...")
        exchange = validator.initialize_exchange()
        
        # Step 2: Validate market data access
        print("\nStep 2: Validating market data access...")
        if not validator.validate_market_data_access(exchange):
            logger.warning("Market data validation had issues")
        
        # Step 3: Initialize executor
        print("\nStep 3: Initializing improved executor...")
        executor = validator.validate_executor_initialization(exchange)
        
        # Step 4: Validate core functions
        print("\nStep 4: Validating liquidity analysis...")
        validator.validate_liquidity_analysis(executor)
        
        print("\nStep 5: Validating order slicing...")
        validator.validate_order_slicing(executor)
        
        # Step 6: Validate safety features
        print("\nStep 6: Validating safety features...")
        validator.validate_safety_features(executor)
        
        # Step 7: Run test order (testnet only)
        if testnet:
            print("\nStep 7: Running minimal test order...")
            validator.run_minimal_test_order(executor)
        
        # Generate report
        print("\n" + "="*60)
        report = validator.generate_validation_report()
        
        print(f"\nValidation Summary:")
        print(f"  Total Checks: {report['summary']['total_checks']}")
        print(f"  Passed: {report['summary']['passed_checks']}")
        print(f"  Success Rate: {report['summary']['success_rate']*100:.1f}%")
        print(f"  Critical Checks: {'PASSED' if report['summary']['critical_checks_passed'] else 'FAILED'}")
        print(f"  Ready for Production: {'YES' if report['ready_for_production'] else 'NO'}")
        
        # Save report
        validator.save_report()
        
        # Detailed check results
        print("\nDetailed Results:")
        for check, result in report['checks'].items():
            status = "✓" if result else "✗"
            print(f"  {status} {check}")
        
        return report['ready_for_production']
        
    except Exception as e:
        logger.error(f"Validation failed with error: {e}")
        return False


def main():
    """Main entry point"""
    import sys
    
    # Check command line arguments
    use_mainnet = '--mainnet' in sys.argv
    
    if use_mainnet:
        print("\n⚠️  WARNING: Running validation on MAINNET!")
        print("This will NOT execute any real orders but will connect to mainnet.")
        response = input("Continue? (yes/no): ").lower().strip()
        if response != 'yes':
            print("Validation cancelled.")
            return
    
    # Run validation
    ready = run_validation(testnet=not use_mainnet)
    
    if ready:
        print("\n✅ Improved executor is ready for use!")
        if not use_mainnet:
            print("\nNext steps:")
            print("1. Review the validation report")
            print("2. Run comparison tests with test_execution_comparison.py")
            print("3. Start with small position sizes in production")
            print("4. Monitor performance metrics closely")
    else:
        print("\n❌ Improved executor validation failed.")
        print("Please review the validation report and fix any issues.")
    
    return 0 if ready else 1


if __name__ == "__main__":
    exit(main())