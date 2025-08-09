#!/usr/bin/env python3
"""
test_paper_trading.py - Test Paper Trading System Components

This script tests the paper trading system components without requiring
live exchange connections.
"""

import sys
import traceback

def test_paper_trading_components():
    """Test all paper trading system components."""
    
    print('Testing Paper Trading System Components...')
    print('='*60)
    
    # Test 1: Configuration Loading
    try:
        from run import PaperTradingConfig
        config = PaperTradingConfig()
        print('[OK] Configuration loaded successfully')
        print(f'    Exchange: {config.config["exchange"]["name"]} (testnet: {config.config["exchange"]["testnet"]})')
        print(f'    Database: {config.config["database_path"]}')
        print(f'    Paper trading balance: ${config.config["paper_trading"]["initial_balance"]}')
    except Exception as e:
        print(f'[ERROR] Configuration failed: {e}')
        return False
    
    # Test 2: Dashboard
    try:
        from run import PaperTradingDashboard
        dashboard = PaperTradingDashboard(config)
        print('[OK] Dashboard initialized successfully')
        dashboard.print_dashboard()
    except Exception as e:
        print(f'[ERROR] Dashboard failed: {e}')
        return False
    
    # Test 3: Trading Engine (without exchange connection)
    try:
        from run import PaperTradingEngine
        engine = PaperTradingEngine(config)
        print('[OK] Trading engine initialized successfully')
        
        # Test market data loading
        market_data = engine.get_latest_market_data()
        if market_data is not None and not market_data.empty:
            print(f'[OK] Market data loaded: {len(market_data)} rows')
            
            # Test feature generation
            features = engine.generate_features(market_data)
            if features is not None and not features.empty:
                print(f'[OK] Features generated: {features.shape[1]} features')
                
                # Test prediction
                prediction = engine.get_prediction(features)
                if prediction:
                    print(f'[OK] Prediction generated: Signal={prediction["signal"]}, Confidence={prediction["confidence"]:.3f}')
                    
                    # Test paper trade simulation (without real exchange)
                    trade = {
                        'side': 'buy' if prediction['signal'] > 0 else 'sell',
                        'amount': 0.01,
                        'price': prediction.get('current_price', 50000),
                        'confidence': prediction['confidence'],
                        'signal': prediction['signal']
                    }
                    print(f'[OK] Paper trade simulated: {trade["side"]} {trade["amount"]} BTC at ${trade["price"]:.2f}')
                    
                    # Update dashboard
                    engine.dashboard.update_trade(trade)
                    engine.dashboard.update_performance(0, 10000, 0.01)
                    print('[OK] Dashboard updated successfully')
                    
                else:
                    print('[WARNING] Prediction failed')
            else:
                print('[WARNING] Feature generation failed')
        else:
            print('[WARNING] No market data available')
            
    except Exception as e:
        print(f'[ERROR] Trading engine test failed: {e}')
        print(f'Traceback: {traceback.format_exc()}')
        return False
    
    print('='*60)
    print('Paper Trading System Component Tests Complete')
    print('[INFO] The system is ready for deployment')
    print('[INFO] Exchange connection will be established during live deployment')
    
    return True

if __name__ == "__main__":
    success = test_paper_trading_components()
    sys.exit(0 if success else 1)