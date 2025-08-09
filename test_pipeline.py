#!/usr/bin/env python3
"""
Test script to verify minimal pipeline functionality
"""

import os
import sys

def test_imports():
    """Test that all core modules can be imported"""
    print("Testing imports...")
    try:
        from database import TradingDatabase
        print("✓ database.py")
        
        from l2_data_collector import L2DataCollector
        print("✓ l2_data_collector.py")
        
        from datahandler import DataHandler
        print("✓ datahandler.py")
        
        from featureengineer import FeatureEngineer
        print("✓ featureengineer.py")
        
        from labelgenerator import LabelGenerator
        print("✓ labelgenerator.py")
        
        from modeltrainer import ModelTrainer
        print("✓ modeltrainer.py")
        
        from modelpredictor import ModelPredictor
        print("✓ modelpredictor.py")
        
        from advancedriskmanager import AdvancedRiskManager
        print("✓ advancedriskmanager.py")
        
        from smartorderexecutor import SmartOrderExecutor
        print("✓ smartorderexecutor.py")
        
        import main
        print("✓ main.py")
        
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False

def test_data_availability():
    """Test that L2 data is available"""
    print("\nTesting data availability...")
    
    # Check L2 data directory
    if os.path.exists('./l2_data'):
        files = [f for f in os.listdir('./l2_data') if f.endswith('.jsonl.gz')]
        print(f"✓ Found {len(files)} L2 data files")
        return len(files) > 0
    else:
        print("✗ L2 data directory not found")
        return False

def test_database():
    """Test database connectivity"""
    print("\nTesting database...")
    
    db_path = './trading_bot.db'
    if os.path.exists(db_path):
        print(f"✓ Database exists: {os.path.getsize(db_path) / 1024 / 1024:.2f} MB")
        
        # Test connection
        try:
            from database import TradingDatabase
            db = TradingDatabase(db_path)
            with db.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM l2_training_data_practical")
                count = cursor.fetchone()[0]
                print(f"✓ L2 training data: {count} records")
            return True
        except Exception as e:
            print(f"✗ Database error: {e}")
            return False
    else:
        print("✗ Database not found")
        return False

def test_config():
    """Test configuration file"""
    print("\nTesting configuration...")
    
    if os.path.exists('config.yaml'):
        try:
            import yaml
            with open('config.yaml', 'r') as f:
                config = yaml.safe_load(f)
            
            # Check critical settings
            checks = [
                ('l2_only_mode', config.get('l2_only_mode', False)),
                ('l2_data_folder', config.get('l2_data_folder') == 'l2_data'),
                ('database_path', 'database_path' in config),
                ('symbol', 'symbol' in config)
            ]
            
            all_good = True
            for name, check in checks:
                if check:
                    print(f"✓ {name}: OK")
                else:
                    print(f"✗ {name}: Failed")
                    all_good = False
                    
            return all_good
        except Exception as e:
            print(f"✗ Config error: {e}")
            return False
    else:
        print("✗ config.yaml not found")
        return False

def main():
    """Run all tests"""
    print("=== Phase 2 Pipeline Test ===\n")
    
    tests = [
        test_imports,
        test_data_availability,
        test_database,
        test_config
    ]
    
    results = []
    for test in tests:
        results.append(test())
    
    print("\n=== Summary ===")
    passed = sum(results)
    total = len(results)
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("\n✓ All tests passed! Pipeline is ready.")
        print("\nNext steps:")
        print("1. python main.py train    # Train model on existing L2 data")
        print("2. python main.py trade --paper    # Run paper trading")
    else:
        print("\n✗ Some tests failed. Fix issues before proceeding.")
    
    return 0 if passed == total else 1

if __name__ == "__main__":
    sys.exit(main())