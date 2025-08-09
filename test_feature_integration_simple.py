#!/usr/bin/env python3
"""
Simple test script for enhanced feature engineering integration.
Tests basic functionality without heavy dependencies.
"""

import sys
import os
import sqlite3
from datetime import datetime, timedelta

def test_database_schema():
    """Test that the database has the expected L2 features schema."""
    print("=" * 60)
    print("TESTING DATABASE SCHEMA")
    print("=" * 60)
    
    db_path = "trading_bot.db"
    if not os.path.exists(db_path):
        print(f"❌ Database {db_path} not found")
        return False
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Check if l2_features table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='l2_features'")
        if not cursor.fetchone():
            print("❌ l2_features table not found")
            return False
        print("✓ l2_features table exists")
        
        # Get schema info
        cursor.execute("PRAGMA table_info(l2_features)")
        columns = cursor.fetchall()
        column_names = [col[1] for col in columns]
        
        print(f"✓ Found {len(columns)} columns in l2_features table")
        
        # Check for Phase 1 features
        phase1_features = [
            'ofi_10s', 'ofi_30s', 'ofi_1m', 'ofi_5m',
            'ofi_normalized_10s', 'bid_pressure', 'ask_pressure', 
            'quote_lifetime', 'book_resilience', 'l2_volatility_50'
        ]
        
        present_features = [f for f in phase1_features if f in column_names]
        missing_features = [f for f in phase1_features if f not in column_names]
        
        print(f"✓ Phase 1 features present: {len(present_features)}/{len(phase1_features)}")
        if missing_features:
            print(f"⚠️  Missing features: {missing_features}")
        
        conn.close()
        return len(missing_features) == 0
        
    except Exception as e:
        print(f"❌ Database test failed: {e}")
        return False

def test_feature_engineer_import():
    """Test that we can import the enhanced feature engineer."""
    print("\n" + "=" * 60)
    print("TESTING FEATURE ENGINEER IMPORT")
    print("=" * 60)
    
    try:
        # Add current directory to path
        sys.path.insert(0, '.')
        
        from featureengineer_enhanced import EnhancedFeatureEngineer
        print("✓ Successfully imported EnhancedFeatureEngineer")
        
        # Test initialization
        config = {'symbol': 'BTC/USDT', 'l2_features': []}
        fe = EnhancedFeatureEngineer(config)
        print("✓ Successfully initialized EnhancedFeatureEngineer")
        
        # Test Phase 1 features definition
        phase1_features = fe._define_phase1_features()
        total_features = sum(len(features) for features in phase1_features.values())
        print(f"✓ Phase 1 features defined: {total_features} features across {len(phase1_features)} categories")
        
        for category, features in phase1_features.items():
            print(f"  - {category}: {len(features)} features")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Initialization failed: {e}")
        return False

def test_database_operations():
    """Test basic database operations."""
    print("\n" + "=" * 60)
    print("TESTING DATABASE OPERATIONS")
    print("=" * 60)
    
    try:
        sys.path.insert(0, '.')
        from featureengineer_enhanced import EnhancedFeatureEngineer
        
        config = {'symbol': 'BTC/USDT', 'l2_features': []}
        fe = EnhancedFeatureEngineer(config)
        
        # Test database connection
        with fe.get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM l2_features")
            count = cursor.fetchone()[0]
            print(f"✓ Database connection successful, {count} existing records")
        
        # Test feature writing
        test_features = {
            'ofi_10s': 0.5,
            'bid_pressure': 100.0,
            'quote_lifetime': 5.0
        }
        
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
        success = fe.write_features_to_db(test_features, timestamp, 'BTCUSDT')
        
        if success:
            print("✓ Successfully wrote test features to database")
            
            # Test reading back
            existing_features = fe.read_existing_features(timestamp, 'BTCUSDT')
            if existing_features:
                print("✓ Successfully read features back from database")
                for key, value in test_features.items():
                    if key in existing_features and existing_features[key] is not None:
                        print(f"  - {key}: {existing_features[key]}")
            else:
                print("⚠️  Could not read features back from database")
        else:
            print("❌ Failed to write features to database")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Database operations test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("Enhanced Feature Engineering Integration Test")
    print("Testing core functionality without heavy dependencies")
    
    tests = [
        ("Database Schema", test_database_schema),
        ("Feature Engineer Import", test_feature_engineer_import),
        ("Database Operations", test_database_operations)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ Test '{test_name}' crashed: {e}")
            results.append((test_name, False))
    
    print("\n" + "=" * 60)
    print("TEST RESULTS SUMMARY")
    print("=" * 60)
    
    passed = 0
    for test_name, result in results:
        status = "✓ PASS" if result else "❌ FAIL"
        print(f"{status} - {test_name}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("🎉 All tests passed! Database integration is ready.")
    else:
        print("⚠️  Some tests failed. Please check the issues above.")

if __name__ == "__main__":
    main()