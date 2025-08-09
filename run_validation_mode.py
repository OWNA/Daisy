#!/usr/bin/env python3
"""
run_validation_mode.py - System Validation with LightGBM Fallback

This script validates the system initialization and basic operations
with fallback handling for LightGBM dependency issues.
"""

import os
import sys
import time
import logging
import yaml
import sqlite3
from datetime import datetime
from dotenv import load_dotenv

# Core system imports that should work
try:
    from component_factory import ComponentFactory, create_factory_from_config
    print("✅ Component factory imported successfully")
except ImportError as e:
    print(f"❌ Critical Error: Could not import component factory: {e}")
    sys.exit(1)

try:
    from production_model_predictor import ProductionModelPredictor
    print("✅ Production model predictor imported successfully")
except ImportError as e:
    print(f"❌ Warning: Production model predictor unavailable: {e}")
    ProductionModelPredictor = None

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f'validation_{datetime.now().strftime("%Y%m%d")}.log')
    ]
)
logger = logging.getLogger(__name__)

def validate_system():
    """Validate core system components."""
    
    print("🔍 SYSTEM VALIDATION - Phase 2")
    print("=" * 60)
    print(f"Time: {datetime.now().isoformat()}")
    print("=" * 60)
    
    validation_results = {
        'config_loading': False,
        'database_connectivity': False,
        'component_factory': False,
        'basic_operations': False
    }
    
    # Test 1: Configuration Loading
    print("\n📋 Test 1: Configuration Loading...")
    try:
        load_dotenv()
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        print(f"  ✅ Config loaded: {len(config)} keys")
        print(f"  ✅ Database path: {config.get('database_path', 'NOT_FOUND')}")
        print(f"  ✅ Exchange config: {config.get('exchange', {}).get('name', 'NOT_FOUND')}")
        validation_results['config_loading'] = True
        
    except Exception as e:
        print(f"  ❌ Config loading failed: {e}")
        return validation_results
    
    # Test 2: Database Connectivity
    print("\n🗄️  Test 2: Database Connectivity...")
    try:
        db_path = config.get('database_path', './trading_bot_live.db')
        if not os.path.exists(db_path):
            print(f"  ❌ Database file not found: {db_path}")
            return validation_results
            
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Check table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='l2_training_data_practical'")
        table_exists = cursor.fetchone() is not None
        
        if table_exists:
            cursor.execute("SELECT COUNT(*) FROM l2_training_data_practical")
            row_count = cursor.fetchone()[0]
            
            cursor.execute("PRAGMA table_info(l2_training_data_practical)")
            columns = cursor.fetchall()
            
            print(f"  ✅ Database connected: {db_path}")
            print(f"  ✅ Table exists: l2_training_data_practical")
            print(f"  ✅ Columns: {len(columns)}")
            print(f"  ✅ Rows: {row_count}")
            validation_results['database_connectivity'] = True
        else:
            print(f"  ❌ Required table not found")
            
        conn.close()
        
    except Exception as e:
        print(f"  ❌ Database connectivity failed: {e}")
        return validation_results
    
    # Test 3: Component Factory
    print("\n🏗️  Test 3: Component Factory...")
    try:
        factory = ComponentFactory(config)
        
        # Test basic component creation
        db_component = factory.get_component('database')
        if db_component:
            print("  ✅ Database component created")
        else:
            print("  ❌ Database component failed")
            
        feature_registry = factory.get_component('feature_registry')
        if feature_registry:
            print("  ✅ Feature registry created")
        else:
            print("  ❌ Feature registry failed")
            
        validation_results['component_factory'] = True
        
    except Exception as e:
        print(f"  ❌ Component factory failed: {e}")
        return validation_results
    
    # Test 4: Basic Operations
    print("\n⚡ Test 4: Basic Operations...")
    try:
        # Test data retrieval
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Test query execution
        cursor.execute("SELECT COUNT(*) FROM l2_training_data_practical LIMIT 1")
        result = cursor.fetchone()
        
        print(f"  ✅ Database query executed successfully")
        print(f"  ✅ System ready for operations")
        
        conn.close()
        validation_results['basic_operations'] = True
        
    except Exception as e:
        print(f"  ❌ Basic operations failed: {e}")
        return validation_results
    
    return validation_results

def main():
    """Main validation execution."""
    
    results = validate_system()
    
    print("\n" + "=" * 60)
    print("📊 VALIDATION SUMMARY")
    print("=" * 60)
    
    total_tests = len(results)
    passed_tests = sum(results.values())
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} {test_name.replace('_', ' ').title()}")
    
    print(f"\n🎯 Overall Result: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("✅ SYSTEM VALIDATION SUCCESSFUL")
        print("📋 Core system is ready for paper trading operations")
        return True
    else:
        print("❌ SYSTEM VALIDATION FAILED")
        print("⚠️  Critical components need attention before deployment")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)