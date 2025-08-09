#!/usr/bin/env python3
"""
verify_demo_trading_config.py - Verify Demo Trading Configuration

CRITICAL_PIVOT_PLAN.md - Task 1.2: Verify Configuration Updates
Ensure all components are properly configured for Demo Trading.
"""

import os
import yaml
import sqlite3
from datetime import datetime

def verify_demo_trading_config():
    """Verify all configurations are updated for Demo Trading."""
    
    print("🔍 DEMO TRADING CONFIGURATION VERIFICATION")
    print("=" * 60)
    print(f"Time: {datetime.now().isoformat()}")
    print("=" * 60)
    
    verification_results = {
        'config_files': {},
        'database': {},
        'component_factory': {},
        'scripts': {}
    }
    
    # Check YAML configuration files
    yaml_files = ['config.yaml', 'config_l2_only.yaml']
    for yaml_file in yaml_files:
        print(f"\n📋 Checking {yaml_file}...")
        try:
            if os.path.exists(yaml_file):
                with open(yaml_file, 'r') as f:
                    config = yaml.safe_load(f)
                
                db_path = config.get('database_path', 'NOT_FOUND')
                testnet = config.get('exchange', {}).get('testnet', 'NOT_FOUND')
                exchange_testnet = config.get('exchange_testnet', 'NOT_SET')
                
                verification_results['config_files'][yaml_file] = {
                    'database_path': db_path,
                    'testnet': testnet,
                    'exchange_testnet': exchange_testnet,
                    'correct_db': db_path == './trading_bot_live.db',
                    'correct_testnet': testnet == False
                }
                
                print(f"  Database path: {db_path}")
                print(f"  Testnet: {testnet}")
                print(f"  ✅ Correct DB: {db_path == './trading_bot_live.db'}")
                print(f"  ✅ Demo Trading: {testnet == False}")
            else:
                print(f"  ❌ File not found: {yaml_file}")
                verification_results['config_files'][yaml_file] = {'error': 'File not found'}
        except Exception as e:
            print(f"  ❌ Error reading {yaml_file}: {e}")
            verification_results['config_files'][yaml_file] = {'error': str(e)}
    
    # Check database
    print(f"\n🗄️  Checking database configuration...")
    db_files = ['./trading_bot_live.db', './trading_bot.db']
    for db_file in db_files:
        exists = os.path.exists(db_file)
        print(f"  {db_file}: {'✅ EXISTS' if exists else '❌ NOT FOUND'}")
        
        if exists and db_file == './trading_bot_live.db':
            try:
                conn = sqlite3.connect(db_file)
                cursor = conn.cursor()
                
                # Check table exists
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='l2_training_data_practical'")
                table_exists = cursor.fetchone() is not None
                
                # Check row count
                if table_exists:
                    cursor.execute("SELECT COUNT(*) FROM l2_training_data_practical")
                    row_count = cursor.fetchone()[0]
                else:
                    row_count = 0
                
                # Check data sources
                if table_exists and row_count > 0:
                    cursor.execute("SELECT DISTINCT data_source FROM l2_training_data_practical")
                    data_sources = [row[0] for row in cursor.fetchall()]
                else:
                    data_sources = []
                
                conn.close()
                
                verification_results['database'][db_file] = {
                    'exists': True,
                    'table_exists': table_exists,
                    'row_count': row_count,
                    'data_sources': data_sources
                }
                
                print(f"    Table exists: {'✅' if table_exists else '❌'}")
                print(f"    Row count: {row_count}")
                print(f"    Data sources: {data_sources}")
                
            except Exception as e:
                print(f"    ❌ Database error: {e}")
                verification_results['database'][db_file] = {'error': str(e)}
        else:
            verification_results['database'][db_file] = {'exists': exists}
    
    # Check key Python files
    print(f"\n🐍 Checking Python configuration files...")
    
    # Check component_factory.py
    factory_file = 'component_factory.py'
    if os.path.exists(factory_file):
        with open(factory_file, 'r') as f:
            content = f.read()
        
        demo_db_count = content.count('./trading_bot_live.db')
        old_db_count = content.count('./trading_bot.db')
        sandbox_false_count = content.count("'sandbox': False")
        api_main_count = content.count('BYBIT_API_KEY_MAIN')
        
        verification_results['component_factory'] = {
            'demo_db_references': demo_db_count,
            'old_db_references': old_db_count,
            'sandbox_false_count': sandbox_false_count,
            'api_main_count': api_main_count
        }
        
        print(f"  component_factory.py:")
        print(f"    Demo DB references: {demo_db_count}")
        print(f"    Old DB references: {old_db_count}")
        print(f"    Sandbox=False: {sandbox_false_count}")
        print(f"    Main API keys: {api_main_count}")
    
    # Check data_ingestor.py
    ingestor_file = 'data_ingestor.py'
    if os.path.exists(ingestor_file):
        with open(ingestor_file, 'r') as f:
            content = f.read()
        
        demo_trading_count = content.count('demo_trading')
        sandbox_false_default = "'sandbox', False" in content
        
        print(f"  data_ingestor.py:")
        print(f"    Demo trading references: {demo_trading_count}")
        print(f"    Sandbox=False default: {'✅' if sandbox_false_default else '❌'}")
    
    # Summary
    print(f"\n" + "=" * 60)
    print("✅ CONFIGURATION VERIFICATION SUMMARY")
    print("=" * 60)
    
    all_configs_correct = True
    
    # Check config files
    for file, config in verification_results['config_files'].items():
        if 'error' in config:
            print(f"❌ {file}: ERROR - {config['error']}")
            all_configs_correct = False
        else:
            correct = config.get('correct_db', False) and config.get('correct_testnet', False)
            print(f"{'✅' if correct else '❌'} {file}: {'CORRECT' if correct else 'NEEDS FIXING'}")
            if not correct:
                all_configs_correct = False
    
    # Check database
    demo_db_ready = verification_results['database'].get('./trading_bot_live.db', {}).get('exists', False)
    print(f"{'✅' if demo_db_ready else '❌'} Demo database: {'READY' if demo_db_ready else 'NOT READY'}")
    
    if not demo_db_ready:
        all_configs_correct = False
    
    print(f"\n🎯 OVERALL STATUS: {'✅ ALL CONFIGURATIONS CORRECT' if all_configs_correct else '❌ SOME CONFIGURATIONS NEED FIXING'}")
    
    if all_configs_correct:
        print("📋 System is ready for Demo Trading implementation!")
    else:
        print("⚠️  Please review and fix the configuration issues above.")
    
    return all_configs_correct

if __name__ == "__main__":
    success = verify_demo_trading_config()
    exit(0 if success else 1)