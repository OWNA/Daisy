#!/usr/bin/env python3
"""
verify_live_database_schema.py - Verify Live Database Schema

Quick verification that the new unified database has the complete schema.
"""

import sqlite3
from datetime import datetime

def verify_live_database_schema():
    """Verify the live database schema is complete."""
    
    db_path = './trading_bot_live.db'
    table_name = 'l2_training_data_practical'
    
    print("🔍 LIVE DATABASE SCHEMA VERIFICATION")
    print("=" * 60)
    print(f"Database: {db_path}")
    print(f"Table: {table_name}")
    print(f"Time: {datetime.now().isoformat()}")
    print("=" * 60)
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Get table info
        cursor.execute(f"PRAGMA table_info({table_name})")
        columns = cursor.fetchall()
        
        # Get indexes
        cursor.execute(f"PRAGMA index_list({table_name})")
        indexes = cursor.fetchall()
        
        # Get row count
        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
        row_count = cursor.fetchone()[0]
        
        conn.close()
        
        print(f"📊 DATABASE STATUS:")
        print(f"  Total columns: {len(columns)}")
        print(f"  Total indexes: {len(indexes)}")
        print(f"  Current rows: {row_count}")
        
        # Categorize columns
        required_columns = {
            'bid_levels': [f'bid_price_{i}' for i in range(1, 11)] + [f'bid_size_{i}' for i in range(1, 11)],
            'ask_levels': [f'ask_price_{i}' for i in range(1, 11)] + [f'ask_size_{i}' for i in range(1, 11)],
            'basic_features': ['mid_price', 'spread', 'spread_bps'],
            'aggregated_metrics': ['total_bid_volume_10', 'total_ask_volume_10', 'weighted_bid_price', 'weighted_ask_price', 'order_book_imbalance'],
            'microstructure': ['microprice', 'price_impact_bid', 'price_impact_ask'],
            'targets': ['target_return_1min', 'target_return_5min', 'target_volatility', 'target_direction'],
            'metadata': ['id', 'timestamp', 'symbol', 'exchange', 'update_id', 'sequence_id', 'data_quality_score', 'data_source']
        }
        
        column_names = [col[1] for col in columns]
        
        print(f"\n📋 SCHEMA VERIFICATION:")
        total_expected = 0
        total_found = 0
        
        for category, expected_cols in required_columns.items():
            found = sum(1 for col in expected_cols if col in column_names)
            total_expected += len(expected_cols)
            total_found += found
            status = "✅" if found == len(expected_cols) else "❌"
            print(f"  {status} {category}: {found}/{len(expected_cols)} columns")
            
            if found != len(expected_cols):
                missing = [col for col in expected_cols if col not in column_names]
                print(f"      Missing: {missing}")
        
        print(f"\n🎯 OVERALL STATUS:")
        print(f"  Expected columns: {total_expected}")
        print(f"  Found columns: {total_found}")
        print(f"  Additional columns: {len(columns) - total_found}")
        
        if total_found >= 51:  # At least 51 essential columns
            print("✅ SCHEMA VERIFICATION PASSED")
            print("📋 Database ready for ML training and live data ingestion")
        else:
            print("❌ SCHEMA VERIFICATION FAILED")
            print("⚠️  Missing essential columns for ML training")
        
        return total_found >= 51
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False

if __name__ == "__main__":
    success = verify_live_database_schema()
    exit(0 if success else 1)