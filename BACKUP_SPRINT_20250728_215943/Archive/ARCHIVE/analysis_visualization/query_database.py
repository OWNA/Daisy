#!/usr/bin/env python3
"""
Database Query Script - Verify L2 Training Data Upload
This script provides comprehensive database statistics and verification.
"""

import sqlite3
import pandas as pd
from datetime import datetime


def query_database_stats():
    """Query database for comprehensive statistics."""
    print("🔍 Querying Database Statistics...\n")
    
    try:
        # Connect to database
        conn = sqlite3.connect('./trading_bot.db')
        cursor = conn.cursor()
        
        # 1. List all tables
        print("📊 Database Tables:")
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = cursor.fetchall()
        for table in tables:
            print(f"  - {table[0]}")
        
        # 2. L2 training data count
        print(f"\n📈 L2 Training Data:")
        cursor.execute("SELECT COUNT(*) FROM l2_training_data")
        count = cursor.fetchone()[0]
        print(f"  - Total rows: {count:,}")
        
        if count > 0:
            # 3. Date range
            cursor.execute("SELECT MIN(timestamp), MAX(timestamp) FROM l2_training_data")
            date_range = cursor.fetchone()
            print(f"  - Date range: {date_range[0]} to {date_range[1]}")
            
            # 4. Symbol information
            cursor.execute("SELECT COUNT(DISTINCT symbol) FROM l2_training_data")
            symbols = cursor.fetchone()[0]
            print(f"  - Unique symbols: {symbols}")
            
            # 5. Column information
            cursor.execute("PRAGMA table_info(l2_training_data)")
            columns = cursor.fetchall()
            print(f"  - Total columns: {len(columns)}")
            
            # 6. Sample data
            print(f"\n📋 Sample Data (first 3 rows):")
            df_sample = pd.read_sql_query(
                "SELECT timestamp, bid_price_1, ask_price_1, mid, symbol FROM l2_training_data LIMIT 3", 
                conn
            )
            print(df_sample.to_string(index=False))
            
            # 7. Data quality checks
            print(f"\n🔍 Data Quality:")
            cursor.execute("SELECT COUNT(*) FROM l2_training_data WHERE bid_price_1 IS NULL")
            null_bids = cursor.fetchone()[0]
            cursor.execute("SELECT COUNT(*) FROM l2_training_data WHERE ask_price_1 IS NULL")
            null_asks = cursor.fetchone()[0]
            cursor.execute("SELECT COUNT(*) FROM l2_training_data WHERE mid IS NULL")
            null_mid = cursor.fetchone()[0]
            
            print(f"  - NULL bid_price_1: {null_bids:,} ({null_bids/count*100:.2f}%)")
            print(f"  - NULL ask_price_1: {null_asks:,} ({null_asks/count*100:.2f}%)")
            print(f"  - NULL mid: {null_mid:,} ({null_mid/count*100:.2f}%)")
            
            # 8. Target columns analysis
            target_columns = [col[1] for col in columns if col[1].startswith('target_')]
            print(f"  - Target columns: {len(target_columns)}")
            if target_columns:
                print(f"  - Sample targets: {target_columns[:3]}...")
            
            # 9. Price statistics
            cursor.execute("SELECT MIN(bid_price_1), MAX(bid_price_1), AVG(bid_price_1) FROM l2_training_data WHERE bid_price_1 IS NOT NULL")
            price_stats = cursor.fetchone()
            print(f"\n💰 Price Statistics:")
            print(f"  - Bid price range: ${price_stats[0]:,.2f} - ${price_stats[1]:,.2f}")
            print(f"  - Average bid price: ${price_stats[2]:,.2f}")
            
        else:
            print("  ❌ No data found in l2_training_data table")
        
        # 10. Database file size
        import os
        db_size = os.path.getsize('./trading_bot.db') / (1024 * 1024)
        print(f"\n💾 Database File Size: {db_size:.1f} MB")
        
        conn.close()
        
        # Summary
        print(f"\n✅ Database Query Complete!")
        if count > 0:
            print(f"🎉 Successfully verified {count:,} rows of L2 training data")
            return True
        else:
            print("❌ No L2 training data found")
            return False
            
    except Exception as e:
        print(f"❌ Database query failed: {e}")
        return False


if __name__ == "__main__":
    success = query_database_stats()
    print(f"\n{'='*50}")
    if success:
        print("🚀 Ready to proceed with Phase 2!")
    else:
        print("⚠️ Database issues detected - need to resolve before Phase 2") 