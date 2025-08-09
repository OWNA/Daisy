#!/usr/bin/env python3
"""
force_purge_data.py - Force Purge Using Python-only Approach

CRITICAL_PIVOT_PLAN.md - Task 1.1: Purge Testnet Data
Use Python to completely clear table data without file operations.
"""

import sqlite3
import os
from datetime import datetime

def force_purge_table_data():
    """Force purge table data using Python sqlite3."""
    
    db_path = './trading_bot.db'
    table_name = 'l2_training_data_practical'
    
    print("🔥 FORCE PURGING TABLE DATA")
    print("=" * 50)
    print(f"Database: {db_path}")
    print(f"Table: {table_name}")
    print(f"Method: Python sqlite3 direct manipulation")
    print(f"Time: {datetime.now().isoformat()}")
    print("=" * 50)
    
    if not os.path.exists(db_path):
        print(f"❌ Database file not found: {db_path}")
        return 0
    
    try:
        # Connect with timeout and specific options
        conn = sqlite3.connect(db_path, timeout=30.0)
        cursor = conn.cursor()
        
        # Enable WAL mode for better concurrency
        cursor.execute("PRAGMA journal_mode=WAL")
        
        # Count existing rows
        print("📊 Counting existing rows...")
        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
        initial_count = cursor.fetchone()[0]
        print(f"Initial row count: {initial_count:,}")
        
        if initial_count == 0:
            print("✅ Table is already empty")
            conn.close()
            return 0
        
        # Get sample of data sources to confirm what we're deleting
        cursor.execute(f"SELECT DISTINCT data_source FROM {table_name} LIMIT 10")
        data_sources = cursor.fetchall()
        print(f"Data sources found: {[row[0] for row in data_sources]}")
        
        # Method 1: Try direct DELETE
        print("\n🚨 Attempting direct DELETE...")
        cursor.execute(f"DELETE FROM {table_name}")
        rows_affected = cursor.rowcount
        print(f"DELETE command affected {rows_affected:,} rows")
        
        # Commit the transaction
        print("💾 Committing transaction...")
        conn.commit()
        
        # Verify deletion
        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
        final_count = cursor.fetchone()[0]
        
        # Method 2: If still have rows, try DROP and recreate
        if final_count > 0:
            print(f"⚠️  Still have {final_count} rows, trying DROP/CREATE approach...")
            
            # Drop table
            cursor.execute(f"DROP TABLE IF EXISTS {table_name}")
            
            # Recreate table
            create_table_sql = f"""
            CREATE TABLE {table_name} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                symbol TEXT NOT NULL,
                bid_price_1 REAL,
                bid_size_1 REAL,
                ask_price_1 REAL,
                ask_size_1 REAL,
                bid_price_2 REAL,
                bid_size_2 REAL,
                ask_price_2 REAL,
                ask_size_2 REAL,
                bid_price_3 REAL,
                bid_size_3 REAL,
                ask_price_3 REAL,
                ask_size_3 REAL,
                bid_price_4 REAL,
                bid_size_4 REAL,
                ask_price_4 REAL,
                ask_size_4 REAL,
                bid_price_5 REAL,
                bid_size_5 REAL,
                ask_price_5 REAL,
                ask_size_5 REAL,
                mid_price REAL,
                spread REAL,
                sequence INTEGER,
                exchange_timestamp INTEGER,
                data_source TEXT DEFAULT 'demo_trading'
            )
            """
            cursor.execute(create_table_sql)
            conn.commit()
            
            # Final count after recreate
            cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            final_count = cursor.fetchone()[0]
        
        conn.close()
        
        # Report results
        print("\n" + "=" * 50)
        print("✅ FORCE PURGE COMPLETE")
        print("=" * 50)
        print(f"Initial rows: {initial_count:,}")
        print(f"Final rows: {final_count:,}")
        print(f"Rows purged: {initial_count - final_count:,}")
        print("=" * 50)
        
        if final_count == 0:
            print("🎉 SUCCESS: All testnet data has been purged!")
            print("📋 Table is clean and ready for Demo Trading")
        else:
            print(f"⚠️  WARNING: {final_count} rows still remain")
        
        return initial_count - final_count
        
    except Exception as e:
        print(f"❌ ERROR during force purge: {e}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        return 0

if __name__ == "__main__":
    purged_count = force_purge_table_data()
    print(f"\n🏁 Final result: {purged_count:,} rows purged from l2_training_data_practical")