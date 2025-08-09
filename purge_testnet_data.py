#!/usr/bin/env python3
"""
purge_testnet_data.py - Purge All Testnet Data from Database

CRITICAL_PIVOT_PLAN.md - Task 1.1: Purge Testnet Data
Clean slate for Demo Trading service implementation.
"""

import sqlite3
import os
from datetime import datetime

def purge_testnet_data():
    """Delete all rows from l2_training_data_practical table."""
    
    db_path = './trading_bot.db'
    table_name = 'l2_training_data_practical'
    
    print("🗑️  PURGING TESTNET DATA")
    print("=" * 50)
    print(f"Database: {db_path}")
    print(f"Table: {table_name}")
    print(f"Time: {datetime.now().isoformat()}")
    print("=" * 50)
    
    if not os.path.exists(db_path):
        print(f"❌ Database file not found: {db_path}")
        return 0
    
    try:
        # Connect to database
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Count existing rows before deletion
        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
        row_count_before = cursor.fetchone()[0]
        print(f"📊 Rows before deletion: {row_count_before:,}")
        
        if row_count_before == 0:
            print("✅ Table is already empty - no data to purge")
            conn.close()
            return 0
        
        # Delete all rows
        print("🚨 DELETING ALL ROWS...")
        cursor.execute(f"DELETE FROM {table_name}")
        rows_deleted = cursor.rowcount
        
        # Commit the transaction
        conn.commit()
        
        # Verify deletion
        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
        row_count_after = cursor.fetchone()[0]
        
        # Vacuum database to reclaim space
        print("🧹 Vacuuming database to reclaim space...")
        cursor.execute("VACUUM")
        conn.commit()
        
        conn.close()
        
        # Report results
        print("\n" + "=" * 50)
        print("✅ TESTNET DATA PURGE COMPLETE")
        print("=" * 50)
        print(f"Rows deleted: {rows_deleted:,}")
        print(f"Rows before: {row_count_before:,}")
        print(f"Rows after: {row_count_after:,}")
        print(f"Database vacuumed: ✅")
        print("=" * 50)
        
        if row_count_after == 0:
            print("🎉 SUCCESS: All testnet data has been purged!")
            print("📋 Ready for Demo Trading service implementation")
        else:
            print(f"⚠️  WARNING: {row_count_after} rows remain in table")
        
        return rows_deleted
        
    except Exception as e:
        print(f"❌ ERROR during purge: {e}")
        return 0

if __name__ == "__main__":
    deleted_count = purge_testnet_data()
    print(f"\n🏁 Final result: {deleted_count:,} rows deleted")