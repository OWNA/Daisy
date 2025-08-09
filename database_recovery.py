#!/usr/bin/env python3
"""
database_recovery.py - Handle Corrupted Database Recovery

CRITICAL_PIVOT_PLAN.md - Task 1.1: Purge Testnet Data
Handle corrupted database and create fresh database for Demo Trading.
"""

import sqlite3
import os
import shutil
from datetime import datetime

def handle_database_recovery():
    """Handle corrupted database and create fresh one."""
    
    db_path = './trading_bot.db'
    backup_suffix = f"_corrupted_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    backup_path = f"{db_path}{backup_suffix}"
    
    print("🛠️  DATABASE RECOVERY PROCESS")
    print("=" * 60)
    print(f"Original DB: {db_path}")
    print(f"Backup path: {backup_path}")
    print(f"Time: {datetime.now().isoformat()}")
    print("=" * 60)
    
    # Check if database exists
    if not os.path.exists(db_path):
        print(f"❌ Database file not found: {db_path}")
        return False, 0
    
    try:
        # First, try to get row count from corrupted database
        initial_row_count = 0
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM l2_training_data_practical")
            initial_row_count = cursor.fetchone()[0]
            conn.close()
            print(f"📊 Corrupted DB contained: {initial_row_count:,} rows")
        except Exception as e:
            print(f"⚠️  Could not read corrupted database: {e}")
            print("🔍 Assuming corrupted database had testnet data")
            initial_row_count = "UNKNOWN (corrupted)"
        
        # Backup the corrupted database
        print("💾 Backing up corrupted database...")
        shutil.copy2(db_path, backup_path)
        print(f"✅ Corrupted database backed up to: {backup_path}")
        
        # Remove corrupted database
        print("🗑️  Removing corrupted database...")
        os.remove(db_path)
        print(f"✅ Corrupted database removed: {db_path}")
        
        # Create fresh database with proper schema
        print("🏗️  Creating fresh database...")
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Create the l2_training_data_practical table with all required columns
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS l2_training_data_practical (
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
        
        # Verify new database
        cursor.execute("SELECT COUNT(*) FROM l2_training_data_practical")
        new_row_count = cursor.fetchone()[0]
        
        cursor.execute("PRAGMA table_info(l2_training_data_practical)")
        columns = cursor.fetchall()
        
        conn.close()
        
        print("\n" + "=" * 60)
        print("✅ DATABASE RECOVERY COMPLETE")
        print("=" * 60)
        print(f"Original rows (corrupted): {initial_row_count}")
        print(f"New database rows: {new_row_count}")
        print(f"Backup location: {backup_path}")
        print(f"Fresh database: {db_path}")
        print(f"Table columns: {len(columns)}")
        print("=" * 60)
        print("🎉 SUCCESS: Fresh database ready for Demo Trading!")
        print("📋 All testnet data has been effectively purged")
        
        return True, initial_row_count
        
    except Exception as e:
        print(f"❌ ERROR during recovery: {e}")
        return False, 0

if __name__ == "__main__":
    success, purged_count = handle_database_recovery()
    if success:
        print(f"\n🏁 Final result: Database recovered, {purged_count} testnet rows purged")
    else:
        print(f"\n❌ Recovery failed")