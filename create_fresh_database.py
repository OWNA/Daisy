#!/usr/bin/env python3
"""
create_fresh_database.py - Create Fresh Database for Live Trading

CRITICAL_PIVOT & REFACTOR PLAN - Task 1.1: Create Unified Database
Create new unified database with complete 51-column schema for live trading.
"""

import sqlite3
import os
from datetime import datetime

def create_fresh_database():
    """Create a completely fresh database for live trading with complete schema."""
    
    # Use unified live database name
    new_db_path = './trading_bot_live.db'
    old_db_path = './trading_bot.db'
    table_name = 'l2_training_data_practical'
    
    print("🏗️  CREATING FRESH LIVE DATABASE")
    print("=" * 60)
    print(f"New database: {new_db_path}")
    print(f"Old DB: {old_db_path}")
    print(f"Table: {table_name}")
    print(f"Schema: Complete 51-column L2 training schema")
    print(f"Time: {datetime.now().isoformat()}")
    print("=" * 60)
    
    # Check existing databases
    print("📊 Checking for existing databases...")
    if os.path.exists(old_db_path):
        file_size = os.path.getsize(old_db_path)
        print(f"  Old database size: {file_size:,} bytes")
    else:
        print(f"  Old database not found: {old_db_path}")
    
    if os.path.exists(new_db_path):
        print(f"  Existing live database found - will be replaced")
    else:
        print(f"  No existing live database found")
    
    try:
        # Remove existing fresh database if it exists
        if os.path.exists(new_db_path):
            os.remove(new_db_path)
            print(f"🗑️  Removed existing {new_db_path}")
        
        # Create fresh database
        print("🏗️  Creating fresh database...")
        conn = sqlite3.connect(new_db_path)
        cursor = conn.cursor()
        
        # Create the l2_training_data_practical table with complete 51-column schema
        create_table_sql = f"""
        CREATE TABLE {table_name} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME NOT NULL,
            symbol TEXT NOT NULL,
            exchange TEXT DEFAULT 'bybit',
            
            -- Top 10 bid levels
            bid_price_1 REAL, bid_size_1 REAL,
            bid_price_2 REAL, bid_size_2 REAL,
            bid_price_3 REAL, bid_size_3 REAL,
            bid_price_4 REAL, bid_size_4 REAL,
            bid_price_5 REAL, bid_size_5 REAL,
            bid_price_6 REAL, bid_size_6 REAL,
            bid_price_7 REAL, bid_size_7 REAL,
            bid_price_8 REAL, bid_size_8 REAL,
            bid_price_9 REAL, bid_size_9 REAL,
            bid_price_10 REAL, bid_size_10 REAL,
            
            -- Top 10 ask levels
            ask_price_1 REAL, ask_size_1 REAL,
            ask_price_2 REAL, ask_size_2 REAL,
            ask_price_3 REAL, ask_size_3 REAL,
            ask_price_4 REAL, ask_size_4 REAL,
            ask_price_5 REAL, ask_size_5 REAL,
            ask_price_6 REAL, ask_size_6 REAL,
            ask_price_7 REAL, ask_size_7 REAL,
            ask_price_8 REAL, ask_size_8 REAL,
            ask_price_9 REAL, ask_size_9 REAL,
            ask_price_10 REAL, ask_size_10 REAL,
            
            -- Calculated fields
            mid_price REAL,
            spread REAL,
            spread_bps REAL,
            
            -- Aggregated order book metrics
            total_bid_volume_10 REAL,
            total_ask_volume_10 REAL,
            weighted_bid_price REAL,
            weighted_ask_price REAL,
            order_book_imbalance REAL,
            
            -- Microstructure features
            microprice REAL,
            price_impact_bid REAL,
            price_impact_ask REAL,
            
            -- Target variables
            target_return_1min REAL,
            target_return_5min REAL,
            target_volatility REAL,
            target_direction INTEGER,
            
            -- Metadata
            update_id INTEGER,
            sequence_id INTEGER,
            data_quality_score REAL,
            data_source TEXT DEFAULT 'live_trading'
        )
        """
        
        cursor.execute(create_table_sql)
        
        # Create indexes for performance (matching the complete schema)
        indexes = [
            f"CREATE INDEX IF NOT EXISTS idx_l2_prac_timestamp ON {table_name}(timestamp)",
            f"CREATE INDEX IF NOT EXISTS idx_l2_prac_symbol ON {table_name}(symbol)",
            f"CREATE INDEX IF NOT EXISTS idx_l2_prac_symbol_timestamp ON {table_name}(symbol, timestamp)",
            f"CREATE INDEX IF NOT EXISTS idx_l2_prac_data_source ON {table_name}(data_source)"
        ]
        
        for index_sql in indexes:
            cursor.execute(index_sql)
        
        conn.commit()
        
        # Verify new database
        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
        row_count = cursor.fetchone()[0]
        
        cursor.execute(f"PRAGMA table_info({table_name})")
        columns = cursor.fetchall()
        
        cursor.execute("PRAGMA journal_mode=WAL")  # Enable WAL mode
        
        conn.close()
        
        print("\n" + "=" * 60)
        print("✅ FRESH LIVE DATABASE CREATED SUCCESSFULLY")
        print("=" * 60)
        print(f"Database file: {new_db_path}")
        print(f"Table created: {table_name}")
        print(f"Current rows: {row_count}")
        print(f"Total columns: {len(columns)} (Complete L2 schema)")
        print(f"Indexes: 4 (timestamp, symbol, symbol+timestamp, data_source)")
        print(f"Journal mode: WAL")
        print(f"Schema: Full 51-column training schema with 10 levels")
        print("=" * 60)
        print("🎉 SUCCESS: Unified live database ready for production!")
        print("📋 Complete schema supports ML training and live data ingestion")
        print(f"⚠️  Note: Update all config files to use '{new_db_path}'")
        
        # Count columns by category
        bid_columns = len([c for c in columns if c[1].startswith('bid_')])
        ask_columns = len([c for c in columns if c[1].startswith('ask_')])
        feature_columns = len([c for c in columns if c[1] in ['mid_price', 'spread', 'spread_bps', 'microprice', 'order_book_imbalance']])
        target_columns = len([c for c in columns if c[1].startswith('target_')])
        
        print(f"\n📊 SCHEMA BREAKDOWN:")
        print(f"  Bid levels: {bid_columns // 2} levels ({bid_columns} columns)")
        print(f"  Ask levels: {ask_columns // 2} levels ({ask_columns} columns)")
        print(f"  Features: {feature_columns} microstructure features")
        print(f"  Targets: {target_columns} ML target variables")
        print(f"  Metadata: {len(columns) - bid_columns - ask_columns - feature_columns - target_columns} system columns")
        
        return True, len(columns)
        
    except Exception as e:
        print(f"❌ ERROR creating fresh database: {e}")
        return False, 0

if __name__ == "__main__":
    success, column_count = create_fresh_database()
    if success:
        print(f"\n🏁 Final result: Unified live database created with {column_count} columns")
    else:
        print(f"\n❌ Failed to create unified live database")