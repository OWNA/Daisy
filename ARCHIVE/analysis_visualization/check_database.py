#!/usr/bin/env python3
"""
Check database tables and L2 data
"""

import sqlite3
import os

def check_database():
    """Check what's in the database."""
    db_path = "trading_bot.db"
    
    if not os.path.exists(db_path):
        print(f"Database file {db_path} does not exist")
        return
    
    print(f"Database file size: {os.path.getsize(db_path) / 1024 / 1024:.2f} MB")
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Get all tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = cursor.fetchall()
    print(f"Tables: {[t[0] for t in tables]}")
    
    # Check each table
    for table in tables:
        table_name = table[0]
        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
        count = cursor.fetchone()[0]
        print(f"Table {table_name}: {count} records")
        
        if table_name == 'l2_training_data_practical':
            # Get sample data
            cursor.execute(f"SELECT * FROM {table_name} LIMIT 1")
            columns = [description[0] for description in cursor.description]
            print(f"L2 table columns: {columns}")
            
            # Check date range
            cursor.execute(f"SELECT MIN(timestamp), MAX(timestamp) FROM {table_name}")
            date_range = cursor.fetchone()
            print(f"L2 date range: {date_range}")
    
    conn.close()

if __name__ == "__main__":
    check_database()
