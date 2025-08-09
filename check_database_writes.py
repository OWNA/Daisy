import sqlite3
import pandas as pd
import os

db_file = './trading_bot_live.db'
table_name = 'l2_training_data_practical'

print(f"--- Checking Database Writes in: {db_file} ---")

if not os.path.exists(db_file):
    print(f"ERROR: Database file not found at '{db_file}'")
    exit()

try:
    conn = sqlite3.connect(db_file)
    
    # Check total row count
    count = pd.read_sql_query(f"SELECT COUNT(*) FROM {table_name}", conn).iloc[0, 0]
    print(f"Total rows in '{table_name}': {count}")
    
    if count > 0:
        # Check most recent entry
        df_recent = pd.read_sql_query(f"SELECT * FROM {table_name} ORDER BY timestamp DESC LIMIT 5", conn)
        print("\n--- 5 Most Recent Rows ---")
        # Display a subset of useful columns
        display_columns = ['timestamp', 'symbol', 'mid_price', 'spread', 'data_source', 'data_quality_score']
        # Filter to columns that actually exist in the dataframe to avoid errors
        existing_display_columns = [col for col in display_columns if col in df_recent.columns]
        print(df_recent[existing_display_columns].to_string())
        
        last_timestamp_str = df_recent['timestamp'].iloc[0]
        print(f"\nLatest data timestamp: {last_timestamp_str}")
    else:
        print("\nDatabase is empty. The data ingestor may need more time to run.")
        
except Exception as e:
    print(f"\nAn error occurred while checking the database: {e}")

finally:
    if 'conn' in locals():
        conn.close()
