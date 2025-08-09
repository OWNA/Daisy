#!/usr/bin/env python3

import sqlite3

def check_database_schema():
    conn = sqlite3.connect('trading_bot.db')
    cursor = conn.cursor()
    
    # Check L2 data schema
    cursor.execute('PRAGMA table_info(l2_training_data_practical)')
    print('L2 Training Data Practical Schema:')
    for row in cursor.fetchall():
        print(f'  {row[1]} {row[2]}')
    
    # Check record count
    cursor.execute('SELECT COUNT(*) FROM l2_training_data_practical')
    print(f'Records: {cursor.fetchone()[0]}')
    
    # Check execution analytics schema
    cursor.execute('PRAGMA table_info(execution_analytics)')
    print('\nExecution Analytics Schema:')
    for row in cursor.fetchall():
        print(f'  {row[1]} {row[2]}')
    
    # Check feature metadata schema  
    cursor.execute('PRAGMA table_info(feature_metadata)')
    print('\nFeature Metadata Schema:')
    for row in cursor.fetchall():
        print(f'  {row[1]} {row[2]}')
    
    conn.close()

if __name__ == "__main__":
    check_database_schema()