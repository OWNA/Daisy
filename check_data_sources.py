import sqlite3
conn = sqlite3.connect('./trading_bot_live.db')
cursor = conn.cursor()

print("Data sources in database:")
cursor.execute('SELECT COUNT(*), data_source FROM l2_training_data_practical GROUP BY data_source')
for row in cursor.fetchall():
    print(f'  {row[1]}: {row[0]} rows')

print("\nMost recent data:")
cursor.execute('SELECT timestamp, data_source FROM l2_training_data_practical ORDER BY timestamp DESC LIMIT 5')
for row in cursor.fetchall():
    print(f'  {row[0]} ({row[1]})')

conn.close()