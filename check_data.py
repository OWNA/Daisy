import sqlite3
conn = sqlite3.connect('./trading_bot_live.db')
cursor = conn.cursor()
cursor.execute('SELECT timestamp, symbol, mid_price FROM l2_training_data_practical WHERE data_source = "mock_data" ORDER BY timestamp DESC LIMIT 5')
rows = cursor.fetchall()
print('Recent mock data rows:')
for row in rows:
    print(f'  {row[0]} | {row[1]} | ${row[2]:.2f}')
conn.close()