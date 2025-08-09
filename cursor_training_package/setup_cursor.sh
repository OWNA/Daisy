#!/bin/bash
# Setup script for Cursor environment

echo "Setting up HHT Training Environment on Cursor..."

# Install Python dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Check if database file exists
if [ ! -f "trading_bot.db" ]; then
    echo "ERROR: trading_bot.db not found!"
    echo "Please copy your trading_bot.db file to this directory"
    exit 1
fi

# Create models directory
mkdir -p models

# Check database content
echo "Checking database content..."
python -c "
import sqlite3
conn = sqlite3.connect('trading_bot.db')
cursor = conn.cursor()
cursor.execute('SELECT COUNT(*) FROM l2_training_data_practical')
count = cursor.fetchone()[0]
print(f'L2 training records available: {count:,}')
conn.close()
"

echo "Setup complete! Ready to train."
echo ""
echo "Run training with:"
echo "  python train_hht_model.py --trials 20"
echo ""
echo "For quick test:"
echo "  python train_hht_model.py --trials 5 --samples 10000"