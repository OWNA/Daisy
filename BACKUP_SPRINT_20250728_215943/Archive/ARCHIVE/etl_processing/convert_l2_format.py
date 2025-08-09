#!/usr/bin/env python3
"""
Convert L2 data between different formats
"""

import json
import gzip
from pathlib import Path
from datetime import datetime

def convert_l2_file(input_file, output_file=None):
    """Convert L2 data to standard format"""
    
    if not output_file:
        # Create output filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = f"l2_data/l2_data_converted_{timestamp}.jsonl.gz"
    
    print(f"Converting {input_file} to standard format...")
    
    converted_count = 0
    skipped_count = 0
    
    with gzip.open(input_file, 'rt') as infile, gzip.open(output_file, 'wt') as outfile:
        for line in infile:
            try:
                record = json.loads(line)
                
                # Convert to standard format
                converted = {
                    'timestamp': record.get('timestamp') or datetime.fromtimestamp(record.get('timestamp_ms', 0) / 1000).isoformat(),
                    'exchange': record.get('exchange', 'bybit'),
                    'symbol': record.get('symbol', 'BTCUSDT'),
                }
                
                # Handle bids/asks format
                if 'bids' in record:
                    converted['b'] = record['bids']
                elif 'b' in record:
                    converted['b'] = record['b']
                else:
                    skipped_count += 1
                    continue
                
                if 'asks' in record:
                    converted['a'] = record['asks']
                elif 'a' in record:
                    converted['a'] = record['a']
                else:
                    skipped_count += 1
                    continue
                
                # Write converted record
                json.dump(converted, outfile)
                outfile.write('\n')
                converted_count += 1
                
                if converted_count % 1000 == 0:
                    print(f"  Converted {converted_count} records...")
                    
            except Exception as e:
                print(f"Error converting record: {e}")
                skipped_count += 1
    
    print(f"\nConversion complete!")
    print(f"  Converted: {converted_count} records")
    print(f"  Skipped: {skipped_count} records")
    print(f"  Output: {output_file}")
    
    return output_file

def check_l2_format(file_path):
    """Check the format of an L2 data file"""
    print(f"\nChecking format of {file_path}...")
    
    with gzip.open(file_path, 'rt') as f:
        # Read first record
        first_line = f.readline()
        record = json.loads(first_line)
        
        print("\nFirst record structure:")
        print(f"  Keys: {list(record.keys())}")
        
        if 'bids' in record or 'b' in record:
            bids_key = 'bids' if 'bids' in record else 'b'
            bids = record[bids_key]
            print(f"  Bids key: '{bids_key}'")
            print(f"  Number of bid levels: {len(bids)}")
            if bids:
                print(f"  First bid: {bids[0]}")
        
        if 'asks' in record or 'a' in record:
            asks_key = 'asks' if 'asks' in record else 'a'
            asks = record[asks_key]
            print(f"  Asks key: '{asks_key}'")
            print(f"  Number of ask levels: {len(asks)}")
            if asks:
                print(f"  First ask: {asks[0]}")
        
        # Count total records
        f.seek(0)
        total_records = sum(1 for _ in f)
        print(f"\nTotal records: {total_records}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--check', help='Check format of L2 file')
    parser.add_argument('--convert', help='Convert L2 file to standard format')
    parser.add_argument('--output', help='Output file for conversion')
    
    args = parser.parse_args()
    
    if args.check:
        check_l2_format(args.check)
    elif args.convert:
        output_file = convert_l2_file(args.convert, args.output)
        print(f"\nYou can now train with: python train_model_robust.py --data {output_file}")
    else:
        # Check the specific file
        target_file = "l2_data/l2_data_BTC_USDT_USDT_20250707_040413.jsonl.gz"
        if Path(target_file).exists():
            check_l2_format(target_file)
            
            print("\nTo convert this file, run:")
            print(f"python convert_l2_format.py --convert {target_file}")
        else:
            print(f"File not found: {target_file}")