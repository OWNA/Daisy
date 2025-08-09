#!/usr/bin/env python3
"""
Fix DataHandler to accept both 'b'/'a' and 'bids'/'asks' formats
"""

import fileinput
import sys

print("Fixing DataHandler to accept both L2 data formats...")

# Read the file
with open('datahandler.py', 'r') as f:
    content = f.read()

# Add format conversion at the beginning of process_l2_order_book_data
fix_code = '''
                # Handle both formats: 'b'/'a' and 'bids'/'asks'
                if 'b' in order_book and 'a' in order_book:
                    order_book['bids'] = order_book['b']
                    order_book['asks'] = order_book['a']
'''

# Find where to insert
if "def process_l2_order_book_data" in content:
    # Find the right place to insert
    lines = content.split('\n')
    new_lines = []
    in_function = False
    inserted = False
    
    for i, line in enumerate(lines):
        new_lines.append(line)
        
        if "def process_l2_order_book_data" in line:
            in_function = True
        
        if in_function and not inserted and "for order_book in records:" in line:
            # Insert after the for loop line
            indent = len(line) - len(line.lstrip())
            for fix_line in fix_code.strip().split('\n'):
                new_lines.append(' ' * (indent + 4) + fix_line.strip())
            inserted = True
    
    # Write back
    with open('datahandler.py', 'w') as f:
        f.write('\n'.join(new_lines))
    
    print("✓ Fixed DataHandler to handle both formats")
else:
    print("✗ Could not find process_l2_order_book_data function")

# Also fix the load_l2_historical_data function
with open('datahandler.py', 'r') as f:
    content = f.read()

# Replace record reading to handle both formats
content = content.replace(
    "if 'b' in record and 'a' in record:",
    "if ('b' in record and 'a' in record) or ('bids' in record and 'asks' in record):"
)

# Add conversion in the record reading loop
if "for line in f:" in content and "record = json.loads(line)" in content:
    lines = content.split('\n')
    new_lines = []
    
    for i, line in enumerate(lines):
        new_lines.append(line)
        if "record = json.loads(line)" in line and i + 1 < len(lines):
            indent = len(line) - len(line.lstrip())
            new_lines.append(indent * ' ' + "# Convert format if needed")
            new_lines.append(indent * ' ' + "if 'bids' in record and 'b' not in record:")
            new_lines.append(indent * ' ' + "    record['b'] = record['bids']")
            new_lines.append(indent * ' ' + "    record['a'] = record['asks']")
    
    with open('datahandler.py', 'w') as f:
        f.write('\n'.join(new_lines))

print("✓ DataHandler now accepts both 'b'/'a' and 'bids'/'asks' formats")
print("\nYou can now train with your original data files!")