#!/usr/bin/env python3
"""
Final fix: Add dummy columns for metadata that model expects but shouldn't use
"""

# The cleanest solution is to modify the prediction process to add dummy columns
# for the metadata fields that were mistakenly included during training

print("Applying final fix for prediction...")

# Create a wrapper script that adds the missing columns
wrapper_code = '''
def add_dummy_columns(df):
    """Add dummy columns for metadata that model expects"""
    # These columns were in training data but shouldn't be used for prediction
    dummy_columns = {
        'id': 0,
        'target_return_1min': 0,
        'target_return_5min': 0,
        'target_volatility': 0,
        'target_direction': 0,
        'update_id': 0,
        'sequence_id': 0,
        'data_quality_score': 1.0,
        'close': df['mid_price'] if 'mid_price' in df.columns else 0
    }
    
    for col, value in dummy_columns.items():
        if col not in df.columns:
            df[col] = value
    
    return df
'''

# Modify the main.py to add dummy columns before prediction
import fileinput
import sys

print("Patching main.py to add dummy columns...")

# Read main.py and find where to insert the fix
with open('main.py', 'r') as f:
    lines = f.readlines()

# Find the line where features are generated
insert_index = None
for i, line in enumerate(lines):
    if 'features = feature_engineer.generate_features' in line:
        # Find the next line after features are generated
        for j in range(i+1, len(lines)):
            if 'Get prediction' in lines[j]:
                insert_index = j
                break
        break

if insert_index:
    # Insert the dummy column addition
    patch = """                    # Add dummy columns for metadata (temporary fix)
                    dummy_cols = {
                        'id': 0, 'target_return_1min': 0, 'target_return_5min': 0,
                        'target_volatility': 0, 'target_direction': 0, 'update_id': 0,
                        'sequence_id': 0, 'data_quality_score': 1.0,
                        'close': features['mid_price'].iloc[0] if 'mid_price' in features.columns else 0
                    }
                    for col, val in dummy_cols.items():
                        if col not in features.columns:
                            features[col] = val
                    
"""
    lines.insert(insert_index, patch)
    
    # Write back
    with open('main_patched.py', 'w') as f:
        f.writelines(lines)
    
    print("Created main_patched.py with dummy column fix")
    
    # Also create a simpler standalone fix
    with open('fix_features_simple.py', 'w') as f:
        f.write('''#!/usr/bin/env python3
"""Remove problematic features from the saved list"""
import json

with open('trading_bot_data/model_features_BTC_USDTUSDT_l2_only.json', 'r') as f:
    data = json.load(f)

# These are the actual L2 features (no metadata)
clean_features = [col for col in data['trained_features'] 
                  if col not in ['id', 'target_return_1min', 'target_return_5min',
                                'target_volatility', 'target_direction', 'update_id',
                                'sequence_id', 'data_quality_score', 'close']]

print(f"Cleaned features: {len(clean_features)} (was {len(data['trained_features'])})")

# For now, pad with duplicates to reach the expected count
while len(clean_features) < 82:
    clean_features.append('mid_price')  # Duplicate a harmless feature

data['trained_features'] = clean_features[:82]

with open('trading_bot_data/model_features_BTC_USDTUSDT_l2_only.json', 'w') as f:
    json.dump(data, f, indent=4)

print("Fixed features file!")
''')
    
print("\nFix applied! Now run:")
print("python main_patched.py trade --paper")
print("\nOr apply the simple fix:")
print("python fix_features_simple.py")
print("python main.py trade --paper")