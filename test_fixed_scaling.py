#!/usr/bin/env python3
"""Test the fixed scaling parameters"""

# Test values
raw_prediction = 0.000002  # Typical raw model output

# Old scaling (wrong)
old_mean = -3.998828611819161
old_std = 786.9006834054989

# New scaling (fixed)
new_mean = 0.0
new_std = 0.001

# Calculate scaled values
old_scaled = (raw_prediction - old_mean) / old_std
new_scaled = (raw_prediction - new_mean) / new_std

# Calculate unscaled values
old_unscaled = (old_scaled * old_std) + old_mean
new_unscaled = (new_scaled * new_std) + new_mean

print("Scaling Test Results")
print("=" * 50)
print(f"Raw prediction: {raw_prediction:.6f}")
print()
print("OLD SCALING (wrong):")
print(f"  Mean: {old_mean}, Std: {old_std}")
print(f"  Scaled: {old_scaled:.6f}")
print(f"  Unscaled: {old_unscaled:.6f}")
print()
print("NEW SCALING (fixed):")
print(f"  Mean: {new_mean}, Std: {new_std}")
print(f"  Scaled: {new_scaled:.6f}")
print(f"  Unscaled: {new_unscaled:.6f}")
print()
print("Interpretation:")
print(f"  Old: {old_unscaled:.2f} (absolute $ change - wrong!)")
print(f"  New: {new_unscaled*100:.4f}% (percentage change - correct!)")