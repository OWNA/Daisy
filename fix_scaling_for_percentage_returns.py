#!/usr/bin/env python3
"""
Fix scaling parameters for percentage returns
The issue: Label generator uses pct_change() but then divides by volatility,
creating tiny values that get incorrectly scaled.
"""

import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def analyze_and_fix_scaling():
    """Analyze the scaling issue and provide fixes"""
    
    # Check current scaling parameters
    scaling_file = "./trading_bot_data/lgbm_model_BTC_USDTUSDT_l2_only_scaling.json"
    
    if os.path.exists(scaling_file):
        with open(scaling_file, 'r') as f:
            current_scaling = json.load(f)
        
        logger.info("Current scaling parameters:")
        logger.info(f"  Mean: {current_scaling['target_mean']}")
        logger.info(f"  Std: {current_scaling['target_std']}")
        
        # The issue: these values suggest absolute price differences
        # Mean of -3.998 and std of 786.9 are typical for $100k BTC price movements
        # But we're using percentage returns which should be much smaller
        
        logger.info("\nDiagnosis:")
        logger.info("- Mean of -3.998 suggests absolute price differences")
        logger.info("- Std of 786.9 is way too large for percentage returns")
        logger.info("- This causes all predictions to be scaled to near-zero")
        
        # Fix 1: Force proper scaling for percentage returns
        logger.info("\nFix 1: Create proper scaling for percentage returns")
        
        # Typical percentage returns for BTC
        pct_return_mean = 0.0  # Should be near zero
        pct_return_std = 0.001  # 0.1% typical volatility for short timeframes
        
        fixed_scaling = {
            "target_mean": pct_return_mean,
            "target_std": pct_return_std,
            "features": current_scaling.get("features", []),
            "scaling_type": "percentage_returns",
            "timestamp": datetime.now().isoformat()
        }
        
        # Save fixed scaling
        fixed_file = scaling_file.replace(".json", "_fixed.json")
        with open(fixed_file, 'w') as f:
            json.dump(fixed_scaling, f, indent=2)
        
        logger.info(f"Fixed scaling saved to: {fixed_file}")
        logger.info(f"  Mean: {pct_return_mean}")
        logger.info(f"  Std: {pct_return_std}")
        
        # Fix 2: Update labelgenerator to use pure percentage returns
        logger.info("\nFix 2: Create simplified label generator")
        
        create_percentage_label_generator()
        
        return fixed_file
    else:
        logger.error(f"Scaling file not found: {scaling_file}")
        return None

def create_percentage_label_generator():
    """Create a simplified label generator for percentage returns"""
    
    code = '''# percentage_label_generator.py
# Simplified label generator that uses pure percentage returns

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

class PercentageLabelGenerator:
    """Generate percentage return labels for trading"""
    
    def __init__(self, config):
        self.config = config
        self.future_window = config.get('label_shift', -1)
        self.clip_quantiles = config.get('label_clip_quantiles', (0.001, 0.999))
        
    def generate_labels(self, df_features):
        """Generate percentage return labels"""
        
        # Get price series (prefer microprice for accuracy)
        price_cols = ['microprice', 'weighted_mid_price', 'mid_price']
        price_series = None
        
        for col in price_cols:
            if col in df_features.columns and not df_features[col].isna().all():
                price_series = df_features[col]
                logger.info(f"Using {col} for labels")
                break
        
        if price_series is None:
            raise ValueError("No price column found")
        
        df = df_features.copy()
        
        # Calculate simple percentage returns
        # No volatility normalization - just pure percentage changes
        df['target'] = price_series.pct_change(periods=abs(self.future_window))
        
        if self.future_window < 0:
            df['target'] = df['target'].shift(self.future_window)
        
        # Calculate statistics before clipping
        valid_targets = df['target'].dropna()
        if len(valid_targets) > 10:
            target_mean = valid_targets.mean()
            target_std = valid_targets.std()
            
            # Clip outliers
            if self.clip_quantiles:
                lower = valid_targets.quantile(self.clip_quantiles[0])
                upper = valid_targets.quantile(self.clip_quantiles[1])
                df['target'] = df['target'].clip(lower, upper)
        else:
            target_mean = 0.0
            target_std = 0.001  # Default 0.1% volatility
        
        # Remove NaN targets
        df = df.dropna(subset=['target'])
        
        logger.info(f"Generated {len(df)} labels")
        logger.info(f"Target mean: {target_mean:.6f} ({target_mean*100:.4f}%)")
        logger.info(f"Target std: {target_std:.6f} ({target_std*100:.4f}%)")
        
        return df, target_mean, target_std
'''
    
    with open('percentage_label_generator.py', 'w') as f:
        f.write(code)
    
    logger.info("Created percentage_label_generator.py")

def test_with_sample_data():
    """Test the fix with sample data"""
    
    logger.info("\nTesting with sample BTC price data...")
    
    # Simulate BTC prices around $118k
    prices = 118000 + np.random.randn(1000) * 100
    
    # Calculate percentage returns
    returns = pd.Series(prices).pct_change()
    
    logger.info(f"Sample price range: ${prices.min():.0f} - ${prices.max():.0f}")
    logger.info(f"Return statistics:")
    logger.info(f"  Mean: {returns.mean():.6f} ({returns.mean()*100:.4f}%)")
    logger.info(f"  Std: {returns.std():.6f} ({returns.std()*100:.4f}%)")
    
    # Show what happens with wrong scaling
    wrong_mean = -3.998
    wrong_std = 786.9
    
    scaled_wrong = (returns - wrong_mean) / wrong_std
    logger.info(f"\nWith wrong scaling (mean={wrong_mean}, std={wrong_std}):")
    logger.info(f"  Scaled values range: {scaled_wrong.min():.6f} to {scaled_wrong.max():.6f}")
    logger.info(f"  All values near: {scaled_wrong.mean():.6f}")
    
    # Show correct scaling
    correct_mean = 0.0
    correct_std = 0.001
    
    scaled_correct = (returns - correct_mean) / correct_std
    logger.info(f"\nWith correct scaling (mean={correct_mean}, std={correct_std}):")
    logger.info(f"  Scaled values range: {scaled_correct.min():.2f} to {scaled_correct.max():.2f}")
    logger.info(f"  Properly distributed around: {scaled_correct.mean():.2f}")

if __name__ == "__main__":
    logger.info("Fixing scaling parameters for percentage returns")
    logger.info("=" * 60)
    
    # Analyze and fix
    fixed_file = analyze_and_fix_scaling()
    
    # Test the fix
    test_with_sample_data()
    
    if fixed_file:
        logger.info("\n" + "="*60)
        logger.info("NEXT STEPS:")
        logger.info("1. Replace the scaling file:")
        logger.info(f"   copy {fixed_file} {fixed_file.replace('_fixed.json', '.json')}")
        logger.info("2. Or update your label generator to use percentage_label_generator.py")
        logger.info("3. Retrain the model with proper scaling")