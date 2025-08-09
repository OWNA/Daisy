# HHT Model Training Package for Cursor

Streamlined training package for L2+HHT enhanced trading models.

## Setup

1. **Copy your database file:**
   ```bash
   # Copy trading_bot.db from your local machine to this directory
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run training:**
   ```bash
   python train_hht_model.py --trials 20 --samples 50000
   ```

## Performance Expectations

**On Cursor's better CPU:**
- HHT calculation: 50-100ms per chunk (vs 1000ms locally)
- Total training time: 1-2 hours (vs 14+ hours locally)
- Memory usage: ~2-4GB for 50k samples

## Output Files

The script generates:
- `models/hht_l2_model.txt` - Trained LightGBM model
- `models/model_features.json` - Feature list (143 features)
- `models/scaling_params.json` - Target scaling parameters
- `models/best_params.json` - Optimal hyperparameters

## Usage Options

```bash
# Quick training (smaller dataset)
python train_hht_model.py --trials 10 --samples 20000

# Full training (production model)
python train_hht_model.py --trials 50 --samples 50000

# Custom database path
python train_hht_model.py --db_path /path/to/trading_bot.db
```

## Features Generated

**L2 Microstructure (122 features):**
- Order book imbalance (multiple levels)
- Order flow imbalance (10s, 30s, 1m, 5m windows)
- Spread dynamics and volatility
- Volume pressure metrics

**HHT Features (21 features):**
- Trend strength and slope
- Cycle phase and frequency
- Regime classification (trending/cyclical/noisy)
- Energy distribution across frequency bands
- Instantaneous amplitude

**Total: 143 features for enhanced signal generation**

## Transfer Back to Local

After training completes, copy these files back to your local `trading_bot_data/` folder:
- `hht_l2_model.txt` → `lgbm_model_BTC_USDTUSDT_l2_only.txt`
- `scaling_params.json` → `lgbm_model_BTC_USDTUSDT_l2_only_scaling.json`
- `model_features.json` → `model_features_BTC_USDTUSDT_l2_only.json`