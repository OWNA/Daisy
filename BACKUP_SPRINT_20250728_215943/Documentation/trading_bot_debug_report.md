# Trading Bot CLI System Debug Report

## 1. Core Scripts Status

### Data Collection Components
✅ **run_l2_collector.py** - EXISTS and appears functional
- Runner script for L2 data collection
- Uses config.yaml for settings
- Supports continuous collection mode
- Dependencies: l2_data_collector.py, yaml, signal, threading

✅ **l2_data_collector.py** - EXISTS and appears functional  
- WebSocket-based L2 order book collector
- Collects from Bybit exchange
- Stores data in compressed gzip files
- Dependencies: ccxt, pandas, websocket-client, gzip

### Model Training Components
✅ **modeltrainer.py** - EXISTS and appears functional
- LightGBM model training for L2-only strategy
- Supports ensemble models
- L2-only mode enforced
- Dependencies: lightgbm, pandas, numpy, optuna (optional)

⚠️ **run_l2_training_pipeline.py** - EXISTS but has issues
- References non-existent scripts:
  - `download_sample_data.py` (missing)
  - `align_l2_data.py` (exists)
  - `run_training_simple.py` (missing)
  - `analyze_predictions.py` (exists)

### Backtesting Components
✅ **run_backtest.py** - EXISTS and appears functional
- L2-only backtest simulation
- Loads data from SQLite database
- Validates L2-only mode
- Dependencies: strategybacktester.py, modelpredictor.py, featureengineer.py

✅ **strategybacktester.py** - EXISTS and appears functional
- Strategy backtesting with L2 data adaptation
- Maps L2 data to OHLCV format for compatibility
- Supports L2 volatility metrics
- Dependencies: pandas, risk_manager

### Live Simulation Components
✅ **start_live_simulation.py** - EXISTS and appears functional
- Entry point for L2-only live simulation
- Validates prerequisites
- Updates config for L2-only mode
- Dependencies: livesimulator.py

⚠️ **livesimulator.py** - EXISTS but has missing dependencies
- L2-only live simulation system
- Missing dependency: `l2_price_reconstructor.py`
- Dependencies: l2_price_reconstructor (missing), l2_volatility_estimator

✅ **run_trading_bot.py** - EXISTS and appears functional
- Main trading bot runner
- Comprehensive L2-only mode support
- Handles API keys and module imports
- Dependencies: tradingbotorchestrator.py

## 2. Data Flow Analysis

### L2 Data Collection Flow
1. **run_l2_collector.py** → Starts collection process
2. **l2_data_collector.py** → WebSocket connection to exchange
3. Data saved to compressed files in `l2_data/` folder
4. Format: gzipped JSON with L2 snapshots

### Feature Engineering Flow
1. **featureengineer.py** → L2-only feature calculation
2. Supports L2 microstructure features:
   - Bid/ask spreads
   - Order book imbalances
   - Price impacts
   - L2 volatility metrics
3. HHT features if PyEMD available

### Model Training Flow
1. Data loaded from database or files
2. **featureengineer.py** → Calculate L2 features
3. **labelgenerator.py** → Generate labels
4. **modeltrainer.py** → Train LightGBM model
5. Model saved as `.txt` file
6. Features saved as `.json` file

### Prediction Flow
1. Live L2 data collected
2. Features calculated in real-time
3. **modelpredictor.py** → Generate predictions
4. Risk management applied
5. Orders executed (paper trading)

## 3. File Path Issues

### Missing Files
❌ **l2_price_reconstructor.py** - Required by livesimulator.py
❌ **download_sample_data.py** - Referenced in run_l2_training_pipeline.py
❌ **run_training_simple.py** - Referenced in run_l2_training_pipeline.py

### Existing L2 Support Files
✅ **l2_data_etl.py**
✅ **l2_microstructure_features.py**
✅ **l2_volatility_estimator.py**
✅ **align_l2_data.py**

## 4. Import Issues

### Common Missing Modules (require pip install)
- ccxt
- websocket-client
- lightgbm
- optuna
- shap
- PyEMD
- pandas_ta

### Internal Import Issues
- livesimulator.py imports missing l2_price_reconstructor
- Some scripts expect specific directory structure

## 5. Configuration Issues

### Config Files
✅ **config.yaml** - Main config with L2 settings
✅ **config_l2_only.yaml** - L2-specific configuration

### Key L2 Configuration
- `l2_only_mode: true` - Enables L2-only operation
- `use_l2_features: true` - Uses L2 features
- `l2_websocket_depth: 50` - Order book depth
- `l2_collection_duration_seconds: 120` - Collection duration

## 6. Functional vs Broken Scripts

### ✅ Functional Scripts
1. **run_l2_collector.py** - Can collect L2 data
2. **l2_data_collector.py** - Core collection logic
3. **modeltrainer.py** - Can train models
4. **strategybacktester.py** - Can run backtests
5. **run_backtest.py** - Backtest runner
6. **start_live_simulation.py** - Simulation starter
7. **run_trading_bot.py** - Main bot runner
8. **featureengineer.py** - Feature calculation

### ⚠️ Partially Functional
1. **run_l2_training_pipeline.py** - Missing some referenced scripts
2. **livesimulator.py** - Missing l2_price_reconstructor dependency

### ❌ Missing Critical Files
1. **l2_price_reconstructor.py**
2. **download_sample_data.py**
3. **run_training_simple.py**

## 7. Recommended Fixes

### Immediate Actions
1. Create missing `l2_price_reconstructor.py` or remove its usage
2. Update `run_l2_training_pipeline.py` to use existing scripts
3. Install missing Python packages: `pip install -r requirements.txt`

### Data Flow Fixes
1. Ensure L2 data collector saves to expected database format
2. Verify feature engineering aligns with model expectations
3. Check that backtesting can load L2 data properly

### Path Fixes
1. Use absolute paths or proper relative paths
2. Ensure `BOT_BASE_DIR` environment variable is set
3. Create required directories before use

## 8. Working Workflow Example

### Collect L2 Data
```bash
python3 run_l2_collector.py --config config.yaml
```

### Train Model (needs fixing)
```bash
# Currently broken due to missing scripts
# Needs manual training using modeltrainer.py directly
```

### Run Backtest
```bash
python3 run_backtest.py --config config.yaml --l2-only
```

### Start Live Simulation
```bash
python3 start_live_simulation.py --config config.yaml
# or
python3 run_trading_bot.py --config config.yaml --workflow l2_live_trading
```

## Summary

The trading bot system is mostly functional but has some missing dependencies and broken references. The core components for L2 data collection, model training, backtesting, and live simulation exist but need some fixes to work together properly. The main issues are:

1. Missing `l2_price_reconstructor.py` file
2. Training pipeline references non-existent scripts
3. Some Python packages need to be installed
4. Path and import issues need resolution

With these fixes, the L2-only trading bot should be fully operational.