# Training Error Fixes Applied

## Fixed Issues

### 1. ModelTrainer Parameter Error
**Problem:** `train_model() got an unexpected keyword argument 'target_mean_for_scaling'`

**Fix Applied:**
- Removed `target_mean_for_scaling` and `target_std_for_scaling` parameters from the call
- Changed from `results = model_trainer.train_model(df_labeled, target_mean_for_scaling=...)`
- To: `booster, features = model_trainer.train_model(df_labeled)`
- The ModelTrainer doesn't need these parameters - it trains on already-scaled data

### 2. Windows Filename Error  
**Problem:** Model filename contained colon (`:`) which is invalid on Windows

**Fix Applied:**
- Added `.replace(':', '_')` to sanitize the symbol
- Changed from: `BTC/USDT:USDT` → `BTC_USDT_USDT`
- Applied in both train and trade functions

## Result

The model file will now be saved as:
- `lgbm_model_BTC_USDT_USDT.txt` (instead of `lgbm_model_BTC_USDT:USDT.txt`)
- `lgbm_model_BTC_USDT_USDT_scaling.json`

## To Run

```powershell
# Train the model
.\venv\Scripts\python.exe main.py train

# Once training completes, run paper trading
.\venv\Scripts\python.exe main.py trade --paper
```

The training should now complete successfully!