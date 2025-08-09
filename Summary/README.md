# Summary Documentation

This folder contains all documentation related to the Phase 2 restructuring and fixes.

## Files

1. **PHASE2_ALL_FIXES.md** - Complete consolidated summary of all fixes applied
2. **PHASE2_COMPLETE.md** - Initial Phase 2 restructuring details
3. **PHASE2_FIX.md** - First round of error fixes
4. **PHASE2_FIXES_COMPLETE.md** - Second round of error fixes
5. **TRAINING_FIXES.md** - Training-specific error fixes
6. **FILENAME_FIX.md** - Model filename mismatch resolution

## Quick Reference

### To Train Model
```powershell
.\venv\Scripts\python.exe main.py train
```

### To Run Paper Trading
```powershell
.\venv\Scripts\python.exe main.py trade --paper
```

### To Check Model Files
```powershell
.\venv\Scripts\python.exe check_model_files.py
```

## Current Model Filename Pattern
For symbol `BTC/USDT:USDT`:
- Model: `lgbm_model_BTC_USDTUSDT_l2_only.txt`
- Scaling: `lgbm_model_BTC_USDTUSDT_l2_only_scaling.json`
- Features: `model_features_BTC_USDTUSDT_l2_only.json`