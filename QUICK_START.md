# 🚀 Quick Start Guide - Advanced Trading CLI

## First Time Setup (One-time only)

```powershell
# 1. Run the setup script
python quick_setup.py

# 2. If dependencies fail, install manually:
pip install -r requirements.txt
pip install rich click plotly seaborn
```

## Basic Usage

### 1. Collect Fresh L2 Data
```powershell
# Collect 5 minutes of data
./trade data collect -d 300

# Collect 30 minutes and append
./trade data collect -d 1800 --append
```

### 2. Train Model with Your Data
```powershell
# Quick training (10 Optuna trials)
./trade model train -o 10

# Full training (100 Optuna trials) 
./trade model train -o 100

# Train with L2 features only
./trade model train -f l2
```

### 3. Run Backtest
```powershell
# Quick backtest (1000 rows)
./trade backtest run -r 1000

# Full backtest
./trade backtest run

# Visualize results
./trade backtest visualize backtest_results/results_20250707.csv
```

### 4. Live Simulation
```powershell
# 5-minute live simulation
./trade simulate run -d 300

# Paper trading mode
./trade simulate run --paper -d 600
```

## 🎯 Complete Workflow (Recommended)

Run everything automatically:
```powershell
# Default: 5min collect, 50 trials, 10k backtest, 5min simulate
./trade workflow full

# Custom: 30min collect, 100 trials, 20k backtest, 10min simulate  
./trade workflow full -c 1800 -o 100 -b 20000 -s 600
```

## 📊 View Your Results

### List Data Files
```powershell
./trade data list
```

### List Trained Models
```powershell
./trade model list
```

### Inspect Model Features
```powershell
./trade model inspect lgbm_model_BTC_USDTUSDT_optuna_l2_hht.txt
```

### Analyze Performance
```powershell
./trade analyze performance
```

### Generate SHAP Analysis
```powershell
./trade analyze shap -m lgbm_model_BTC_USDTUSDT_optuna_l2_hht.txt
```

## 🔧 Configuration

### View Config
```powershell
./trade config show
```

### Update Settings
```powershell
./trade config set symbol "ETH/USDT:USDT"
./trade config set initial_balance 50000
```

## 💡 Tips

1. **Start Small**: Test with short durations first
   ```powershell
   ./trade data collect -d 60
   ./trade model train -o 10
   ./trade backtest run -r 1000
   ```

2. **Use the Best Model**: The system kept your best model from Optuna training:
   - Model: `lgbm_model_BTC_USDTUSDT_optuna_l2_hht.txt`
   - R² score: 0.4525
   - HHT features: 14.9% contribution

3. **Monitor Progress**: The CLI shows progress bars and status updates

4. **Check Logs**: Results are saved in:
   - `l2_data/` - Collected market data
   - `backtest_results/` - Backtest outputs
   - `paper_trading_results/` - Simulation results
   - `shap_plots/` - Model interpretation

## ❓ Need Help?

- Run `./trade --help` for all commands
- Run `./trade <command> --help` for specific command help
- Check `CLI_ADVANCED_GUIDE.md` for detailed documentation

## 🎉 You're Ready!

Start with collecting fresh data and training a model:
```powershell
./trade workflow full
```