# Advanced Trading Bot CLI Guide

## Overview
The advanced CLI provides full flexibility for managing L2 data, training models, running backtests, and visualizing results.

## Quick Start

First, clean up the project (one-time):
```powershell
python cleanup_project.py
```

Then use the CLI:
```powershell
# Using PowerShell
./trade <command>

# Or directly
python trade_cli_advanced.py <command>
```

## Commands

### 1. Data Collection & Management

#### Collect L2 Data
```powershell
# Collect for 5 minutes
./trade data collect -d 300

# Collect for 30 minutes and append to existing
./trade data collect -d 1800 --append

# Collect different symbol
./trade data collect -s ETH/USDT:USDT -d 600
```

#### List Data Files
```powershell
# Show last 5 files
./trade data list

# Show last 10 files
./trade data list -l 10
```

#### Inspect Data
```powershell
# Inspect specific file
./trade data inspect l2_data_BTC_USDT_USDT_20250707_173241.jsonl.gz

# Show 200 rows
./trade data inspect <filename> -r 200
```

### 2. Model Training

#### Train with Full Control
```powershell
# Train with all features
./trade model train

# Train with L2 features only
./trade model train -f l2

# Train with HHT features only
./trade model train -f hht

# Train with 100 Optuna trials
./trade model train -o 100

# Train with specific data file
./trade model train -d l2_data_BTC_USDT_USDT_20250707_173241.jsonl.gz

# Remove specific features before training
./trade model train -r bid_price_2 -r ask_price_2 -r hht_amp_imf2
```

#### List Models
```powershell
./trade model list
```

#### Inspect Model
```powershell
# Show top 20 features
./trade model inspect lgbm_model_BTC_USDTUSDT_optuna_l2_hht.txt

# Show top 50 features
./trade model inspect <model_file> -t 50
```

### 3. Backtesting

#### Run Backtest
```powershell
# Basic backtest
./trade backtest run

# With specific model
./trade backtest run -m lgbm_model_BTC_USDTUSDT_optuna_l2_hht.txt

# With date range
./trade backtest run -s 2025-01-01 -e 2025-01-07

# Limit data rows (faster)
./trade backtest run -r 5000
```

#### Visualize Results
```powershell
# Interactive Plotly chart
./trade backtest visualize backtest_results/results_20250707.csv

# Static Seaborn chart
./trade backtest visualize backtest_results/results_20250707.csv -p seaborn
```

### 4. Live Simulation

#### Run Simulation
```powershell
# 5-minute simulation
./trade simulate run -d 300

# With specific model
./trade simulate run -m lgbm_model_BTC_USDTUSDT_optuna_l2_hht.txt

# Paper trading mode
./trade simulate run --paper -d 600
```

### 5. Analysis

#### SHAP Analysis
```powershell
# Generate SHAP plots for model interpretation
./trade analyze shap -m lgbm_model_BTC_USDTUSDT_optuna_l2_hht.txt

# With specific data
./trade analyze shap -m <model> -d prepared_data.csv
```

#### Performance Analysis
```powershell
# Analyze all backtest results
./trade analyze performance

# From specific directory
./trade analyze performance -r paper_trading_results
```

### 6. Configuration

#### View Config
```powershell
./trade config show
```

#### Update Config
```powershell
# Set simple value
./trade config set symbol "ETH/USDT:USDT"
./trade config set initial_balance 50000

# Set nested value
./trade config set risk_management.max_drawdown 0.2
./trade config set l2_features.use_hht_features true
```

### 7. Workflows

#### Full Automated Workflow
```powershell
# Default: 5min collect, 50 Optuna trials, 10k backtest rows, 5min simulate
./trade workflow full

# Custom parameters
./trade workflow full -c 1800 -o 100 -b 20000 -s 600
```

This runs:
1. Collect L2 data (30 min)
2. Train model with Optuna (100 trials)
3. Backtest (20k rows)
4. Live simulation (10 min)

## Output Files

### Models
- `lgbm_model_BTC_USDTUSDT_*.txt` - Trained models
- `model_features_BTC_USDTUSDT_*.json` - Feature lists

### Results
- `backtest_results/` - Backtest outputs
- `paper_trading_results/` - Simulation results
- `shap_plots/` - SHAP analysis

### Visualizations
- `backtest_visualization.html` - Interactive Plotly charts
- `backtest_visualization.png` - Static Seaborn charts
- `shap_summary.png` - SHAP summary plot
- `shap_importance.png` - SHAP feature importance

## Tips

1. **Start Fresh**: Run `python cleanup_project.py` to remove old files

2. **Quick Test**:
   ```powershell
   ./trade data collect -d 60
   ./trade model train -o 10
   ./trade backtest run -r 1000
   ```

3. **Production Workflow**:
   ```powershell
   ./trade workflow full -c 1800 -o 100
   ```

4. **Feature Selection**:
   - Use `-f l2` for L2 features only
   - Use `-f hht` for HHT features only
   - Use `-r feature_name` to remove specific features

5. **Visualization**:
   - Plotly creates interactive HTML files
   - Seaborn creates static PNG images

## Advanced Examples

### Custom Feature Training
```powershell
# Train without certain noisy features
./trade model train -r hht_freq_imf2 -r hht_amp_imf2 -r bid_price_10 -r ask_price_10
```

### Specific Data Analysis
```powershell
# Collect morning data
./trade data collect -d 3600

# List files to find the one you want
./trade data list

# Train on specific file
./trade model train -d l2_data_BTC_USDT_USDT_20250708_090000.jsonl.gz

# Backtest on same period
./trade backtest run
```

### Model Comparison
```powershell
# Train multiple models
./trade model train -f l2 -o 50  # L2 only
./trade model train -f all -o 50  # L2 + HHT

# Compare performance
./trade model list
./trade analyze performance
```