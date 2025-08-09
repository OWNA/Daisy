# Configuration Files Guide

## 📋 Overview

This guide explains all configuration files in the trading bot project and their specific purposes.

## 🎯 Configuration Files Summary

### 1. **`config.yaml`** - Main Configuration ✅ **RECOMMENDED**
**Purpose**: Primary configuration with optimized L2 integration  
**Use Case**: Default configuration for training and backtesting with L2 features  
**L2 Status**: ✅ Enabled with optimized settings  

**Key Features:**
- Optimized L2 depth levels: `[2, 3]`
- 30-second alignment tolerance
- 37 features (24 technical + 13 L2)
- Realistic order book expectations

**Usage:**
```bash
python run_training_simple.py --config config.yaml
python run_backtest.py  # Uses config.yaml by default
```

### 2. **`config_l2.yaml`** - L2 Data Collection
**Purpose**: Specialized configuration for L2 data collection  
**Use Case**: Running the L2 data collector  
**L2 Status**: ✅ Collection settings optimized  

**Key Features:**
- WebSocket depth: 50 levels
- Collection duration: 6 hours
- Optimized feature list with realistic depths
- Quality filtering settings

**Usage:**
```bash
python run_l2_collector.py --config config_l2.yaml
```

### 3. **`config_no_l2.yaml`** - Baseline Configuration
**Purpose**: Baseline comparison without L2 features  
**Use Case**: Performance comparison and fallback option  
**L2 Status**: ❌ Disabled  

**Key Features:**
- Only OHLCV + technical indicators
- 24 features total
- Faster training and execution
- Baseline performance reference

**Usage:**
```bash
python run_training_simple.py --config config_no_l2.yaml
```

### 4. **`config_wfo.yaml`** - Walk-Forward Optimization
**Purpose**: Optimized for walk-forward optimization  
**Use Case**: WFO backtesting and strategy validation  
**L2 Status**: ❌ Disabled (for speed)  

**Key Features:**
- Reduced Optuna trials: 50 (vs 100)
- Smaller model complexity
- Faster hyperparameter search
- WFO-specific settings

**Usage:**
```bash
python run_wfo.py --config config_wfo.yaml
```

## 🔧 Configuration Comparison

| Feature | config.yaml | config_l2.yaml | config_no_l2.yaml | config_wfo.yaml |
|---------|-------------|----------------|-------------------|-----------------|
| **L2 Features** | ✅ Optimized | ✅ Collection | ❌ Disabled | ❌ Disabled |
| **Feature Count** | 37 | N/A | 24 | 24 |
| **Optuna Trials** | 100 | N/A | 100 | 50 |
| **Use Case** | Main training | Data collection | Baseline | WFO testing |
| **Performance** | Best | N/A | Fast | Fastest |

## 📊 L2 Configuration Details

### Optimized L2 Settings (config.yaml & config_l2.yaml)

```yaml
# Realistic depth levels based on analysis of 1M+ snapshots
l2_depth_imbalance_levels: [2, 3]  # Only 53-56% of snapshots have 2-4 levels
l2_price_impact_depth_idx: 1       # 2nd level (more available than 5th)
l2_curve_fit_levels: 3             # 3 levels for slope analysis
l2_max_time_diff_ms: 30000         # 30 seconds tolerance for alignment

# Core L2 features (100% coverage expected)
l2_features:
  - bid_ask_spread
  - bid_ask_spread_pct
  - weighted_mid_price
  - order_book_imbalance_2
  - order_book_imbalance_3
  - total_bid_volume_2
  - total_ask_volume_2
  - total_bid_volume_3
  - total_ask_volume_3
  - price_impact_buy      # ~45% coverage
  - price_impact_sell     # ~41% coverage
  - bid_slope            # ~20% coverage
  - ask_slope            # ~20% coverage
```

## 🎯 Usage Recommendations

### For New Users
1. **Start with**: `config.yaml` (main configuration)
2. **Collect L2 data**: `config_l2.yaml`
3. **Compare performance**: `config_no_l2.yaml`

### For Development
1. **Main development**: `config.yaml`
2. **L2 data collection**: `config_l2.yaml`
3. **Performance testing**: `config_wfo.yaml`

### For Production
1. **Live trading**: `config.yaml` (with live L2 collection)
2. **Fallback**: `config_no_l2.yaml` (if L2 unavailable)

## 🔍 Configuration Validation

### Check Your Configuration
```bash
# Validate main config
python -c "import yaml; print('✅ Valid' if yaml.safe_load(open('config.yaml')) else '❌ Invalid')"

# Check L2 settings
python -c "
import yaml
config = yaml.safe_load(open('config.yaml'))
print(f'L2 enabled: {config.get(\"use_l2_features\", False)}')
print(f'Depth levels: {config.get(\"l2_depth_imbalance_levels\", \"Not set\")}')
print(f'Tolerance: {config.get(\"l2_max_time_diff_ms\", \"Not set\")}ms')
"
```

### Expected Output
```
L2 enabled: True
Depth levels: [2, 3]
Tolerance: 30000ms
```

## 🚨 Common Configuration Issues

### Issue 1: Wrong L2 Depth Levels
**Problem**: Using old depth levels `[5, 10, 20]`  
**Solution**: Update to optimized levels `[2, 3]`  
**Impact**: Reduces NaN values by 22%

### Issue 2: Strict Alignment Tolerance
**Problem**: Using `l2_max_time_diff_ms: 1000`  
**Solution**: Increase to `30000` (30 seconds)  
**Impact**: Improves alignment from 31% to 100%

### Issue 3: Missing L2 Configuration
**Problem**: L2 enabled but missing depth settings  
**Solution**: Add all required L2 parameters  
**Impact**: Prevents feature calculation errors

## 📈 Performance Impact

### Configuration Performance Comparison

| Config | Features | Training Time | MAE | Win Rate | Use Case |
|--------|----------|---------------|-----|----------|----------|
| config.yaml | 37 | ~2 min | 0.817 | 8.33% | Main (needs strategy optimization) |
| config_no_l2.yaml | 24 | ~1 min | ~0.850 | TBD | Baseline comparison |
| config_wfo.yaml | 24 | ~30 sec | TBD | TBD | Fast WFO testing |

## 🔧 Customization Guide

### Adding Custom L2 Features
```yaml
# In config.yaml, add to l2_features list:
l2_features:
  - bid_ask_spread
  - your_custom_feature  # Add your feature here
```

### Adjusting Depth Levels
```yaml
# For more conservative (faster) approach:
l2_depth_imbalance_levels: [2]

# For more aggressive (more features) approach:
l2_depth_imbalance_levels: [2, 3, 5]  # Note: 5 levels only available 14% of time
```

### Modifying Alignment Tolerance
```yaml
# For high-frequency trading (stricter):
l2_max_time_diff_ms: 5000  # 5 seconds

# For research (more lenient):
l2_max_time_diff_ms: 60000  # 1 minute
```

## 📚 Related Documentation

- **[README.md](README.md)**: Project overview and quick start
- **[L2_INTEGRATION_GUIDE.md](L2_INTEGRATION_GUIDE.md)**: Detailed L2 implementation
- **[TEST_RESULTS.md](TEST_RESULTS.md)**: Performance testing results

---

**Last Updated**: Current session - Configuration optimization complete  
**Status**: ✅ All configs updated with optimized L2 settings 