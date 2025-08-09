# EMD Package Performance Benchmarking Suite

## Overview

This suite provides comprehensive benchmarking and analysis tools for Empirical Mode Decomposition (EMD) packages in high-frequency trading (HFT) applications. It specifically tests PyHHT, PyEMD, and custom SciPy implementations with realistic financial time series data sampled at 100ms intervals.

## Features

### 🚀 Core Capabilities
- **Multi-package benchmarking**: PyHHT, PyEMD, and SciPy-based implementations
- **Realistic HFT data simulation**: 100ms sampling intervals with microstructure effects
- **Memory profiling**: Detailed memory usage tracking with peak detection
- **Performance metrics**: Execution time, reconstruction error, IMF analysis
- **Pandas integration**: Complete DataFrame workflow examples
- **Visualization**: Comprehensive performance plots and analysis charts

### 📊 Signal Lengths Tested
- **100 samples**: 10 seconds of data (ultra-high frequency)
- **1,000 samples**: ~1.7 minutes of data (typical intraday analysis window)  
- **10,000 samples**: ~16.7 minutes of data (extended analysis period)

### 🎯 HFT-Specific Features
- Order book imbalance simulation
- Microstructure noise modeling
- Volatility clustering (GARCH-like effects)
- Jump event simulation
- Bid-ask spread dynamics
- Market regime detection using EMD

## Files Structure

```
├── emd_performance_benchmark.py      # Main benchmarking suite
├── install_and_test_emd_packages.py  # Package installation & testing
├── hft_microstructure_emd_example.py # HFT microstructure analysis demo
├── run_emd_tests.py                  # Test runner script
└── EMD_BENCHMARK_README.md           # This documentation
```

## Quick Start

### 1. Install Required Packages

The suite requires PyEMD (already in your requirements.txt). PyHHT will be installed automatically if missing.

```bash
# PyEMD should already be installed from requirements.txt
pip install PyEMD

# PyHHT will be auto-installed by the test scripts if needed
pip install PyHHT
```

### 2. Run Complete Test Suite

```bash
python run_emd_tests.py
```

This will:
1. Check and install missing packages
2. Run basic functionality tests
3. Execute comprehensive performance benchmarks
4. Generate HFT microstructure analysis examples

### 3. Individual Script Usage

#### Basic Installation & Testing
```bash
python install_and_test_emd_packages.py
```
- Tests package imports
- Runs basic functionality tests
- Creates simple visualization
- ~2-3 minutes runtime

#### Comprehensive Benchmarking
```bash
python emd_performance_benchmark.py
```
- Full performance comparison across packages
- Multiple signal lengths and trials
- Detailed memory profiling
- Comprehensive visualizations
- ~10-15 minutes runtime

#### HFT Microstructure Analysis
```bash
python hft_microstructure_emd_example.py
```
- Realistic market microstructure simulation
- EMD-based regime detection
- Trading signal generation
- Strategy performance analysis
- ~3-5 minutes runtime

## Output Files

### 📈 Visualizations
- `benchmark_visualization.png` - Comprehensive performance comparison
- `emd_test_visualization.png` - Basic EMD decomposition example
- `hft_microstructure_analysis.png` - HFT analysis dashboard

### 📊 Data Files
- `benchmark_results/benchmark_results.csv` - Raw benchmark data
- `benchmark_results/performance_report.txt` - Detailed analysis report
- `hft_sample_data.csv` - Sample HFT microstructure data
- `emd_decomposition_results.csv` - EMD analysis results

## Benchmark Results Interpretation

### Performance Metrics

#### Execution Time
- **PyEMD.EMD**: Fastest for signals < 1,000 samples
- **PyEMD.EEMD**: 10-20x slower due to ensemble averaging
- **PyEMD.CEEMDAN**: Slowest but most accurate decomposition
- **PyHHT.EMD**: Variable performance, good for research use

#### Memory Usage
- **Linear scaling** with signal length for most methods
- **EEMD/CEEMDAN**: Higher memory due to ensemble trials
- **Typical usage**: 50-200MB for 10,000 sample signals

#### Reconstruction Accuracy
- **EMD**: Perfect reconstruction (by definition)
- **EEMD**: Small residual noise from averaging
- **CEEMDAN**: Near-perfect reconstruction with best mode separation

### HFT Application Recommendations

#### For Real-Time Trading (< 100ms latency)
```python
# Use EMD with short rolling windows
signal_window = prices[-500:]  # Last 500 ticks
emd = EMD()
imfs = emd(signal_window)
```

#### For Regime Detection (1-5 second updates)
```python
# Use EEMD with moderate ensemble size
signal_window = prices[-1000:]  # Last 1000 ticks  
eemd = EEMD(trials=50)
imfs = eemd(signal_window)
```

#### For Research/Backtesting (accuracy priority)
```python
# Use CEEMDAN for best decomposition quality
ceemdan = CEEMDAN(trials=100)
imfs = ceemdan(full_signal)
```

## Integration with Trading System

### Real-Time Pipeline Example
```python
from PyEMD import EMD
import numpy as np

class EMDTradingSignal:
    def __init__(self, window_size=500):
        self.emd = EMD()
        self.window_size = window_size
        self.price_buffer = []
    
    def update(self, new_price):
        self.price_buffer.append(new_price)
        if len(self.price_buffer) > self.window_size:
            self.price_buffer.pop(0)
        
        if len(self.price_buffer) == self.window_size:
            return self.generate_signal()
        return 0
    
    def generate_signal(self):
        prices = np.array(self.price_buffer)
        imfs = self.emd(prices)
        
        # Simple regime detection
        trend = imfs[-1]  # Last IMF is trend
        recent_trend = np.mean(trend[-10:]) - np.mean(trend[-20:-10])
        
        # High frequency component
        noise = imfs[0]
        recent_noise = np.std(noise[-20:])
        
        # Generate signal based on trend and noise
        if abs(recent_trend) > recent_noise * 0.5:
            return 1 if recent_trend > 0 else -1
        return 0
```

### Pandas DataFrame Integration
```python
import pandas as pd
from PyEMD import EMD

def add_emd_features(df, price_col='price', window=200):
    """Add EMD-based features to trading DataFrame"""
    emd = EMD()
    
    # Rolling EMD analysis
    trend_strength = []
    cycle_dominance = []
    noise_level = []
    
    for i in range(window, len(df)):
        prices = df[price_col].iloc[i-window:i].values
        imfs = emd(prices)
        
        # Calculate regime features
        trend_var = np.var(imfs[-1]) if len(imfs) > 1 else 0
        cycle_var = np.mean([np.var(imf) for imf in imfs[1:-1]]) if len(imfs) > 2 else 0
        noise_var = np.var(imfs[0]) if len(imfs) > 0 else 0
        
        total_var = trend_var + cycle_var + noise_var
        
        trend_strength.append(trend_var / total_var if total_var > 0 else 0)
        cycle_dominance.append(cycle_var / total_var if total_var > 0 else 0)
        noise_level.append(noise_var / total_var if total_var > 0 else 0)
    
    # Add to DataFrame
    df = df.copy()
    df['emd_trend_strength'] = [0] * window + trend_strength
    df['emd_cycle_dominance'] = [0] * window + cycle_dominance
    df['emd_noise_level'] = [1] * window + noise_level
    
    return df
```

## Performance Optimization Tips

### 1. Rolling Window Strategy
```python
# Instead of full signal decomposition
full_imfs = emd(entire_price_history)  # Slow, memory intensive

# Use rolling windows
window_imfs = emd(prices[-500:])  # Fast, constant memory
```

### 2. Caching for Repeated Analysis
```python
from functools import lru_cache

@lru_cache(maxsize=128)
def cached_emd_decomposition(price_tuple):
    prices = np.array(price_tuple)
    return tuple(emd(prices))
```

### 3. GPU Acceleration (Advanced)
```python
# For EEMD/CEEMDAN with large ensembles
import cupy as cp  # Requires NVIDIA GPU

# Move computations to GPU for speed
gpu_prices = cp.asarray(prices)
# ... GPU-accelerated EMD operations
```

## Troubleshooting

### Common Issues

#### 1. Import Errors
```
ImportError: No module named 'PyEMD'
```
**Solution**: Install PyEMD
```bash
pip install PyEMD
```

#### 2. Memory Issues with Large Signals
```
MemoryError: Unable to allocate array
```
**Solution**: Use shorter windows or reduce ensemble trials
```python
# Reduce ensemble size
eemd = EEMD(trials=20)  # Instead of 100

# Use shorter windows
signal_window = prices[-1000:]  # Instead of full history
```

#### 3. Slow Performance
```
# EMD taking too long
```
**Solution**: Profile and optimize
```python
import time

start = time.time()
imfs = emd(prices)
print(f"EMD took {time.time() - start:.2f} seconds")

# Consider switching to simpler methods for real-time use
```

### Getting Help

1. **Check the performance report**: `benchmark_results/performance_report.txt`
2. **Review visualizations**: Look for outliers or unexpected patterns
3. **Adjust parameters**: Reduce signal length or ensemble trials
4. **Hardware considerations**: More RAM helps with larger signals

## Advanced Configuration

### Custom EMD Parameters
```python
# PyEMD EMD with custom stopping criteria
emd = EMD()
emd.FIXE = 10  # Fixed number of sifting iterations
emd.FIXE_H = 5  # Number of sifting iterations for each mode

# EEMD with noise configuration
eemd = EEMD(trials=100)
eemd.noise_scale = 0.01  # Reduce noise amplitude
eemd.ext_EMD = EMD()  # Custom EMD instance

# CEEMDAN with adaptive parameters
ceemdan = CEEMDAN(trials=50)
ceemdan.beta = 0.01  # Noise scaling factor
```

### Performance Monitoring
```python
import psutil
import time

def profile_emd_performance(emd_func, signal, method_name):
    """Profile EMD performance with detailed metrics"""
    process = psutil.Process()
    
    # Measure memory before
    mem_before = process.memory_info().rss / 1024 / 1024  # MB
    
    # Run EMD
    start_time = time.perf_counter()
    imfs = emd_func(signal)
    end_time = time.perf_counter()
    
    # Measure memory after
    mem_after = process.memory_info().rss / 1024 / 1024  # MB
    
    print(f"{method_name}:")
    print(f"  Time: {(end_time - start_time)*1000:.1f}ms")
    print(f"  Memory: {mem_after - mem_before:.1f}MB increase")
    print(f"  IMFs: {len(imfs)}")
    
    return imfs
```

## License and Citation

This benchmarking suite is designed for the Trading System project. When using EMD methods in trading applications, consider citing the original EMD papers:

- Huang, N. E. et al. (1998). "The empirical mode decomposition and the Hilbert spectrum for nonlinear and non-stationary time series analysis"
- Wu, Z., & Huang, N. E. (2009). "Ensemble empirical mode decomposition"
- Torres, M. E. et al. (2011). "A complete ensemble empirical mode decomposition with adaptive noise"

---

**Created**: 2025-07-30  
**Author**: Trading System  
**Version**: 1.0  
**Python**: 3.8+