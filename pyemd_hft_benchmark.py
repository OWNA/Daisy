#!/usr/bin/env python3
"""
PyEMD High-Frequency Trading Benchmark
=====================================

Focused benchmarking suite for PyEMD package with HFT data simulation.
This script works with just PyEMD and provides comprehensive performance analysis.

Features:
- Realistic HFT price simulation (100ms intervals)
- PyEMD EMD, EEMD, CEEMDAN benchmarking  
- Memory usage profiling
- Performance visualization
- Pandas DataFrame integration
- Trading signal examples

Author: Trading System
Date: 2025-07-30
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import psutil
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

try:
    from PyEMD import EMD, EEMD, CEEMDAN
    PYEMD_AVAILABLE = True
    print("PyEMD successfully imported")
except ImportError as e:
    print(f"PyEMD import failed: {e}")
    PYEMD_AVAILABLE = False


@dataclass
class BenchmarkResult:
    """Container for benchmark results"""
    method: str
    signal_length: int
    execution_time: float
    memory_mb: float  
    num_imfs: int
    reconstruction_error: float
    success: bool
    error_msg: Optional[str] = None


class HFTDataGenerator:
    """Generate realistic HFT price data"""
    
    def __init__(self, base_price: float = 50000.0, sampling_ms: float = 100.0):
        self.base_price = base_price
        self.dt = sampling_ms / 1000.0
        
    def generate_hft_series(self, length: int) -> np.ndarray:
        """Generate HFT price series with realistic components"""
        t = np.arange(length) * self.dt
        
        # Base trend
        trend = self.base_price * (1 + 0.0001 * t)
        
        # Multiple cycles (like market sessions)
        cycle1 = 100 * np.sin(2 * np.pi * 0.1 * t)  # 10-second cycle
        cycle2 = 50 * np.sin(2 * np.pi * 0.05 * t)  # 20-second cycle  
        cycle3 = 20 * np.sin(2 * np.pi * 0.2 * t)   # 5-second cycle
        
        # Volatility clustering
        vol = self._garch_volatility(length, 0.02)
        random_walk = np.cumsum(np.random.normal(0, vol * np.sqrt(self.dt), length))
        
        # Microstructure noise
        noise = 10 * np.random.normal(0, 1, length)
        
        # Jump events
        jumps = self._add_jumps(length)
        
        return trend + cycle1 + cycle2 + cycle3 + random_walk + noise + jumps
    
    def _garch_volatility(self, length: int, base_vol: float) -> np.ndarray:
        """Generate GARCH-like volatility"""
        vol_sq = np.zeros(length)
        vol_sq[0] = base_vol ** 2
        
        for i in range(1, length):
            vol_sq[i] = 0.0001 + 0.1 * vol_sq[i-1] + 0.85 * vol_sq[i-1]
        
        return np.sqrt(vol_sq) * self.base_price
    
    def _add_jumps(self, length: int) -> np.ndarray:
        """Add occasional jump events"""
        jumps = np.zeros(length)
        jump_prob = 0.001  # 0.1% chance per tick
        
        for i in range(length):
            if np.random.random() < jump_prob:
                jumps[i] = np.random.normal(0, self.base_price * 0.002)
        
        return np.cumsum(jumps)


class PyEMDBenchmark:
    """Benchmark PyEMD methods"""
    
    def __init__(self):
        self.generator = HFTDataGenerator()
        
    def benchmark_method(self, method_name: str, signal: np.ndarray) -> BenchmarkResult:
        """Benchmark a specific EMD method"""
        if not PYEMD_AVAILABLE:
            return BenchmarkResult(
                method=method_name,
                signal_length=len(signal), 
                execution_time=0,
                memory_mb=0,
                num_imfs=0,
                reconstruction_error=0,
                success=False,
                error_msg="PyEMD not available"
            )
        
        try:
            # Memory before
            process = psutil.Process()
            mem_before = process.memory_info().rss / 1024 / 1024
            
            # Run EMD method
            start_time = time.perf_counter()
            
            if method_name == 'EMD':
                emd = EMD()
                imfs = emd(signal)
            elif method_name == 'EEMD':
                eemd = EEMD(trials=50)  # Reduced for speed
                imfs = eemd(signal)
            elif method_name == 'CEEMDAN':
                ceemdan = CEEMDAN(trials=30)  # Reduced for speed
                imfs = ceemdan(signal)
            else:
                raise ValueError(f"Unknown method: {method_name}")
            
            end_time = time.perf_counter()
            
            # Memory after
            mem_after = process.memory_info().rss / 1024 / 1024
            
            # Calculate reconstruction error
            reconstruction = np.sum(imfs, axis=0)
            error = np.mean((signal - reconstruction) ** 2)
            
            return BenchmarkResult(
                method=method_name,
                signal_length=len(signal),
                execution_time=end_time - start_time,
                memory_mb=mem_after - mem_before,
                num_imfs=len(imfs),
                reconstruction_error=error,
                success=True
            )
            
        except Exception as e:
            return BenchmarkResult(
                method=method_name,
                signal_length=len(signal),
                execution_time=0,
                memory_mb=0,
                num_imfs=0,
                reconstruction_error=0,
                success=False,
                error_msg=str(e)
            )
    
    def run_comprehensive_benchmark(self) -> pd.DataFrame:
        """Run comprehensive benchmark across methods and signal lengths"""
        print("Starting PyEMD HFT Benchmark...")
        
        methods = ['EMD', 'EEMD', 'CEEMDAN']
        signal_lengths = [100, 500, 1000, 2000]  # HFT relevant sizes
        trials = 3
        
        results = []
        
        for length in signal_lengths:
            print(f"\nTesting signal length: {length}")
            
            for trial in range(trials):
                print(f"  Trial {trial + 1}/{trials}")
                
                # Generate test signal
                signal = self.generator.generate_hft_series(length)
                
                for method in methods:
                    print(f"    {method}...", end=" ", flush=True)
                    result = self.benchmark_method(method, signal)
                    results.append(result)
                    
                    if result.success:
                        print(f"{result.execution_time*1000:.1f}ms, {result.num_imfs} IMFs")
                    else:
                        print(f"FAILED: {result.error_msg}")
        
        # Convert to DataFrame
        df = pd.DataFrame([
            {
                'method': r.method,
                'signal_length': r.signal_length,
                'execution_time_ms': r.execution_time * 1000,
                'memory_mb': r.memory_mb,
                'num_imfs': r.num_imfs,
                'reconstruction_error': r.reconstruction_error,
                'success': r.success
            }
            for r in results
        ])
        
        return df
    
    def create_benchmark_plots(self, df: pd.DataFrame):
        """Create comprehensive benchmark visualizations"""
        if df.empty or not df['success'].any():
            print("No successful results to plot")
            return
        
        # Filter successful results
        df_success = df[df['success'] == True].copy()
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('PyEMD Performance Benchmark - HFT Data', fontsize=14, fontweight='bold')
        
        # 1. Execution time vs signal length
        ax1 = axes[0, 0]
        for method in df_success['method'].unique():
            method_data = df_success[df_success['method'] == method]
            grouped = method_data.groupby('signal_length')['execution_time_ms'].mean()
            ax1.plot(grouped.index, grouped.values, 'o-', label=method, linewidth=2, markersize=6)
        
        ax1.set_xlabel('Signal Length')
        ax1.set_ylabel('Execution Time (ms)')
        ax1.set_title('Execution Time vs Signal Length')
        ax1.set_yscale('log')  
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Memory usage vs signal length
        ax2 = axes[0, 1]
        for method in df_success['method'].unique():
            method_data = df_success[df_success['method'] == method]
            grouped = method_data.groupby('signal_length')['memory_mb'].mean()
            ax2.plot(grouped.index, grouped.values, 's-', label=method, linewidth=2, markersize=6)
        
        ax2.set_xlabel('Signal Length')
        ax2.set_ylabel('Memory Usage (MB)')
        ax2.set_title('Memory Usage vs Signal Length')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Number of IMFs
        ax3 = axes[1, 0]
        imf_data = df_success.groupby(['method', 'signal_length'])['num_imfs'].mean().reset_index()
        
        for method in imf_data['method'].unique():
            method_data = imf_data[imf_data['method'] == method]
            ax3.plot(method_data['signal_length'], method_data['num_imfs'], 
                    '^-', label=method, linewidth=2, markersize=6)
        
        ax3.set_xlabel('Signal Length')
        ax3.set_ylabel('Number of IMFs')
        ax3.set_title('IMFs Generated vs Signal Length')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Reconstruction error
        ax4 = axes[1, 1]
        for method in df_success['method'].unique():
            method_data = df_success[df_success['method'] == method]
            grouped = method_data.groupby('signal_length')['reconstruction_error'].mean()
            ax4.plot(grouped.index, grouped.values, 'd-', label=method, linewidth=2, markersize=6)
        
        ax4.set_xlabel('Signal Length')
        ax4.set_ylabel('Reconstruction Error')
        ax4.set_title('Reconstruction Error vs Signal Length')
        ax4.set_yscale('log')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('pyemd_hft_benchmark.png', dpi=300, bbox_inches='tight')
        print("Benchmark plot saved as 'pyemd_hft_benchmark.png'")
        
    def demonstrate_trading_signals(self):
        """Demonstrate EMD-based trading signal generation"""
        print("\nDemonstrating EMD trading signals...")
        
        # Generate longer signal for meaningful analysis
        signal = self.generator.generate_hft_series(1500)  # 2.5 minutes @ 100ms
        
        # Apply EMD
        emd = EMD()
        imfs = emd(signal)
        
        # Create DataFrame
        df = pd.DataFrame({
            'price': signal,
            'returns': np.append([0], np.diff(signal))
        })
        
        # Add IMF-based features
        if len(imfs) >= 3:
            df['noise'] = np.append(imfs[0], [imfs[0][-1]])[:len(df)]  # High freq noise
            df['trend'] = np.append(imfs[-1], [imfs[-1][-1]])[:len(df)]  # Trend
            
            # Calculate trend strength
            df['trend_change'] = df['trend'].diff()
            df['trend_strength'] = df['trend_change'].rolling(20).mean()
            
            # Generate simple signals
            df['signal'] = 0
            df.loc[(df['trend_strength'] > 0) & (df['noise'] < -10), 'signal'] = 1  # Buy dips in uptrend
            df.loc[(df['trend_strength'] < 0) & (df['noise'] > 10), 'signal'] = -1  # Sell rallies in downtrend
        
        # Calculate strategy performance
        df['strategy_returns'] = df['signal'].shift(1) * df['returns']
        
        # Plot results
        fig, axes = plt.subplots(3, 1, figsize=(15, 12))
        
        # Price and trend
        ax1 = axes[0]
        ax1.plot(df.index, df['price'], 'b-', linewidth=1, label='Price', alpha=0.7)
        if 'trend' in df.columns:
            ax1.plot(df.index, df['trend'], 'r-', linewidth=2, label='EMD Trend')
        ax1.set_title('Price and EMD Trend Component')
        ax1.set_ylabel('Price')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # IMF components  
        ax2 = axes[1]
        if len(imfs) >= 3:
            ax2.plot(df.index[:len(imfs[0])], imfs[0], 'g-', linewidth=1, label='IMF 1 (noise)', alpha=0.7)
            ax2.plot(df.index[:len(imfs[1])], imfs[1], 'orange', linewidth=1, label='IMF 2 (short cycle)', alpha=0.7)
            if len(imfs) > 3:
                ax2.plot(df.index[:len(imfs[2])], imfs[2], 'purple', linewidth=1, label='IMF 3 (med cycle)', alpha=0.7)
        ax2.set_title('EMD Components')
        ax2.set_ylabel('Amplitude')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Trading signals
        ax3 = axes[2]
        ax3.plot(df.index, df['price'], 'k-', linewidth=1, alpha=0.6, label='Price')
        
        buy_signals = df[df['signal'] == 1]
        sell_signals = df[df['signal'] == -1]
        
        if len(buy_signals) > 0:
            ax3.scatter(buy_signals.index, buy_signals['price'], c='green', marker='^', s=50, label='Buy')
        if len(sell_signals) > 0:
            ax3.scatter(sell_signals.index, sell_signals['price'], c='red', marker='v', s=50, label='Sell')
        
        ax3.set_title('Trading Signals Based on EMD Analysis')
        ax3.set_xlabel('Time (100ms ticks)')
        ax3.set_ylabel('Price')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('emd_trading_signals.png', dpi=300, bbox_inches='tight')
        print("Trading signals plot saved as 'emd_trading_signals.png'")
        
        # Performance summary
        if 'strategy_returns' in df.columns:
            total_return = df['strategy_returns'].sum()
            num_trades = (df['signal'] != 0).sum()
            win_rate = (df['strategy_returns'] > 0).sum() / num_trades if num_trades > 0 else 0
            
            print(f"\nTrading Strategy Performance:")
            print(f"  Total Return: {total_return:.2f}")
            print(f"  Number of Trades: {num_trades}")
            print(f"  Win Rate: {win_rate:.1%}")


def main():
    """Main execution function"""
    print("PyEMD High-Frequency Trading Benchmark")
    print("=" * 40)
    
    if not PYEMD_AVAILABLE:
        print("ERROR: PyEMD not available")
        print("Install with: pip install EMD-signal")
        return
    
    # Initialize benchmark
    benchmark = PyEMDBenchmark()
    
    try:
        # Run comprehensive benchmark
        print("\n1. Running comprehensive benchmark...")
        df_results = benchmark.run_comprehensive_benchmark()
        
        # Save results
        df_results.to_csv('pyemd_benchmark_results.csv', index=False)
        print(f"\nResults saved to: pyemd_benchmark_results.csv")
        
        # Create visualizations
        print("\n2. Creating benchmark visualizations...")
        benchmark.create_benchmark_plots(df_results)
        
        # Demonstrate trading signals
        print("\n3. Demonstrating trading signal generation...")
        benchmark.demonstrate_trading_signals()
        
        # Summary statistics
        print("\n" + "=" * 50)
        print("BENCHMARK SUMMARY")
        print("=" * 50)
        
        success_df = df_results[df_results['success'] == True]
        if not success_df.empty:
            for method in success_df['method'].unique():
                method_data = success_df[success_df['method'] == method]
                avg_time = method_data['execution_time_ms'].mean()
                avg_memory = method_data['memory_mb'].mean()
                avg_imfs = method_data['num_imfs'].mean()
                
                print(f"{method}:")
                print(f"  Average Time: {avg_time:.1f}ms")
                print(f"  Average Memory: {avg_memory:.1f}MB")
                print(f"  Average IMFs: {avg_imfs:.1f}")
                print()
        
        print("Files generated:")
        print("  - pyemd_benchmark_results.csv")
        print("  - pyemd_hft_benchmark.png")  
        print("  - emd_trading_signals.png")
        
        print("\nBenchmark completed successfully!")
        
    except Exception as e:
        print(f"Benchmark failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()