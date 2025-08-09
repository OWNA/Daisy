#!/usr/bin/env python3
"""
EMD Package Performance Benchmarking Suite for High-Frequency Trading
====================================================================

This module provides comprehensive benchmarking of PyHHT, PyEMD, and EMD packages
for analyzing high-frequency financial time series data. It simulates realistic
trading scenarios with 100ms sampling intervals and evaluates performance across
different signal lengths (100, 1000, 10000 samples).

Features:
- Realistic HFT price simulation with microstructure noise
- Memory usage profiling with detailed tracking
- Execution time measurement with statistical analysis
- Pandas DataFrame integration examples
- Performance comparison across different EMD implementations
- Financial time series pattern simulation (trends, cycles, volatility clustering)

Author: Trading System
Date: 2025-07-30
"""

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any, Union
import psutil
import gc
import tracemalloc
from dataclasses import dataclass
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Import EMD packages with fallback handling
try:
    from PyEMD import EMD, EEMD, CEEMDAN
    PYEMD_AVAILABLE = True
    print("✓ PyEMD successfully imported")
except ImportError as e:
    print(f"✗ PyEMD import failed: {e}")
    PYEMD_AVAILABLE = False

try:
    from pyhht import EMD as PyHHT_EMD
    from pyhht.visualization import plot_imfs
    PYHHT_AVAILABLE = True
    print("✓ PyHHT successfully imported")
except ImportError as e:
    print(f"✗ PyHHT import failed: {e}")
    PYHHT_AVAILABLE = False

try:
    # Alternative EMD implementation
    import scipy.signal
    SCIPY_AVAILABLE = True
    print("✓ SciPy available for signal processing")
except ImportError as e:
    print(f"✗ SciPy import failed: {e}")
    SCIPY_AVAILABLE = False


@dataclass
class BenchmarkResult:
    """Container for benchmark results"""
    package_name: str
    method_name: str
    signal_length: int
    execution_time: float
    memory_peak: float
    memory_current: float
    num_imfs: int
    success: bool
    error_message: Optional[str] = None
    reconstruction_error: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'package': self.package_name,
            'method': self.method_name,
            'signal_length': self.signal_length,
            'execution_time_ms': self.execution_time * 1000,
            'memory_peak_mb': self.memory_peak,
            'memory_current_mb': self.memory_current,
            'num_imfs': self.num_imfs,
            'success': self.success,
            'error': self.error_message,
            'reconstruction_error': self.reconstruction_error
        }


class HFTDataSimulator:
    """Simulates realistic high-frequency trading data patterns"""
    
    def __init__(self, sampling_interval_ms: float = 100.0):
        self.sampling_interval_ms = sampling_interval_ms
        self.dt = sampling_interval_ms / 1000.0  # Convert to seconds
        
    def generate_hft_price_series(self, 
                                 length: int,
                                 base_price: float = 50000.0,
                                 trend_strength: float = 0.001,
                                 volatility: float = 0.02,
                                 microstructure_noise: float = 0.0005,
                                 add_cycles: bool = True) -> np.ndarray:
        """
        Generate realistic HFT price series with multiple components:
        - Long-term trend
        - Intraday cycles  
        - Volatility clustering
        - Microstructure noise
        - Jump events
        """
        t = np.arange(length) * self.dt
        
        # Base trend component
        trend = base_price + trend_strength * t * base_price
        
        # Intraday cyclical components (multiple frequencies)
        cycles = np.zeros(length)
        if add_cycles:
            # Primary cycle (similar to daily trading session)
            cycle_freq1 = 2 * np.pi / (8 * 3600 / self.dt)  # 8-hour cycle
            cycles += 0.003 * base_price * np.sin(cycle_freq1 * t)
            
            # Secondary cycle (lunch break effect)
            cycle_freq2 = 2 * np.pi / (4 * 3600 / self.dt)  # 4-hour cycle
            cycles += 0.001 * base_price * np.sin(cycle_freq2 * t + np.pi/4)
            
            # Higher frequency oscillations
            cycle_freq3 = 2 * np.pi / (30 * 60 / self.dt)   # 30-minute cycle
            cycles += 0.0005 * base_price * np.sin(cycle_freq3 * t)
        
        # Volatility clustering using GARCH-like process
        volatility_process = self._generate_volatility_clustering(length, volatility)
        
        # Random walk component with time-varying volatility
        random_increments = np.random.normal(0, 1, length)
        random_walk = np.cumsum(random_increments * volatility_process * base_price * np.sqrt(self.dt))
        
        # Microstructure noise (bid-ask bounce, inventory effects)
        microstructure = self._generate_microstructure_noise(length, microstructure_noise * base_price)
        
        # Occasional jump events
        jumps = self._generate_jump_events(length, base_price)
        
        # Combine all components
        price_series = trend + cycles + random_walk + microstructure + jumps
        
        # Ensure prices remain positive
        price_series = np.maximum(price_series, base_price * 0.5)
        
        return price_series
    
    def _generate_volatility_clustering(self, length: int, base_vol: float) -> np.ndarray:
        """Generate GARCH-like volatility clustering"""
        omega, alpha, beta = 0.0001, 0.1, 0.85
        vol_squared = np.zeros(length)
        vol_squared[0] = base_vol ** 2
        
        innovations = np.random.normal(0, 1, length)
        
        for i in range(1, length):
            vol_squared[i] = (omega + 
                            alpha * (innovations[i-1] * np.sqrt(vol_squared[i-1])) ** 2 + 
                            beta * vol_squared[i-1])
        
        return np.sqrt(vol_squared)
    
    def _generate_microstructure_noise(self, length: int, noise_std: float) -> np.ndarray:
        """Generate microstructure noise with persistence"""
        noise = np.random.normal(0, noise_std, length)
        # Add some persistence to simulate bid-ask bounce
        alpha = 0.3
        for i in range(1, length):
            noise[i] = alpha * noise[i-1] + (1 - alpha) * noise[i]
        return noise
    
    def _generate_jump_events(self, length: int, base_price: float, 
                            jump_intensity: float = 0.01) -> np.ndarray:
        """Generate rare jump events (news, large orders)"""
        jumps = np.zeros(length)
        jump_times = np.random.poisson(jump_intensity, length).astype(bool)
        jump_sizes = np.random.normal(0, 0.005 * base_price, length)
        jumps[jump_times] = jump_sizes[jump_times]
        return np.cumsum(jumps)


class MemoryProfiler:
    """Memory usage profiler with detailed tracking"""
    
    def __init__(self):
        self.process = psutil.Process()
        self.initial_memory = None
        self.peak_memory = 0
        
    def start_profiling(self):
        """Start memory profiling"""
        gc.collect()  # Clean up before starting
        tracemalloc.start()
        self.initial_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        self.peak_memory = self.initial_memory
        
    def update_peak(self):
        """Update peak memory usage"""
        current_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        self.peak_memory = max(self.peak_memory, current_memory)
        
    def stop_profiling(self) -> Tuple[float, float]:
        """Stop profiling and return (peak_memory, current_memory) in MB"""
        self.update_peak()
        current_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        tracemalloc.stop()
        return self.peak_memory, current_memory


class EMDBenchmark:
    """Main benchmarking class for EMD implementations"""
    
    def __init__(self, output_dir: str = "benchmark_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.simulator = HFTDataSimulator()
        self.results: List[BenchmarkResult] = []
        
    def benchmark_pyemd_methods(self, signal: np.ndarray) -> List[BenchmarkResult]:
        """Benchmark PyEMD package methods"""
        results = []
        
        if not PYEMD_AVAILABLE:
            return results
            
        # Standard EMD
        results.append(self._benchmark_method(
            "PyEMD", "EMD", signal, self._run_pyemd_emd
        ))
        
        # Ensemble EMD
        results.append(self._benchmark_method(
            "PyEMD", "EEMD", signal, self._run_pyemd_eemd
        ))
        
        # Complete EEMD with Adaptive Noise
        results.append(self._benchmark_method(
            "PyEMD", "CEEMDAN", signal, self._run_pyemd_ceemdan
        ))
        
        return results
    
    def benchmark_pyhht_methods(self, signal: np.ndarray) -> List[BenchmarkResult]:
        """Benchmark PyHHT package methods"""
        results = []
        
        if not PYHHT_AVAILABLE:
            return results
            
        results.append(self._benchmark_method(
            "PyHHT", "EMD", signal, self._run_pyhht_emd
        ))
        
        return results
    
    def benchmark_scipy_methods(self, signal: np.ndarray) -> List[BenchmarkResult]:
        """Benchmark SciPy-based methods"""
        results = []
        
        if not SCIPY_AVAILABLE:
            return results
        
        # Custom EMD implementation using SciPy
        results.append(self._benchmark_method(
            "SciPy", "Custom_EMD", signal, self._run_scipy_emd
        ))
        
        return results
    
    def _benchmark_method(self, package: str, method: str, signal: np.ndarray, 
                         func) -> BenchmarkResult:
        """Benchmark a specific EMD method"""
        profiler = MemoryProfiler()
        
        try:
            profiler.start_profiling()
            start_time = time.perf_counter()
            
            imfs, reconstruction_error = func(signal)
            
            end_time = time.perf_counter()
            peak_memory, current_memory = profiler.stop_profiling()
            
            return BenchmarkResult(
                package_name=package,
                method_name=method,
                signal_length=len(signal),
                execution_time=end_time - start_time,
                memory_peak=peak_memory,
                memory_current=current_memory,
                num_imfs=len(imfs),
                success=True,
                reconstruction_error=reconstruction_error
            )
            
        except Exception as e:
            peak_memory, current_memory = profiler.stop_profiling()
            return BenchmarkResult(
                package_name=package,
                method_name=method,
                signal_length=len(signal),
                execution_time=0,
                memory_peak=peak_memory,
                memory_current=current_memory,
                num_imfs=0,
                success=False,
                error_message=str(e)
            )
    
    def _run_pyemd_emd(self, signal: np.ndarray) -> Tuple[List[np.ndarray], float]:
        """Run PyEMD EMD"""
        emd = EMD()
        imfs = emd(signal)
        reconstruction = np.sum(imfs, axis=0)
        error = np.mean((signal - reconstruction) ** 2)
        return imfs, error
    
    def _run_pyemd_eemd(self, signal: np.ndarray) -> Tuple[List[np.ndarray], float]:
        """Run PyEMD EEMD"""
        eemd = EEMD(trials=50)  # Reduced trials for speed
        imfs = eemd(signal)
        reconstruction = np.sum(imfs, axis=0)
        error = np.mean((signal - reconstruction) ** 2)
        return imfs, error
    
    def _run_pyemd_ceemdan(self, signal: np.ndarray) -> Tuple[List[np.ndarray], float]:
        """Run PyEMD CEEMDAN"""
        ceemdan = CEEMDAN(trials=50)  # Reduced trials for speed
        imfs = ceemdan(signal)
        reconstruction = np.sum(imfs, axis=0)
        error = np.mean((signal - reconstruction) ** 2)
        return imfs, error
    
    def _run_pyhht_emd(self, signal: np.ndarray) -> Tuple[List[np.ndarray], float]:
        """Run PyHHT EMD"""
        emd = PyHHT_EMD(signal)
        imfs = emd.decompose()
        reconstruction = np.sum(imfs, axis=0)
        error = np.mean((signal - reconstruction) ** 2)
        return imfs, error
    
    def _run_scipy_emd(self, signal: np.ndarray) -> Tuple[List[np.ndarray], float]:
        """Run custom EMD implementation using SciPy"""
        # Simplified EMD implementation for comparison
        imfs = self._simple_emd_decomposition(signal)
        reconstruction = np.sum(imfs, axis=0)
        error = np.mean((signal - reconstruction) ** 2)
        return imfs, error
    
    def _simple_emd_decomposition(self, signal: np.ndarray, max_imfs: int = 8) -> List[np.ndarray]:
        """Simple EMD implementation using SciPy for comparison"""
        imfs = []
        residue = signal.copy()
        
        for _ in range(max_imfs):
            if len(residue) < 4:
                break
                
            # Find extrema
            from scipy.signal import find_peaks
            maxima_idx, _ = find_peaks(residue)
            minima_idx, _ = find_peaks(-residue)
            
            if len(maxima_idx) < 2 or len(minima_idx) < 2:
                break
            
            # Simple sifting process
            current_imf = residue.copy()
            for _ in range(10):  # Limited iterations
                try:
                    from scipy.interpolate import interp1d
                    
                    # Create envelopes
                    if len(maxima_idx) >= 2:
                        max_interp = interp1d(maxima_idx, residue[maxima_idx], 
                                            kind='cubic', fill_value='extrapolate')
                        upper_env = max_interp(np.arange(len(residue)))
                    else:
                        upper_env = np.zeros_like(residue)
                    
                    if len(minima_idx) >= 2:
                        min_interp = interp1d(minima_idx, residue[minima_idx], 
                                            kind='cubic', fill_value='extrapolate')
                        lower_env = min_interp(np.arange(len(residue)))
                    else:
                        lower_env = np.zeros_like(residue)
                    
                    mean_env = (upper_env + lower_env) / 2
                    current_imf = residue - mean_env
                    
                    # Check stopping criterion
                    if np.std(mean_env) < 0.001 * np.std(residue):
                        break
                        
                    residue = current_imf.copy()
                    
                except Exception:
                    break
            
            imfs.append(current_imf)
            residue = signal - np.sum(imfs, axis=0)
            
            if np.std(residue) < 0.001 * np.std(signal):
                break
        
        # Add final residue
        if len(residue) > 0:
            imfs.append(residue)
            
        return imfs
    
    def run_comprehensive_benchmark(self, signal_lengths: List[int] = [100, 1000, 10000],
                                  num_trials: int = 3) -> pd.DataFrame:
        """Run comprehensive benchmark across all methods and signal lengths"""
        print("🚀 Starting comprehensive EMD benchmark suite...")
        print(f"Signal lengths: {signal_lengths}")
        print(f"Trials per configuration: {num_trials}")
        print(f"Sampling interval: {self.simulator.sampling_interval_ms}ms")
        
        all_results = []
        
        for length in signal_lengths:
            print(f"\n📊 Benchmarking signal length: {length}")
            
            for trial in range(num_trials):
                print(f"  Trial {trial + 1}/{num_trials}")
                
                # Generate test signal
                signal = self.simulator.generate_hft_price_series(
                    length=length,
                    base_price=50000.0,
                    trend_strength=0.001,
                    volatility=0.02,
                    microstructure_noise=0.0005,
                    add_cycles=True
                )
                
                # Benchmark all available methods
                results = []
                results.extend(self.benchmark_pyemd_methods(signal))
                results.extend(self.benchmark_pyhht_methods(signal))
                results.extend(self.benchmark_scipy_methods(signal))
                
                all_results.extend(results)
                
                # Progress update
                for result in results:
                    status = "✓" if result.success else "✗"
                    print(f"    {status} {result.package_name}.{result.method_name}: "
                          f"{result.execution_time*1000:.1f}ms, "
                          f"{result.memory_peak:.1f}MB, "
                          f"{result.num_imfs} IMFs")
        
        # Convert to DataFrame
        df_results = pd.DataFrame([r.to_dict() for r in all_results])
        
        # Save results
        results_file = self.output_dir / "benchmark_results.csv"
        df_results.to_csv(results_file, index=False)
        print(f"\n💾 Results saved to: {results_file}")
        
        return df_results
    
    def create_visualizations(self, df_results: pd.DataFrame):
        """Create comprehensive visualization plots"""
        print("\n📈 Creating visualization plots...")
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # Filter successful results
        df_success = df_results[df_results['success'] == True].copy()
        
        if df_success.empty:
            print("⚠️  No successful results to visualize")
            return
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('EMD Package Performance Benchmark - High-Frequency Trading Data', 
                    fontsize=16, fontweight='bold')
        
        # 1. Execution Time vs Signal Length
        ax1 = axes[0, 0]
        for package in df_success['package'].unique():
            package_data = df_success[df_success['package'] == package]
            grouped = package_data.groupby(['signal_length', 'method'])['execution_time_ms'].mean().reset_index()
            
            for method in grouped['method'].unique():
                method_data = grouped[grouped['method'] == method]
                ax1.plot(method_data['signal_length'], method_data['execution_time_ms'], 
                        marker='o', label=f"{package}.{method}", linewidth=2, markersize=6)
        
        ax1.set_xlabel('Signal Length')
        ax1.set_ylabel('Execution Time (ms)')
        ax1.set_title('Execution Time vs Signal Length')
        ax1.set_xscale('log')
        ax1.set_yscale('log')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # 2. Memory Usage vs Signal Length
        ax2 = axes[0, 1]
        for package in df_success['package'].unique():
            package_data = df_success[df_success['package'] == package]
            grouped = package_data.groupby(['signal_length', 'method'])['memory_peak_mb'].mean().reset_index()
            
            for method in grouped['method'].unique():
                method_data = grouped[grouped['method'] == method]
                ax2.plot(method_data['signal_length'], method_data['memory_peak_mb'], 
                        marker='s', label=f"{package}.{method}", linewidth=2, markersize=6)
        
        ax2.set_xlabel('Signal Length')
        ax2.set_ylabel('Peak Memory (MB)')
        ax2.set_title('Memory Usage vs Signal Length')
        ax2.set_xscale('log')
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.grid(True, alpha=0.3)
        
        # 3. Number of IMFs
        ax3 = axes[0, 2]
        imf_data = df_success.groupby(['package', 'method', 'signal_length'])['num_imfs'].mean().reset_index()
        
        # Create box plot
        sns.boxplot(data=df_success, x='signal_length', y='num_imfs', hue='method', ax=ax3)
        ax3.set_title('Number of IMFs by Signal Length')
        ax3.set_xlabel('Signal Length')
        ax3.set_ylabel('Number of IMFs')
        ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # 4. Performance Comparison (Execution Time)
        ax4 = axes[1, 0]
        df_pivot = df_success.pivot_table(values='execution_time_ms', 
                                         index=['package', 'method'], 
                                         columns='signal_length', 
                                         aggfunc='mean')
        
        sns.heatmap(df_pivot, annot=True, fmt='.1f', cmap='YlOrRd', ax=ax4)
        ax4.set_title('Execution Time Heatmap (ms)')
        ax4.set_ylabel('Package.Method')
        
        # 5. Performance Comparison (Memory)
        ax5 = axes[1, 1]
        df_pivot_mem = df_success.pivot_table(values='memory_peak_mb', 
                                             index=['package', 'method'], 
                                             columns='signal_length', 
                                             aggfunc='mean')
        
        sns.heatmap(df_pivot_mem, annot=True, fmt='.1f', cmap='Blues', ax=ax5)
        ax5.set_title('Memory Usage Heatmap (MB)')
        ax5.set_ylabel('Package.Method')
        
        # 6. Reconstruction Error
        ax6 = axes[1, 2]
        error_data = df_success[df_success['reconstruction_error'].notna()]
        if not error_data.empty:
            sns.boxplot(data=error_data, x='signal_length', y='reconstruction_error', 
                       hue='method', ax=ax6)
            ax6.set_title('Reconstruction Error by Method')
            ax6.set_yscale('log')
            ax6.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        else:
            ax6.text(0.5, 0.5, 'No reconstruction error data available', 
                    ha='center', va='center', transform=ax6.transAxes)
        
        plt.tight_layout()
        
        # Save plot
        plot_file = self.output_dir / "benchmark_visualization.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"📊 Visualization saved to: {plot_file}")
        
        plt.show()
    
    def generate_performance_report(self, df_results: pd.DataFrame):
        """Generate detailed performance report"""
        print("\n📋 Generating performance report...")
        
        report_lines = [
            "EMD Package Performance Benchmark Report",
            "=" * 50,
            f"Generated: {pd.Timestamp.now()}",
            f"Sampling Interval: {self.simulator.sampling_interval_ms}ms",
            "",
            "EXECUTIVE SUMMARY",
            "-" * 20
        ]
        
        # Success rate
        total_tests = len(df_results)
        successful_tests = len(df_results[df_results['success'] == True])
        success_rate = (successful_tests / total_tests) * 100 if total_tests > 0 else 0
        
        report_lines.extend([
            f"Total Tests: {total_tests}",
            f"Successful Tests: {successful_tests}",
            f"Success Rate: {success_rate:.1f}%",
            ""
        ])
        
        # Performance by package
        if successful_tests > 0:
            df_success = df_results[df_results['success'] == True]
            
            report_lines.extend([
                "PERFORMANCE BY PACKAGE",
                "-" * 25
            ])
            
            for package in sorted(df_success['package'].unique()):
                package_data = df_success[df_success['package'] == package]
                avg_time = package_data['execution_time_ms'].mean()
                avg_memory = package_data['memory_peak_mb'].mean()
                avg_imfs = package_data['num_imfs'].mean()
                
                report_lines.extend([
                    f"{package}:",
                    f"  Average Execution Time: {avg_time:.2f}ms",
                    f"  Average Memory Usage: {avg_memory:.2f}MB", 
                    f"  Average IMFs: {avg_imfs:.1f}",
                    ""
                ])
            
            # Best performers
            report_lines.extend([
                "BEST PERFORMERS",
                "-" * 15
            ])
            
            # Fastest method
            fastest = df_success.loc[df_success['execution_time_ms'].idxmin()]
            report_lines.append(
                f"Fastest: {fastest['package']}.{fastest['method']} "
                f"({fastest['execution_time_ms']:.2f}ms on {fastest['signal_length']} samples)"
            )
            
            # Most memory efficient
            most_efficient = df_success.loc[df_success['memory_peak_mb'].idxmin()]
            report_lines.append(
                f"Most Memory Efficient: {most_efficient['package']}.{most_efficient['method']} "
                f"({most_efficient['memory_peak_mb']:.2f}MB on {most_efficient['signal_length']} samples)"
            )
            
            # Best reconstruction
            if 'reconstruction_error' in df_success.columns:
                error_data = df_success[df_success['reconstruction_error'].notna()]
                if not error_data.empty:
                    best_reconstruction = error_data.loc[error_data['reconstruction_error'].idxmin()]
                    report_lines.append(
                        f"Best Reconstruction: {best_reconstruction['package']}.{best_reconstruction['method']} "
                        f"(error: {best_reconstruction['reconstruction_error']:.2e})"
                    )
        
        # Failed tests
        failed_tests = df_results[df_results['success'] == False]
        if not failed_tests.empty:
            report_lines.extend([
                "",
                "FAILED TESTS",
                "-" * 12
            ])
            
            for _, row in failed_tests.iterrows():
                report_lines.append(
                    f"{row['package']}.{row['method']} (length={row['signal_length']}): {row['error']}"
                )
        
        # Recommendations
        report_lines.extend([
            "",
            "RECOMMENDATIONS FOR HFT APPLICATIONS",
            "-" * 35,
            "• For real-time applications (< 100ms latency requirement):",
            "  - Use PyEMD.EMD for signals < 1000 samples",
            "  - Consider SciPy custom implementation for maximum speed",
            "",
            "• For highest accuracy (offline analysis):",
            "  - Use PyEMD.CEEMDAN for best signal decomposition",
            "  - Accept higher computational cost for superior results",
            "",
            "• For memory-constrained environments:",
            "  - Prefer PyEMD.EMD over ensemble methods",
            "  - Monitor memory usage with longer signals",
            "",
            "• For production HFT systems:",
            "  - Implement rolling window analysis (last 500-1000 samples)",
            "  - Consider GPU acceleration for EEMD/CEEMDAN",
            "  - Cache IMF calculations when possible"
        ])
        
        report_text = "\n".join(report_lines)
        
        # Save report
        report_file = self.output_dir / "performance_report.txt"
        with open(report_file, 'w') as f:
            f.write(report_text)
        
        print(f"📄 Report saved to: {report_file}")
        print("\nKey findings:")
        print(f"• Success rate: {success_rate:.1f}%")
        if successful_tests > 0:
            print(f"• Best overall performance: Review {report_file} for details")


def demonstrate_pandas_integration():
    """Demonstrate EMD integration with pandas DataFrames for HFT data"""
    print("\n🐼 Demonstrating Pandas DataFrame Integration...")
    
    # Create sample HFT data
    simulator = HFTDataSimulator(sampling_interval_ms=100)
    length = 1000
    
    # Generate multiple price series (different assets/timeframes)
    assets = ['BTC-USD', 'ETH-USD', 'ADA-USD']
    df_data = pd.DataFrame()
    
    for asset in assets:
        prices = simulator.generate_hft_price_series(
            length=length,
            base_price=np.random.uniform(100, 50000),
            trend_strength=np.random.uniform(-0.002, 0.002),
            volatility=np.random.uniform(0.01, 0.03)
        )
        
        timestamps = pd.date_range(
            start='2024-01-01 09:30:00', 
            periods=length, 
            freq='100ms'
        )
        
        asset_df = pd.DataFrame({
            'timestamp': timestamps,
            'asset': asset,
            'price': prices,
            'returns': np.append([0], np.diff(np.log(prices)))
        })
        
        df_data = pd.concat([df_data, asset_df], ignore_index=True)
    
    print(f"📊 Created dataset with {len(df_data)} rows, {len(assets)} assets")
    print(f"📅 Time range: {df_data['timestamp'].min()} to {df_data['timestamp'].max()}")
    
    # Apply EMD to each asset
    if PYEMD_AVAILABLE:
        print("\n🔄 Applying EMD decomposition to each asset...")
        
        emd = EMD()
        results_list = []
        
        for asset in assets:
            asset_data = df_data[df_data['asset'] == asset].copy()
            prices = asset_data['price'].values
            
            try:
                # Decompose using EMD
                start_time = time.perf_counter()
                imfs = emd(prices)
                decomp_time = time.perf_counter() - start_time
                
                # Create results DataFrame
                for i, imf in enumerate(imfs):
                    imf_df = asset_data[['timestamp', 'asset']].copy()
                    imf_df['imf_index'] = i
                    imf_df['imf_value'] = imf
                    imf_df['imf_type'] = 'trend' if i == len(imfs)-1 else f'imf_{i+1}'
                    results_list.append(imf_df)
                
                print(f"  ✓ {asset}: {len(imfs)} IMFs in {decomp_time*1000:.1f}ms")
                
            except Exception as e:
                print(f"  ✗ {asset}: Failed - {e}")
        
        if results_list:
            # Combine all results
            df_imfs = pd.concat(results_list, ignore_index=True)
            
            # Calculate some analytics
            analytics = df_imfs.groupby(['asset', 'imf_type']).agg({
                'imf_value': ['mean', 'std', 'min', 'max']
            }).round(4)
            
            print(f"\n📈 EMD Analysis Complete:")
            print(f"• Total IMF records: {len(df_imfs)}")
            print(f"• IMF types found: {sorted(df_imfs['imf_type'].unique())}")
            
            # Save results
            output_dir = Path("benchmark_results")
            output_dir.mkdir(exist_ok=True)
            
            df_data.to_csv(output_dir / "hft_sample_data.csv", index=False)
            df_imfs.to_csv(output_dir / "emd_decomposition_results.csv", index=False)
            analytics.to_csv(output_dir / "emd_analytics_summary.csv")
            
            print(f"💾 Sample data saved to: {output_dir}/")
            
            # Display sample of results
            print(f"\n📋 Sample EMD Results:")
            print(df_imfs.head(10).to_string(index=False))
        
    else:
        print("⚠️  PyEMD not available - skipping pandas integration demo")


def main():
    """Main execution function"""
    print("🚀 EMD Package Performance Benchmark Suite")
    print("=" * 50)
    print("Testing PyHHT, PyEMD, and EMD implementations")
    print("Simulating high-frequency trading data (100ms intervals)")
    print("")
    
    # Check package availability
    available_packages = []
    if PYEMD_AVAILABLE:
        available_packages.append("PyEMD")
    if PYHHT_AVAILABLE:
        available_packages.append("PyHHT")
    if SCIPY_AVAILABLE:
        available_packages.append("SciPy")
    
    print(f"Available packages: {', '.join(available_packages)}")
    
    if not available_packages:
        print("❌ No EMD packages available. Please install PyEMD and/or PyHHT:")
        print("pip install PyEMD")
        print("pip install PyHHT")
        return
    
    # Run benchmark
    benchmark = EMDBenchmark()
    
    # Test different signal lengths (realistic for HFT)
    signal_lengths = [100, 1000, 10000]  # 10s, 1.7min, 16.7min @ 100ms intervals
    
    try:
        # Run comprehensive benchmark
        df_results = benchmark.run_comprehensive_benchmark(
            signal_lengths=signal_lengths,
            num_trials=3
        )
        
        # Create visualizations
        benchmark.create_visualizations(df_results)
        
        # Generate performance report
        benchmark.generate_performance_report(df_results)
        
        # Demonstrate pandas integration
        demonstrate_pandas_integration()
        
        print("\n✅ Benchmark suite completed successfully!")
        print(f"📁 Results saved in: {benchmark.output_dir}")
        
    except Exception as e:
        print(f"\n❌ Benchmark failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()