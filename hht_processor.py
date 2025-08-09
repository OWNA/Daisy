#!/usr/bin/env python3
"""
Hilbert-Huang Transform (HHT) Processor for High-Frequency Trading
Optimized for real-time L2 order book analysis

Based on the HHT_L2.md blueprint requirements:
- Real-time EMD decomposition for regime classification
- Computational performance optimized for <50ms execution
- Point-in-time calculation to avoid look-ahead bias
- Integration with existing L2 microstructure features
"""

import numpy as np
import pandas as pd
import time
import collections
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime
import warnings
from numba import jit
import logging

# Suppress EMD warnings for cleaner output
warnings.filterwarnings('ignore', category=RuntimeWarning)

try:
    from PyEMD import EMD, EEMD, CEEMDAN
    HHT_AVAILABLE = True
except ImportError:
    HHT_AVAILABLE = False
    print("Warning: PyEMD not installed. Install with: pip install EMD-signal")

class OptimizedHHTProcessor:
    """
    High-performance HHT processor optimized for live trading applications
    
    Features:
    - Real-time EMD decomposition with <50ms target latency
    - Rolling window processing to avoid look-ahead bias
    - Intelligent caching to reduce computational overhead
    - Market regime classification (trending/cyclical/noisy)
    - Integration with L2 microstructure features
    """
    
    def __init__(self, 
                 window_size: int = 1000,
                 max_imfs: int = 5,
                 update_frequency: int = 5,
                 method: str = 'emd',
                 enable_caching: bool = True,
                 performance_monitoring: bool = True):
        """
        Initialize HHT processor
        
        Args:
            window_size: Rolling window size (default 1000 = ~10 seconds at 100ms sampling)
            max_imfs: Maximum number of IMFs to extract (balance insight vs speed)
            update_frequency: Update HHT every N ticks (default 5 = every 500ms)
            method: 'emd', 'eemd', or 'ceemdan' (emd recommended for real-time)
            enable_caching: Use intelligent caching to reduce computation
            performance_monitoring: Track execution times and log warnings
        """
        if not HHT_AVAILABLE:
            raise ImportError("PyEMD not available. Install with: pip install EMD-signal")
        
        self.window_size = window_size
        self.max_imfs = max_imfs
        self.update_frequency = update_frequency
        self.method = method.lower()
        self.enable_caching = enable_caching
        self.performance_monitoring = performance_monitoring
        
        # Initialize EMD processor based on method
        self._init_emd_processor()
        
        # Rolling data buffers
        self.price_buffer = collections.deque(maxlen=window_size)
        self.timestamp_buffer = collections.deque(maxlen=window_size)
        
        # Caching and state management
        self.tick_count = 0
        self.last_update_tick = 0
        self.cached_imfs = None
        self.cached_features = {}
        
        # Performance monitoring
        self.execution_times = collections.deque(maxlen=100)
        self.total_calculations = 0
        
        # Feature storage for current values
        self.current_features = {
            'hht_trend_strength': 0.0,
            'hht_cycle_phase': 0.0,
            'hht_dominant_freq': 0.0,
            'hht_inst_amplitude': 0.0,
            'hht_regime_classifier': 0,  # 1=trending_up, -1=trending_down, 0=ranging, 2=noisy
            'hht_trend_slope': 0.0,
            'hht_energy_distribution': [0.0, 0.0, 0.0],  # [high_freq, mid_freq, residual]
            'hht_calculation_time_ms': 0.0,
            'hht_data_quality': 1.0
        }
        
        logging.info(f"HHTProcessor initialized: method={method}, window={window_size}, max_imfs={max_imfs}")
    
    def _init_emd_processor(self):
        """Initialize the appropriate EMD processor"""
        if self.method == 'emd':
            self.emd_processor = EMD(max_imf=self.max_imfs)
        elif self.method == 'eemd':
            # EEMD with reduced trials for better performance
            self.emd_processor = EEMD(trials=50, max_imf=self.max_imfs)
        elif self.method == 'ceemdan':
            # CEEMDAN - high quality but slow (research only)
            self.emd_processor = CEEMDAN(trials=50, max_imf=self.max_imfs)
        else:
            raise ValueError(f"Unknown EMD method: {self.method}")
    
    def update(self, 
               price: float, 
               timestamp: Optional[datetime] = None) -> Dict[str, Union[float, int, List[float]]]:
        """
        Update HHT features with new price data
        
        Args:
            price: New price observation (WAP, mid-price, etc.)
            timestamp: Optional timestamp for the observation
            
        Returns:
            Dictionary with current HHT features
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        # Add new data to rolling buffers
        self.price_buffer.append(float(price))
        self.timestamp_buffer.append(timestamp)
        self.tick_count += 1
        
        # Decide whether to recalculate HHT
        should_update = self._should_update_hht()
        
        if should_update and len(self.price_buffer) >= 100:  # Minimum data for HHT
            self._calculate_hht_features()
            self.last_update_tick = self.tick_count
        
        return self.current_features.copy()
    
    def _should_update_hht(self) -> bool:
        """
        Intelligent update logic to balance accuracy and performance
        """
        if not self.enable_caching:
            return True
        
        # Always update on first calculation
        if self.cached_imfs is None:
            return True
        
        # Update based on frequency
        ticks_since_update = self.tick_count - self.last_update_tick
        if ticks_since_update >= self.update_frequency:
            return True
        
        # Force update if buffer is full (new data regime)
        if len(self.price_buffer) >= self.window_size:
            return True
        
        return False
    
    def _calculate_hht_features(self):
        """
        Calculate HHT features using current price buffer
        Point-in-time calculation to avoid look-ahead bias
        """
        if len(self.price_buffer) < 50:
            return
        
        start_time = time.time()
        
        try:
            # Convert buffer to numpy array for processing
            price_data = np.array(list(self.price_buffer), dtype=np.float32)
            
            # Preprocess data to reduce noise and improve EMD stability
            price_data = self._preprocess_price_data(price_data)
            
            # Perform EMD decomposition
            imfs = self.emd_processor(price_data)
            
            # Cache results
            self.cached_imfs = imfs
            self.total_calculations += 1
            
            # Extract features from IMFs
            self._extract_features_from_imfs(imfs)
            
            # Calculate execution time
            execution_time = (time.time() - start_time) * 1000
            self.execution_times.append(execution_time)
            self.current_features['hht_calculation_time_ms'] = execution_time
            
            if self.performance_monitoring and execution_time > 50:
                logging.warning(f"HHT calculation took {execution_time:.1f}ms (target <50ms)")
            
            # Update data quality score
            self.current_features['hht_data_quality'] = self._assess_data_quality(imfs)
            
        except Exception as e:
            logging.error(f"HHT calculation failed: {e}")
            # Keep previous values on failure
            self.current_features['hht_data_quality'] = 0.5  # Mark as degraded
    
    @staticmethod
    @jit(nopython=True)
    def _preprocess_price_data(data: np.ndarray) -> np.ndarray:
        """
        JIT-compiled preprocessing for improved performance
        
        Applies light smoothing to reduce high-frequency noise
        while preserving meaningful price movements
        """
        if len(data) < 3:
            return data
        
        # Simple 3-point moving average to reduce noise
        smoothed = np.zeros_like(data)
        smoothed[0] = data[0]
        
        for i in range(1, len(data) - 1):
            smoothed[i] = (data[i-1] + data[i] + data[i+1]) / 3.0
        
        smoothed[-1] = data[-1]
        return smoothed
    
    def _extract_features_from_imfs(self, imfs: np.ndarray):
        """
        Extract trading-relevant features from IMF decomposition
        
        Features align with HHT_L2.md blueprint requirements:
        - Trend strength and direction
        - Cycle phase and frequency
        - Market regime classification
        - Energy distribution analysis
        """
        n_imfs = len(imfs)
        if n_imfs < 2:
            return
        
        # Feature 1: Trend Strength and Direction
        trend_strength, trend_slope = self._calculate_trend_metrics(imfs)
        self.current_features['hht_trend_strength'] = trend_strength
        self.current_features['hht_trend_slope'] = trend_slope
        
        # Feature 2: Dominant Cycle Analysis
        cycle_phase, dominant_freq = self._calculate_cycle_metrics(imfs)
        self.current_features['hht_cycle_phase'] = cycle_phase
        self.current_features['hht_dominant_freq'] = dominant_freq
        
        # Feature 3: Instantaneous Amplitude
        inst_amplitude = self._calculate_instantaneous_amplitude(imfs)
        self.current_features['hht_inst_amplitude'] = inst_amplitude
        
        # Feature 4: Energy Distribution
        energy_dist = self._calculate_energy_distribution(imfs)
        self.current_features['hht_energy_distribution'] = energy_dist
        
        # Feature 5: Market Regime Classification
        regime = self._classify_market_regime(imfs, trend_strength, trend_slope, energy_dist)
        self.current_features['hht_regime_classifier'] = regime
    
    def _calculate_trend_metrics(self, imfs: np.ndarray) -> Tuple[float, float]:
        """
        Calculate trend strength and slope from residual component
        
        Returns:
            (trend_strength, trend_slope)
        """
        if len(imfs) < 2:
            return 0.0, 0.0
        
        residual = imfs[-1]  # Trend component is the final residual
        
        # Calculate trend strength as relative energy in residual
        residual_energy = np.var(residual)
        total_energy = sum(np.var(imf) for imf in imfs)
        trend_strength = residual_energy / (total_energy + 1e-8)
        
        # Calculate trend slope using linear regression
        if len(residual) >= 10:
            x = np.arange(len(residual))
            slope = np.polyfit(x, residual, 1)[0]
            # Normalize slope by price scale
            mean_price = np.mean(residual)
            trend_slope = slope / (abs(mean_price) + 1e-8)
        else:
            trend_slope = 0.0
        
        return float(trend_strength), float(trend_slope)
    
    def _calculate_cycle_metrics(self, imfs: np.ndarray) -> Tuple[float, float]:
        """
        Calculate cycle phase and dominant frequency
        
        Returns:
            (cycle_phase, dominant_frequency)
        """
        if len(imfs) < 2:
            return 0.0, 0.0
        
        # Use second IMF as primary cycle (first is usually noise)
        cycle_imf = imfs[1] if len(imfs) > 1 else imfs[0]
        
        try:
            # Calculate instantaneous phase using Hilbert transform
            from scipy.signal import hilbert
            analytic_signal = hilbert(cycle_imf)
            instantaneous_phase = np.angle(analytic_signal)
            
            # Current phase (normalized to -1, 1)
            current_phase = np.sin(instantaneous_phase[-1])
            
            # Estimate dominant frequency from phase differences
            if len(instantaneous_phase) > 10:
                phase_diff = np.diff(np.unwrap(instantaneous_phase))
                dominant_freq = np.mean(phase_diff) / (2 * np.pi)
            else:
                dominant_freq = 0.0
                
        except ImportError:
            # Fallback if scipy not available
            current_phase = 0.0
            dominant_freq = 0.0
        
        return float(current_phase), float(abs(dominant_freq))
    
    def _calculate_instantaneous_amplitude(self, imfs: np.ndarray) -> float:
        """
        Calculate instantaneous amplitude of dominant cycle
        """
        if len(imfs) < 2:
            return 0.0
        
        # Use the IMF with highest energy as dominant component
        energies = [np.var(imf) for imf in imfs[:-1]]  # Exclude residual
        if not energies:
            return 0.0
        
        dominant_idx = np.argmax(energies)
        dominant_imf = imfs[dominant_idx]
        
        # Simple amplitude estimate as recent RMS
        recent_window = min(50, len(dominant_imf))
        recent_data = dominant_imf[-recent_window:]
        amplitude = np.sqrt(np.mean(recent_data**2))
        
        return float(amplitude)
    
    def _calculate_energy_distribution(self, imfs: np.ndarray) -> List[float]:
        """
        Calculate energy distribution across frequency bands
        
        Returns:
            [high_freq_energy, mid_freq_energy, residual_energy] (normalized)
        """
        if len(imfs) < 2:
            return [0.0, 0.0, 1.0]
        
        # Calculate energy for each component
        energies = [np.var(imf) for imf in imfs]
        total_energy = sum(energies) + 1e-8
        
        # Categorize by frequency (index position)
        high_freq_energy = energies[0] / total_energy if len(energies) > 0 else 0.0
        
        mid_freq_energy = 0.0
        if len(energies) > 2:
            mid_freq_energy = sum(energies[1:-1]) / total_energy
        elif len(energies) == 2:
            mid_freq_energy = 0.0
        
        residual_energy = energies[-1] / total_energy if len(energies) > 1 else 0.0
        
        return [float(high_freq_energy), float(mid_freq_energy), float(residual_energy)]
    
    def _classify_market_regime(self, 
                               imfs: np.ndarray,
                               trend_strength: float,
                               trend_slope: float,
                               energy_dist: List[float]) -> int:
        """
        Classify market regime based on HHT analysis
        
        Returns:
            1: Trending Up
            -1: Trending Down  
            0: Ranging/Cyclical
            2: Noisy/Choppy
        """
        high_freq_energy, mid_freq_energy, residual_energy = energy_dist
        
        # Regime classification logic based on energy distribution
        
        # High noise regime
        if high_freq_energy > 0.6:
            return 2
        
        # Trending regime (strong residual energy + significant slope)
        if residual_energy > 0.4 and trend_strength > 0.3:
            if abs(trend_slope) > 0.0001:  # Significant slope threshold
                return 1 if trend_slope > 0 else -1
            else:
                return 0  # Flat trend = ranging
        
        # Cyclical regime (dominant mid-frequency energy)
        if mid_freq_energy > 0.4:
            return 0
        
        # Default to ranging if unclear
        return 0
    
    def _assess_data_quality(self, imfs: np.ndarray) -> float:
        """
        Assess the quality of HHT decomposition
        
        Returns:
            Quality score between 0.0 (poor) and 1.0 (excellent)
        """
        if len(imfs) < 2:
            return 0.3
        
        # Quality factors:
        # 1. Number of IMFs (more IMFs generally better, up to a point)
        imf_count_score = min(len(imfs) / 5.0, 1.0)
        
        # 2. Energy concentration (avoid over-decomposition)
        energies = [np.var(imf) for imf in imfs]
        total_energy = sum(energies)
        max_energy_ratio = max(energies) / (total_energy + 1e-8)
        concentration_score = 1.0 - max_energy_ratio  # Penalize over-concentration
        
        # 3. Reconstruction quality (should be perfect for EMD)
        reconstruction_score = 1.0  # EMD guarantees perfect reconstruction
        
        # Combine scores
        quality_score = (imf_count_score + concentration_score + reconstruction_score) / 3.0
        return float(np.clip(quality_score, 0.0, 1.0))
    
    def get_performance_stats(self) -> Dict[str, float]:
        """
        Get performance statistics for monitoring
        """
        if not self.execution_times:
            return {}
        
        times = list(self.execution_times)
        return {
            'avg_execution_time_ms': np.mean(times),
            'max_execution_time_ms': np.max(times),
            'min_execution_time_ms': np.min(times),
            'total_calculations': self.total_calculations,
            'buffer_size': len(self.price_buffer),
            'update_frequency_actual': self.tick_count / (self.total_calculations + 1)
        }
    
    def reset(self):
        """Reset all buffers and cached data"""
        self.price_buffer.clear()
        self.timestamp_buffer.clear()
        self.cached_imfs = None
        self.cached_features.clear()
        self.tick_count = 0
        self.last_update_tick = 0
        
        # Reset features to defaults
        for key in self.current_features:
            if isinstance(self.current_features[key], (int, float)):
                self.current_features[key] = 0.0
            elif isinstance(self.current_features[key], list):
                self.current_features[key] = [0.0] * len(self.current_features[key])
        
        logging.info("HHT processor reset")


def test_hht_processor():
    """Test function for HHT processor"""
    import matplotlib.pyplot as plt
    
    print("Testing HHT Processor...")
    
    # Generate synthetic price data with trend and cycle
    np.random.seed(42)
    t = np.linspace(0, 100, 1000)
    trend = 0.01 * t  # Linear trend
    cycle = 0.5 * np.sin(0.1 * t)  # Slow cycle
    noise = 0.1 * np.random.randn(len(t))  # Noise
    price_data = 100 + trend + cycle + noise
    
    # Initialize processor
    processor = OptimizedHHTProcessor(
        window_size=500,
        update_frequency=10,
        method='emd',
        performance_monitoring=True
    )
    
    # Process data and collect features
    features_history = []
    execution_times = []
    
    for i, price in enumerate(price_data):
        start_time = time.time()
        features = processor.update(price)
        exec_time = (time.time() - start_time) * 1000
        execution_times.append(exec_time)
        
        features_history.append({
            'price': price,
            'trend_strength': features['hht_trend_strength'],
            'regime': features['hht_regime_classifier'],
            'cycle_phase': features['hht_cycle_phase'],
            'inst_amplitude': features['hht_inst_amplitude']
        })
        
        if i % 100 == 0:
            print(f"Processed {i+1}/1000 samples...")
    
    # Analysis
    df = pd.DataFrame(features_history)
    
    print(f"\nPerformance Statistics:")
    print(f"Average execution time: {np.mean(execution_times):.2f}ms")
    print(f"Max execution time: {np.max(execution_times):.2f}ms")
    print(f"95th percentile: {np.percentile(execution_times, 95):.2f}ms")
    
    print(f"\nFeature Statistics:")
    print(f"Trend strength range: {df['trend_strength'].min():.3f} to {df['trend_strength'].max():.3f}")
    print(f"Regime distribution: {df['regime'].value_counts().to_dict()}")
    
    performance_stats = processor.get_performance_stats()
    print(f"\nProcessor Stats: {performance_stats}")


if __name__ == "__main__":
    test_hht_processor()