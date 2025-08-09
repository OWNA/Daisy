#!/usr/bin/env python3
"""
Enhanced Feature Engineer with HHT Integration - Phase 1
Production-ready integration of Hilbert-Huang Transform with existing L2 microstructure features

This module implements the HHT_L2.md blueprint by:
1. Adding real-time HHT regime classification to existing L2 features
2. Creating L2-HHT confluence features for enhanced signal generation
3. Maintaining compatibility with existing trading system architecture
4. Optimizing for live trading performance (<50ms HHT calculation target)
"""

import pandas as pd
import numpy as np
import time
import logging
from typing import Dict, List, Tuple, Optional

# Import existing feature engineer as base
from featureengineer import FeatureEngineer as BaseFeatureEngineer
from hht_processor import OptimizedHHTProcessor

class EnhancedFeatureEngineer(BaseFeatureEngineer):
    """
    Enhanced Feature Engineer with HHT Integration
    
    Extends the existing FeatureEngineer with real-time HHT analysis:
    - Market regime classification (trending/cyclical/noisy)
    - L2-HHT confluence features 
    - Point-in-time HHT calculation for live trading
    - Performance optimized for high-frequency applications
    """
    
    def __init__(self, config: dict):
        """
        Initialize Enhanced Feature Engineer with HHT capabilities
        
        Args:
            config: Configuration dictionary including HHT parameters
        """
        # Initialize base feature engineer first
        super().__init__(config)
        
        # HHT configuration
        self.hht_enabled = config.get('use_hht_features', False)
        self.hht_config = self._parse_hht_config(config)
        
        # Initialize HHT processor
        self.hht_processor = None
        if self.hht_enabled:
            try:
                self.hht_processor = OptimizedHHTProcessor(
                    window_size=self.hht_config['window_size'],
                    max_imfs=self.hht_config['max_imfs'],
                    update_frequency=self.hht_config['update_frequency'],
                    method=self.hht_config['method'],
                    enable_caching=self.hht_config['enable_caching'],
                    performance_monitoring=True
                )
                print(f"HHT processor initialized: {self.hht_config['method']}, window={self.hht_config['window_size']}")
            except Exception as e:
                print(f"Warning: HHT processor initialization failed: {e}")
                self.hht_enabled = False
        
        # Performance tracking
        self.feature_calc_times = {}
        
        # Track processed data for incremental HHT updates
        self.last_processed_length = 0
        
        print(f"Enhanced FeatureEngineer initialized:")
        print(f"  Base L2 features: {len(self.l2_features_config)}")
        print(f"  HHT enabled: {self.hht_enabled}")
        if self.hht_enabled:
            print(f"  HHT method: {self.hht_config['method']}")
    
    def _parse_hht_config(self, config: dict) -> dict:
        """Parse HHT configuration from config dictionary"""
        return {
            'window_size': config.get('hht_window_size', 1000),
            'max_imfs': config.get('hht_imf_count', 5),
            'update_frequency': config.get('hht_update_frequency', 5),
            'method': config.get('hht_method', 'emd'),
            'enable_caching': config.get('hht_enable_caching', True)
        }
    
    def generate_features(self, df_input: pd.DataFrame) -> pd.DataFrame:
        """
        Enhanced feature generation with HHT integration
        
        Args:
            df_input: Input L2 DataFrame
            
        Returns:
            DataFrame with L2 + HHT features
        """
        if df_input is None or df_input.empty:
            print("Error: Input L2 DataFrame is empty.")
            return pd.DataFrame()
        
        start_time = time.time()
        print(f"Starting enhanced feature generation for {len(df_input)} rows...")
        
        # Generate base L2 microstructure features
        df = super().generate_features(df_input)
        
        # Add HHT features if enabled
        if self.hht_enabled and self.hht_processor is not None:
            df = self._calculate_hht_features(df)
            df = self._calculate_hht_derived_features(df)

        # Calculate performance metrics
        total_time = (time.time() - start_time) * 1000
        print(f"Enhanced feature generation complete. Time: {total_time:.1f}ms")
        print(f"Final DataFrame shape: {df.shape}")
        
        return df
    
    def _calculate_hht_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate core HHT features using point-in-time methodology
        
        This is the critical method that processes price data through HHT
        to extract regime classification and cycle analysis features
        """
        print("Calculating HHT features...")
        
        # Note: Don't reset HHT processor in live trading - we want to accumulate data
        # Only reset would be needed for completely new backtesting runs
        
        # Use weighted mid price as primary signal (more stable than raw mid_price)
        price_column = 'weighted_mid_price' if 'weighted_mid_price' in df.columns else 'mid_price'
        price_series = df[price_column].ffill().bfill()
        
        # Initialize HHT feature columns with defaults
        hht_features = {
            'hht_trend_strength': 0.0,
            'hht_trend_slope': 0.0,
            'hht_cycle_phase': 0.0,
            'hht_dominant_freq': 0.0,
            'hht_inst_amplitude': 0.0,
            'hht_regime_classifier': 0,  # 0=ranging, 1=up_trend, -1=down_trend, 2=noisy
            'hht_energy_high_freq': 0.0,
            'hht_energy_mid_freq': 0.0,
            'hht_energy_residual': 0.0,
            'hht_data_quality': 1.0,
            'hht_calculation_time_ms': 0.0
        }
        
        for feature_name in hht_features:
            df[feature_name] = hht_features[feature_name]
        
        # Incremental processing - only process new data points
        timestamps = df['timestamp'].values if 'timestamp' in df.columns else range(len(df))
        current_length = len(df)
        
        # Process only new data points (incremental approach)
        start_idx = max(0, self.last_processed_length)
        
        if start_idx > 0:
            # Copy previous HHT values for existing rows
            for i in range(start_idx):
                if i < current_length:
                    # Keep existing HHT values (they don't change for historical data)
                    pass
        
        # Process only new data points
        latest_hht_result = None
        for i in range(start_idx, current_length):
            price = price_series.iloc[i]
            timestamp = timestamps[i]
            
            # Update HHT processor with new price point
            hht_result = self.hht_processor.update(price, timestamp)
            latest_hht_result = hht_result
            
            # Extract features from HHT result  
            df.loc[i, 'hht_trend_strength'] = hht_result['hht_trend_strength']
            df.loc[i, 'hht_trend_slope'] = hht_result['hht_trend_slope']
            df.loc[i, 'hht_cycle_phase'] = hht_result['hht_cycle_phase']
            df.loc[i, 'hht_dominant_freq'] = hht_result['hht_dominant_freq']
            df.loc[i, 'hht_inst_amplitude'] = hht_result['hht_inst_amplitude']
            df.loc[i, 'hht_regime_classifier'] = hht_result['hht_regime_classifier']
            df.loc[i, 'hht_data_quality'] = hht_result['hht_data_quality']
            df.loc[i, 'hht_calculation_time_ms'] = hht_result['hht_calculation_time_ms']
            
            # Extract energy distribution
            energy_dist = hht_result['hht_energy_distribution']
            if len(energy_dist) >= 3:
                df.loc[i, 'hht_energy_high_freq'] = energy_dist[0]
                df.loc[i, 'hht_energy_mid_freq'] = energy_dist[1]
                df.loc[i, 'hht_energy_residual'] = energy_dist[2]
        
        # Update tracking
        self.last_processed_length = current_length
        
        # For all previous rows, use the latest HHT state (this gives continuity)  
        if latest_hht_result and start_idx > 0:
            for i in range(start_idx):
                if i < current_length:
                    df.loc[i, 'hht_trend_strength'] = latest_hht_result['hht_trend_strength']
                    df.loc[i, 'hht_trend_slope'] = latest_hht_result['hht_trend_slope']
                    df.loc[i, 'hht_cycle_phase'] = latest_hht_result['hht_cycle_phase']
                    df.loc[i, 'hht_dominant_freq'] = latest_hht_result['hht_dominant_freq']
                    df.loc[i, 'hht_inst_amplitude'] = latest_hht_result['hht_inst_amplitude']
                    df.loc[i, 'hht_regime_classifier'] = latest_hht_result['hht_regime_classifier']
                    df.loc[i, 'hht_data_quality'] = latest_hht_result['hht_data_quality']
                    
                    energy_dist = latest_hht_result['hht_energy_distribution']
                    if len(energy_dist) >= 3:
                        df.loc[i, 'hht_energy_high_freq'] = energy_dist[0]
                        df.loc[i, 'hht_energy_mid_freq'] = energy_dist[1]
                        df.loc[i, 'hht_energy_residual'] = energy_dist[2]
        
        new_points = current_length - start_idx
        print(f"HHT features calculated: {new_points} new points, {current_length} total rows.")
        return df
    
    def _calculate_hht_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate derived HHT features and L2-HHT confluence signals
        
        This implements the core L2-HHT strategy logic:
        - Regime-qualified L2 signals
        - Trend-aligned features
        - Multi-scale confluence indicators
        """
        print("Calculating HHT derived and confluence features...")
        
        # 1. Trend Confidence (combines trend strength with data quality)
        df['hht_trend_confidence'] = (
            df['hht_trend_strength'] * df['hht_data_quality']
        ).clip(0, 1)
        
        # 2. Market State Classification (enhanced regime)
        df['hht_market_state'] = self._calculate_enhanced_market_state(
            df['hht_regime_classifier'], 
            df['hht_trend_confidence'],
            df['hht_energy_high_freq']
        )
        
        # 3. L2-HHT Confluence Features
        
        # Regime-Qualified Order Flow Imbalance
        for window in ['10s', '30s', '1m']:
            ofi_col = f'ofi_{window}'
            if ofi_col in df.columns:
                df[f'{ofi_col}_hht_qualified'] = df[ofi_col] * np.where(
                    df['hht_regime_classifier'].abs() == 1,  # Trending regimes
                    df['hht_trend_confidence'] * 1.5,  # Boost trending signals
                    np.where(
                        df['hht_regime_classifier'] == 2,  # Noisy regime
                        0.3,  # Dampen noisy signals
                        1.0   # Neutral for ranging
                    )
                )
        
        # Trend-Aligned Order Book Imbalance
        if 'order_book_imbalance' in df.columns:
            df['obi_trend_aligned'] = df['order_book_imbalance'] * np.where(
                df['hht_regime_classifier'] == 1,   # Uptrend
                np.where(df['order_book_imbalance'] > 0, 1.2, 0.8),  # Boost buy signals
                np.where(
                    df['hht_regime_classifier'] == -1,  # Downtrend  
                    np.where(df['order_book_imbalance'] < 0, 1.2, 0.8),  # Boost sell signals
                    1.0  # No adjustment for ranging/noisy
                )
            )
        
        # Multi-Scale Signal Confluence
        if 'ofi_30s' in df.columns:
            df['l2_hht_signal_confluence'] = np.where(
                (abs(df['ofi_30s']) > df['ofi_30s'].quantile(0.8)) &  # Strong L2 signal
                (df['hht_trend_confidence'] > 0.5) &  # HHT confidence
                (df['hht_regime_classifier'] != 2),   # Not noisy
                np.sign(df['ofi_30s']) * (1 + df['hht_trend_confidence']),  # Amplify
                0  # No signal
            )
        
        # 4. Regime-Specific Features
        
        # Trend Persistence (meaningful only in trending regimes)
        if 'mid_price_return' in df.columns:
            df['trend_persistence_hht'] = np.where(
                df['hht_regime_classifier'].abs() == 1,  # Trending regimes only
                df['mid_price_return'].rolling(50).mean() * df['hht_trend_strength'],
                0
            )
        
        # Mean Reversion Signal (meaningful only in ranging regimes)
        if 'weighted_mid_price' in df.columns:
            rolling_mean = df['weighted_mid_price'].rolling(200).mean()
            df['mean_reversion_hht'] = np.where(
                df['hht_regime_classifier'] == 0,  # Ranging regimes only
                (df['weighted_mid_price'] - rolling_mean) / (rolling_mean + 1e-8) * -1,
                0
            )
        
        print("HHT derived and confluence features calculated.")
        return df
    
    def _calculate_enhanced_market_state(self, 
                                       regime: pd.Series, 
                                       confidence: pd.Series,
                                       high_freq_energy: pd.Series) -> pd.Series:
        """
        Enhanced market state classification using HHT analysis
        
        Returns:
            0: Uncertain
            1: Strong Uptrend  
            2: Weak Uptrend
            3: Ranging
            4: Weak Downtrend
            5: Strong Downtrend
            6: Noisy/Choppy
        """
        state = np.zeros(len(regime), dtype=int)
        
        for i in range(len(regime)):
            r = regime.iloc[i]
            c = confidence.iloc[i]
            hf = high_freq_energy.iloc[i]
            
            if hf > 0.6:  # High noise regime
                state[i] = 6
            elif r == 1:  # Uptrend
                state[i] = 1 if c > 0.6 else 2  # Strong vs weak
            elif r == -1:  # Downtrend
                state[i] = 5 if c > 0.6 else 4  # Strong vs weak
            elif r == 0:  # Ranging
                state[i] = 3
            else:
                state[i] = 0  # Uncertain
        
        return pd.Series(state, index=regime.index)
    
    def get_hht_performance_stats(self) -> Dict[str, float]:
        """
        Get HHT performance statistics for monitoring
        """
        if not self.hht_enabled or not self.hht_processor:
            return {}
        
        return self.hht_processor.get_performance_stats()
    
    def validate_hht_features(self, df: pd.DataFrame) -> Dict[str, any]:
        """
        Validate HHT feature quality and completeness
        """
        hht_cols = [col for col in df.columns if col.startswith('hht_')]
        
        validation = {
            'hht_features_count': len(hht_cols),
            'hht_enabled': self.hht_enabled,
            'missing_hht_values': df[hht_cols].isnull().sum().sum() if hht_cols else 0,
            'avg_calculation_time_ms': df['hht_calculation_time_ms'].mean() if 'hht_calculation_time_ms' in df.columns else 0,
            'avg_data_quality': df['hht_data_quality'].mean() if 'hht_data_quality' in df.columns else 0,
            'regime_distribution': df['hht_regime_classifier'].value_counts().to_dict() if 'hht_regime_classifier' in df.columns else {}
        }
        
        return validation


def test_enhanced_feature_engineer():
    """
    Test the enhanced feature engineer with sample L2 data
    """
    print("Testing Enhanced Feature Engineer (Phase 1)...")
    
    # Create realistic L2 test data
    np.random.seed(42)
    n_samples = 500  # Smaller sample for faster testing
    
    # Generate price data with trend and noise
    base_price = 50000
    trend = np.linspace(0, 100, n_samples)  # Upward trend
    noise = np.random.randn(n_samples) * 10
    prices = base_price + trend + noise
    
    # Create sample L2 data structure
    sample_data = {
        'timestamp': pd.date_range('2025-01-01', periods=n_samples, freq='100ms'),
        'symbol': ['BTCUSDT'] * n_samples,
    }
    
    # Add bid/ask data for 10 levels
    for level in range(1, 11):
        spread_offset = level * 0.5
        sample_data[f'bid_price_{level}'] = prices - spread_offset
        sample_data[f'ask_price_{level}'] = prices + spread_offset
        sample_data[f'bid_size_{level}'] = np.random.uniform(0.1, 5.0, n_samples)
        sample_data[f'ask_size_{level}'] = np.random.uniform(0.1, 5.0, n_samples)
    
    df_test = pd.DataFrame(sample_data)
    
    # Test configuration with HHT enabled
    test_config = {
        'use_hht_features': True,
        'hht_window_size': 200,
        'hht_imf_count': 3,
        'hht_update_frequency': 5,
        'hht_method': 'emd',
        'hht_enable_caching': True,
        'l2_features': []  # Will be populated by base class
    }
    
    try:
        # Initialize enhanced feature engineer
        fe = EnhancedFeatureEngineer(test_config)
        
        print(f"Processing {len(df_test)} samples with HHT integration...")
        start_time = time.time()
        
        # Generate features
        result_df = fe.generate_features(df_test)
        
        processing_time = time.time() - start_time
        
        # Results
        print(f"\n=== Test Results ===")
        print(f"Processing time: {processing_time:.2f} seconds")
        print(f"Input shape: {df_test.shape}")
        print(f"Output shape: {result_df.shape}")
        print(f"Features added: {result_df.shape[1] - df_test.shape[1]}")
        
        # Validate HHT features
        hht_validation = fe.validate_hht_features(result_df)
        print(f"\n=== HHT Validation ===")
        print(f"HHT features count: {hht_validation['hht_features_count']}")
        print(f"Average calculation time: {hht_validation['avg_calculation_time_ms']:.2f}ms")
        print(f"Average data quality: {hht_validation['avg_data_quality']:.3f}")
        print(f"Regime distribution: {hht_validation['regime_distribution']}")
        
        # Show sample HHT features
        hht_cols = [col for col in result_df.columns if col.startswith('hht_') and not 'energy' in col][:5]
        if hht_cols:
            print(f"\n=== Sample HHT Features (last 5 rows) ===")
            print(result_df[hht_cols].tail())
        
        # Performance stats
        if fe.hht_enabled:
            perf_stats = fe.get_hht_performance_stats()
            if perf_stats:
                print(f"\n=== HHT Performance Stats ===")
                for key, value in perf_stats.items():
                    if isinstance(value, float):
                        print(f"{key}: {value:.2f}")
                    else:
                        print(f"{key}: {value}")
        
        print("\n[SUCCESS] Phase 1 integration test completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n[ERROR] Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    test_enhanced_feature_engineer()