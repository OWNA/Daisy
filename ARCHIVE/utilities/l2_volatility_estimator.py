#!/usr/bin/env python3
"""
Module: l2_volatility_estimator.py
Description: L2-based volatility estimation for high-frequency trading
Author: L2-Only Strategy Implementation
Date: 2025-01-27
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple
import logging
from datetime import datetime


class L2VolatilityEstimator:
    """
    Advanced volatility estimation using L2 order book data.
    
    Features:
    - Realized volatility from tick-by-tick price changes
    - Order book imbalance-based volatility
    - Microstructure noise estimation
    - Multi-timeframe volatility (1min, 5min, 15min)
    - Volatility clustering detection
    - Risk-adjusted volatility metrics
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize L2 volatility estimator.
        
        Args:
            config: Configuration dictionary with volatility parameters
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Volatility estimation parameters
        self.sampling_frequency_ms = config.get('l2_sampling_frequency_ms', 100)
        self.volatility_windows = config.get('volatility_windows', {
            '1min': 600,    # 600 samples at 100ms = 1 minute
            '5min': 3000,   # 3000 samples at 100ms = 5 minutes
            '15min': 9000   # 9000 samples at 100ms = 15 minutes
        })
        
        # Microstructure parameters
        self.noise_threshold = config.get('microstructure_noise_threshold', 0.001)
        self.min_price_change = config.get('min_price_change', 1e-8)
        self.outlier_threshold = config.get('volatility_outlier_threshold', 5.0)
        
        # Volatility model parameters
        self.decay_factor = config.get('volatility_decay_factor', 0.94)
        self.min_observations = config.get('min_volatility_observations', 10)
        
        self.logger.info("L2VolatilityEstimator initialized")
        self.logger.info(f"Sampling frequency: {self.sampling_frequency_ms}ms")
        self.logger.info(f"Volatility windows: {self.volatility_windows}")

    def estimate_realized_volatility(self, df: pd.DataFrame, 
                                   price_col: str = 'reconstructed_price',
                                   window: str = '1min') -> pd.Series:
        """
        Calculate realized volatility from high-frequency price data.
        
        Args:
            df: DataFrame with price data
            price_col: Column name for price data
            window: Volatility window ('1min', '5min', '15min')
            
        Returns:
            Series with realized volatility estimates
        """
        if price_col not in df.columns:
            self.logger.warning(f"Price column '{price_col}' not found")
            return pd.Series(index=df.index, dtype=float)
        
        # Get window size
        window_size = self.volatility_windows.get(window, 600)
        min_periods = max(self.min_observations, window_size // 10)
        
        # Calculate log returns
        prices = df[price_col].dropna()
        if len(prices) < 2:
            return pd.Series(index=df.index, dtype=float)
        
        log_returns = np.log(prices / prices.shift(1)).dropna()
        
        # Remove outliers
        return_std = log_returns.std()
        outlier_mask = np.abs(log_returns) > (self.outlier_threshold * return_std)
        clean_returns = log_returns[~outlier_mask]
        
        # Calculate realized volatility (sum of squared returns)
        squared_returns = clean_returns ** 2
        
        # Rolling realized volatility
        realized_vol = squared_returns.rolling(
            window=window_size, 
            min_periods=min_periods
        ).sum().apply(np.sqrt)
        
        # Annualize to the specified timeframe
        samples_per_period = window_size
        realized_vol = realized_vol * np.sqrt(samples_per_period)
        
        # Reindex to original DataFrame
        result = pd.Series(index=df.index, dtype=float)
        result.loc[realized_vol.index] = realized_vol
        
        return result.fillna(method='ffill').fillna(0.01)

    def estimate_microstructure_volatility(self, df: pd.DataFrame) -> pd.Series:
        """
        Estimate volatility from order book microstructure.
        
        Args:
            df: DataFrame with L2 order book data
            
        Returns:
            Series with microstructure-based volatility
        """
        required_cols = ['bid_price_1', 'ask_price_1', 'bid_size_1', 'ask_size_1']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            self.logger.warning(f"Missing columns for microstructure volatility: {missing_cols}")
            return pd.Series(index=df.index, dtype=float).fillna(0.01)
        
        # Calculate bid-ask spread volatility
        spread = df['ask_price_1'] - df['bid_price_1']
        spread_pct = spread / df['bid_price_1']
        
        # Calculate order flow imbalance volatility
        total_volume = df['bid_size_1'] + df['ask_size_1']
        imbalance = np.where(
            total_volume > 0,
            (df['bid_size_1'] - df['ask_size_1']) / total_volume,
            0
        )
        
        # Calculate price impact volatility
        mid_price = (df['bid_price_1'] + df['ask_price_1']) / 2
        price_impact = spread / mid_price
        
        # Combine microstructure signals
        window_size = self.volatility_windows['1min']
        min_periods = max(self.min_observations, window_size // 10)
        
        # Spread volatility component
        spread_vol = spread_pct.rolling(
            window=window_size, min_periods=min_periods
        ).std()
        
        # Imbalance volatility component
        imbalance_vol = pd.Series(imbalance).rolling(
            window=window_size, min_periods=min_periods
        ).std()
        
        # Price impact volatility component
        impact_vol = price_impact.rolling(
            window=window_size, min_periods=min_periods
        ).std()
        
        # Weighted combination
        microstructure_vol = (
            0.4 * spread_vol.fillna(0) +
            0.3 * imbalance_vol.fillna(0) +
            0.3 * impact_vol.fillna(0)
        )
        
        return microstructure_vol.fillna(method='ffill').fillna(0.01)

    def estimate_garch_volatility(self, returns: pd.Series, 
                                alpha: float = 0.1, 
                                beta: float = 0.85) -> pd.Series:
        """
        Estimate GARCH(1,1) volatility for L2 data.
        
        Args:
            returns: Return series
            alpha: GARCH alpha parameter
            beta: GARCH beta parameter
            
        Returns:
            Series with GARCH volatility estimates
        """
        if len(returns) < self.min_observations:
            return pd.Series(index=returns.index, dtype=float).fillna(0.01)
        
        # Initialize
        omega = 0.01 * (1 - alpha - beta)  # Long-term variance
        garch_var = pd.Series(index=returns.index, dtype=float)
        
        # Initial variance estimate
        initial_var = returns.var()
        garch_var.iloc[0] = initial_var
        
        # GARCH recursion
        for i in range(1, len(returns)):
            if pd.notna(returns.iloc[i-1]) and pd.notna(garch_var.iloc[i-1]):
                garch_var.iloc[i] = (
                    omega + 
                    alpha * (returns.iloc[i-1] ** 2) + 
                    beta * garch_var.iloc[i-1]
                )
            else:
                garch_var.iloc[i] = garch_var.iloc[i-1]
        
        # Convert variance to volatility
        garch_vol = np.sqrt(garch_var)
        
        return garch_vol.fillna(method='ffill').fillna(0.01)

    def detect_volatility_regime(self, volatility: pd.Series, 
                                threshold_multiplier: float = 1.5) -> pd.Series:
        """
        Detect volatility regime changes.
        
        Args:
            volatility: Volatility series
            threshold_multiplier: Multiplier for regime detection
            
        Returns:
            Series with regime indicators (0=low, 1=normal, 2=high)
        """
        if len(volatility) < self.min_observations:
            return pd.Series(index=volatility.index, dtype=int).fillna(1)
        
        # Calculate rolling statistics
        window_size = self.volatility_windows['5min']
        min_periods = max(self.min_observations, window_size // 10)
        
        vol_mean = volatility.rolling(
            window=window_size, min_periods=min_periods
        ).mean()
        vol_std = volatility.rolling(
            window=window_size, min_periods=min_periods
        ).std()
        
        # Define regime thresholds
        low_threshold = vol_mean - threshold_multiplier * vol_std
        high_threshold = vol_mean + threshold_multiplier * vol_std
        
        # Classify regimes
        regime = pd.Series(index=volatility.index, dtype=int)
        regime = np.where(volatility < low_threshold, 0, regime)  # Low vol
        regime = np.where(
            (volatility >= low_threshold) & (volatility <= high_threshold), 
            1, regime
        )  # Normal vol
        regime = np.where(volatility > high_threshold, 2, regime)  # High vol
        
        return pd.Series(regime, index=volatility.index).fillna(1)

    def calculate_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate comprehensive volatility features for L2 data.
        
        Args:
            df: DataFrame with L2 order book data
            
        Returns:
            DataFrame with volatility features added
        """
        result_df = df.copy()
        
        self.logger.info(f"Calculating volatility features for {len(df)} rows")
        
        # 1. Realized volatility for different timeframes
        for window in ['1min', '5min', '15min']:
            col_name = f'l2_realized_vol_{window}'
            result_df[col_name] = self.estimate_realized_volatility(
                df, window=window
            )
        
        # 2. Microstructure volatility
        result_df['l2_microstructure_vol'] = self.estimate_microstructure_volatility(df)
        
        # 3. GARCH volatility (if we have price data)
        if 'reconstructed_price' in df.columns:
            prices = df['reconstructed_price'].dropna()
            if len(prices) > 1:
                returns = np.log(prices / prices.shift(1)).dropna()
                garch_vol = self.estimate_garch_volatility(returns)
                result_df['l2_garch_vol'] = garch_vol.reindex(df.index).fillna(0.01)
            else:
                result_df['l2_garch_vol'] = 0.01
        else:
            result_df['l2_garch_vol'] = 0.01
        
        # 4. Volatility regime detection
        primary_vol_col = 'l2_realized_vol_1min'
        if primary_vol_col in result_df.columns:
            result_df['l2_vol_regime'] = self.detect_volatility_regime(
                result_df[primary_vol_col]
            )
        else:
            result_df['l2_vol_regime'] = 1
        
        # 5. Volatility ratios and relative measures
        if 'l2_realized_vol_1min' in result_df.columns and 'l2_realized_vol_5min' in result_df.columns:
            result_df['l2_vol_ratio_1m_5m'] = (
                result_df['l2_realized_vol_1min'] / 
                (result_df['l2_realized_vol_5min'] + 1e-8)
            )
        else:
            result_df['l2_vol_ratio_1m_5m'] = 1.0
        
        # 6. Volatility momentum
        if 'l2_realized_vol_1min' in result_df.columns:
            vol_momentum_window = min(300, len(df) // 4)  # 30 seconds at 100ms
            result_df['l2_vol_momentum'] = result_df['l2_realized_vol_1min'].pct_change(
                periods=vol_momentum_window
            ).fillna(0)
        else:
            result_df['l2_vol_momentum'] = 0.0
        
        # Fill any remaining NaN values
        vol_columns = [col for col in result_df.columns if 'vol' in col.lower()]
        for col in vol_columns:
            if col in result_df.columns:
                result_df[col] = result_df[col].fillna(method='ffill').fillna(0.01)
        
        self.logger.info(f"Generated {len(vol_columns)} volatility features")
        
        return result_df

    def get_current_volatility_estimate(self, df: pd.DataFrame, 
                                      method: str = 'realized') -> float:
        """
        Get current volatility estimate for risk management.
        
        Args:
            df: DataFrame with recent L2 data
            method: Volatility estimation method ('realized', 'microstructure', 'garch')
            
        Returns:
            Current volatility estimate
        """
        if df.empty:
            return 0.01  # Default volatility
        
        if method == 'realized':
            vol_series = self.estimate_realized_volatility(df, window='1min')
        elif method == 'microstructure':
            vol_series = self.estimate_microstructure_volatility(df)
        elif method == 'garch':
            if 'reconstructed_price' in df.columns:
                prices = df['reconstructed_price'].dropna()
                if len(prices) > 1:
                    returns = np.log(prices / prices.shift(1)).dropna()
                    vol_series = self.estimate_garch_volatility(returns)
                else:
                    return 0.01
            else:
                return 0.01
        else:
            self.logger.warning(f"Unknown volatility method: {method}")
            return 0.01
        
        # Return most recent non-null estimate
        current_vol = vol_series.dropna()
        if len(current_vol) > 0:
            return float(current_vol.iloc[-1])
        else:
            return 0.01

    def validate_volatility_estimates(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate volatility estimates for quality control.
        
        Args:
            df: DataFrame with volatility features
            
        Returns:
            Dictionary with validation results
        """
        validation_results = {
            'total_rows': len(df),
            'volatility_columns': [],
            'quality_metrics': {},
            'warnings': []
        }
        
        # Find volatility columns
        vol_columns = [col for col in df.columns if 'vol' in col.lower()]
        validation_results['volatility_columns'] = vol_columns
        
        for col in vol_columns:
            if col in df.columns:
                vol_data = df[col].dropna()
                
                if len(vol_data) > 0:
                    metrics = {
                        'non_null_count': len(vol_data),
                        'mean': float(vol_data.mean()),
                        'std': float(vol_data.std()),
                        'min': float(vol_data.min()),
                        'max': float(vol_data.max()),
                        'negative_count': int((vol_data < 0).sum()),
                        'zero_count': int((vol_data == 0).sum()),
                        'extreme_count': int((vol_data > 1.0).sum())  # >100% volatility
                    }
                    
                    validation_results['quality_metrics'][col] = metrics
                    
                    # Quality warnings
                    if metrics['negative_count'] > 0:
                        validation_results['warnings'].append(
                            f"{col}: {metrics['negative_count']} negative values"
                        )
                    
                    if metrics['extreme_count'] > len(vol_data) * 0.01:  # >1% extreme values
                        validation_results['warnings'].append(
                            f"{col}: {metrics['extreme_count']} extreme values (>100%)"
                        )
                    
                    if metrics['zero_count'] > len(vol_data) * 0.1:  # >10% zero values
                        validation_results['warnings'].append(
                            f"{col}: {metrics['zero_count']} zero values"
                        )
        
        return validation_results 