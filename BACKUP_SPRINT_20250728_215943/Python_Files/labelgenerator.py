# labelgenerator.py
# L2-Only Label Generator - Phase 3 Implementation
# Converted from OHLCV-based to pure L2-derived labeling

import pandas as pd
import numpy as np
import traceback
import logging

class LabelGenerator:
    """
    Generates target labels for model training from L2-derived price series.
    
    This L2-only implementation uses L2 order book derived prices instead of OHLCV data:
    - weighted_mid_price for primary price series
    - microprice for high-precision labeling
    - L2-derived volatility for normalization
    """
    
    def __init__(self, config):
        """
        Initializes the L2-Only LabelGenerator.
        
        Args:
            config (dict): L2-only configuration dictionary
        """
        self.config = config
        self.l2_only_mode = config.get('l2_only_mode', True)
        
        # Validate L2-only mode
        if not self.l2_only_mode:
            raise ValueError("LabelGenerator requires l2_only_mode=True for Phase 3 implementation")
        
        # L2-specific labeling method
        self.labeling_method = config.get('labeling_method', 'l2_volatility_normalized_return')
        
        # L2 price columns for labeling (in order of preference)
        self.l2_price_columns = [
            'weighted_mid_price',
            'microprice', 
            'mid_price',
            'best_bid',
            'best_ask'
        ]
        
        # Parameters for L2 volatility normalized return
        self.vol_norm_volatility_window = config.get('label_volatility_window', 50)  # Increased for L2
        self.vol_norm_clip_lower_quantile = config.get('label_clip_quantiles', (0.005, 0.995))[0]  # Tighter for L2
        self.vol_norm_clip_upper_quantile = config.get('label_clip_quantiles', (0.005, 0.995))[1]
        self.vol_norm_label_shift = config.get('label_shift', -1)
        
        # Parameters for L2 triple barrier
        self.tb_profit_target_spread_mult = config.get('l2_triple_barrier_profit_target_spread_mult', 3.0)
        self.tb_stop_loss_spread_mult = config.get('l2_triple_barrier_stop_loss_spread_mult', 2.0)
        self.tb_time_horizon_ticks = config.get('l2_triple_barrier_time_horizon_ticks', 100)  # L2 ticks instead of bars
        self.tb_spread_column = config.get('l2_triple_barrier_spread_column', 'bid_ask_spread')
        
        # Parameters for L2 microstructure labeling
        self.micro_imbalance_threshold = config.get('l2_microstructure_imbalance_threshold', 0.1)
        self.micro_price_impact_threshold = config.get('l2_microstructure_price_impact_threshold', 0.05)
        
        # Store calculated parameters
        self.target_mean_calculated = None
        self.target_std_calculated = None
        
        # Initialize logging
        self.logger = logging.getLogger(__name__)
        
        self.logger.info(f"L2-Only LabelGenerator initialized. Using method: {self.labeling_method}")

    def _get_l2_price_series(self, df_features):
        """
        Gets the best available L2-derived price series from the DataFrame.
        
        Args:
            df_features (pd.DataFrame): L2 features DataFrame
            
        Returns:
            pd.Series: L2-derived price series
        """
        for price_col in self.l2_price_columns:
            if price_col in df_features.columns and not df_features[price_col].isna().all():
                self.logger.info(f"Using L2 price column: {price_col}")
                return df_features[price_col]
        
        # Fallback: construct mid price from bid/ask if available
        if 'best_bid' in df_features.columns and 'best_ask' in df_features.columns:
            self.logger.info("Constructing mid price from best_bid and best_ask")
            return (df_features['best_bid'] + df_features['best_ask']) / 2
        
        raise ValueError("No suitable L2 price column found. Available columns: " + 
                        str(list(df_features.columns)))

    def _generate_l2_volatility_normalized_return_labels(self, df_features):
        """
        Generates volatility-normalized returns using L2-derived price series.
        
        Args:
            df_features (pd.DataFrame): L2 features DataFrame
            
        Returns:
            tuple: (labeled_df, target_mean, target_std)
        """
        try:
            # Get L2 price series
            l2_price_series = self._get_l2_price_series(df_features)
            
            print("L2 Price Series:")
            print(l2_price_series)
            
            if l2_price_series.empty:
                self.logger.warning("L2 price series is empty. Cannot generate labels.")
                return df_features.copy(), None, None
            
            df = df_features.copy()
            
            # Calculate L2-based returns
            returns = l2_price_series.pct_change()
            
            if returns.empty:
                self.logger.warning("Returns series is empty. Cannot generate labels.")
                return df_features.copy(), None, None
            
            # Use L2 volatility if available, otherwise calculate from returns
            if 'l2_volatility_1min' in df_features.columns:
                volatility = df_features['l2_volatility_1min']
                self.logger.info("Using pre-calculated L2 volatility for normalization")
            else:
                # Calculate rolling volatility from L2 returns
                min_periods_vol = max(1, self.vol_norm_volatility_window // 2)
                volatility = returns.rolling(
                    window=self.vol_norm_volatility_window, 
                    min_periods=min_periods_vol
                ).std()
                self.logger.info("Calculating L2 volatility from returns")
            
            # Generate normalized target
            df['target'] = returns.shift(self.vol_norm_label_shift) / (volatility.fillna(method='ffill').fillna(1e-9) + 1e-9)
            
            # Calculate statistics
            target_mean_raw = None
            target_std_raw = None
            
            valid_targets = df['target'].dropna()
            if len(valid_targets) > self.vol_norm_volatility_window:
                target_mean_raw = valid_targets.mean()
                target_std_raw = valid_targets.std()
                
                if target_std_raw == 0 or not pd.notna(target_std_raw) or target_std_raw < 1e-9:
                    target_std_raw = 1e-9
                
                # Apply L2-specific clipping (tighter bounds for high-frequency data)
                if (pd.notna(self.vol_norm_clip_lower_quantile) and 
                    pd.notna(self.vol_norm_clip_upper_quantile)):
                    
                    lower_bound = valid_targets.quantile(self.vol_norm_clip_lower_quantile)
                    upper_bound = valid_targets.quantile(self.vol_norm_clip_upper_quantile)
                    
                    if (pd.notna(lower_bound) and pd.notna(upper_bound) and 
                        lower_bound < upper_bound):
                        df['target'] = df['target'].clip(lower_bound, upper_bound)
                        self.logger.info(f"L2 target clipped to [{lower_bound:.6f}, {upper_bound:.6f}]")
                    else:
                        self.logger.warning("Could not clip L2 target due to invalid quantile bounds")
            else:
                self.logger.warning("Not enough valid L2 targets for robust statistics")
                target_mean_raw = 0.0
                target_std_raw = 1.0
            
            self.target_mean_calculated = target_mean_raw
            self.target_std_calculated = target_std_raw
            
            # Remove invalid targets
            df.dropna(subset=['target'], inplace=True)
            
            self.logger.info(f"L2 volatility normalized labels generated. "
                           f"Mean: {target_mean_raw:.6f}, Std: {target_std_raw:.6f}")
            
            return df, self.target_mean_calculated, self.target_std_calculated
            
        except Exception as e:
            self.logger.error(f"Error generating L2 volatility normalized labels: {e}")
            return df_features.copy(), None, None

    def _generate_l2_triple_barrier_labels(self, df_features):
        """
        Generates labels using L2-based Triple-Barrier Method.
        
        Uses L2 spread and price impact instead of ATR for barrier calculation.
        
        Args:
            df_features (pd.DataFrame): L2 features DataFrame
            
        Returns:
            tuple: (labeled_df, target_mean, target_std)
        """
        try:
            # Get L2 price series
            l2_price_series = self._get_l2_price_series(df_features)
            
            # Check for required L2 columns
            required_cols = [self.tb_spread_column]
            missing_cols = [col for col in required_cols if col not in df_features.columns]
            
            if missing_cols:
                self.logger.error(f"Missing L2 columns for triple barrier: {missing_cols}")
                return df_features.copy(), None, None
            
            df = df_features.copy()
            df['target'] = 0  # Default to 0 (time barrier)
            df['event_time'] = pd.NaT
            
            spread_series = df_features[self.tb_spread_column]
            
            for i in range(len(df) - self.tb_time_horizon_ticks):
                entry_price = l2_price_series.iloc[i]
                spread_at_entry = spread_series.iloc[i]
                
                if pd.isna(entry_price) or pd.isna(spread_at_entry) or spread_at_entry <= 1e-9:
                    df.loc[df.index[i], 'target'] = np.nan
                    continue
                
                # L2-based barriers using spread
                upper_barrier = entry_price + (spread_at_entry * self.tb_profit_target_spread_mult)
                lower_barrier = entry_price - (spread_at_entry * self.tb_stop_loss_spread_mult)
                
                # Look ahead for barrier hits
                for k in range(1, self.tb_time_horizon_ticks + 1):
                    future_idx = i + k
                    if future_idx >= len(df):
                        break
                    
                    future_price = l2_price_series.iloc[future_idx]
                    future_timestamp = df.index[future_idx] if hasattr(df.index, 'to_pydatetime') else df.iloc[future_idx].get('timestamp')
                    
                    # Check barriers
                    if future_price >= upper_barrier:
                        df.loc[df.index[i], 'target'] = 1  # Profit target hit
                        df.loc[df.index[i], 'event_time'] = future_timestamp
                        break
                    elif future_price <= lower_barrier:
                        df.loc[df.index[i], 'target'] = -1  # Stop loss hit
                        df.loc[df.index[i], 'event_time'] = future_timestamp
                        break
            
            # Clean up
            df.dropna(subset=['target'], inplace=True)
            df['target'] = df['target'].astype(int)
            
            # For classification, mean/std are less relevant
            self.target_mean_calculated = None
            self.target_std_calculated = None
            
            class_distribution = df['target'].value_counts(normalize=True)
            self.logger.info(f"L2 triple barrier labels generated. Class distribution:\n{class_distribution}")
            
            return df, self.target_mean_calculated, self.target_std_calculated
            
        except Exception as e:
            self.logger.error(f"Error generating L2 triple barrier labels: {e}")
            return df_features.copy(), None, None

    def _generate_l2_microstructure_labels(self, df_features):
        """
        Generates labels based on L2 microstructure signals.
        
        Uses order book imbalance and price impact to predict short-term price movements.
        
        Args:
            df_features (pd.DataFrame): L2 features DataFrame
            
        Returns:
            tuple: (labeled_df, target_mean, target_std)
        """
        try:
            # Get L2 price series
            l2_price_series = self._get_l2_price_series(df_features)
            
            # Required L2 microstructure features
            required_features = ['order_book_imbalance_2', 'price_impact_buy']
            missing_features = [feat for feat in required_features if feat not in df_features.columns]
            
            if missing_features:
                self.logger.error(f"Missing L2 microstructure features: {missing_features}")
                return df_features.copy(), None, None
            
            df = df_features.copy()
            
            # Calculate future price change (next tick)
            future_returns = l2_price_series.pct_change().shift(-1)
            
            # Generate microstructure-based targets
            imbalance = df_features['order_book_imbalance_2']
            price_impact = df_features['price_impact_buy']
            
            # Create target based on microstructure signals
            df['target'] = np.where(
                (imbalance > self.micro_imbalance_threshold) & 
                (price_impact < self.micro_price_impact_threshold),
                1,  # Strong buy signal
                np.where(
                    (imbalance < -self.micro_imbalance_threshold) & 
                    (price_impact < self.micro_price_impact_threshold),
                    -1,  # Strong sell signal
                    0   # Neutral
                )
            )
            
            # Weight targets by actual future returns for regression
            df['target'] = df['target'] * future_returns
            
            # Calculate statistics
            valid_targets = df['target'].dropna()
            if len(valid_targets) > 10:
                self.target_mean_calculated = valid_targets.mean()
                self.target_std_calculated = valid_targets.std()
                
                if self.target_std_calculated <= 1e-9:
                    self.target_std_calculated = 1e-9
            else:
                self.target_mean_calculated = 0.0
                self.target_std_calculated = 1.0
            
            # Clean up
            df.dropna(subset=['target'], inplace=True)
            
            self.logger.info(f"L2 microstructure labels generated. "
                           f"Mean: {self.target_mean_calculated:.6f}, "
                           f"Std: {self.target_std_calculated:.6f}")
            
            return df, self.target_mean_calculated, self.target_std_calculated
            
        except Exception as e:
            self.logger.error(f"Error generating L2 microstructure labels: {e}")
            return df_features.copy(), None, None

    def generate_labels(self, df_features):
        """
        Generates target labels based on L2-only methods.
        
        Args:
            df_features (pd.DataFrame): L2 features DataFrame
            
        Returns:
            tuple: (labeled_df, target_mean, target_std)
        """
        if df_features is None or df_features.empty:
            self.logger.error("Input DataFrame is empty")
            return df_features, None, None
        
        # Validate L2-only mode
        if not self.l2_only_mode:
            raise ValueError("LabelGenerator must be in L2-only mode for Phase 3")
        
        self.logger.info(f"Generating L2-only labels using method: {self.labeling_method}")
        
        try:
            if self.labeling_method == 'l2_volatility_normalized_return':
                df_labeled, mean_val, std_val = self._generate_l2_volatility_normalized_return_labels(df_features)
            elif self.labeling_method == 'l2_triple_barrier':
                df_labeled, mean_val, std_val = self._generate_l2_triple_barrier_labels(df_features)
            elif self.labeling_method == 'l2_microstructure':
                df_labeled, mean_val, std_val = self._generate_l2_microstructure_labels(df_features)
            elif self.labeling_method == 'volatility_normalized_return':
                # Legacy fallback - convert to L2 method
                self.logger.warning("Converting legacy method to L2 volatility normalized return")
                self.labeling_method = 'l2_volatility_normalized_return'
                df_labeled, mean_val, std_val = self._generate_l2_volatility_normalized_return_labels(df_features)
            else:
                self.logger.error(f"Unknown L2 labeling method: {self.labeling_method}")
                self.logger.info("Defaulting to l2_volatility_normalized_return")
                df_labeled, mean_val, std_val = self._generate_l2_volatility_normalized_return_labels(df_features)
            
            if df_labeled.empty:
                self.logger.warning("DataFrame is empty after L2 label generation")
            else:
                self.logger.info(f"L2 labels generated successfully. "
                               f"Rows: {len(df_labeled)}, "
                               f"Target mean: {mean_val}, "
                               f"Target std: {std_val}")
            
            return df_labeled, mean_val, std_val
            
        except Exception as e:
            self.logger.error(f"Error in L2 label generation: {e}")
            traceback.print_exc()
            return df_features.copy(), None, None

    def get_l2_labeling_info(self):
        """
        Get information about L2 labeling configuration.
        
        Returns:
            dict: L2 labeling configuration and status
        """
        return {
            'l2_only_mode': self.l2_only_mode,
            'labeling_method': self.labeling_method,
            'l2_price_columns': self.l2_price_columns,
            'volatility_window': self.vol_norm_volatility_window,
            'clip_quantiles': (self.vol_norm_clip_lower_quantile, self.vol_norm_clip_upper_quantile),
            'label_shift': self.vol_norm_label_shift,
            'target_mean': self.target_mean_calculated,
            'target_std': self.target_std_calculated,
            'triple_barrier_config': {
                'profit_target_spread_mult': self.tb_profit_target_spread_mult,
                'stop_loss_spread_mult': self.tb_stop_loss_spread_mult,
                'time_horizon_ticks': self.tb_time_horizon_ticks,
                'spread_column': self.tb_spread_column
            },
            'microstructure_config': {
                'imbalance_threshold': self.micro_imbalance_threshold,
                'price_impact_threshold': self.micro_price_impact_threshold
            }
        }