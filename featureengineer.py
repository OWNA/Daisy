# featureengineer.py
# L2-Only Trading Strategy Implementation
# Restructured for L2-only mode with advanced microstructure features
# Enhanced with order flow imbalance, book pressure metrics, and stability indicators

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional


class FeatureEngineer:
    """
    Calculates L2 microstructure features for the L2-only trading strategy.
    This version is adapted to work with the 'l2_training_data_practical'
    schema.
    """
    def __init__(self, config: dict):
        """Initializes the FeatureEngineer for L2-only mode."""
        self.config = config
        self.l2_features_config = config.get('l2_features', [])
        
        # Define time windows for order flow analysis (in rows, assuming 100ms sampling)
        # 10s = 100 rows, 30s = 300 rows, 1m = 600 rows, 5m = 3000 rows
        self.ofi_windows = {
            '10s': 100,
            '30s': 300, 
            '1m': 600,
            '5m': 3000
        }
        
        # Book pressure decay parameters
        self.price_decay_factor = 0.95  # Decay factor per price level
        self.max_levels = 10  # Maximum levels to consider
        
        print("FeatureEngineer initialized for L2 practical schema with enhanced features.")
        print(
            "Number of L2 features to generate: "
            f"{len(self.l2_features_config)}"
        )
        print(f"Order flow imbalance windows: {list(self.ofi_windows.keys())}")

    def generate_features(self, df_input: pd.DataFrame) -> pd.DataFrame:
        """
        Generate all features from the L2 data.
        This is the main entry point for feature generation.
        """
        if df_input is None or df_input.empty:
            print("Error: Input L2 DataFrame is empty.")
            return pd.DataFrame()

        print(f"Starting feature generation for {len(df_input)} rows...")
        df = df_input.copy()

        # All feature calculation is now consolidated here
        df = self._calculate_microstructure_features(df)
        
        # Add new enhanced features
        df = self._calculate_order_flow_imbalance(df)
        df = self._calculate_book_pressure_metrics(df)
        df = self._calculate_stability_indicators(df)
        
        # Handle missing values last
        df = self._handle_missing_values(df)

        print(f"Feature generation complete. DataFrame shape: {df.shape}")
        print(f"Total features generated: {len([c for c in df.columns if c not in ['timestamp', 'symbol']])}")
        return df

    def _calculate_microstructure_features(
        self, df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Calculates L2 microstructure features from the practical L2 data
        schema.
        """
        print("Calculating L2 microstructure features...")

        # Basic spread and mid-price
        df['spread'] = df['ask_price_1'] - df['bid_price_1']
        df['mid_price'] = (df['ask_price_1'] + df['bid_price_1']) / 2
        df['spread_bps'] = (df['spread'] / df['mid_price']) * 10000
        
        # Additional spread features (aliases)
        df['bid_ask_spread'] = df['spread']
        df['bid_ask_spread_pct'] = df['spread_bps'] / 10000

        # Weighted prices and microprice
        df['weighted_bid_price'] = df['bid_price_1'] * df['ask_size_1']
        df['weighted_ask_price'] = df['ask_price_1'] * df['bid_size_1']
        
        # Weighted mid price (volume-weighted)
        total_volume = df['bid_size_1'] + df['ask_size_1']
        df['weighted_mid_price'] = (
            df['bid_price_1'] * df['bid_size_1'] + 
            df['ask_price_1'] * df['ask_size_1']
        ) / total_volume
        df['weighted_mid_price'] = df['weighted_mid_price'].fillna(df['mid_price'])

        sum_of_sizes = df['bid_size_1'] + df['ask_size_1']
        df['microprice'] = (
            df['weighted_bid_price'] + df['weighted_ask_price']
        ) / sum_of_sizes
        df['microprice'] = df['microprice'].fillna(df['mid_price'])

        # Order book imbalance
        df['order_book_imbalance'] = (
            df['bid_size_1'] - df['ask_size_1']
        ) / sum_of_sizes
        
        # Order book imbalance at different levels
        for level in [2, 3, 5]:
            bid_sum = df[[f'bid_size_{i}' for i in range(1, level+1)]].sum(axis=1)
            ask_sum = df[[f'ask_size_{i}' for i in range(1, level+1)]].sum(axis=1)
            total_sum = bid_sum + ask_sum
            df[f'order_book_imbalance_{level}'] = (bid_sum - ask_sum) / total_sum
            df[f'order_book_imbalance_{level}'] = df[f'order_book_imbalance_{level}'].fillna(0)

        # Aggregated volumes and slopes
        for i in range(1, 11):
            bid_cols = [f'bid_size_{j}' for j in range(1, i + 1)]
            ask_cols = [f'ask_size_{j}' for j in range(1, i + 1)]
            df[f'total_bid_volume_{i}'] = df[bid_cols].sum(axis=1)
            df[f'total_ask_volume_{i}'] = df[ask_cols].sum(axis=1)

        # Price impact (simplified)
        df['price_impact_ask'] = (
            df['ask_price_1'] - df['microprice']
        ) / df['mid_price']
        df['price_impact_bid'] = (
            df['microprice'] - df['bid_price_1']
        ) / df['mid_price']
        
        # Additional price impact features
        df['price_impact_buy'] = df['price_impact_ask']
        df['price_impact_sell'] = df['price_impact_bid']
        
        # Price impact at different levels
        for level in [1, 5, 10]:
            if level == 1:
                df['price_impact_1'] = (df['ask_price_1'] - df['bid_price_1']) / (2 * df['mid_price'])
            else:
                # Weighted average price to buy/sell at level
                bid_prices = [df[f'bid_price_{i}'] for i in range(1, min(level+1, 11))]
                ask_prices = [df[f'ask_price_{i}'] for i in range(1, min(level+1, 11))]
                df[f'price_impact_{level}'] = (
                    sum(ask_prices) / len(ask_prices) - sum(bid_prices) / len(bid_prices)
                ) / (2 * df['mid_price'])

        # Rolling volatility of mid-price returns
        df['mid_price_return'] = df['mid_price'].pct_change()
        for window in [10, 50, 200]:  # 1-sec, 5-sec, 20-sec at 100ms
            df[f'l2_volatility_{window}'] = df['mid_price_return'].rolling(
                window=window).std()

        print("Microstructure feature calculation completed.")
        return df
    
    def _calculate_order_flow_imbalance(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate order flow imbalance at multiple time windows.
        OFI measures the net order flow (bid vs ask volume changes) over time.
        """
        print("Calculating order flow imbalance features...")
        
        # Calculate volume changes at each level
        for level in range(1, 6):  # Top 5 levels
            df[f'bid_volume_change_{level}'] = df[f'bid_size_{level}'].diff()
            df[f'ask_volume_change_{level}'] = df[f'ask_size_{level}'].diff()
        
        # Calculate OFI for each window
        for window_name, window_size in self.ofi_windows.items():
            # Skip if window is too large for the data
            if window_size > len(df):
                print(f"Skipping OFI window {window_name} - need {window_size} rows, have {len(df)}")
                # Add missing features with default values
                df[f'ofi_{window_name}'] = 0.0
                df[f'ofi_normalized_{window_name}'] = 0.0
                df[f'ofi_weighted_{window_name}'] = 0.0
                continue
                
            # Basic OFI: sum of bid volume changes - sum of ask volume changes
            bid_changes = df[[f'bid_volume_change_{i}' for i in range(1, 6)]].sum(axis=1)
            ask_changes = df[[f'ask_volume_change_{i}' for i in range(1, 6)]].sum(axis=1)
            
            df[f'ofi_{window_name}'] = (
                bid_changes.rolling(window=window_size, min_periods=1).sum() - 
                ask_changes.rolling(window=window_size, min_periods=1).sum()
            )
            
            # Normalized OFI (by total volume)
            total_volume = df[[f'bid_size_{i}' for i in range(1, 6)] + 
                            [f'ask_size_{i}' for i in range(1, 6)]].sum(axis=1)
            df[f'ofi_normalized_{window_name}'] = (
                df[f'ofi_{window_name}'] / 
                total_volume.rolling(window=window_size, min_periods=1).mean()
            )
            
            # Weighted OFI (closer to mid price = higher weight)
            weighted_bid_changes = 0
            weighted_ask_changes = 0
            
            for level in range(1, 6):
                weight = 1.0 / level  # Simple inverse distance weighting
                weighted_bid_changes += df[f'bid_volume_change_{level}'] * weight
                weighted_ask_changes += df[f'ask_volume_change_{level}'] * weight
            
            df[f'ofi_weighted_{window_name}'] = (
                weighted_bid_changes.rolling(window=window_size, min_periods=1).sum() - 
                weighted_ask_changes.rolling(window=window_size, min_periods=1).sum()
            )
        
        # Drop intermediate columns
        cols_to_drop = [f'bid_volume_change_{i}' for i in range(1, 6)] + \
                      [f'ask_volume_change_{i}' for i in range(1, 6)]
        df.drop(columns=cols_to_drop, inplace=True)
        
        print(f"Order flow imbalance features calculated: {len([c for c in df.columns if 'ofi_' in c])}")
        return df
    
    def _calculate_book_pressure_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate book pressure metrics weighted by distance from mid price.
        Captures the "pressure" exerted by orders at different price levels.
        """
        print("Calculating book pressure metrics...")
        
        # Calculate mid price if not already present
        if 'mid_price' not in df.columns:
            df['mid_price'] = (df['ask_price_1'] + df['bid_price_1']) / 2
        
        # Bid pressure (sum of size * price proximity)
        df['bid_pressure'] = 0
        df['bid_pressure_weighted'] = 0
        
        for level in range(1, min(self.max_levels + 1, 11)):
            # Distance from mid price (in basis points)
            bid_distance_bps = ((df['mid_price'] - df[f'bid_price_{level}']) / 
                               df['mid_price']) * 10000
            
            # Linear decay: 1 at mid price, 0 at 100 bps away
            linear_weight = np.maximum(0, 1 - bid_distance_bps / 100)
            
            # Exponential decay
            exp_weight = np.exp(-bid_distance_bps / 50)  # 50 bps decay constant
            
            df['bid_pressure'] += df[f'bid_size_{level}']
            df['bid_pressure_weighted'] += df[f'bid_size_{level}'] * exp_weight
        
        # Ask pressure
        df['ask_pressure'] = 0
        df['ask_pressure_weighted'] = 0
        
        for level in range(1, min(self.max_levels + 1, 11)):
            # Distance from mid price (in basis points)
            ask_distance_bps = ((df[f'ask_price_{level}'] - df['mid_price']) / 
                               df['mid_price']) * 10000
            
            # Exponential decay
            exp_weight = np.exp(-ask_distance_bps / 50)
            
            df['ask_pressure'] += df[f'ask_size_{level}']
            df['ask_pressure_weighted'] += df[f'ask_size_{level}'] * exp_weight
        
        # Pressure imbalance
        df['pressure_imbalance'] = (df['bid_pressure'] - df['ask_pressure']) / \
                                   (df['bid_pressure'] + df['ask_pressure'])
        df['pressure_imbalance_weighted'] = (df['bid_pressure_weighted'] - df['ask_pressure_weighted']) / \
                                            (df['bid_pressure_weighted'] + df['ask_pressure_weighted'])
        
        # Book depth asymmetry (how symmetric is the book?)
        df['book_depth_asymmetry'] = 0
        for level in range(1, 6):
            level_asymmetry = (df[f'bid_size_{level}'] - df[f'ask_size_{level}']) / \
                             (df[f'bid_size_{level}'] + df[f'ask_size_{level}'])
            df['book_depth_asymmetry'] += level_asymmetry / 5
        
        print("Book pressure metrics calculated.")
        return df
    
    def _calculate_stability_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate microstructure stability indicators.
        These help identify when the order book is stable vs. experiencing rapid changes.
        """
        print("Calculating stability indicators...")
        
        # Quote lifetime (how long does best bid/ask stay unchanged)
        df['bid_price_changed'] = (df['bid_price_1'] != df['bid_price_1'].shift(1)).astype(int)
        df['ask_price_changed'] = (df['ask_price_1'] != df['ask_price_1'].shift(1)).astype(int)
        
        # Calculate quote lifetime using expanding window
        df['bid_quote_lifetime'] = df.groupby(df['bid_price_changed'].cumsum()).cumcount()
        df['ask_quote_lifetime'] = df.groupby(df['ask_price_changed'].cumsum()).cumcount()
        df['quote_lifetime'] = (df['bid_quote_lifetime'] + df['ask_quote_lifetime']) / 2
        
        # Book resilience (how quickly does the book replenish after trades)
        # Measured as the ratio of volume at best levels to volume at deeper levels
        top_volume = df[['bid_size_1', 'ask_size_1']].sum(axis=1)
        deep_volume = df[[f'bid_size_{i}' for i in range(2, 6)] + 
                        [f'ask_size_{i}' for i in range(2, 6)]].sum(axis=1)
        df['book_resilience'] = top_volume / (deep_volume + 1)  # +1 to avoid division by zero
        
        # Spread stability (volatility of spread over different windows)
        if 'spread' not in df.columns:
            df['spread'] = df['ask_price_1'] - df['bid_price_1']
        
        for window in [10, 50, 100]:  # 1s, 5s, 10s
            df[f'spread_stability_{window}'] = df['spread'].rolling(window=window, min_periods=1).std()
            # Normalized by mean spread
            mean_spread = df['spread'].rolling(window=window, min_periods=1).mean()
            df[f'spread_stability_norm_{window}'] = df[f'spread_stability_{window}'] / (mean_spread + 1e-8)
        
        # Order book shape stability (how much does the shape change)
        # Calculate a simple shape metric: ratio of volumes at different levels
        df['book_shape_1_5'] = (df['bid_size_1'] + df['ask_size_1']) / \
                               (df['bid_size_5'] + df['ask_size_5'] + 1)
        df['book_shape_stability'] = df['book_shape_1_5'].rolling(window=50, min_periods=1).std()
        
        # Volume concentration (how concentrated is volume at top levels)
        total_bid_volume = df[[f'bid_size_{i}' for i in range(1, 11)]].sum(axis=1)
        total_ask_volume = df[[f'ask_size_{i}' for i in range(1, 11)]].sum(axis=1)
        top_3_bid_volume = df[[f'bid_size_{i}' for i in range(1, 4)]].sum(axis=1)
        top_3_ask_volume = df[[f'ask_size_{i}' for i in range(1, 4)]].sum(axis=1)
        
        df['bid_volume_concentration'] = top_3_bid_volume / (total_bid_volume + 1)
        df['ask_volume_concentration'] = top_3_ask_volume / (total_ask_volume + 1)
        df['volume_concentration'] = (df['bid_volume_concentration'] + df['ask_volume_concentration']) / 2
        
        # Clean up temporary columns
        df.drop(columns=['bid_price_changed', 'ask_price_changed'], inplace=True)
        
        print("Stability indicators calculated.")
        return df

    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fills or handles NaN values in the feature DataFrame."""
        print("Handling missing values...")
        # Forward-fill most features to propagate last known state
        df.ffill(inplace=True)
        # Backward-fill to handle NaNs at the beginning
        df.bfill(inplace=True)
        # Any remaining NaNs can be filled with 0
        df.fillna(0, inplace=True)
        print("Missing values handled.")
        return df
