# featureengineer_enhanced.py
# Advanced L2 Microstructure Feature Engineering
# Implements sophisticated features to reduce false signals

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class EnhancedFeatureEngineer:
    """
    Enhanced feature engineer with advanced microstructure features designed
    to capture market dynamics and reduce false signals.
    """
    
    def __init__(self, config: dict):
        """Initialize enhanced feature engineer."""
        self.config = config
        self.l2_features_config = config.get('l2_features', [])
        
        # Feature computation windows (in ticks)
        self.imbalance_windows = [10, 30, 100, 300]  # ~1s, 3s, 10s, 30s at 100ms
        self.volatility_windows = [10, 50, 100, 200, 500]  # Multiple timescales
        self.flow_windows = [50, 100, 300]  # For order flow features
        
        logger.info("Enhanced FeatureEngineer initialized")

    def generate_features(self, df_input: pd.DataFrame) -> pd.DataFrame:
        """Generate all enhanced features from L2 data."""
        if df_input is None or df_input.empty:
            logger.error("Input L2 DataFrame is empty")
            return pd.DataFrame()
        
        logger.info(f"Starting enhanced feature generation for {len(df_input)} rows")
        df = df_input.copy()
        
        # Generate features in order of complexity
        df = self._calculate_basic_features(df)
        df = self._calculate_order_flow_imbalance(df)
        df = self._calculate_book_pressure_metrics(df)
        df = self._calculate_microstructure_stability(df)
        df = self._calculate_temporal_patterns(df)
        df = self._calculate_advanced_volatility(df)
        df = self._calculate_market_regime_indicators(df)
        df = self._handle_missing_values(df)
        
        logger.info(f"Enhanced feature generation complete. Shape: {df.shape}")
        return df

    def _calculate_basic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate basic L2 features (keeping compatibility)."""
        # Basic spread and mid-price
        df['spread'] = df['ask_price_1'] - df['bid_price_1']
        df['mid_price'] = (df['ask_price_1'] + df['bid_price_1']) / 2
        df['spread_bps'] = (df['spread'] / df['mid_price']) * 10000
        
        # Aliases for compatibility
        df['bid_ask_spread'] = df['spread']
        df['bid_ask_spread_pct'] = df['spread_bps'] / 10000
        
        # Weighted prices
        total_volume = df['bid_size_1'] + df['ask_size_1']
        df['weighted_mid_price'] = (
            df['bid_price_1'] * df['bid_size_1'] + 
            df['ask_price_1'] * df['ask_size_1']
        ) / total_volume.replace(0, 1)
        
        # Microprice
        df['microprice'] = (
            df['bid_price_1'] * df['ask_size_1'] + 
            df['ask_price_1'] * df['bid_size_1']
        ) / total_volume.replace(0, 1)
        
        # Basic imbalances (keeping for compatibility)
        df['order_book_imbalance'] = (
            df['bid_size_1'] - df['ask_size_1']
        ) / total_volume.replace(0, 1)
        
        # Multi-level imbalances
        for level in [2, 3, 5]:
            bid_sum = df[[f'bid_size_{i}' for i in range(1, level+1)]].sum(axis=1)
            ask_sum = df[[f'ask_size_{i}' for i in range(1, level+1)]].sum(axis=1)
            total_sum = bid_sum + ask_sum
            df[f'order_book_imbalance_{level}'] = (
                (bid_sum - ask_sum) / total_sum.replace(0, 1)
            )
        
        # Volume aggregations
        for i in range(1, 11):
            df[f'total_bid_volume_{i}'] = df[
                [f'bid_size_{j}' for j in range(1, i + 1)]
            ].sum(axis=1)
            df[f'total_ask_volume_{i}'] = df[
                [f'ask_size_{j}' for j in range(1, i + 1)]
            ].sum(axis=1)
        
        # Price impacts
        df['price_impact_bid'] = (df['microprice'] - df['bid_price_1']) / df['mid_price']
        df['price_impact_ask'] = (df['ask_price_1'] - df['microprice']) / df['mid_price']
        df['price_impact_buy'] = df['price_impact_ask']
        df['price_impact_sell'] = df['price_impact_bid']
        
        # Multi-level price impacts
        for level in [1, 5, 10]:
            if level <= 10:
                weighted_bid = 0
                weighted_ask = 0
                total_bid_size = 0
                total_ask_size = 0
                
                for i in range(1, min(level + 1, 11)):
                    weighted_bid += df[f'bid_price_{i}'] * df[f'bid_size_{i}']
                    weighted_ask += df[f'ask_price_{i}'] * df[f'ask_size_{i}']
                    total_bid_size += df[f'bid_size_{i}']
                    total_ask_size += df[f'ask_size_{i}']
                
                avg_bid = weighted_bid / total_bid_size.replace(0, 1)
                avg_ask = weighted_ask / total_ask_size.replace(0, 1)
                df[f'price_impact_{level}'] = (avg_ask - avg_bid) / (2 * df['mid_price'])
        
        # Basic volatility
        df['mid_price_return'] = df['mid_price'].pct_change()
        for window in [10, 50, 200]:
            df[f'l2_volatility_{window}'] = df['mid_price_return'].rolling(
                window=window, min_periods=1
            ).std()
        
        return df

    def _calculate_order_flow_imbalance(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate sophisticated order flow imbalance features."""
        logger.info("Calculating order flow imbalance features")
        
        # Volume-weighted order flow imbalance across multiple windows
        for window in self.imbalance_windows:
            # Calculate rolling imbalances
            bid_flow = df['total_bid_volume_5'].rolling(window, min_periods=1).sum()
            ask_flow = df['total_ask_volume_5'].rolling(window, min_periods=1).sum()
            total_flow = bid_flow + ask_flow
            
            df[f'flow_imbalance_{window}'] = (
                (bid_flow - ask_flow) / total_flow.replace(0, 1)
            )
            
            # Exponentially weighted version for faster adaptation
            alpha = 2 / (window + 1)
            df[f'flow_imbalance_ema_{window}'] = (
                df['order_book_imbalance'].ewm(alpha=alpha, min_periods=1).mean()
            )
        
        # Cross-level flow correlation
        for i in range(1, 6):
            df[f'level_{i}_flow_ratio'] = (
                df[f'bid_size_{i}'] / (df[f'bid_size_{i}'] + df[f'ask_size_{i}']).replace(0, 1)
            )
        
        # Flow persistence (autocorrelation of imbalance)
        df['flow_persistence'] = df['order_book_imbalance'].rolling(20).apply(
            lambda x: x.autocorr(lag=1) if len(x) >= 2 else 0
        )
        
        return df

    def _calculate_book_pressure_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate order book pressure metrics weighted by price distance."""
        logger.info("Calculating book pressure metrics")
        
        # Weighted book pressure (volume weighted by inverse distance from mid)
        for side in ['bid', 'ask']:
            weighted_pressure = 0
            total_weight = 0
            
            for level in range(1, 11):
                price_col = f'{side}_price_{level}'
                size_col = f'{side}_size_{level}'
                
                # Distance from mid price (in bps)
                if side == 'bid':
                    distance = (df['mid_price'] - df[price_col]) / df['mid_price'] * 10000
                else:
                    distance = (df[price_col] - df['mid_price']) / df['mid_price'] * 10000
                
                # Weight inversely proportional to distance (closer = more weight)
                weight = 1 / (1 + distance / 10)  # Normalize by 10 bps
                weighted_pressure += df[size_col] * weight
                total_weight += weight
            
            df[f'{side}_pressure'] = weighted_pressure / total_weight.replace(0, 1)
        
        # Pressure imbalance
        df['book_pressure_imbalance'] = (
            (df['bid_pressure'] - df['ask_pressure']) / 
            (df['bid_pressure'] + df['ask_pressure']).replace(0, 1)
        )
        
        # Pressure concentration (how much volume is near the touch)
        for side in ['bid', 'ask']:
            near_touch = df[f'{side}_size_1'] + df[f'{side}_size_2']
            total_10 = df[f'total_{side}_volume_10']
            df[f'{side}_concentration'] = near_touch / total_10.replace(0, 1)
        
        return df

    def _calculate_microstructure_stability(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate features measuring order book stability."""
        logger.info("Calculating microstructure stability features")
        
        # Spread stability (rolling std of spread)
        for window in [20, 50, 100]:
            df[f'spread_stability_{window}'] = df['spread_bps'].rolling(
                window, min_periods=1
            ).std()
        
        # Quote life proxy (how long has best bid/ask been stable)
        df['bid_change'] = (df['bid_price_1'] != df['bid_price_1'].shift(1)).astype(int)
        df['ask_change'] = (df['ask_price_1'] != df['ask_price_1'].shift(1)).astype(int)
        
        # Count ticks since last change
        df['bid_life'] = df.groupby(df['bid_change'].cumsum()).cumcount()
        df['ask_life'] = df.groupby(df['ask_change'].cumsum()).cumcount()
        df['quote_life'] = (df['bid_life'] + df['ask_life']) / 2
        
        # Book depth resilience (how quickly depth recovers after depletion)
        df['depth_ratio'] = df['total_bid_volume_5'] / df['total_ask_volume_5'].replace(0, 1)
        df['depth_ratio_stability'] = df['depth_ratio'].rolling(50, min_periods=1).std()
        
        # Order book shape stability
        for level in [3, 5, 10]:
            imbalance_col = f'order_book_imbalance_{level}'
            df[f'imbalance_stability_{level}'] = df[imbalance_col].rolling(
                50, min_periods=1
            ).std()
        
        return df

    def _calculate_temporal_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate features capturing temporal patterns in order flow."""
        logger.info("Calculating temporal pattern features")
        
        # Order arrival intensity (changes in order book)
        df['book_updates'] = (
            (df['bid_size_1'] != df['bid_size_1'].shift(1)) |
            (df['ask_size_1'] != df['ask_size_1'].shift(1))
        ).astype(int)
        
        for window in self.flow_windows:
            df[f'update_intensity_{window}'] = df['book_updates'].rolling(
                window, min_periods=1
            ).sum() / window
        
        # Size clustering (are large orders arriving together?)
        df['large_bid'] = (df['bid_size_1'] > df['bid_size_1'].rolling(100).quantile(0.9)).astype(int)
        df['large_ask'] = (df['ask_size_1'] > df['ask_size_1'].rolling(100).quantile(0.9)).astype(int)
        
        df['size_clustering'] = (df['large_bid'] | df['large_ask']).rolling(
            20, min_periods=1
        ).sum()
        
        # Momentum features
        for window in [10, 30, 100]:
            df[f'price_momentum_{window}'] = df['mid_price'].pct_change(window)
            df[f'imbalance_momentum_{window}'] = df['order_book_imbalance'].diff(window)
        
        return df

    def _calculate_advanced_volatility(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate advanced volatility features."""
        logger.info("Calculating advanced volatility features")
        
        # Realized volatility at multiple scales
        returns = df['mid_price_return']
        
        for window in self.volatility_windows:
            # Standard volatility
            df[f'volatility_{window}'] = returns.rolling(window, min_periods=1).std()
            
            # Upside/downside volatility
            upside_returns = returns.clip(lower=0)
            downside_returns = returns.clip(upper=0)
            
            df[f'upside_vol_{window}'] = upside_returns.rolling(window, min_periods=1).std()
            df[f'downside_vol_{window}'] = downside_returns.rolling(window, min_periods=1).std()
            df[f'vol_skew_{window}'] = (
                df[f'upside_vol_{window}'] - df[f'downside_vol_{window}']
            ) / df[f'volatility_{window}'].replace(0, 1)
        
        # GARCH-style features (simplified)
        df['squared_returns'] = returns ** 2
        df['garch_vol'] = df['squared_returns'].ewm(span=50, min_periods=1).mean() ** 0.5
        
        # Volatility of volatility
        df['vol_of_vol'] = df['volatility_50'].rolling(100, min_periods=1).std()
        
        return df

    def _calculate_market_regime_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate features indicating market regime (trending/ranging)."""
        logger.info("Calculating market regime indicators")
        
        # Price efficiency ratio (similar to Kaufman's Efficiency Ratio)
        for window in [50, 100, 200]:
            price_change = (df['mid_price'] - df['mid_price'].shift(window)).abs()
            path_sum = df['mid_price'].diff().abs().rolling(window, min_periods=1).sum()
            df[f'efficiency_ratio_{window}'] = price_change / path_sum.replace(0, 1)
        
        # Trend strength
        for window in [30, 100]:
            ma = df['mid_price'].rolling(window, min_periods=1).mean()
            df[f'trend_strength_{window}'] = (df['mid_price'] - ma) / ma
            
            # Trend consistency
            returns_sign = returns.rolling(window, min_periods=1).apply(
                lambda x: (x > 0).sum() / len(x) if len(x) > 0 else 0.5
            )
            df[f'trend_consistency_{window}'] = (returns_sign - 0.5).abs() * 2
        
        # Range indicators
        for window in [50, 100]:
            high = df['mid_price'].rolling(window, min_periods=1).max()
            low = df['mid_price'].rolling(window, min_periods=1).min()
            range_pct = (high - low) / df['mid_price']
            df[f'range_pct_{window}'] = range_pct
            
            # Position in range
            df[f'range_position_{window}'] = (
                (df['mid_price'] - low) / (high - low).replace(0, 1)
            )
        
        return df

    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values appropriately for each feature type."""
        logger.info("Handling missing values")
        
        # Features that should be forward-filled (state features)
        state_features = [col for col in df.columns if any(
            keyword in col for keyword in ['life', 'pressure', 'imbalance', 'volume']
        )]
        df[state_features] = df[state_features].fillna(method='ffill')
        
        # Features that should be zero-filled (flow/momentum features)
        flow_features = [col for col in df.columns if any(
            keyword in col for keyword in ['momentum', 'flow', 'update', 'change']
        )]
        df[flow_features] = df[flow_features].fillna(0)
        
        # Volatility features should use expanding window for initial values
        vol_features = [col for col in df.columns if 'vol' in col]
        for col in vol_features:
            df[col] = df[col].fillna(df['mid_price_return'].expanding(min_periods=2).std())
        
        # Final cleanup
        df = df.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        return df

    def get_feature_groups(self) -> Dict[str, List[str]]:
        """Return feature groups for analysis and selection."""
        return {
            'basic': ['spread', 'mid_price', 'microprice', 'order_book_imbalance'],
            'flow_imbalance': [col for col in self.trained_features if 'flow_imbalance' in col],
            'book_pressure': [col for col in self.trained_features if 'pressure' in col],
            'stability': [col for col in self.trained_features if 'stability' in col or 'life' in col],
            'temporal': [col for col in self.trained_features if any(
                keyword in col for keyword in ['momentum', 'intensity', 'clustering']
            )],
            'volatility': [col for col in self.trained_features if 'vol' in col],
            'regime': [col for col in self.trained_features if any(
                keyword in col for keyword in ['efficiency', 'trend', 'range']
            )]
        }