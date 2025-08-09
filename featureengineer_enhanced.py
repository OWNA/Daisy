# featureengineer_enhanced.py
# Advanced L2 Microstructure Feature Engineering
# Implements sophisticated features to reduce false signals

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
import sqlite3
import time
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class EnhancedFeatureEngineer:
    """
    Enhanced feature engineer with advanced microstructure features designed
    to capture market dynamics and reduce false signals.
    
    Features database integration with read-before-write pattern for performance optimization.
    """
    
    def __init__(self, config: dict, db_path: str = "trading_bot.db"):
        """Initialize enhanced feature engineer with database integration."""
        self.config = config
        self.db_path = db_path
        self.l2_features_config = config.get('l2_features', [])
        
        # Feature computation windows (in ticks)
        self.imbalance_windows = [10, 30, 100, 300]  # ~1s, 3s, 10s, 30s at 100ms
        self.volatility_windows = [10, 50, 100, 200, 500]  # Multiple timescales
        self.flow_windows = [50, 100, 300]  # For order flow features
        
        # Define Phase 1 features for database integration
        self.phase1_features = self._define_phase1_features()
        
        # Performance tracking
        self.cache_hits = 0
        self.cache_misses = 0
        
        logger.info(f"Enhanced FeatureEngineer initialized with database integration (DB: {db_path})")
        logger.info(f"Phase 1 features defined: {len(self.phase1_features)} features")

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
        
        # Phase 1 OFI Features - Time-based windows
        # Convert window ticks to approximate time windows (assuming 100ms per tick)
        ofi_windows = {
            'ofi_10s': 100,    # ~10 seconds = 100 ticks
            'ofi_30s': 300,    # ~30 seconds = 300 ticks  
            'ofi_1m': 600,     # ~1 minute = 600 ticks
            'ofi_5m': 3000     # ~5 minutes = 3000 ticks
        }
        
        # Calculate basic OFI for each time window
        for ofi_name, window_ticks in ofi_windows.items():
            bid_flow = df['total_bid_volume_5'].rolling(window_ticks, min_periods=1).sum()
            ask_flow = df['total_ask_volume_5'].rolling(window_ticks, min_periods=1).sum()
            df[ofi_name] = bid_flow - ask_flow
            
            # Normalized OFI (divide by total volume)
            total_volume = bid_flow + ask_flow
            df[f'{ofi_name.replace("ofi_", "ofi_normalized_")}'] = (
                df[ofi_name] / total_volume.replace(0, 1)
            )
            
            # Distance-weighted OFI (weight by inverse distance from mid)
            # Simplified version - weight by level 1 vs deeper levels
            l1_bid_flow = df['bid_size_1'].rolling(window_ticks, min_periods=1).sum()
            l1_ask_flow = df['ask_size_1'].rolling(window_ticks, min_periods=1).sum()
            deeper_bid_flow = bid_flow - l1_bid_flow
            deeper_ask_flow = ask_flow - l1_ask_flow
            
            # Give higher weight to level 1 (closer to mid)
            weighted_bid_flow = l1_bid_flow * 2.0 + deeper_bid_flow * 1.0
            weighted_ask_flow = l1_ask_flow * 2.0 + deeper_ask_flow * 1.0
            df[f'{ofi_name.replace("ofi_", "ofi_weighted_")}'] = weighted_bid_flow - weighted_ask_flow
        
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
        
        # Phase 1 Book Pressure Features
        # Add book depth asymmetry
        df['book_depth_asymmetry'] = (
            (df['total_bid_volume_10'] - df['total_ask_volume_10']) / 
            (df['total_bid_volume_10'] + df['total_ask_volume_10']).replace(0, 1)
        )
        
        # Volume concentration features for Phase 1
        df['bid_volume_concentration'] = df['bid_size_1'] / df['total_bid_volume_5'].replace(0, 1)
        df['ask_volume_concentration'] = df['ask_size_1'] / df['total_ask_volume_5'].replace(0, 1)
        
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
        
        # Phase 1 Stability Features
        # Individual quote lifetimes (already calculated above but renamed for Phase 1)
        df['bid_quote_lifetime'] = df['bid_life']
        df['ask_quote_lifetime'] = df['ask_life']
        
        # Book resilience (how stable is the book structure)
        df['book_resilience'] = 1 / (df['spread_stability_50'].replace(0, 0.001))
        
        # Book shape ratio between levels 1 and 5
        bid_shape_1_5 = df['bid_size_1'] / df['bid_size_5'].replace(0, 1)
        ask_shape_1_5 = df['ask_size_1'] / df['ask_size_5'].replace(0, 1)
        df['book_shape_1_5'] = (bid_shape_1_5 + ask_shape_1_5) / 2
        
        # Book shape stability (volatility of book shape over time)
        df['book_shape_stability'] = df['book_shape_1_5'].rolling(50, min_periods=1).std()
        
        # Volume concentration (already calculated in book pressure, but ensure it exists)
        if 'volume_concentration' not in df.columns:
            total_volume_l1 = df['bid_size_1'] + df['ask_size_1']
            total_volume_l5 = df['total_bid_volume_5'] + df['total_ask_volume_5']
            df['volume_concentration'] = total_volume_l1 / total_volume_l5.replace(0, 1)
        
        # Normalized spread stability features
        for window in [10, 50, 100]:
            df[f'spread_stability_norm_{window}'] = (
                df[f'spread_stability_{window}'] / df['spread_bps'].replace(0, 1)
            )
        
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

    def _define_phase1_features(self) -> Dict[str, List[str]]:
        """Define the 51 Phase 1 features organized by category."""
        return {
            'order_flow_imbalance': [
                'ofi_10s', 'ofi_30s', 'ofi_1m', 'ofi_5m',
                'ofi_normalized_10s', 'ofi_normalized_30s', 'ofi_normalized_1m', 'ofi_normalized_5m',
                'ofi_weighted_10s', 'ofi_weighted_30s', 'ofi_weighted_1m', 'ofi_weighted_5m'
            ],
            'book_pressure': [
                'bid_pressure', 'ask_pressure', 'bid_pressure_weighted', 'ask_pressure_weighted',
                'pressure_imbalance', 'pressure_imbalance_weighted', 'book_depth_asymmetry'
            ],
            'stability_indicators': [
                'bid_quote_lifetime', 'ask_quote_lifetime', 'quote_lifetime', 'book_resilience',
                'book_shape_1_5', 'book_shape_stability', 'volume_concentration',
                'spread_stability_10', 'spread_stability_50', 'spread_stability_100',
                'spread_stability_norm_10', 'spread_stability_norm_50', 'spread_stability_norm_100',
                'bid_volume_concentration', 'ask_volume_concentration'
            ],
            'enhanced_volatility': [
                'l2_volatility_10', 'l2_volatility_50', 'l2_volatility_200',
                'mid_price_return', 'volatility_10', 'volatility_30', 'volatility_100', 
                'volatility_200', 'volatility_500',
                'upside_vol_10', 'upside_vol_30', 'upside_vol_100', 'upside_vol_200', 'upside_vol_500',
                'downside_vol_10', 'downside_vol_30', 'downside_vol_100'
            ]
        }

    @contextmanager
    def get_db_connection(self):
        """Context manager for database connections with proper error handling."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.error(f"Database error: {e}")
            raise
        finally:
            conn.close()

    def read_existing_features(self, timestamp: str, symbol: str = 'BTCUSDT') -> Optional[Dict[str, float]]:
        """
        Read existing features from database for given timestamp.
        
        Args:
            timestamp: Timestamp to lookup (ISO format string)
            symbol: Trading symbol
            
        Returns:
            Dictionary of existing features or None if not found
        """
        try:
            with self.get_db_connection() as conn:
                cursor = conn.cursor()
                
                # Get all Phase 1 feature columns
                all_features = []
                for feature_group in self.phase1_features.values():
                    all_features.extend(feature_group)
                
                feature_columns = ', '.join(all_features)
                
                cursor.execute(f"""
                    SELECT {feature_columns}
                    FROM l2_features 
                    WHERE timestamp = ? AND symbol = ?
                    ORDER BY id DESC
                    LIMIT 1
                """, (timestamp, symbol))
                
                row = cursor.fetchone()
                if row:
                    self.cache_hits += 1
                    return dict(row)
                else:
                    self.cache_misses += 1
                    return None
                    
        except Exception as e:
            logger.error(f"Error reading existing features: {e}")
            return None

    def write_features_to_db(self, features_dict: Dict[str, float], timestamp: str, 
                           symbol: str = 'BTCUSDT', snapshot_id: Optional[int] = None) -> bool:
        """
        Write calculated features to database.
        
        Args:
            features_dict: Dictionary of feature name -> value
            timestamp: Timestamp for the features
            symbol: Trading symbol
            snapshot_id: Optional L2 snapshot ID for linking
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with self.get_db_connection() as conn:
                cursor = conn.cursor()
                
                # Filter to only Phase 1 features that exist in the database
                all_phase1_features = []
                for feature_group in self.phase1_features.values():
                    all_phase1_features.extend(feature_group)
                
                filtered_features = {k: v for k, v in features_dict.items() 
                                   if k in all_phase1_features and v is not None}
                
                if not filtered_features:
                    logger.warning("No valid Phase 1 features to write to database")
                    return False
                
                # Build dynamic INSERT statement
                base_columns = ['timestamp', 'symbol']
                base_values = [timestamp, symbol]
                
                if snapshot_id is not None:
                    base_columns.append('snapshot_id')
                    base_values.append(snapshot_id)
                
                feature_columns = list(filtered_features.keys())
                feature_values = list(filtered_features.values())
                
                all_columns = base_columns + feature_columns
                all_values = base_values + feature_values
                
                placeholders = ', '.join(['?' for _ in all_values])
                columns_str = ', '.join(all_columns)
                
                # Use INSERT OR REPLACE to handle duplicates
                cursor.execute(f"""
                    INSERT OR REPLACE INTO l2_features ({columns_str})
                    VALUES ({placeholders})
                """, all_values)
                
                logger.debug(f"Wrote {len(filtered_features)} features to database for {timestamp}")
                return True
                
        except Exception as e:
            logger.error(f"Error writing features to database: {e}")
            return False

    def generate_features_with_db_integration(self, df_input: pd.DataFrame, 
                                            force_recalculate: bool = False) -> pd.DataFrame:
        """
        Generate features with database integration using read-before-write pattern.
        
        Args:
            df_input: Input L2 DataFrame
            force_recalculate: If True, skip database lookup and recalculate all features
            
        Returns:
            DataFrame with features (either from DB or calculated)
        """
        if df_input is None or df_input.empty:
            logger.error("Input L2 DataFrame is empty")
            return pd.DataFrame()
        
        start_time = time.time()
        logger.info(f"Starting feature generation with DB integration for {len(df_input)} rows")
        
        df = df_input.copy()
        symbol = self.config.get('symbol', 'BTCUSDT').replace('/', '')  # Remove / for DB
        
        # Track performance metrics
        rows_from_db = 0
        rows_calculated = 0
        
        if not force_recalculate and 'timestamp' in df.columns:
            # Try to read existing features from database for each timestamp
            for idx, row in df.iterrows():
                timestamp = row['timestamp']
                if pd.isna(timestamp):
                    continue
                    
                # Convert timestamp to string format expected by database
                if hasattr(timestamp, 'strftime'):
                    timestamp_str = timestamp.strftime('%Y-%m-%d %H:%M:%S.%f')
                else:
                    timestamp_str = str(timestamp)
                
                existing_features = self.read_existing_features(timestamp_str, symbol)
                
                if existing_features:
                    # Use existing features from database
                    for feature_name, feature_value in existing_features.items():
                        if feature_value is not None:
                            df.at[idx, feature_name] = feature_value
                    rows_from_db += 1
                else:
                    # Mark this row for calculation
                    df.at[idx, '_needs_calculation'] = True
                    rows_calculated += 1
        else:
            # Mark all rows for calculation
            df['_needs_calculation'] = True
            rows_calculated = len(df)
        
        # Calculate features for rows that need them
        rows_needing_calculation = df[df.get('_needs_calculation', True) == True]
        
        if not rows_needing_calculation.empty:
            logger.info(f"Calculating features for {len(rows_needing_calculation)} rows")
            
            # Use existing feature calculation logic
            calculated_df = self.generate_features(rows_needing_calculation)
            
            # Write new features to database
            if not calculated_df.empty and 'timestamp' in calculated_df.columns:
                self._batch_write_features_to_db(calculated_df, symbol)
            
            # Merge calculated features back into main dataframe
            feature_columns = []
            for feature_group in self.phase1_features.values():
                feature_columns.extend(feature_group)
            
            # Update the main dataframe with calculated features
            for feature_col in feature_columns:
                if feature_col in calculated_df.columns:
                    mask = df.get('_needs_calculation', True) == True
                    df.loc[mask, feature_col] = calculated_df[feature_col].values
        
        # Clean up temporary column
        if '_needs_calculation' in df.columns:
            df = df.drop('_needs_calculation', axis=1)
        
        elapsed_time = time.time() - start_time
        cache_hit_rate = (self.cache_hits / (self.cache_hits + self.cache_misses) * 100) if (self.cache_hits + self.cache_misses) > 0 else 0
        
        logger.info(f"Feature generation complete in {elapsed_time:.3f}s")
        logger.info(f"Performance: {rows_from_db} from DB, {rows_calculated} calculated")
        logger.info(f"Cache hit rate: {cache_hit_rate:.1f}% ({self.cache_hits}/{self.cache_hits + self.cache_misses})")
        
        return df

    def _batch_write_features_to_db(self, df: pd.DataFrame, symbol: str):
        """
        Write features to database in batches for better performance.
        
        Args:
            df: DataFrame with calculated features
            symbol: Trading symbol
        """
        if df.empty or 'timestamp' not in df.columns:
            return
            
        batch_size = 100
        success_count = 0
        
        for start_idx in range(0, len(df), batch_size):
            end_idx = min(start_idx + batch_size, len(df))
            batch_df = df.iloc[start_idx:end_idx]
            
            for _, row in batch_df.iterrows():
                timestamp = row['timestamp']
                if pd.isna(timestamp):
                    continue
                    
                # Convert timestamp to string format
                if hasattr(timestamp, 'strftime'):
                    timestamp_str = timestamp.strftime('%Y-%m-%d %H:%M:%S.%f')
                else:
                    timestamp_str = str(timestamp)
                
                # Create features dictionary
                features_dict = {}
                all_phase1_features = []
                for feature_group in self.phase1_features.values():
                    all_phase1_features.extend(feature_group)
                
                for feature_name in all_phase1_features:
                    if feature_name in row and not pd.isna(row[feature_name]):
                        features_dict[feature_name] = row[feature_name]
                
                if features_dict:
                    if self.write_features_to_db(features_dict, timestamp_str, symbol):
                        success_count += 1
        
        logger.info(f"Successfully wrote {success_count} feature records to database")

    def get_performance_stats(self) -> Dict[str, any]:
        """Get performance statistics for the feature engineering process."""
        total_requests = self.cache_hits + self.cache_misses
        cache_hit_rate = (self.cache_hits / total_requests * 100) if total_requests > 0 else 0
        
        return {
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'total_requests': total_requests,
            'cache_hit_rate_percent': cache_hit_rate,
            'phase1_feature_count': sum(len(features) for features in self.phase1_features.values())
        }

    def get_feature_groups(self) -> Dict[str, List[str]]:
        """Return feature groups for analysis and selection."""
        # Return the Phase 1 features plus basic features
        feature_groups = self.phase1_features.copy()
        feature_groups['basic'] = [
            'spread', 'mid_price', 'microprice', 'order_book_imbalance',
            'spread_bps', 'weighted_mid_price'
        ]
        return feature_groups