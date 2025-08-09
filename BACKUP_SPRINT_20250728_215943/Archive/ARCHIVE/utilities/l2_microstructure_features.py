#!/usr/bin/env python3
"""
Module: l2_microstructure_features.py
Description: Advanced L2 microstructure features for high-frequency trading
Author: L2-Only Strategy Implementation
Date: 2025-01-27
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
import logging
from scipy import stats
from scipy.optimize import minimize_scalar


class L2MicrostructureFeatures:
    """
    Advanced microstructure features from L2 order book data.
    
    Features:
    - Order flow toxicity
    - Market impact estimation
    - Liquidity measures (depth, resilience)
    - Information content metrics
    - Adverse selection indicators
    - Market maker inventory effects
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize L2 microstructure features calculator.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Microstructure parameters
        self.max_depth_levels = config.get('max_l2_depth_levels', 10)
        self.tick_size = config.get('tick_size', 0.01)
        self.min_volume_threshold = config.get('min_volume_threshold', 0.001)
        
        # Feature calculation windows
        self.short_window = config.get('microstructure_short_window', 100)  # 10 seconds
        self.medium_window = config.get('microstructure_medium_window', 300)  # 30 seconds
        self.long_window = config.get('microstructure_long_window', 600)  # 1 minute
        
        # Market impact parameters
        self.impact_volume_levels = config.get('impact_volume_levels', [0.1, 0.5, 1.0, 2.0])
        self.impact_decay_factor = config.get('impact_decay_factor', 0.95)
        
        self.logger.info("L2MicrostructureFeatures initialized")
        self.logger.info(f"Max depth levels: {self.max_depth_levels}")
        self.logger.info(f"Analysis windows: {self.short_window}, {self.medium_window}, {self.long_window}")

    def calculate_order_flow_toxicity(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate order flow toxicity (VPIN - Volume-Synchronized Probability of Informed Trading).
        
        Args:
            df: DataFrame with L2 order book data
            
        Returns:
            Series with toxicity measures
        """
        required_cols = ['bid_size_1', 'ask_size_1', 'bid_price_1', 'ask_price_1']
        if not all(col in df.columns for col in required_cols):
            self.logger.warning("Missing columns for toxicity calculation")
            return pd.Series(index=df.index, dtype=float).fillna(0.0)
        
        # Calculate volume imbalance
        total_volume = df['bid_size_1'] + df['ask_size_1']
        volume_imbalance = np.where(
            total_volume > 0,
            np.abs(df['bid_size_1'] - df['ask_size_1']) / total_volume,
            0
        )
        
        # Calculate price volatility
        mid_price = (df['bid_price_1'] + df['ask_price_1']) / 2
        price_returns = mid_price.pct_change().fillna(0)
        
        # Rolling volatility
        rolling_vol = price_returns.rolling(
            window=self.medium_window, min_periods=self.short_window // 2
        ).std().fillna(0)
        
        # VPIN calculation
        rolling_imbalance = pd.Series(volume_imbalance).rolling(
            window=self.medium_window, min_periods=self.short_window // 2
        ).mean().fillna(0)
        
        # Toxicity = Volume Imbalance / Price Volatility (normalized)
        toxicity = np.where(
            rolling_vol > 1e-8,
            rolling_imbalance / (rolling_vol + 1e-8),
            0
        )
        
        return pd.Series(toxicity, index=df.index).fillna(0.0)

    def calculate_market_impact(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate market impact for different volume levels.
        
        Args:
            df: DataFrame with L2 order book data
            
        Returns:
            DataFrame with market impact features
        """
        impact_features = pd.DataFrame(index=df.index)
        
        # Ensure we have sufficient depth data
        available_levels = 0
        for i in range(1, self.max_depth_levels + 1):
            if f'bid_price_{i}' in df.columns and f'ask_price_{i}' in df.columns:
                available_levels = i
            else:
                break
        
        if available_levels < 2:
            self.logger.warning("Insufficient depth for market impact calculation")
            for vol_level in self.impact_volume_levels:
                impact_features[f'market_impact_{vol_level}'] = 0.0
            return impact_features
        
        # Calculate cumulative volumes and prices
        bid_cumvol = pd.DataFrame(index=df.index)
        ask_cumvol = pd.DataFrame(index=df.index)
        bid_prices = pd.DataFrame(index=df.index)
        ask_prices = pd.DataFrame(index=df.index)
        
        for i in range(1, available_levels + 1):
            bid_vol_col = f'bid_size_{i}'
            ask_vol_col = f'ask_size_{i}'
            bid_price_col = f'bid_price_{i}'
            ask_price_col = f'ask_price_{i}'
            
            if all(col in df.columns for col in [bid_vol_col, ask_vol_col, bid_price_col, ask_price_col]):
                # Cumulative volumes
                if i == 1:
                    bid_cumvol[i] = df[bid_vol_col]
                    ask_cumvol[i] = df[ask_vol_col]
                else:
                    bid_cumvol[i] = bid_cumvol[i-1] + df[bid_vol_col]
                    ask_cumvol[i] = ask_cumvol[i-1] + df[ask_vol_col]
                
                bid_prices[i] = df[bid_price_col]
                ask_prices[i] = df[ask_price_col]
        
        # Calculate impact for each volume level
        mid_price = (df['bid_price_1'] + df['ask_price_1']) / 2
        
        for vol_level in self.impact_volume_levels:
            buy_impact = []
            sell_impact = []
            
            for idx in df.index:
                # Buy impact (walking up the ask side)
                target_volume = vol_level
                cumulative_vol = 0
                weighted_price = 0
                total_weight = 0
                
                for level in range(1, available_levels + 1):
                    if level in ask_cumvol.columns and level in ask_prices.columns:
                        level_vol = ask_cumvol.loc[idx, level] if level == 1 else (
                            ask_cumvol.loc[idx, level] - ask_cumvol.loc[idx, level-1]
                        )
                        level_price = ask_prices.loc[idx, level]
                        
                        if pd.notna(level_vol) and pd.notna(level_price) and level_vol > 0:
                            remaining_vol = min(level_vol, target_volume - cumulative_vol)
                            weighted_price += level_price * remaining_vol
                            total_weight += remaining_vol
                            cumulative_vol += remaining_vol
                            
                            if cumulative_vol >= target_volume:
                                break
                
                if total_weight > 0:
                    avg_execution_price = weighted_price / total_weight
                    buy_impact.append((avg_execution_price - mid_price.loc[idx]) / mid_price.loc[idx])
                else:
                    buy_impact.append(0.0)
                
                # Sell impact (walking down the bid side)
                cumulative_vol = 0
                weighted_price = 0
                total_weight = 0
                
                for level in range(1, available_levels + 1):
                    if level in bid_cumvol.columns and level in bid_prices.columns:
                        level_vol = bid_cumvol.loc[idx, level] if level == 1 else (
                            bid_cumvol.loc[idx, level] - bid_cumvol.loc[idx, level-1]
                        )
                        level_price = bid_prices.loc[idx, level]
                        
                        if pd.notna(level_vol) and pd.notna(level_price) and level_vol > 0:
                            remaining_vol = min(level_vol, target_volume - cumulative_vol)
                            weighted_price += level_price * remaining_vol
                            total_weight += remaining_vol
                            cumulative_vol += remaining_vol
                            
                            if cumulative_vol >= target_volume:
                                break
                
                if total_weight > 0:
                    avg_execution_price = weighted_price / total_weight
                    sell_impact.append((mid_price.loc[idx] - avg_execution_price) / mid_price.loc[idx])
                else:
                    sell_impact.append(0.0)
            
            # Average buy and sell impact
            impact_features[f'market_impact_{vol_level}'] = [
                (buy + sell) / 2 for buy, sell in zip(buy_impact, sell_impact)
            ]
        
        return impact_features.fillna(0.0)

    def calculate_liquidity_measures(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate various liquidity measures.
        
        Args:
            df: DataFrame with L2 order book data
            
        Returns:
            DataFrame with liquidity features
        """
        liquidity_features = pd.DataFrame(index=df.index)
        
        # Basic liquidity measures
        if 'bid_price_1' in df.columns and 'ask_price_1' in df.columns:
            # Bid-ask spread
            spread = df['ask_price_1'] - df['bid_price_1']
            mid_price = (df['bid_price_1'] + df['ask_price_1']) / 2
            liquidity_features['relative_spread'] = spread / mid_price
            
            # Effective spread (2x relative spread)
            liquidity_features['effective_spread'] = 2 * liquidity_features['relative_spread']
        
        # Depth measures
        total_bid_depth = 0
        total_ask_depth = 0
        depth_levels = 0
        
        for i in range(1, self.max_depth_levels + 1):
            bid_size_col = f'bid_size_{i}'
            ask_size_col = f'ask_size_{i}'
            
            if bid_size_col in df.columns and ask_size_col in df.columns:
                total_bid_depth += df[bid_size_col].fillna(0)
                total_ask_depth += df[ask_size_col].fillna(0)
                depth_levels = i
        
        if depth_levels > 0:
            liquidity_features['total_depth'] = total_bid_depth + total_ask_depth
            liquidity_features['depth_imbalance'] = (
                (total_bid_depth - total_ask_depth) / (total_bid_depth + total_ask_depth + 1e-8)
            )
            liquidity_features['avg_depth_per_level'] = liquidity_features['total_depth'] / depth_levels
        else:
            liquidity_features['total_depth'] = 0.0
            liquidity_features['depth_imbalance'] = 0.0
            liquidity_features['avg_depth_per_level'] = 0.0
        
        # Liquidity concentration (Herfindahl index)
        if depth_levels > 1:
            concentration_scores = []
            for idx in df.index:
                level_volumes = []
                for i in range(1, depth_levels + 1):
                    bid_vol = df.loc[idx, f'bid_size_{i}'] if f'bid_size_{i}' in df.columns else 0
                    ask_vol = df.loc[idx, f'ask_size_{i}'] if f'ask_size_{i}' in df.columns else 0
                    level_volumes.extend([bid_vol, ask_vol])
                
                total_vol = sum(level_volumes)
                if total_vol > 0:
                    shares = [vol / total_vol for vol in level_volumes]
                    hhi = sum(share ** 2 for share in shares)
                    concentration_scores.append(hhi)
                else:
                    concentration_scores.append(0.0)
            
            liquidity_features['liquidity_concentration'] = concentration_scores
        else:
            liquidity_features['liquidity_concentration'] = 0.0
        
        # Resilience measure (price recovery after impact)
        if 'bid_price_1' in df.columns and 'ask_price_1' in df.columns:
            mid_price = (df['bid_price_1'] + df['ask_price_1']) / 2
            price_changes = mid_price.pct_change().fillna(0)
            
            # Rolling standard deviation as resilience proxy
            resilience = price_changes.rolling(
                window=self.short_window, min_periods=self.short_window // 4
            ).std().fillna(0)
            
            # Invert so higher values = more resilient
            liquidity_features['resilience'] = 1 / (1 + resilience)
        else:
            liquidity_features['resilience'] = 0.5
        
        return liquidity_features.fillna(0.0)

    def calculate_information_content(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate information content measures.
        
        Args:
            df: DataFrame with L2 order book data
            
        Returns:
            DataFrame with information content features
        """
        info_features = pd.DataFrame(index=df.index)
        
        if 'bid_price_1' in df.columns and 'ask_price_1' in df.columns:
            mid_price = (df['bid_price_1'] + df['ask_price_1']) / 2
            
            # Price discovery efficiency
            price_changes = mid_price.pct_change().fillna(0)
            
            # Autocorrelation of price changes (lower = more efficient)
            rolling_autocorr = price_changes.rolling(
                window=self.medium_window, min_periods=self.short_window
            ).apply(lambda x: x.autocorr(lag=1) if len(x) > 1 else 0, raw=False).fillna(0)
            
            info_features['price_discovery_efficiency'] = 1 - np.abs(rolling_autocorr)
            
            # Information asymmetry (based on order flow imbalance)
            if 'bid_size_1' in df.columns and 'ask_size_1' in df.columns:
                total_volume = df['bid_size_1'] + df['ask_size_1']
                order_imbalance = np.where(
                    total_volume > 0,
                    (df['bid_size_1'] - df['ask_size_1']) / total_volume,
                    0
                )
                
                # Correlation between imbalance and future price changes
                future_returns = price_changes.shift(-1).fillna(0)
                
                rolling_corr = pd.Series(order_imbalance).rolling(
                    window=self.medium_window, min_periods=self.short_window
                ).corr(future_returns).fillna(0)
                
                info_features['information_asymmetry'] = np.abs(rolling_corr)
            else:
                info_features['information_asymmetry'] = 0.0
            
            # Volatility clustering (GARCH effect)
            squared_returns = price_changes ** 2
            volatility_autocorr = squared_returns.rolling(
                window=self.medium_window, min_periods=self.short_window
            ).apply(lambda x: x.autocorr(lag=1) if len(x) > 1 else 0, raw=False).fillna(0)
            
            info_features['volatility_clustering'] = np.abs(volatility_autocorr)
        else:
            info_features['price_discovery_efficiency'] = 0.5
            info_features['information_asymmetry'] = 0.0
            info_features['volatility_clustering'] = 0.0
        
        return info_features.fillna(0.0)

    def calculate_adverse_selection(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate adverse selection indicators.
        
        Args:
            df: DataFrame with L2 order book data
            
        Returns:
            Series with adverse selection measures
        """
        if not all(col in df.columns for col in ['bid_price_1', 'ask_price_1', 'bid_size_1', 'ask_size_1']):
            return pd.Series(index=df.index, dtype=float).fillna(0.0)
        
        # Calculate effective spread
        spread = df['ask_price_1'] - df['bid_price_1']
        mid_price = (df['bid_price_1'] + df['ask_price_1']) / 2
        effective_spread = spread / mid_price
        
        # Calculate realized spread (price reversal after trade)
        mid_price_changes = mid_price.pct_change().fillna(0)
        
        # Rolling correlation between spread and price changes
        spread_price_corr = effective_spread.rolling(
            window=self.medium_window, min_periods=self.short_window
        ).corr(mid_price_changes).fillna(0)
        
        # Adverse selection = portion of spread due to information
        # Higher correlation = more adverse selection
        adverse_selection = np.abs(spread_price_corr)
        
        return pd.Series(adverse_selection, index=df.index).fillna(0.0)

    def calculate_all_microstructure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all microstructure features.
        
        Args:
            df: DataFrame with L2 order book data
            
        Returns:
            DataFrame with all microstructure features
        """
        self.logger.info(f"Calculating microstructure features for {len(df)} rows")
        
        result_df = df.copy()
        
        # 1. Order flow toxicity
        result_df['order_flow_toxicity'] = self.calculate_order_flow_toxicity(df)
        
        # 2. Market impact features
        impact_features = self.calculate_market_impact(df)
        for col in impact_features.columns:
            result_df[col] = impact_features[col]
        
        # 3. Liquidity measures
        liquidity_features = self.calculate_liquidity_measures(df)
        for col in liquidity_features.columns:
            result_df[col] = liquidity_features[col]
        
        # 4. Information content
        info_features = self.calculate_information_content(df)
        for col in info_features.columns:
            result_df[col] = info_features[col]
        
        # 5. Adverse selection
        result_df['adverse_selection'] = self.calculate_adverse_selection(df)
        
        # Count microstructure features
        microstructure_cols = [col for col in result_df.columns 
                             if col not in df.columns]
        
        self.logger.info(f"Generated {len(microstructure_cols)} microstructure features")
        
        return result_df

    def validate_microstructure_features(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate microstructure features for quality control.
        
        Args:
            df: DataFrame with microstructure features
            
        Returns:
            Dictionary with validation results
        """
        validation_results = {
            'total_rows': len(df),
            'microstructure_columns': [],
            'quality_metrics': {},
            'warnings': []
        }
        
        # Identify microstructure columns
        microstructure_keywords = [
            'toxicity', 'impact', 'liquidity', 'spread', 'depth', 
            'resilience', 'information', 'adverse_selection'
        ]
        
        microstructure_cols = []
        for col in df.columns:
            if any(keyword in col.lower() for keyword in microstructure_keywords):
                microstructure_cols.append(col)
        
        validation_results['microstructure_columns'] = microstructure_cols
        
        for col in microstructure_cols:
            if col in df.columns:
                data = df[col].dropna()
                
                if len(data) > 0:
                    metrics = {
                        'non_null_count': len(data),
                        'mean': float(data.mean()),
                        'std': float(data.std()),
                        'min': float(data.min()),
                        'max': float(data.max()),
                        'negative_count': int((data < 0).sum()),
                        'zero_count': int((data == 0).sum()),
                        'extreme_count': int((np.abs(data) > 10).sum())
                    }
                    
                    validation_results['quality_metrics'][col] = metrics
                    
                    # Quality warnings
                    if metrics['negative_count'] > len(data) * 0.1:  # >10% negative
                        validation_results['warnings'].append(
                            f"{col}: {metrics['negative_count']} negative values"
                        )
                    
                    if metrics['zero_count'] > len(data) * 0.5:  # >50% zeros
                        validation_results['warnings'].append(
                            f"{col}: {metrics['zero_count']} zero values"
                        )
                    
                    if metrics['extreme_count'] > len(data) * 0.01:  # >1% extreme
                        validation_results['warnings'].append(
                            f"{col}: {metrics['extreme_count']} extreme values"
                        )
        
        return validation_results 