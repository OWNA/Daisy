#!/usr/bin/env python3
"""
feature_registry.py - Feature Registry for BTC Trading System

This module provides a centralized registry for defining, managing, and computing
all L2 microstructure features. It replaces scattered feature calculation logic
with a unified, scalable system.

Sprint 2 - Priority 0.2: Create Feature Registry
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class FeatureCategory(Enum):
    """Categories for organizing features."""
    SPREAD = "spread"
    VOLUME = "volume"
    VOLATILITY = "volatility"
    IMBALANCE = "imbalance"
    PRESSURE = "pressure"
    MOMENTUM = "momentum"
    PRICE = "price"
    MICROSTRUCTURE = "microstructure"
    FLOW = "flow"
    STABILITY = "stability"


class FeatureDataType(Enum):
    """Data types for features."""
    FLOAT = "float"
    INTEGER = "integer"
    BOOLEAN = "boolean"
    PERCENTAGE = "percentage"


@dataclass
class FeatureDependency:
    """Represents a feature dependency."""
    column_name: str
    source: str = "raw_data"
    window: Optional[int] = None
    lag: Optional[int] = None


@dataclass
class FeatureDefinition:
    """Complete definition of a trading feature."""
    
    # Basic metadata
    name: str
    category: FeatureCategory
    data_type: FeatureDataType
    description: str
    
    # Technical details
    calculation_func: Callable[[pd.DataFrame], pd.Series]
    dependencies: List[FeatureDependency] = field(default_factory=list)
    
    # Performance and validation
    expected_range: Optional[tuple] = None
    null_fill_method: str = "forward_fill"
    performance_weight: float = 1.0
    
    # Feature engineering metadata
    stability_score: Optional[float] = None
    interpretability: str = "high"  # high, medium, low
    computation_cost: str = "low"   # low, medium, high
    
    # Version and history
    version: str = "1.0"
    created_date: Optional[str] = None
    last_modified: Optional[str] = None
    
    def __post_init__(self):
        """Validate feature definition after initialization."""
        if not self.name:
            raise ValueError("Feature name cannot be empty")
        
        if not callable(self.calculation_func):
            raise ValueError("calculation_func must be callable")
        
        # Validate expected_range if provided
        if self.expected_range and len(self.expected_range) != 2:
            raise ValueError("expected_range must be a tuple of (min, max)")


class FeatureCalculator:
    """Utility class for common feature calculations."""
    
    @staticmethod
    def safe_divide(numerator: pd.Series, denominator: pd.Series, fill_value: float = 0.0) -> pd.Series:
        """Safely divide two series, handling division by zero."""
        return numerator / (denominator + 1e-8).replace(0, fill_value)
    
    @staticmethod
    def rolling_safe(series: pd.Series, window: int, min_periods: int = None, func: str = 'mean') -> pd.Series:
        """Safely apply rolling window function."""
        if min_periods is None:
            min_periods = max(1, window // 4)
        
        rolling_obj = series.rolling(window=window, min_periods=min_periods)
        
        if func == 'mean':
            return rolling_obj.mean()
        elif func == 'std':
            return rolling_obj.std()
        elif func == 'sum':
            return rolling_obj.sum()
        elif func == 'max':
            return rolling_obj.max()
        elif func == 'min':
            return rolling_obj.min()
        else:
            raise ValueError(f"Unsupported rolling function: {func}")
    
    @staticmethod
    def percentage_change(series: pd.Series, periods: int = 1) -> pd.Series:
        """Calculate percentage change with proper handling."""
        return series.pct_change(periods=periods)
    
    @staticmethod
    def z_score(series: pd.Series, window: int = 50) -> pd.Series:
        """Calculate rolling z-score."""
        rolling_mean = FeatureCalculator.rolling_safe(series, window, func='mean')
        rolling_std = FeatureCalculator.rolling_safe(series, window, func='std')
        return (series - rolling_mean) / (rolling_std + 1e-8)


class FeatureRegistry:
    """
    Central registry for all trading system features.
    
    Features:
    - Centralized feature definitions
    - Dependency management
    - Batch computation
    - Performance tracking
    - Feature validation
    """
    
    def __init__(self):
        """Initialize the feature registry."""
        self._features: Dict[str, FeatureDefinition] = {}
        self._feature_order: List[str] = []
        self._categories: Dict[FeatureCategory, List[str]] = {}
        self._computation_cache: Dict[str, pd.Series] = {}
        
        # Register basic L2 features
        self._register_basic_l2_features()
        
        logger.info(f"FeatureRegistry initialized with {len(self._features)} features")
    
    def register_feature(self, feature: FeatureDefinition, force_overwrite: bool = False):
        """Register a new feature definition."""
        if feature.name in self._features and not force_overwrite:
            raise ValueError(f"Feature '{feature.name}' already registered. Use force_overwrite=True to replace.")
        
        # Register the feature
        self._features[feature.name] = feature
        
        # Update feature order if new feature
        if feature.name not in self._feature_order:
            self._feature_order.append(feature.name)
        
        # Update category mapping
        if feature.category not in self._categories:
            self._categories[feature.category] = []
        
        if feature.name not in self._categories[feature.category]:
            self._categories[feature.category].append(feature.name)
        
        logger.info(f"Registered feature: {feature.name} ({feature.category.value})")
    
    def get_feature(self, name: str) -> Optional[FeatureDefinition]:
        """Get a feature definition by name."""
        return self._features.get(name)
    
    def get_features_by_category(self, category: FeatureCategory) -> List[FeatureDefinition]:
        """Get all features in a specific category."""
        feature_names = self._categories.get(category, [])
        return [self._features[name] for name in feature_names]
    
    def list_features(self) -> List[str]:
        """List all registered feature names."""
        return list(self._feature_order)
    
    def compute_feature(self, feature_name: str, data: pd.DataFrame, use_cache: bool = True) -> pd.Series:
        """Compute a single feature."""
        if use_cache and feature_name in self._computation_cache:
            cached_result = self._computation_cache[feature_name]
            if len(cached_result) == len(data):
                return cached_result
        
        feature = self.get_feature(feature_name)
        if feature is None:
            raise ValueError(f"Feature '{feature_name}' not found in registry")
        
        try:
            # Compute the feature
            result = feature.calculation_func(data)
            
            # Ensure result has correct name
            if hasattr(result, 'name'):
                result.name = feature_name
            
            # Apply null fill method
            result = self._apply_null_fill(result, feature.null_fill_method)
            
            # Validate result
            self._validate_feature_result(feature, result)
            
            # Cache result
            if use_cache:
                self._computation_cache[feature_name] = result
            
            return result
            
        except Exception as e:
            logger.error(f"Error computing feature '{feature_name}': {e}")
            # Return zero-filled series as fallback
            return pd.Series(0.0, index=data.index, name=feature_name)
    
    def compute_features(self, feature_names: List[str], data: pd.DataFrame, 
                        use_cache: bool = True) -> pd.DataFrame:
        """Compute multiple features efficiently."""
        if not feature_names:
            return pd.DataFrame(index=data.index)
        
        results = {}
        
        for feature_name in feature_names:
            try:
                results[feature_name] = self.compute_feature(feature_name, data, use_cache)
            except Exception as e:
                logger.error(f"Failed to compute feature '{feature_name}': {e}")
                # Add zero-filled column as fallback
                results[feature_name] = pd.Series(0.0, index=data.index, name=feature_name)
        
        return pd.DataFrame(results, index=data.index)
    
    def compute_all_features(self, data: pd.DataFrame, use_cache: bool = True) -> pd.DataFrame:
        """Compute all registered features."""
        return self.compute_features(self.list_features(), data, use_cache)
    
    def _apply_null_fill(self, series: pd.Series, method: str) -> pd.Series:
        """Apply null filling method to a series."""
        if method == "forward_fill":
            return series.ffill().bfill().fillna(0)
        elif method == "backward_fill":
            return series.bfill().ffill().fillna(0)
        elif method == "zero_fill":
            return series.fillna(0)
        elif method == "interpolate":
            return series.interpolate().fillna(0)
        else:
            logger.warning(f"Unknown null fill method: {method}, using forward_fill")
            return series.ffill().bfill().fillna(0)
    
    def _validate_feature_result(self, feature: FeatureDefinition, result: pd.Series):
        """Validate computed feature result."""
        if feature.expected_range:
            min_val, max_val = feature.expected_range
            if result.min() < min_val or result.max() > max_val:
                logger.warning(f"Feature '{feature.name}' values outside expected range "
                             f"[{min_val}, {max_val}]: actual range [{result.min():.4f}, {result.max():.4f}]")
    
    def clear_cache(self):
        """Clear the computation cache."""
        self._computation_cache.clear()
        logger.info("Feature computation cache cleared")
    
    def get_feature_info(self) -> pd.DataFrame:
        """Get information about all registered features."""
        info_data = []
        
        for feature_name in self._feature_order:
            feature = self._features[feature_name]
            info_data.append({
                'name': feature.name,
                'category': feature.category.value,
                'data_type': feature.data_type.value,
                'description': feature.description,
                'dependencies': len(feature.dependencies),
                'computation_cost': feature.computation_cost,
                'interpretability': feature.interpretability,
                'version': feature.version
            })
        
        return pd.DataFrame(info_data)
    
    def _register_basic_l2_features(self):
        """Register the initial 12 basic L2 features."""
        
        # 1. Spread Features
        spread_bps = FeatureDefinition(
            name="spread_bps",
            category=FeatureCategory.SPREAD,
            data_type=FeatureDataType.FLOAT,
            description="Bid-ask spread in basis points",
            calculation_func=lambda df: (df['spread'] / df['mid_price']) * 10000,
            dependencies=[
                FeatureDependency('spread', 'raw_data'),
                FeatureDependency('mid_price', 'raw_data')
            ],
            expected_range=(0, 1000),
            interpretability="high",
            computation_cost="low"
        )
        
        # 2. Volume Features
        total_bid_volume_5 = FeatureDefinition(
            name="total_bid_volume_5",
            category=FeatureCategory.VOLUME,
            data_type=FeatureDataType.FLOAT,
            description="Total bid volume across top 5 levels",
            calculation_func=lambda df: (
                df['bid_size_1'] + df['bid_size_2'] + df['bid_size_3'] + 
                df['bid_size_4'] + df['bid_size_5']
            ),
            dependencies=[
                FeatureDependency(f'bid_size_{i}', 'raw_data') for i in range(1, 6)
            ],
            expected_range=(0, float('inf')),
            interpretability="high",
            computation_cost="low"
        )
        
        total_ask_volume_5 = FeatureDefinition(
            name="total_ask_volume_5",
            category=FeatureCategory.VOLUME,
            data_type=FeatureDataType.FLOAT,
            description="Total ask volume across top 5 levels",
            calculation_func=lambda df: (
                df['ask_size_1'] + df['ask_size_2'] + df['ask_size_3'] + 
                df['ask_size_4'] + df['ask_size_5']
            ),
            dependencies=[
                FeatureDependency(f'ask_size_{i}', 'raw_data') for i in range(1, 6)
            ],
            expected_range=(0, float('inf')),
            interpretability="high",
            computation_cost="low"
        )
        
        # 3. Price Return Feature
        mid_price_return = FeatureDefinition(
            name="mid_price_return",
            category=FeatureCategory.PRICE,
            data_type=FeatureDataType.PERCENTAGE,
            description="Mid price percentage return",
            calculation_func=lambda df: df['mid_price'].pct_change(),
            dependencies=[FeatureDependency('mid_price', 'raw_data')],
            expected_range=(-0.1, 0.1),
            interpretability="high",
            computation_cost="low"
        )
        
        # 4. Volatility Features
        l2_volatility_10 = FeatureDefinition(
            name="l2_volatility_10",
            category=FeatureCategory.VOLATILITY,
            data_type=FeatureDataType.FLOAT,
            description="Rolling 10-period volatility of mid price returns",
            calculation_func=lambda df: FeatureCalculator.rolling_safe(
                df['mid_price'].pct_change(), 10, min_periods=2, func='std'
            ),
            dependencies=[FeatureDependency('mid_price', 'raw_data')],
            expected_range=(0, 1),
            interpretability="medium",
            computation_cost="medium"
        )
        
        l2_volatility_50 = FeatureDefinition(
            name="l2_volatility_50",
            category=FeatureCategory.VOLATILITY,
            data_type=FeatureDataType.FLOAT,
            description="Rolling 50-period volatility of mid price returns",
            calculation_func=lambda df: FeatureCalculator.rolling_safe(
                df['mid_price'].pct_change(), 50, min_periods=5, func='std'
            ),
            dependencies=[FeatureDependency('mid_price', 'raw_data')],
            expected_range=(0, 1),
            interpretability="medium",
            computation_cost="medium"
        )
        
        # 5. Imbalance Features
        order_book_imbalance_2 = FeatureDefinition(
            name="order_book_imbalance_2",
            category=FeatureCategory.IMBALANCE,
            data_type=FeatureDataType.FLOAT,
            description="Order book imbalance using top 2 levels",
            calculation_func=lambda df: FeatureCalculator.safe_divide(
                (df['bid_size_1'] + df['bid_size_2'] - df['ask_size_1'] - df['ask_size_2']),
                (df['bid_size_1'] + df['bid_size_2'] + df['ask_size_1'] + df['ask_size_2'])
            ),
            dependencies=[
                FeatureDependency('bid_size_1', 'raw_data'),
                FeatureDependency('bid_size_2', 'raw_data'),
                FeatureDependency('ask_size_1', 'raw_data'),
                FeatureDependency('ask_size_2', 'raw_data')
            ],
            expected_range=(-1, 1),
            interpretability="high",
            computation_cost="low"
        )
        
        # 6. Pressure Features
        bid_pressure = FeatureDefinition(
            name="bid_pressure",
            category=FeatureCategory.PRESSURE,
            data_type=FeatureDataType.PERCENTAGE,
            description="Bid pressure as ratio of bid volume to total volume",
            calculation_func=lambda df: FeatureCalculator.safe_divide(
                df['bid_size_1'] + df['bid_size_2'] + df['bid_size_3'] + df['bid_size_4'] + df['bid_size_5'],
                (df['bid_size_1'] + df['bid_size_2'] + df['bid_size_3'] + df['bid_size_4'] + df['bid_size_5'] +
                 df['ask_size_1'] + df['ask_size_2'] + df['ask_size_3'] + df['ask_size_4'] + df['ask_size_5'])
            ),
            dependencies=[
                FeatureDependency(f'bid_size_{i}', 'raw_data') for i in range(1, 6)
            ] + [
                FeatureDependency(f'ask_size_{i}', 'raw_data') for i in range(1, 6)
            ],
            expected_range=(0, 1),
            interpretability="high",
            computation_cost="low"
        )
        
        ask_pressure = FeatureDefinition(
            name="ask_pressure",
            category=FeatureCategory.PRESSURE,
            data_type=FeatureDataType.PERCENTAGE,
            description="Ask pressure as ratio of ask volume to total volume",
            calculation_func=lambda df: FeatureCalculator.safe_divide(
                df['ask_size_1'] + df['ask_size_2'] + df['ask_size_3'] + df['ask_size_4'] + df['ask_size_5'],
                (df['bid_size_1'] + df['bid_size_2'] + df['bid_size_3'] + df['bid_size_4'] + df['bid_size_5'] +
                 df['ask_size_1'] + df['ask_size_2'] + df['ask_size_3'] + df['ask_size_4'] + df['ask_size_5'])
            ),
            dependencies=[
                FeatureDependency(f'bid_size_{i}', 'raw_data') for i in range(1, 6)
            ] + [
                FeatureDependency(f'ask_size_{i}', 'raw_data') for i in range(1, 6)
            ],
            expected_range=(0, 1),
            interpretability="high",
            computation_cost="low"
        )
        
        # Register all basic features
        features_to_register = [
            spread_bps,
            total_bid_volume_5,
            total_ask_volume_5,
            mid_price_return,
            l2_volatility_10,
            l2_volatility_50,
            order_book_imbalance_2,
            bid_pressure,
            ask_pressure
        ]
        
        for feature in features_to_register:
            self.register_feature(feature)
        
        logger.info(f"Registered {len(features_to_register)} basic L2 features")


# Global registry instance
_default_registry = None

def get_default_registry() -> FeatureRegistry:
    """Get the default feature registry instance."""
    global _default_registry
    
    if _default_registry is None:
        _default_registry = FeatureRegistry()
    
    return _default_registry


def reset_default_registry():
    """Reset the default registry (useful for testing)."""
    global _default_registry
    _default_registry = None