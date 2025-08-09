#!/usr/bin/env python3
"""
Missing L2 Features Migration List
Generated from comprehensive analysis of database schema vs feature engineering code

This module contains the definitive list of missing L2 features that need to be added
to the database schema for complete feature storage and retrieval.
"""

# Complete list of missing features organized by category and priority
MISSING_L2_FEATURES = {
    
    # HIGH PRIORITY - Core microstructure features used in current models
    "order_flow_imbalance": [
        # Time-windowed Order Flow Imbalance features (12 features)
        "ofi_10s",           # 10-second window OFI
        "ofi_30s",           # 30-second window OFI  
        "ofi_1m",            # 1-minute window OFI
        "ofi_5m",            # 5-minute window OFI
        "ofi_normalized_10s", # Volume-normalized OFI 10s
        "ofi_normalized_30s", # Volume-normalized OFI 30s
        "ofi_normalized_1m",  # Volume-normalized OFI 1m
        "ofi_normalized_5m",  # Volume-normalized OFI 5m
        "ofi_weighted_10s",   # Distance-weighted OFI 10s
        "ofi_weighted_30s",   # Distance-weighted OFI 30s
        "ofi_weighted_1m",    # Distance-weighted OFI 1m
        "ofi_weighted_5m",    # Distance-weighted OFI 5m
    ],
    
    "book_pressure": [
        # Book pressure metrics (6 features)
        "bid_pressure",              # Raw bid pressure
        "ask_pressure",              # Raw ask pressure
        "bid_pressure_weighted",     # Distance-weighted bid pressure
        "ask_pressure_weighted",     # Distance-weighted ask pressure
        "pressure_imbalance",        # Raw pressure imbalance
        "pressure_imbalance_weighted", # Weighted pressure imbalance
        "book_depth_asymmetry",      # Book symmetry measure
    ],
    
    "stability_indicators": [
        # Order book stability features (13 features)
        "bid_quote_lifetime",        # Ticks since bid price change
        "ask_quote_lifetime",        # Ticks since ask price change
        "quote_lifetime",            # Combined quote lifetime
        "book_resilience",           # Volume ratio metric
        "spread_stability_10",       # 10-tick spread volatility
        "spread_stability_50",       # 50-tick spread volatility
        "spread_stability_100",      # 100-tick spread volatility
        "spread_stability_norm_10",  # Normalized spread stability 10
        "spread_stability_norm_50",  # Normalized spread stability 50
        "spread_stability_norm_100", # Normalized spread stability 100
        "book_shape_1_5",           # Book shape ratio
        "book_shape_stability",      # Book shape volatility
        "volume_concentration",      # Volume concentration measure
        "bid_volume_concentration",  # Bid-side concentration
        "ask_volume_concentration",  # Ask-side concentration
    ],
    
    "enhanced_volatility": [
        # Missing volatility windows and calculations (15 features)
        "l2_volatility_10",          # 10-tick volatility (1 second)
        "l2_volatility_50",          # 50-tick volatility (5 seconds)  
        "l2_volatility_200",         # 200-tick volatility (20 seconds)
        "mid_price_return",          # Price returns for volatility calc
        "volatility_10", "volatility_30", "volatility_100", "volatility_200", "volatility_500",
        "upside_vol_10", "upside_vol_30", "upside_vol_100", "upside_vol_200", "upside_vol_500",
        # Note: More volatility features exist in enhanced engineer
    ],
    
    # HIGH PRIORITY - HHT Features (if HHT is enabled)
    "hht_core": [
        # Core HHT analysis features (10 features)
        "hht_trend_strength",        # Trend component strength
        "hht_trend_slope",           # Trend direction and slope
        "hht_cycle_phase",           # Current cycle phase
        "hht_dominant_freq",         # Dominant frequency component
        "hht_inst_amplitude",        # Instantaneous amplitude
        "hht_regime_classifier",     # Market regime (trend/range/noise)
        "hht_energy_high_freq",      # High frequency energy
        "hht_energy_mid_freq",       # Mid frequency energy
        "hht_energy_residual",       # Residual/trend energy
        "hht_data_quality",          # HHT calculation quality score
        "hht_calculation_time_ms",   # Performance tracking
    ],
    
    "hht_derived": [
        # HHT-derived and confluence features (12 features)
        "hht_trend_confidence",      # Trend confidence score
        "hht_market_state",          # Enhanced market state classification
        "ofi_10s_hht_qualified",     # HHT-qualified OFI signals
        "ofi_30s_hht_qualified",
        "ofi_1m_hht_qualified", 
        "obi_trend_aligned",         # Trend-aligned order book imbalance
        "l2_hht_signal_confluence",  # Multi-scale signal confluence
        "trend_persistence_hht",     # HHT trend persistence
        "mean_reversion_hht",        # HHT mean reversion signal
    ],
    
    # MEDIUM PRIORITY - Advanced microstructure features
    "advanced_flow": [
        # Enhanced flow analysis (20+ features)
        "flow_imbalance_10", "flow_imbalance_30", "flow_imbalance_100", "flow_imbalance_300",
        "flow_imbalance_ema_10", "flow_imbalance_ema_30", "flow_imbalance_ema_100", "flow_imbalance_ema_300",
        "level_1_flow_ratio", "level_2_flow_ratio", "level_3_flow_ratio", "level_4_flow_ratio", "level_5_flow_ratio",
        "flow_persistence",          # Flow autocorrelation
        "bid_concentration", "ask_concentration",  # Enhanced concentration metrics
    ],
    
    "temporal_patterns": [
        # Temporal and momentum features (15+ features)
        "book_updates",             # Order book update indicator
        "update_intensity_50", "update_intensity_100", "update_intensity_300",
        "large_bid", "large_ask",   # Large order indicators
        "size_clustering",          # Order size clustering
        "price_momentum_10", "price_momentum_30", "price_momentum_100",
        "imbalance_momentum_10", "imbalance_momentum_30", "imbalance_momentum_100",
    ],
    
    "market_regime": [
        # Market regime and efficiency features (10+ features)
        "efficiency_ratio_50", "efficiency_ratio_100", "efficiency_ratio_200",
        "trend_strength_30", "trend_strength_100",
        "trend_consistency_30", "trend_consistency_100", 
        "range_pct_50", "range_pct_100",
        "range_position_50", "range_position_100",
    ],
    
    # MEDIUM PRIORITY - Enhanced stability and regime features
    "enhanced_stability": [
        # Additional stability metrics
        "spread_stability_20",       # 20-tick spread stability
        "bid_change", "ask_change",  # Quote change indicators
        "bid_life", "ask_life",      # Individual quote lifetimes
        "depth_ratio",               # Bid/ask depth ratio
        "depth_ratio_stability",     # Depth ratio stability
        "imbalance_stability_3", "imbalance_stability_5", "imbalance_stability_10",
    ],
    
    "advanced_volatility": [
        # GARCH and advanced volatility features
        "downside_vol_10", "downside_vol_30", "downside_vol_100", "downside_vol_200", "downside_vol_500",
        "vol_skew_10", "vol_skew_30", "vol_skew_100", "vol_skew_200", "vol_skew_500",
        "squared_returns",           # For GARCH calculation
        "garch_vol",                # GARCH volatility estimate
        "vol_of_vol",               # Volatility of volatility
    ],
}

# Database column definitions for SQL migration
FEATURE_COLUMN_DEFINITIONS = {
    # Standard REAL columns for most features
    "REAL": [
        # Order Flow Imbalance
        "ofi_10s", "ofi_30s", "ofi_1m", "ofi_5m",
        "ofi_normalized_10s", "ofi_normalized_30s", "ofi_normalized_1m", "ofi_normalized_5m",
        "ofi_weighted_10s", "ofi_weighted_30s", "ofi_weighted_1m", "ofi_weighted_5m",
        
        # Book Pressure
        "bid_pressure", "ask_pressure", "bid_pressure_weighted", "ask_pressure_weighted",
        "pressure_imbalance", "pressure_imbalance_weighted", "book_depth_asymmetry",
        
        # Stability Indicators  
        "book_resilience", "spread_stability_10", "spread_stability_50", "spread_stability_100",
        "spread_stability_norm_10", "spread_stability_norm_50", "spread_stability_norm_100",
        "book_shape_1_5", "book_shape_stability", "volume_concentration",
        "bid_volume_concentration", "ask_volume_concentration",
        
        # Volatility Features
        "l2_volatility_10", "l2_volatility_50", "l2_volatility_200", "mid_price_return",
        "volatility_10", "volatility_30", "volatility_100", "volatility_200", "volatility_500",
        "upside_vol_10", "upside_vol_30", "upside_vol_100", "upside_vol_200", "upside_vol_500",
        "downside_vol_10", "downside_vol_30", "downside_vol_100", "downside_vol_200", "downside_vol_500",
        "vol_skew_10", "vol_skew_30", "vol_skew_100", "vol_skew_200", "vol_skew_500",
        "squared_returns", "garch_vol", "vol_of_vol",
        
        # HHT Core Features
        "hht_trend_strength", "hht_trend_slope", "hht_cycle_phase", "hht_dominant_freq",
        "hht_inst_amplitude", "hht_energy_high_freq", "hht_energy_mid_freq", "hht_energy_residual",
        "hht_data_quality", "hht_calculation_time_ms",
        
        # HHT Derived Features
        "hht_trend_confidence", "ofi_10s_hht_qualified", "ofi_30s_hht_qualified", "ofi_1m_hht_qualified",
        "obi_trend_aligned", "l2_hht_signal_confluence", "trend_persistence_hht", "mean_reversion_hht",
        
        # Advanced Flow
        "flow_imbalance_10", "flow_imbalance_30", "flow_imbalance_100", "flow_imbalance_300",
        "flow_imbalance_ema_10", "flow_imbalance_ema_30", "flow_imbalance_ema_100", "flow_imbalance_ema_300",
        "level_1_flow_ratio", "level_2_flow_ratio", "level_3_flow_ratio", "level_4_flow_ratio", "level_5_flow_ratio",
        "flow_persistence", "bid_concentration", "ask_concentration",
        
        # Temporal Patterns
        "size_clustering", "price_momentum_10", "price_momentum_30", "price_momentum_100",
        "imbalance_momentum_10", "imbalance_momentum_30", "imbalance_momentum_100",
        "update_intensity_50", "update_intensity_100", "update_intensity_300",
        
        # Market Regime
        "efficiency_ratio_50", "efficiency_ratio_100", "efficiency_ratio_200",
        "trend_strength_30", "trend_strength_100", "trend_consistency_30", "trend_consistency_100",
        "range_pct_50", "range_pct_100", "range_position_50", "range_position_100",
        
        # Enhanced Stability
        "spread_stability_20", "depth_ratio", "depth_ratio_stability",
        "imbalance_stability_3", "imbalance_stability_5", "imbalance_stability_10",
    ],
    
    # INTEGER columns for counters and classifiers
    "INTEGER": [
        "bid_quote_lifetime", "ask_quote_lifetime", "quote_lifetime",
        "hht_regime_classifier", "hht_market_state",
        "book_updates", "large_bid", "large_ask",
        "bid_change", "ask_change", "bid_life", "ask_life",
    ],
}

# Priority groups for phased migration
MIGRATION_PRIORITIES = {
    "phase_1_critical": [
        "order_flow_imbalance", "book_pressure", "stability_indicators", "enhanced_volatility"
    ],
    "phase_2_hht": [
        "hht_core", "hht_derived"
    ],
    "phase_3_advanced": [
        "advanced_flow", "temporal_patterns", "market_regime", "enhanced_stability", "advanced_volatility"
    ]
}

def get_all_missing_features():
    """Return flat list of all missing features"""
    all_features = []
    for category_features in MISSING_L2_FEATURES.values():
        all_features.extend(category_features)
    return sorted(list(set(all_features)))  # Remove duplicates

def get_features_by_priority(priority="phase_1_critical"):
    """Get features for specific migration phase"""
    if priority not in MIGRATION_PRIORITIES:
        return []
    
    features = []
    for category in MIGRATION_PRIORITIES[priority]:
        if category in MISSING_L2_FEATURES:
            features.extend(MISSING_L2_FEATURES[category])
    return sorted(list(set(features)))

def get_feature_sql_type(feature_name):
    """Get SQL column type for feature"""
    if feature_name in FEATURE_COLUMN_DEFINITIONS["INTEGER"]:
        return "INTEGER"
    else:
        return "REAL"  # Default for most features

def print_migration_summary():
    """Print summary of missing features by category"""
    print("L2 Database Migration - Missing Features Summary")
    print("=" * 60)
    
    total_features = 0
    for category, features in MISSING_L2_FEATURES.items():
        print(f"\n{category.upper().replace('_', ' ')}: {len(features)} features")
        for feature in features[:5]:  # Show first 5
            print(f"  - {feature}")
        if len(features) > 5:
            print(f"  ... and {len(features) - 5} more")
        total_features += len(features)
    
    print(f"\nTOTAL MISSING FEATURES: {total_features}")
    print(f"HIGH PRIORITY (Phase 1): {len(get_features_by_priority('phase_1_critical'))} features")
    print(f"HHT FEATURES (Phase 2): {len(get_features_by_priority('phase_2_hht'))} features")
    print(f"ADVANCED (Phase 3): {len(get_features_by_priority('phase_3_advanced'))} features")

if __name__ == "__main__":
    print_migration_summary()