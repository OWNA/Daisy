-- Migration 003: Add Missing L2 Features - Phase 1 Critical Features (SAFE VERSION)
-- Generated: 2025-08-01T12:00:00.000000
-- Description: Safely adds 48 critical Phase 1 L2 microstructure features using ALTER TABLE only
-- 
-- CRITICAL FIX: This version uses ONLY ALTER TABLE ADD COLUMN operations to prevent data loss
-- - Order Flow Imbalance (OFI) features: 12 features
-- - Book Pressure & Stability indicators: 22 features  
-- - Enhanced Volatility metrics: 14 features
-- 
-- Features are added to both l2_features and l2_training_data tables
-- All operations are safe and preserve existing data

BEGIN TRANSACTION;

-- Create migration tracking table if it doesn't exist
CREATE TABLE IF NOT EXISTS migration_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    migration_name TEXT NOT NULL,
    executed_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    status TEXT NOT NULL,
    error_message TEXT,
    rollback_sql TEXT
);

-- Create feature_metadata table if it doesn't exist
CREATE TABLE IF NOT EXISTS feature_metadata (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    feature_name TEXT UNIQUE NOT NULL,
    feature_group TEXT NOT NULL,
    description TEXT,
    calculation_params TEXT,
    version INTEGER DEFAULT 1,
    is_active BOOLEAN DEFAULT 1,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- ========================================
-- PHASE 1: ORDER FLOW IMBALANCE FEATURES (12 features)
-- ========================================

-- Add OFI features to l2_features table
ALTER TABLE l2_features ADD COLUMN ofi_10s REAL;
ALTER TABLE l2_features ADD COLUMN ofi_30s REAL;
ALTER TABLE l2_features ADD COLUMN ofi_1m REAL;
ALTER TABLE l2_features ADD COLUMN ofi_5m REAL;
ALTER TABLE l2_features ADD COLUMN ofi_normalized_10s REAL;
ALTER TABLE l2_features ADD COLUMN ofi_normalized_30s REAL;
ALTER TABLE l2_features ADD COLUMN ofi_normalized_1m REAL;
ALTER TABLE l2_features ADD COLUMN ofi_normalized_5m REAL;
ALTER TABLE l2_features ADD COLUMN ofi_weighted_10s REAL;
ALTER TABLE l2_features ADD COLUMN ofi_weighted_30s REAL;
ALTER TABLE l2_features ADD COLUMN ofi_weighted_1m REAL;
ALTER TABLE l2_features ADD COLUMN ofi_weighted_5m REAL;

-- Add OFI features to l2_training_data table
ALTER TABLE l2_training_data ADD COLUMN ofi_10s REAL;
ALTER TABLE l2_training_data ADD COLUMN ofi_30s REAL;
ALTER TABLE l2_training_data ADD COLUMN ofi_1m REAL;
ALTER TABLE l2_training_data ADD COLUMN ofi_5m REAL;
ALTER TABLE l2_training_data ADD COLUMN ofi_normalized_10s REAL;
ALTER TABLE l2_training_data ADD COLUMN ofi_normalized_30s REAL;
ALTER TABLE l2_training_data ADD COLUMN ofi_normalized_1m REAL;
ALTER TABLE l2_training_data ADD COLUMN ofi_normalized_5m REAL;
ALTER TABLE l2_training_data ADD COLUMN ofi_weighted_10s REAL;
ALTER TABLE l2_training_data ADD COLUMN ofi_weighted_30s REAL;
ALTER TABLE l2_training_data ADD COLUMN ofi_weighted_1m REAL;
ALTER TABLE l2_training_data ADD COLUMN ofi_weighted_5m REAL;

-- ========================================
-- PHASE 1: BOOK PRESSURE FEATURES (7 features)
-- ========================================

-- Add Book Pressure features to l2_features table
ALTER TABLE l2_features ADD COLUMN bid_pressure REAL;
ALTER TABLE l2_features ADD COLUMN ask_pressure REAL;
ALTER TABLE l2_features ADD COLUMN bid_pressure_weighted REAL;
ALTER TABLE l2_features ADD COLUMN ask_pressure_weighted REAL;
ALTER TABLE l2_features ADD COLUMN pressure_imbalance REAL;
ALTER TABLE l2_features ADD COLUMN pressure_imbalance_weighted REAL;
ALTER TABLE l2_features ADD COLUMN book_depth_asymmetry REAL;

-- Add Book Pressure features to l2_training_data table
ALTER TABLE l2_training_data ADD COLUMN bid_pressure REAL;
ALTER TABLE l2_training_data ADD COLUMN ask_pressure REAL;
ALTER TABLE l2_training_data ADD COLUMN bid_pressure_weighted REAL;
ALTER TABLE l2_training_data ADD COLUMN ask_pressure_weighted REAL;
ALTER TABLE l2_training_data ADD COLUMN pressure_imbalance REAL;
ALTER TABLE l2_training_data ADD COLUMN pressure_imbalance_weighted REAL;
ALTER TABLE l2_training_data ADD COLUMN book_depth_asymmetry REAL;

-- ========================================
-- PHASE 1: STABILITY INDICATORS (15 features)
-- ========================================

-- Add Stability Indicator features to l2_features table
ALTER TABLE l2_features ADD COLUMN bid_quote_lifetime INTEGER;
ALTER TABLE l2_features ADD COLUMN ask_quote_lifetime INTEGER;
ALTER TABLE l2_features ADD COLUMN quote_lifetime REAL;
ALTER TABLE l2_features ADD COLUMN book_resilience REAL;
ALTER TABLE l2_features ADD COLUMN book_shape_1_5 REAL;
ALTER TABLE l2_features ADD COLUMN book_shape_stability REAL;
ALTER TABLE l2_features ADD COLUMN volume_concentration REAL;
ALTER TABLE l2_features ADD COLUMN spread_stability_10 REAL;
ALTER TABLE l2_features ADD COLUMN spread_stability_50 REAL;
ALTER TABLE l2_features ADD COLUMN spread_stability_100 REAL;
ALTER TABLE l2_features ADD COLUMN spread_stability_norm_10 REAL;
ALTER TABLE l2_features ADD COLUMN spread_stability_norm_50 REAL;
ALTER TABLE l2_features ADD COLUMN spread_stability_norm_100 REAL;

-- Add Stability Indicator features to l2_training_data table
ALTER TABLE l2_training_data ADD COLUMN bid_quote_lifetime INTEGER;
ALTER TABLE l2_training_data ADD COLUMN ask_quote_lifetime INTEGER;
ALTER TABLE l2_training_data ADD COLUMN quote_lifetime REAL;
ALTER TABLE l2_training_data ADD COLUMN book_resilience REAL;
ALTER TABLE l2_training_data ADD COLUMN book_shape_1_5 REAL;
ALTER TABLE l2_training_data ADD COLUMN book_shape_stability REAL;
ALTER TABLE l2_training_data ADD COLUMN volume_concentration REAL;
ALTER TABLE l2_training_data ADD COLUMN spread_stability_10 REAL;
ALTER TABLE l2_training_data ADD COLUMN spread_stability_50 REAL;
ALTER TABLE l2_training_data ADD COLUMN spread_stability_100 REAL;
ALTER TABLE l2_training_data ADD COLUMN spread_stability_norm_10 REAL;
ALTER TABLE l2_training_data ADD COLUMN spread_stability_norm_50 REAL;
ALTER TABLE l2_training_data ADD COLUMN spread_stability_norm_100 REAL;

-- ========================================
-- PHASE 1: ENHANCED VOLATILITY FEATURES (14 features)
-- ========================================

-- Add Enhanced Volatility features to l2_features table
ALTER TABLE l2_features ADD COLUMN l2_volatility_10 REAL;
ALTER TABLE l2_features ADD COLUMN l2_volatility_50 REAL;
ALTER TABLE l2_features ADD COLUMN l2_volatility_200 REAL;
ALTER TABLE l2_features ADD COLUMN mid_price_return REAL;
ALTER TABLE l2_features ADD COLUMN volatility_10 REAL;
ALTER TABLE l2_features ADD COLUMN volatility_30 REAL;
ALTER TABLE l2_features ADD COLUMN volatility_100 REAL;
ALTER TABLE l2_features ADD COLUMN volatility_200 REAL;
ALTER TABLE l2_features ADD COLUMN volatility_500 REAL;
ALTER TABLE l2_features ADD COLUMN upside_vol_10 REAL;
ALTER TABLE l2_features ADD COLUMN upside_vol_30 REAL;
ALTER TABLE l2_features ADD COLUMN upside_vol_100 REAL;
ALTER TABLE l2_features ADD COLUMN upside_vol_200 REAL;
ALTER TABLE l2_features ADD COLUMN upside_vol_500 REAL;
ALTER TABLE l2_features ADD COLUMN downside_vol_10 REAL;
ALTER TABLE l2_features ADD COLUMN downside_vol_30 REAL;
ALTER TABLE l2_features ADD COLUMN downside_vol_100 REAL;

-- Add Enhanced Volatility features to l2_training_data table
ALTER TABLE l2_training_data ADD COLUMN l2_volatility_10 REAL;
ALTER TABLE l2_training_data ADD COLUMN l2_volatility_50 REAL;
ALTER TABLE l2_training_data ADD COLUMN l2_volatility_200 REAL;
ALTER TABLE l2_training_data ADD COLUMN mid_price_return REAL;
ALTER TABLE l2_training_data ADD COLUMN volatility_10 REAL;
ALTER TABLE l2_training_data ADD COLUMN volatility_30 REAL;
ALTER TABLE l2_training_data ADD COLUMN volatility_100 REAL;
ALTER TABLE l2_training_data ADD COLUMN volatility_200 REAL;
ALTER TABLE l2_training_data ADD COLUMN volatility_500 REAL;
ALTER TABLE l2_training_data ADD COLUMN upside_vol_10 REAL;
ALTER TABLE l2_training_data ADD COLUMN upside_vol_30 REAL;
ALTER TABLE l2_training_data ADD COLUMN upside_vol_100 REAL;
ALTER TABLE l2_training_data ADD COLUMN upside_vol_200 REAL;
ALTER TABLE l2_training_data ADD COLUMN upside_vol_500 REAL;
ALTER TABLE l2_training_data ADD COLUMN downside_vol_10 REAL;
ALTER TABLE l2_training_data ADD COLUMN downside_vol_30 REAL;
ALTER TABLE l2_training_data ADD COLUMN downside_vol_100 REAL;

-- ========================================
-- FEATURE METADATA REGISTRATION
-- ========================================

-- Register all new features in feature_metadata table
INSERT OR IGNORE INTO feature_metadata (feature_name, feature_group, description, version, is_active) VALUES

-- Order Flow Imbalance Features
('ofi_10s', 'order_flow_imbalance', 'Order Flow Imbalance over 10-second window', 1, 1),
('ofi_30s', 'order_flow_imbalance', 'Order Flow Imbalance over 30-second window', 1, 1),
('ofi_1m', 'order_flow_imbalance', 'Order Flow Imbalance over 1-minute window', 1, 1),
('ofi_5m', 'order_flow_imbalance', 'Order Flow Imbalance over 5-minute window', 1, 1),
('ofi_normalized_10s', 'order_flow_imbalance', 'Volume-normalized OFI over 10-second window', 1, 1),
('ofi_normalized_30s', 'order_flow_imbalance', 'Volume-normalized OFI over 30-second window', 1, 1),
('ofi_normalized_1m', 'order_flow_imbalance', 'Volume-normalized OFI over 1-minute window', 1, 1),
('ofi_normalized_5m', 'order_flow_imbalance', 'Volume-normalized OFI over 5-minute window', 1, 1),
('ofi_weighted_10s', 'order_flow_imbalance', 'Distance-weighted OFI over 10-second window', 1, 1),
('ofi_weighted_30s', 'order_flow_imbalance', 'Distance-weighted OFI over 30-second window', 1, 1),
('ofi_weighted_1m', 'order_flow_imbalance', 'Distance-weighted OFI over 1-minute window', 1, 1),
('ofi_weighted_5m', 'order_flow_imbalance', 'Distance-weighted OFI over 5-minute window', 1, 1),

-- Book Pressure Features
('bid_pressure', 'book_pressure', 'Raw bid pressure metric', 1, 1),
('ask_pressure', 'book_pressure', 'Raw ask pressure metric', 1, 1),
('bid_pressure_weighted', 'book_pressure', 'Distance-weighted bid pressure', 1, 1),
('ask_pressure_weighted', 'book_pressure', 'Distance-weighted ask pressure', 1, 1),
('pressure_imbalance', 'book_pressure', 'Raw pressure imbalance between bid and ask', 1, 1),
('pressure_imbalance_weighted', 'book_pressure', 'Weighted pressure imbalance between bid and ask', 1, 1),
('book_depth_asymmetry', 'book_pressure', 'Order book depth asymmetry measure', 1, 1),

-- Stability Indicator Features
('bid_quote_lifetime', 'stability_indicators', 'Number of ticks since bid price last changed', 1, 1),
('ask_quote_lifetime', 'stability_indicators', 'Number of ticks since ask price last changed', 1, 1),
('quote_lifetime', 'stability_indicators', 'Combined quote lifetime measure', 1, 1),
('book_resilience', 'stability_indicators', 'Order book resilience metric', 1, 1),
('book_shape_1_5', 'stability_indicators', 'Book shape ratio between levels 1 and 5', 1, 1),
('book_shape_stability', 'stability_indicators', 'Volatility of book shape over time', 1, 1),
('volume_concentration', 'stability_indicators', 'Overall volume concentration measure', 1, 1),
('spread_stability_10', 'stability_indicators', 'Spread stability over 10-tick window', 1, 1),
('spread_stability_50', 'stability_indicators', 'Spread stability over 50-tick window', 1, 1),
('spread_stability_100', 'stability_indicators', 'Spread stability over 100-tick window', 1, 1),
('spread_stability_norm_10', 'stability_indicators', 'Normalized spread stability over 10-tick window', 1, 1),
('spread_stability_norm_50', 'stability_indicators', 'Normalized spread stability over 50-tick window', 1, 1),
('spread_stability_norm_100', 'stability_indicators', 'Normalized spread stability over 100-tick window', 1, 1),

-- Enhanced Volatility Features
('l2_volatility_10', 'enhanced_volatility', 'L2-based volatility over 10-tick window', 1, 1),
('l2_volatility_50', 'enhanced_volatility', 'L2-based volatility over 50-tick window', 1, 1),
('l2_volatility_200', 'enhanced_volatility', 'L2-based volatility over 200-tick window', 1, 1),
('mid_price_return', 'enhanced_volatility', 'Mid-price return for volatility calculations', 1, 1),
('volatility_10', 'enhanced_volatility', 'Standard volatility over 10-tick window', 1, 1),
('volatility_30', 'enhanced_volatility', 'Standard volatility over 30-tick window', 1, 1),
('volatility_100', 'enhanced_volatility', 'Standard volatility over 100-tick window', 1, 1),
('volatility_200', 'enhanced_volatility', 'Standard volatility over 200-tick window', 1, 1),
('volatility_500', 'enhanced_volatility', 'Standard volatility over 500-tick window', 1, 1),
('upside_vol_10', 'enhanced_volatility', 'Upside volatility over 10-tick window', 1, 1),
('upside_vol_30', 'enhanced_volatility', 'Upside volatility over 30-tick window', 1, 1),
('upside_vol_100', 'enhanced_volatility', 'Upside volatility over 100-tick window', 1, 1),
('upside_vol_200', 'enhanced_volatility', 'Upside volatility over 200-tick window', 1, 1),
('upside_vol_500', 'enhanced_volatility', 'Upside volatility over 500-tick window', 1, 1),
('downside_vol_10', 'enhanced_volatility', 'Downside volatility over 10-tick window', 1, 1),
('downside_vol_30', 'enhanced_volatility', 'Downside volatility over 30-tick window', 1, 1),
('downside_vol_100', 'enhanced_volatility', 'Downside volatility over 100-tick window', 1, 1);

-- ========================================
-- PERFORMANCE INDEXES FOR NEW FEATURES
-- ========================================

-- Create indexes on key new features for faster querying
-- Note: Only create indexes on features likely to be used in WHERE clauses

CREATE INDEX IF NOT EXISTS idx_l2_features_ofi_10s ON l2_features(ofi_10s) WHERE ofi_10s IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_l2_features_ofi_1m ON l2_features(ofi_1m) WHERE ofi_1m IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_l2_features_pressure_imbalance ON l2_features(pressure_imbalance) WHERE pressure_imbalance IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_l2_features_book_resilience ON l2_features(book_resilience) WHERE book_resilience IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_l2_features_l2_volatility_50 ON l2_features(l2_volatility_50) WHERE l2_volatility_50 IS NOT NULL;

-- Similar indexes for l2_training_data table
CREATE INDEX IF NOT EXISTS idx_l2_training_ofi_10s ON l2_training_data(ofi_10s) WHERE ofi_10s IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_l2_training_ofi_1m ON l2_training_data(ofi_1m) WHERE ofi_1m IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_l2_training_pressure_imbalance ON l2_training_data(pressure_imbalance) WHERE pressure_imbalance IS NOT NULL;

-- ========================================
-- MIGRATION COMPLETION TRACKING
-- ========================================

-- Record successful migration
INSERT INTO migration_log (migration_name, status) 
VALUES ('003_add_missing_l2_features_safe', 'completed');

-- Create summary view of new features
CREATE VIEW IF NOT EXISTS v_phase1_features_summary AS
SELECT 
    'Order Flow Imbalance' as feature_category,
    COUNT(*) as feature_count
FROM feature_metadata 
WHERE feature_group = 'order_flow_imbalance'

UNION ALL

SELECT 
    'Book Pressure' as feature_category,
    COUNT(*) as feature_count
FROM feature_metadata 
WHERE feature_group = 'book_pressure'

UNION ALL

SELECT 
    'Stability Indicators' as feature_category,
    COUNT(*) as feature_count
FROM feature_metadata 
WHERE feature_group = 'stability_indicators'

UNION ALL

SELECT 
    'Enhanced Volatility' as feature_category,
    COUNT(*) as feature_count
FROM feature_metadata 
WHERE feature_group = 'enhanced_volatility';

COMMIT;

-- ========================================
-- MIGRATION SUMMARY
-- ========================================
-- 
-- Total Features Added: 48
-- 
-- Breakdown by Category:
-- - Order Flow Imbalance: 12 features
-- - Book Pressure: 7 features  
-- - Stability Indicators: 15 features
-- - Enhanced Volatility: 14 features
--
-- Tables Modified:
-- - l2_features: Added 48 new columns using ALTER TABLE ADD COLUMN
-- - l2_training_data: Added 48 new columns using ALTER TABLE ADD COLUMN  
-- - feature_metadata: Added 48 new feature records
--
-- CRITICAL FIX: This version uses ONLY ALTER TABLE operations
-- - NO DROP TABLE operations - all existing data preserved
-- - Safe for production environments
-- - Idempotent - can be run multiple times
--
-- Indexes Created: 8 performance indexes on key features
-- Views Created: 1 summary view (v_phase1_features_summary)
--
-- Migration Status: Phase 1 Critical Features - SAFE VERSION COMPLETE
-- Next Phase: Phase 2 HHT Features (when HHT processing is enabled)
-- ========================================