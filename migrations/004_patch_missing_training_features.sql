-- Migration 004: Patch Missing Training Features
-- Generated: 2025-08-01T12:30:00.000000
-- Description: Adds the two missing volume concentration features to l2_training_data table
-- 
-- PATCH PURPOSE: Complete the Phase 1 L2 features migration by adding the two columns
-- that were inadvertently missed in migration 003 for the l2_training_data table.
--
-- Missing Features:
-- - bid_volume_concentration: Bid-side volume concentration measure
-- - ask_volume_concentration: Ask-side volume concentration measure
--
-- This patch ensures schema consistency between l2_features and l2_training_data tables.

BEGIN TRANSACTION;

-- ========================================
-- PATCH: ADD MISSING VOLUME CONCENTRATION FEATURES
-- ========================================

-- Add the two missing volume concentration features to l2_training_data table
ALTER TABLE l2_training_data ADD COLUMN bid_volume_concentration REAL;
ALTER TABLE l2_training_data ADD COLUMN ask_volume_concentration REAL;

-- ========================================
-- FEATURE METADATA REGISTRATION
-- ========================================

-- Register the missing features in feature_metadata table (if not already present)
INSERT OR IGNORE INTO feature_metadata (feature_name, feature_group, description, version, is_active) VALUES
('bid_volume_concentration', 'stability_indicators', 'Bid-side volume concentration measure', 1, 1),
('ask_volume_concentration', 'stability_indicators', 'Ask-side volume concentration measure', 1, 1);

-- ========================================
-- PERFORMANCE INDEXES
-- ========================================

-- Create indexes for the new columns (optional, for query optimization)
CREATE INDEX IF NOT EXISTS idx_l2_training_bid_vol_concentration ON l2_training_data(bid_volume_concentration) WHERE bid_volume_concentration IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_l2_training_ask_vol_concentration ON l2_training_data(ask_volume_concentration) WHERE ask_volume_concentration IS NOT NULL;

-- ========================================
-- MIGRATION COMPLETION TRACKING
-- ========================================

-- Record successful patch migration
INSERT INTO migration_log (migration_name, status) 
VALUES ('004_patch_missing_training_features', 'completed');

-- Update the feature summary view to reflect the patch
-- (The view will automatically include the new features on next query)

COMMIT;

-- ========================================
-- MIGRATION SUMMARY
-- ========================================
-- 
-- Patch Applied: 2 missing columns added to l2_training_data
-- 
-- Features Added:
-- - bid_volume_concentration (REAL)
-- - ask_volume_concentration (REAL)
--
-- Tables Modified:
-- - l2_training_data: Added 2 missing columns using ALTER TABLE ADD COLUMN
-- - feature_metadata: Added 2 feature records (if not already present)
--
-- Indexes Created: 2 performance indexes on new columns
--
-- Schema Consistency: l2_training_data now matches l2_features for Phase 1 features
--
-- Migration Status: Phase 1 L2 Features - FULLY COMPLETE
-- Ready for: Feature engineering integration and model retraining
-- ========================================