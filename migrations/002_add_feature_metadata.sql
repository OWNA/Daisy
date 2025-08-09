
-- Migration 002: Populate feature metadata
BEGIN TRANSACTION;

-- Basic features
INSERT INTO feature_metadata (feature_name, feature_group, description) VALUES
('mid_price', 'basic', 'Mid price between best bid and ask'),
('spread', 'basic', 'Bid-ask spread in price units'),
('spread_bps', 'basic', 'Bid-ask spread in basis points'),
('microprice', 'basic', 'Volume-weighted mid price');

-- OFI features
INSERT INTO feature_metadata (feature_name, feature_group, description, calculation_params) VALUES
('ofi_10s', 'ofi', 'Order flow imbalance over 10 seconds', '{"window": 100, "unit": "rows"}'),
('ofi_30s', 'ofi', 'Order flow imbalance over 30 seconds', '{"window": 300, "unit": "rows"}'),
('ofi_1m', 'ofi', 'Order flow imbalance over 1 minute', '{"window": 600, "unit": "rows"}'),
('ofi_5m', 'ofi', 'Order flow imbalance over 5 minutes', '{"window": 3000, "unit": "rows"}');

-- Pressure features
INSERT INTO feature_metadata (feature_name, feature_group, description, calculation_params) VALUES
('bid_pressure_weighted', 'pressure', 'Exponentially weighted bid pressure', '{"decay_constant": 50}'),
('ask_pressure_weighted', 'pressure', 'Exponentially weighted ask pressure', '{"decay_constant": 50}'),
('pressure_imbalance_weighted', 'pressure', 'Weighted pressure imbalance', '{"decay_constant": 50}');

-- Stability features
INSERT INTO feature_metadata (feature_name, feature_group, description) VALUES
('quote_lifetime', 'stability', 'Average lifetime of best quotes'),
('book_resilience', 'stability', 'Ratio of top-level to deep-level volume'),
('spread_stability_50', 'stability', 'Spread volatility over 50 rows (5 seconds)');

INSERT INTO migration_log (migration_name, status) 
VALUES ('002_add_feature_metadata', 'completed');

COMMIT;
