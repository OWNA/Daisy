
-- Migration 003: Create optimized views for fast feature retrieval
BEGIN TRANSACTION;

-- View for latest features by symbol
CREATE VIEW IF NOT EXISTS v_latest_l2_features AS
SELECT 
    f.*,
    s.bid_prices,
    s.bid_sizes,
    s.ask_prices,
    s.ask_sizes
FROM l2_features f
JOIN l2_snapshots s ON f.snapshot_id = s.id
WHERE f.id IN (
    SELECT MAX(id) FROM l2_features 
    GROUP BY symbol
);

-- View for feature time series
CREATE VIEW IF NOT EXISTS v_l2_feature_timeseries AS
SELECT 
    symbol,
    timestamp,
    mid_price,
    spread_bps,
    order_book_imbalance,
    ofi_1m,
    pressure_imbalance_weighted,
    l2_volatility_50
FROM l2_features
ORDER BY symbol, timestamp;

-- Index for performance
CREATE INDEX IF NOT EXISTS idx_l2_features_composite 
ON l2_features(symbol, timestamp, mid_price, spread_bps);

INSERT INTO migration_log (migration_name, status) 
VALUES ('003_create_optimized_views', 'completed');

COMMIT;
