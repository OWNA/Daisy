-- Migration 001: Add L2 feature tables
-- Fixed for SQLite compatibility

BEGIN TRANSACTION;

-- Store computed L2 features cache
CREATE TABLE IF NOT EXISTS l2_features_cache (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp DATETIME NOT NULL,
    symbol TEXT NOT NULL,
    features_json TEXT NOT NULL,  -- JSON blob of all features
    feature_count INTEGER,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes separately
CREATE INDEX IF NOT EXISTS idx_l2_cache_symbol_timestamp 
ON l2_features_cache (symbol, timestamp);

-- Store feature metadata
CREATE TABLE IF NOT EXISTS feature_metadata (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    feature_name TEXT UNIQUE NOT NULL,
    feature_group TEXT NOT NULL,
    description TEXT,
    importance_score REAL,
    version INTEGER DEFAULT 1,
    is_active BOOLEAN DEFAULT 1,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Store execution analytics
CREATE TABLE IF NOT EXISTS execution_analytics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp DATETIME NOT NULL,
    symbol TEXT NOT NULL,
    strategy TEXT NOT NULL,
    ml_features_json TEXT,
    signal_strength REAL,
    order_size REAL,
    execution_price REAL,
    slippage_bps REAL,
    cost_bps REAL,
    fill_time_seconds REAL,
    success BOOLEAN,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Create index for execution analytics
CREATE INDEX IF NOT EXISTS idx_execution_symbol_timestamp 
ON execution_analytics (symbol, timestamp);

-- Insert key feature metadata
INSERT OR IGNORE INTO feature_metadata (feature_name, feature_group, description, importance_score) VALUES
('spread_stability_norm_100', 'stability', 'Normalized spread stability over 100 samples', 2.67),
('ofi_normalized_1m', 'order_flow', 'Normalized Order Flow Imbalance (1 minute)', 1.85),
('pressure_imbalance_weighted', 'book_pressure', 'Weighted pressure imbalance', 1.62),
('book_resilience', 'stability', 'Order book resilience metric', 1.43),
('volume_concentration', 'volume', 'Volume concentration at best levels', 1.21);

COMMIT;