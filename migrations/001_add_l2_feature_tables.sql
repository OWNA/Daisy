
-- Migration 001: Add L2 feature tables
-- Generated: 2025-07-29T16:02:32.264049

BEGIN TRANSACTION;


        -- Store raw L2 orderbook snapshots for feature computation
        CREATE TABLE IF NOT EXISTS l2_snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            timestamp DATETIME NOT NULL,
            bid_prices TEXT NOT NULL,  -- JSON array of bid prices
            bid_sizes TEXT NOT NULL,   -- JSON array of bid sizes
            ask_prices TEXT NOT NULL,  -- JSON array of ask prices
            ask_sizes TEXT NOT NULL,   -- JSON array of ask sizes
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            INDEX idx_l2_symbol_timestamp (symbol, timestamp)
        );
        


        -- Store computed L2 features with proper indexing
        CREATE TABLE IF NOT EXISTS l2_features (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            snapshot_id INTEGER NOT NULL,
            symbol TEXT NOT NULL,
            timestamp DATETIME NOT NULL,
            
            -- Basic features
            mid_price REAL NOT NULL,
            spread REAL NOT NULL,
            spread_bps REAL NOT NULL,
            
            -- Microstructure features
            microprice REAL,
            weighted_mid_price REAL,
            order_book_imbalance REAL,
            order_book_imbalance_2 REAL,
            order_book_imbalance_3 REAL,
            order_book_imbalance_5 REAL,
            
            -- Volume features
            total_bid_volume REAL,
            total_ask_volume REAL,
            bid_volume_concentration REAL,
            ask_volume_concentration REAL,
            
            -- Price impact features
            price_impact_buy REAL,
            price_impact_sell REAL,
            price_impact_1 REAL,
            price_impact_5 REAL,
            price_impact_10 REAL,
            
            -- Order Flow Imbalance (OFI) features
            ofi_10s REAL,
            ofi_30s REAL,
            ofi_1m REAL,
            ofi_5m REAL,
            ofi_normalized_10s REAL,
            ofi_normalized_30s REAL,
            ofi_normalized_1m REAL,
            ofi_normalized_5m REAL,
            ofi_weighted_10s REAL,
            ofi_weighted_30s REAL,
            ofi_weighted_1m REAL,
            ofi_weighted_5m REAL,
            
            -- Book pressure metrics
            bid_pressure REAL,
            bid_pressure_weighted REAL,
            ask_pressure REAL,
            ask_pressure_weighted REAL,
            pressure_imbalance REAL,
            pressure_imbalance_weighted REAL,
            book_depth_asymmetry REAL,
            
            -- Stability indicators
            bid_quote_lifetime INTEGER,
            ask_quote_lifetime INTEGER,
            quote_lifetime REAL,
            book_resilience REAL,
            spread_stability_10 REAL,
            spread_stability_50 REAL,
            spread_stability_100 REAL,
            book_shape_stability REAL,
            volume_concentration REAL,
            
            -- Volatility features
            l2_volatility_10 REAL,
            l2_volatility_50 REAL,
            l2_volatility_200 REAL,
            
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (snapshot_id) REFERENCES l2_snapshots(id),
            INDEX idx_features_symbol_timestamp (symbol, timestamp),
            INDEX idx_features_snapshot (snapshot_id)
        );
        


        -- Track feature versions and configurations
        CREATE TABLE IF NOT EXISTS feature_metadata (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            feature_name TEXT UNIQUE NOT NULL,
            feature_group TEXT NOT NULL,
            description TEXT,
            calculation_params TEXT,  -- JSON of parameters used
            version INTEGER DEFAULT 1,
            is_active BOOLEAN DEFAULT 1,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
        );
        


        -- Track schema migrations
        CREATE TABLE IF NOT EXISTS migration_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            migration_name TEXT NOT NULL,
            executed_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            status TEXT NOT NULL,
            error_message TEXT,
            rollback_sql TEXT
        );
        

-- Record migration
INSERT INTO migration_log (migration_name, status) 
VALUES ('001_add_l2_feature_tables', 'completed');

COMMIT;
