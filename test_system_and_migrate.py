#!/usr/bin/env python3
"""
Test system operations and design database migrations for enhanced features
"""

import os
import sys
import json
import sqlite3
from datetime import datetime
from typing import Dict, List, Optional

# Test basic system functionality
def test_main_operations():
    """Test basic main.py operations"""
    print("=" * 80)
    print("TESTING MAIN.PY OPERATIONS")
    print("=" * 80)
    
    tests = [
        ("Data Collection (1 min)", "python main.py collect --duration 1"),
        ("Model Training (1 trial)", "python main.py train --trials 1"),
        ("System Monitoring", "python main.py monitor --refresh 1")
    ]
    
    results = []
    for test_name, command in tests:
        print(f"\n[TEST] {test_name}")
        print(f"Command: {command}")
        # In production, these would actually run the commands
        # For now, we'll simulate the testing
        results.append({
            "test": test_name,
            "command": command,
            "status": "SIMULATED",
            "notes": "Requires Windows Python environment"
        })
    
    return results

# Analyze current database schema
def analyze_current_schema(db_path: str = "trading_bot.db"):
    """Analyze existing database schema"""
    print("\n" + "=" * 80)
    print("CURRENT DATABASE SCHEMA ANALYSIS")
    print("=" * 80)
    
    if not os.path.exists(db_path):
        print(f"Database not found at {db_path}")
        return None
        
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Get all tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = cursor.fetchall()
    
    schema_info = {}
    for table in tables:
        table_name = table[0]
        cursor.execute(f"PRAGMA table_info({table_name})")
        columns = cursor.fetchall()
        schema_info[table_name] = [
            {"name": col[1], "type": col[2], "nullable": not col[3], "default": col[4], "pk": col[5]}
            for col in columns
        ]
    
    conn.close()
    return schema_info

# Design new database schema for enhanced features
def design_enhanced_schema():
    """Design database schema for enhanced L2 features"""
    print("\n" + "=" * 80)
    print("PROPOSED DATABASE SCHEMA MIGRATIONS")
    print("=" * 80)
    
    migrations = {
        "l2_snapshots": """
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
        """,
        
        "l2_features": """
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
        """,
        
        "l2_features_optimized": """
        -- Optimized feature storage using compression
        CREATE TABLE IF NOT EXISTS l2_features_optimized (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            timestamp DATETIME NOT NULL,
            feature_group TEXT NOT NULL,  -- 'basic', 'ofi', 'pressure', 'stability'
            features_json TEXT NOT NULL,   -- Compressed JSON of features
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            INDEX idx_optimized_lookup (symbol, timestamp, feature_group)
        );
        """,
        
        "feature_metadata": """
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
        """,
        
        "migration_log": """
        -- Track schema migrations
        CREATE TABLE IF NOT EXISTS migration_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            migration_name TEXT NOT NULL,
            executed_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            status TEXT NOT NULL,
            error_message TEXT,
            rollback_sql TEXT
        );
        """
    }
    
    # Migration strategy
    migration_strategy = {
        "phase1": {
            "description": "Add new tables without modifying existing ones",
            "tables": ["l2_snapshots", "l2_features", "feature_metadata", "migration_log"],
            "risk": "low",
            "rollback": "DROP new tables"
        },
        "phase2": {
            "description": "Migrate existing L2 data to new schema",
            "steps": [
                "Copy data from existing features table to l2_features",
                "Validate data integrity",
                "Create optimized views for fast querying"
            ],
            "risk": "medium",
            "rollback": "Restore from backup"
        },
        "phase3": {
            "description": "Optimize storage and add compression",
            "tables": ["l2_features_optimized"],
            "steps": [
                "Compress feature groups into JSON",
                "Create materialized views for common queries",
                "Add partitioning by date if needed"
            ],
            "risk": "low",
            "rollback": "Use uncompressed tables"
        }
    }
    
    return migrations, migration_strategy

# Generate migration scripts
def generate_migration_scripts():
    """Generate SQL migration scripts"""
    print("\n" + "=" * 80)
    print("MIGRATION SCRIPTS")
    print("=" * 80)
    
    migrations, strategy = design_enhanced_schema()
    
    # Create migration files
    migration_files = {
        "001_add_l2_feature_tables.sql": f"""
-- Migration 001: Add L2 feature tables
-- Generated: {datetime.now().isoformat()}

BEGIN TRANSACTION;

{migrations['l2_snapshots']}

{migrations['l2_features']}

{migrations['feature_metadata']}

{migrations['migration_log']}

-- Record migration
INSERT INTO migration_log (migration_name, status) 
VALUES ('001_add_l2_feature_tables', 'completed');

COMMIT;
""",
        
        "002_add_feature_metadata.sql": """
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
""",
        
        "003_create_optimized_views.sql": """
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
"""
    }
    
    return migration_files

# Generate feature storage requirements document
def document_feature_requirements():
    """Document feature storage requirements for ML Specialist"""
    print("\n" + "=" * 80)
    print("FEATURE STORAGE REQUIREMENTS")
    print("=" * 80)
    
    requirements = {
        "overview": {
            "total_features": 118,
            "new_features": 34,
            "feature_groups": ["basic", "microstructure", "ofi", "pressure", "stability", "volatility"],
            "update_frequency": "100ms (real-time)",
            "retention_period": "30 days active, archive older"
        },
        
        "storage_estimates": {
            "per_snapshot": {
                "l2_snapshot": "~2KB (20 levels × 4 values)",
                "computed_features": "~1KB (118 features × 8 bytes)",
                "total_per_row": "~3KB"
            },
            "daily_volume": {
                "snapshots_per_day": 864000,  # 10 per second × 86400 seconds
                "storage_per_day": "~2.6GB",
                "monthly_storage": "~78GB"
            }
        },
        
        "performance_requirements": {
            "write_latency": "<10ms per feature set",
            "read_latency": "<5ms for latest features",
            "batch_read": "<100ms for 1000 rows",
            "concurrent_operations": "Support 10+ concurrent readers"
        },
        
        "feature_categories": {
            "basic_features": {
                "count": 10,
                "examples": ["mid_price", "spread", "microprice"],
                "update_frequency": "every snapshot",
                "storage": "always compute and store"
            },
            "ofi_features": {
                "count": 12,
                "examples": ["ofi_10s", "ofi_normalized_1m", "ofi_weighted_5m"],
                "update_frequency": "rolling window computation",
                "storage": "store computed values, not raw deltas"
            },
            "pressure_features": {
                "count": 7,
                "examples": ["bid_pressure_weighted", "pressure_imbalance"],
                "update_frequency": "every snapshot",
                "storage": "compute with exponential decay"
            },
            "stability_features": {
                "count": 9,
                "examples": ["quote_lifetime", "book_resilience", "spread_stability_50"],
                "update_frequency": "rolling statistics",
                "storage": "maintain rolling buffers"
            }
        },
        
        "optimization_strategies": {
            "compression": {
                "method": "Group related features into JSON",
                "benefit": "~60% storage reduction",
                "tradeoff": "Slightly higher CPU usage"
            },
            "partitioning": {
                "method": "Partition by date",
                "benefit": "Faster queries for recent data",
                "implementation": "Monthly partitions"
            },
            "caching": {
                "method": "Redis cache for latest features",
                "benefit": "<1ms read latency",
                "size": "~100MB for active symbols"
            }
        },
        
        "integration_notes": {
            "model_training": "Batch read historical features with timestamp filtering",
            "real_time_prediction": "Stream latest features from cache or DB",
            "backtesting": "Efficient time-range queries with proper indexing",
            "monitoring": "Track feature computation latency and storage growth"
        }
    }
    
    return requirements

# Identify architecture issues
def identify_architecture_issues():
    """Identify issues with the unified architecture"""
    print("\n" + "=" * 80)
    print("ARCHITECTURE ISSUES AND FIXES")
    print("=" * 80)
    
    issues = [
        {
            "issue": "Database schema lacks support for new features",
            "impact": "Cannot store enhanced features (OFI, pressure, stability)",
            "fix": "Implement proposed migration schema",
            "priority": "HIGH"
        },
        {
            "issue": "No feature versioning system",
            "impact": "Cannot track feature changes or A/B test",
            "fix": "Add feature_metadata table and version tracking",
            "priority": "MEDIUM"
        },
        {
            "issue": "Missing feature computation pipeline optimization",
            "impact": "Redundant calculations, high latency",
            "fix": "Implement feature computation caching and batch processing",
            "priority": "MEDIUM"
        },
        {
            "issue": "No data retention policy",
            "impact": "Database will grow unbounded",
            "fix": "Implement data archival and cleanup policies",
            "priority": "LOW"
        },
        {
            "issue": "Lack of feature monitoring",
            "impact": "Cannot detect feature drift or computation errors",
            "fix": "Add feature statistics tracking and alerting",
            "priority": "MEDIUM"
        }
    ]
    
    return issues

# Main execution
def main():
    """Run all tests and generate reports"""
    
    # 1. Test basic operations
    test_results = test_main_operations()
    
    # 2. Analyze current schema
    current_schema = analyze_current_schema()
    
    # 3. Design enhanced schema
    migrations, strategy = design_enhanced_schema()
    
    # 4. Generate migration scripts
    migration_files = generate_migration_scripts()
    
    # 5. Document requirements
    requirements = document_feature_requirements()
    
    # 6. Identify issues
    issues = identify_architecture_issues()
    
    # Save results
    report = {
        "timestamp": datetime.now().isoformat(),
        "test_results": test_results,
        "current_schema": current_schema,
        "migration_strategy": strategy,
        "feature_requirements": requirements,
        "architecture_issues": issues
    }
    
    with open("migration_report.json", "w") as f:
        json.dump(report, f, indent=2)
    
    # Save migration scripts
    os.makedirs("migrations", exist_ok=True)
    for filename, content in migration_files.items():
        with open(f"migrations/{filename}", "w") as f:
            f.write(content)
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"✓ Generated {len(migration_files)} migration scripts in ./migrations/")
    print(f"✓ Identified {len(issues)} architecture issues")
    print(f"✓ Documented requirements for {requirements['overview']['total_features']} features")
    print(f"✓ Full report saved to migration_report.json")
    
    return report

if __name__ == "__main__":
    main()