"""
Module: data_upload_manager.py
Description: Manages large file uploads for L2 training data with database
Author: L2-Only Strategy Implementation
Date: 2025-01-27
"""

import os
import pandas as pd
import sqlite3
from pathlib import Path
import hashlib
from typing import Dict, Any
import logging
import yaml
from datetime import datetime


class DataUploadManager:
    """
    Manages large file uploads for L2 training data with database integration.
    
    Features:
    - Chunked processing for large files
    - File integrity checking with SHA256 hashes
    - Database field mapping and validation
    - Progress tracking and error handling
    - Support for multiple data types (L2 training, OHLCV, labels)
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the DataUploadManager.
        
        Args:
            config: Configuration dictionary containing upload settings
        """
        self.config = config
        self.upload_dir = Path(config.get('upload_dir', './uploads'))
        self.processed_dir = Path(
            config.get('processed_dir', './processed_data')
        )
        self.db_path = config.get('database_path', './trading_bot.db')
        print(f"DataUploadManager using database at: {os.path.abspath(self.db_path)}")
        self.chunk_size = config.get('upload_chunk_size', 5000)
        
        # Create directories
        self.upload_dir.mkdir(exist_ok=True)
        self.processed_dir.mkdir(exist_ok=True)
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
        
        # Load field mapping configuration
        self.field_mapping = self._load_field_mapping()
        
        # Initialize database
        self._initialize_database()
    
    def _load_field_mapping(self) -> Dict[str, Any]:
        """Load field mapping configuration from YAML file."""
        mapping_file = Path('field_mapping.yaml')
        if mapping_file.exists():
            with open(mapping_file, 'r') as f:
                return yaml.safe_load(f)
        else:
            # Default field mapping for L2 training data
            return {
                'l2_training_data_mapping': {
                    'timestamp': 'timestamp',
                    'symbol': 'symbol',
                    'bid_ask_spread': 'bid_ask_spread',
                    'bid_ask_spread_pct': 'bid_ask_spread_pct',
                    'weighted_mid_price': 'weighted_mid_price',
                    'microprice': 'microprice',
                    'order_book_imbalance_2': 'order_book_imbalance_2',
                    'order_book_imbalance_3': 'order_book_imbalance_3',
                    'order_book_imbalance_5': 'order_book_imbalance_5',
                    'total_bid_volume_2': 'total_bid_volume_2',
                    'total_ask_volume_2': 'total_ask_volume_2',
                    'total_bid_volume_3': 'total_bid_volume_3',
                    'total_ask_volume_3': 'total_ask_volume_3',
                    'price_impact_buy': 'price_impact_buy',
                    'price_impact_sell': 'price_impact_sell',
                    'price_impact_1': 'price_impact_1',
                    'price_impact_5': 'price_impact_5',
                    'price_impact_10': 'price_impact_10',
                    'bid_slope': 'bid_slope',
                    'ask_slope': 'ask_slope',
                    'hht_freq_imf0': 'hht_freq_imf0',
                    'hht_freq_imf1': 'hht_freq_imf1',
                    'hht_freq_imf2': 'hht_freq_imf2',
                    'hht_amp_imf0': 'hht_amp_imf0',
                    'hht_amp_imf1': 'hht_amp_imf1',
                    'hht_amp_imf2': 'hht_amp_imf2',
                    'l2_volatility_1min': 'l2_volatility_1min',
                    'l2_volatility_5min': 'l2_volatility_5min',
                    'realized_volatility': 'realized_volatility',
                    'order_flow_imbalance': 'order_flow_imbalance',
                    'trade_intensity': 'trade_intensity',
                    'effective_spread': 'effective_spread'
                },
                'validation_rules': {
                    'required_fields': ['timestamp'],
                    'numeric_fields': [
                        'bid_ask_spread', 'weighted_mid_price', 
                        'order_book_imbalance_2', 'price_impact_buy'
                    ],
                    'range_checks': {
                        'bid_ask_spread': [0, 1000]
                    },
                    'null_tolerance': {
                        'hht_features': 0.1,
                        'l2_features': 0.05
                    }
                }
            }
    
    def _initialize_database(self):
        """Initialize database tables for L2 training data."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Create L2 training data table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS l2_training_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME NOT NULL,
                    symbol VARCHAR(20) DEFAULT 'BTC/USDT:USDT',
                    
                    -- Raw L2 Data
                    bid_price_1 REAL,
                    bid_size_1 REAL,
                    ask_price_1 REAL,
                    ask_size_1 REAL,
                    mid REAL,
                    bid_ask_spread REAL,
                    bid_ask_spread_pct REAL,
                    
                    -- L2 Microstructure Features
                    weighted_mid_price REAL,
                    microprice REAL,
                    
                    -- Order Book Imbalance Features
                    order_book_imbalance_2 REAL,
                    order_book_imbalance_3 REAL,
                    order_book_imbalance_5 REAL,
                    
                    -- Volume Features
                    total_bid_volume_2 REAL,
                    total_ask_volume_2 REAL,
                    total_bid_volume_3 REAL,
                    total_ask_volume_3 REAL,
                    
                    -- Price Impact Features
                    price_impact_buy REAL,
                    price_impact_sell REAL,
                    price_impact_1 REAL,
                    price_impact_5 REAL,
                    price_impact_10 REAL,
                    
                    -- Slope Features
                    bid_slope REAL,
                    ask_slope REAL,
                    
                    -- HHT Features (L2-derived)
                    hht_freq_imf0 REAL,
                    hht_freq_imf1 REAL,
                    hht_freq_imf2 REAL,
                    hht_amp_imf0 REAL,
                    hht_amp_imf1 REAL,
                    hht_amp_imf2 REAL,
                    
                    -- Volatility Features
                    l2_volatility_1min REAL,
                    l2_volatility_5min REAL,
                    realized_volatility REAL,
                    
                    -- Order Flow Features
                    order_flow_imbalance REAL,
                    trade_intensity REAL,
                    effective_spread REAL,
                    
                    -- Metadata
                    data_quality_score REAL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create indexes for performance
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_timestamp "
                "ON l2_training_data(timestamp)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_symbol "
                "ON l2_training_data(symbol)"
            )
            
            
            # Create upload metadata table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS upload_metadata (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    file_name VARCHAR(255) NOT NULL,
                    file_hash VARCHAR(64) NOT NULL UNIQUE,
                    data_type VARCHAR(50) NOT NULL,
                    total_rows INTEGER NOT NULL,
                    upload_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    processing_time_seconds REAL,
                    status VARCHAR(20) DEFAULT 'completed'
                )
            """)
            
            conn.commit()
            self.logger.info("Database initialized successfully")
    
    def upload_large_csv(self, file_path: str, 
                         data_type: str = 'l2_training') -> Dict[str, Any]:
        """
        Upload and process large CSV files for training.
        
        Args:
            file_path: Path to the CSV file
            data_type: Type of data ('l2_training', 'ohlcv', 'labels')
            
        Returns:
            Dictionary with upload status and metadata
        """
        start_time = datetime.now()
        
        try:
            file_path_obj = Path(file_path)
            if not file_path_obj.exists():
                raise FileNotFoundError(f"File not found: {file_path}")
            
            self.logger.info(
                f"Starting upload of {file_path_obj.name} ({data_type})"
            )
            
            # Calculate file hash for integrity
            file_hash = self._calculate_file_hash(file_path_obj)
            
            # Check if already processed
            if self._is_file_processed(file_hash):
                return {
                    'status': 'already_processed',
                    'file_hash': file_hash,
                    'message': 'File already exists in database'
                }
            
            # Process file in chunks
            total_rows = 0
            chunk_count = 0
            
            for chunk in pd.read_csv(file_path, chunksize=self.chunk_size):
                chunk_count += 1
                rows_processed = self._process_chunk(
                    chunk, data_type, chunk_count
                )
                total_rows += rows_processed
                
                if chunk_count % 10 == 0:
                    self.logger.info(
                        f"Processed {chunk_count} chunks, "
                        f"{total_rows} total rows"
                    )
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Record upload metadata
            self._record_upload_metadata(
                file_path_obj, file_hash, data_type, 
                total_rows, processing_time
            )
            
            self.logger.info(
                f"Upload completed: {total_rows} rows "
                f"in {processing_time:.2f}s"
            )
            
            return {
                'status': 'success',
                'file_hash': file_hash,
                'total_rows': total_rows,
                'chunks_processed': chunk_count,
                'processing_time_seconds': processing_time,
                'message': f'Successfully uploaded {total_rows} rows'
            }
            
        except Exception as e:
            self.logger.error(f"Upload failed: {str(e)}")
            return {
                'status': 'error',
                'message': str(e)
            }
    
    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA256 hash of file for integrity checking."""
        hash_sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()
    
    def _is_file_processed(self, file_hash: str) -> bool:
        """Check if file with this hash has already been processed."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT COUNT(*) FROM upload_metadata WHERE file_hash = ?",
                (file_hash,)
            )
            return cursor.fetchone()[0] > 0
    
    def _process_chunk(self, chunk: pd.DataFrame, data_type: str, 
                       chunk_id: int) -> int:
        """Process a single chunk of data."""
        try:
            # Validate and clean chunk
            chunk_clean = self._validate_chunk(chunk, data_type)
            
            if chunk_clean.empty:
                self.logger.warning(
                    f"Chunk {chunk_id} is empty after validation"
                )
                return 0
            
            # Insert into appropriate table
            table_name = self._get_table_name(data_type)
            self._insert_chunk_to_db(chunk_clean, table_name)
            
            print(f"Uploaded {len(chunk_clean)} rows to {table_name}")
            
            return len(chunk_clean)
            
        except Exception as e:
            self.logger.error(f"Error processing chunk {chunk_id}: {str(e)}")
            return 0
    
    def _validate_chunk(self, chunk: pd.DataFrame, 
                        data_type: str) -> pd.DataFrame:
        """Validate and clean chunk data according to data type."""
        if data_type == 'l2_training':
            return self._validate_l2_training_chunk(chunk)
        elif data_type == 'ohlcv':
            return self._validate_ohlcv_chunk(chunk)
        else:
            return chunk
    
    def _validate_l2_training_chunk(self, chunk: pd.DataFrame) -> pd.DataFrame:
        """Validate L2 training data chunk with updated requirements."""
        required_columns = ['timestamp', 'bid_price_1', 'ask_price_1', 'mid']
        
        # Check required columns exist
        missing_cols = [
            col for col in required_columns if col not in chunk.columns
        ]
        if missing_cols:
            self.logger.warning(f"Missing required columns: {missing_cols}")
        
        # Convert timestamp
        if 'timestamp' in chunk.columns:
            chunk['timestamp'] = pd.to_datetime(chunk['timestamp'], utc=True)
        
        # Convert target column names from CSV format to database format
        # CSV: target_0.0015_50 -> DB: target_0_0015_50
        target_columns = [
            col for col in chunk.columns if col.startswith('target_')
        ]
        for col in target_columns:
            new_col = col.replace('.', '_')
            if new_col != col:
                chunk[new_col] = chunk[col]
                chunk.drop(col, axis=1, inplace=True)
        
        # Add symbol column if missing
        if 'symbol' not in chunk.columns:
            chunk['symbol'] = 'BTC/USDT'
        
        # Ensure numeric columns are numeric
        for col in ['bid_price_1', 'ask_price_1', 'bid_size_1', 'ask_size_1']:
            if col in chunk.columns:
                chunk[col] = pd.to_numeric(chunk[col], errors='coerce')
        
        print("Processed chunk:")
        print(chunk.head())
        
        # Validate numeric columns
        numeric_cols = chunk.select_dtypes(include=['number']).columns
        chunk[numeric_cols] = chunk[numeric_cols].replace(
            [float('inf'), -float('inf')], None
        )
        
        return chunk
    
    def _validate_ohlcv_chunk(self, chunk: pd.DataFrame) -> pd.DataFrame:
        """Validate OHLCV data chunk."""
        required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        
        # Check required columns
        missing_cols = [
            col for col in required_cols if col not in chunk.columns
        ]
        if missing_cols:
            raise ValueError(f"Missing required OHLCV columns: {missing_cols}")
        
        # Convert timestamp
        chunk['timestamp'] = pd.to_datetime(chunk['timestamp'], utc=True)
        
        # Validate OHLCV data integrity
        chunk = chunk[
            (chunk['high'] >= chunk['low']) &
            (chunk['high'] >= chunk['open']) &
            (chunk['high'] >= chunk['close']) &
            (chunk['low'] <= chunk['open']) &
            (chunk['low'] <= chunk['close']) &
            (chunk['volume'] >= 0)
        ]
        
        return chunk
    
    def _get_table_name(self, data_type: str) -> str:
        """Get database table name for data type."""
        table_mapping = {
            'l2_training': 'l2_training_data',
            'ohlcv': 'ohlcv_data',
            'labels': 'labels_data'
        }
        return table_mapping.get(data_type, 'l2_training_data')
    
    def _insert_chunk_to_db(self, chunk: pd.DataFrame, table_name: str):
        """Insert chunk data into database."""
        with sqlite3.connect(self.db_path) as conn:
            chunk.to_sql(table_name, conn, if_exists='append', index=False)
    
    def _record_upload_metadata(self, file_path: Path, file_hash: str, 
                                data_type: str, total_rows: int, 
                                processing_time: float):
        """Record upload metadata in database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO upload_metadata 
                (file_name, file_hash, data_type, total_rows, 
                 processing_time_seconds)
                VALUES (?, ?, ?, ?, ?)
            """, (file_path.name, file_hash, data_type, total_rows, 
                  processing_time))
            conn.commit()
    
    def get_database_stats(self) -> Dict[str, int]:
        """Get database statistics."""
        stats = {}
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # L2 training data rows
            cursor.execute("SELECT COUNT(*) FROM l2_training_data")
            stats['l2_training_rows'] = cursor.fetchone()[0]
            
            # Upload metadata
            cursor.execute("SELECT COUNT(*) FROM upload_metadata")
            stats['total_files'] = cursor.fetchone()[0]
            
            # OHLCV rows (if table exists)
            try:
                cursor.execute("SELECT COUNT(*) FROM ohlcv_data")
                stats['ohlcv_rows'] = cursor.fetchone()[0]
            except sqlite3.OperationalError:
                stats['ohlcv_rows'] = 0
        
        return stats
    
    def get_upload_history(self) -> pd.DataFrame:
        """Get upload history from database."""
        with sqlite3.connect(self.db_path) as conn:
            return pd.read_sql_query("""
                SELECT file_name, data_type, total_rows, 
                       upload_timestamp, processing_time_seconds, status
                FROM upload_metadata 
                ORDER BY upload_timestamp DESC
            """, conn)
    
    def validate_data_quality(self, 
                              data_type: str = 'l2_training') -> Dict[str, Any]:
        """Validate data quality in database."""
        table_name = self._get_table_name(data_type)
        
        with sqlite3.connect(self.db_path) as conn:
            # Get basic statistics
            df = pd.read_sql_query(
                f"SELECT * FROM {table_name} LIMIT 10000", conn
            )
            
            if df.empty:
                return {
                    'status': 'no_data', 
                    'message': 'No data found in database'
                }
            
            # Calculate quality metrics
            quality_metrics = {
                'total_rows': len(df),
                'null_percentages': (
                    df.isnull().sum() / len(df) * 100
                ).to_dict(),
                'duplicate_timestamps': df.duplicated(
                    subset=['timestamp']
                ).sum(),
                'date_range': {
                    'start': (
                        df['timestamp'].min().isoformat() 
                        if 'timestamp' in df.columns else None
                    ),
                    'end': (
                        df['timestamp'].max().isoformat() 
                        if 'timestamp' in df.columns else None
                    )
                },
                'numeric_ranges': {}
            }
            
            # Numeric column ranges
            numeric_cols = df.select_dtypes(include=['number']).columns
            for col in numeric_cols:
                quality_metrics['numeric_ranges'][col] = {
                    'min': float(df[col].min()),
                    'max': float(df[col].max()),
                    'mean': float(df[col].mean()),
                    'std': float(df[col].std())
                }
            
            return {
                'status': 'success',
                'quality_metrics': quality_metrics
            }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Data Upload Manager CLI')
    parser.add_argument('--delete-all', action='store_true', help='Delete all data from the specified tables')
    parser.add_argument('--l2-training', action='store_true', help='Apply actions to l2_training_data table')
    parser.add_argument('--l2-aligned', action='store_true', help='Apply actions to l2_aligned_data table')
    parser.add_argument('--l2-paper-trading', action='store_true', help='Apply actions to l2_paper_trading_data table')
    args = parser.parse_args()

    # Load configuration
    try:
        with open('config_l2_only.yaml', 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        config = {
            'upload_dir': './uploads',
            'processed_dir': './processed_data',
            'database_path': './trading_bot.db',
            'upload_chunk_size': 5000
        }
    
    # Initialize upload manager
    upload_manager = DataUploadManager(config)

    if args.delete_all:
        tables_to_delete = []
        if args.l2_training:
            tables_to_delete.append('l2_training_data')
        if args.l2_aligned:
            tables_to_delete.append('l2_aligned_data')
        if args.l2_paper_trading:
            tables_to_delete.append('l2_paper_trading_data')
        
        if tables_to_delete:
            with sqlite3.connect(upload_manager.db_path) as conn:
                cursor = conn.cursor()
                for table in tables_to_delete:
                    print(f"Deleting table: {table}")
                    cursor.execute(f"DROP TABLE IF EXISTS {table}")
                conn.commit()
            print("All specified tables have been deleted.")
        else:
            print("No tables specified for deletion.")
    else:
        # Upload the generated L2 features CSV
        l2_features_csv = "data/l2_features_for_upload.csv"
        if os.path.exists(l2_features_csv):
            print(f"\nAttempting to upload {l2_features_csv}...")
            result = upload_manager.upload_large_csv(l2_features_csv, 'l2_training')
            print(f"Upload result: {result}")
        else:
            print(f"Error: {l2_features_csv} not found. Please run convert_l2_to_csv.py first.")
    
    # Get database statistics
    stats = upload_manager.get_database_stats()
    print(f"Database stats: {stats}") 