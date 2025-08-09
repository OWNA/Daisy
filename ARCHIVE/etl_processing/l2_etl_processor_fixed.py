#!/usr/bin/env python3
"""
L2 ETL Processor (Fixed) - Convert existing L2 data files to database format
This script processes the compressed L2 data files and loads them into the database
with a practical schema that works within SQLite limits.
"""

import os
import gzip
import json
import sqlite3
import pandas as pd
from datetime import datetime, timezone
from typing import Dict, List, Any
import traceback


class L2ETLProcessorFixed:
    """
    Processes L2 data files and loads them into the database with practical schema.
    """
    
    def __init__(self, db_path: str = "trading_bot.db", l2_data_dir: str = "l2_data"):
        """
        Initialize the ETL processor.
        
        Args:
            db_path: Path to the SQLite database
            l2_data_dir: Directory containing L2 data files
        """
        self.db_path = db_path
        self.l2_data_dir = l2_data_dir
        self.processed_files = []
        self.failed_files = []
        
    def _log(self, message: str) -> None:
        """Log a message with timestamp."""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f"[{timestamp}] {message}")
    
    def create_practical_l2_schema(self) -> None:
        """
        Create a practical L2 database schema that works within SQLite limits.
        Focus on the most important order book levels (top 10) plus aggregated data.
        """
        self._log("Creating practical L2 database schema...")
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create table with practical schema (top 10 levels + aggregated data)
            create_table_sql = """
            CREATE TABLE IF NOT EXISTS l2_training_data_practical (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME NOT NULL,
                symbol TEXT NOT NULL,
                exchange TEXT DEFAULT 'bybit',
                
                -- Top 10 bid levels
                bid_price_1 REAL, bid_size_1 REAL,
                bid_price_2 REAL, bid_size_2 REAL,
                bid_price_3 REAL, bid_size_3 REAL,
                bid_price_4 REAL, bid_size_4 REAL,
                bid_price_5 REAL, bid_size_5 REAL,
                bid_price_6 REAL, bid_size_6 REAL,
                bid_price_7 REAL, bid_size_7 REAL,
                bid_price_8 REAL, bid_size_8 REAL,
                bid_price_9 REAL, bid_size_9 REAL,
                bid_price_10 REAL, bid_size_10 REAL,
                
                -- Top 10 ask levels
                ask_price_1 REAL, ask_size_1 REAL,
                ask_price_2 REAL, ask_size_2 REAL,
                ask_price_3 REAL, ask_size_3 REAL,
                ask_price_4 REAL, ask_size_4 REAL,
                ask_price_5 REAL, ask_size_5 REAL,
                ask_price_6 REAL, ask_size_6 REAL,
                ask_price_7 REAL, ask_size_7 REAL,
                ask_price_8 REAL, ask_size_8 REAL,
                ask_price_9 REAL, ask_size_9 REAL,
                ask_price_10 REAL, ask_size_10 REAL,
                
                -- Calculated fields
                mid_price REAL,
                spread REAL,
                spread_bps REAL,
                
                -- Aggregated order book metrics
                total_bid_volume_10 REAL,
                total_ask_volume_10 REAL,
                weighted_bid_price REAL,
                weighted_ask_price REAL,
                order_book_imbalance REAL,
                
                -- Microstructure features
                microprice REAL,
                price_impact_bid REAL,
                price_impact_ask REAL,
                
                -- Target variables
                target_return_1min REAL,
                target_return_5min REAL,
                target_volatility REAL,
                target_direction INTEGER,
                
                -- Metadata
                update_id INTEGER,
                sequence_id INTEGER,
                data_quality_score REAL
            )
            """
            
            cursor.execute(create_table_sql)
            
            # Create indexes for performance
            indexes = [
                "CREATE INDEX IF NOT EXISTS idx_l2_prac_timestamp ON l2_training_data_practical(timestamp)",
                "CREATE INDEX IF NOT EXISTS idx_l2_prac_symbol ON l2_training_data_practical(symbol)",
                "CREATE INDEX IF NOT EXISTS idx_l2_prac_symbol_timestamp ON l2_training_data_practical(symbol, timestamp)"
            ]
            
            for index_sql in indexes:
                cursor.execute(index_sql)
            
            conn.commit()
            conn.close()
            
            self._log("Practical L2 database schema created successfully")
            
        except Exception as e:
            self._log(f"Error creating database schema: {e}")
            raise
    
    def calculate_microstructure_features(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate microstructure features from order book data.
        
        Args:
            record: L2 record with bid/ask data
            
        Returns:
            Record with additional microstructure features
        """
        try:
            # Get best bid and ask
            best_bid = record.get('bid_price_1')
            best_ask = record.get('ask_price_1')
            bid_size_1 = record.get('bid_size_1')
            ask_size_1 = record.get('ask_size_1')
            
            if best_bid and best_ask:
                # Basic calculations
                record['mid_price'] = (best_bid + best_ask) / 2
                record['spread'] = best_ask - best_bid
                record['spread_bps'] = (record['spread'] / record['mid_price']) * 10000
                
                # Microprice calculation (size-weighted mid)
                if bid_size_1 and ask_size_1:
                    total_size = bid_size_1 + ask_size_1
                    record['microprice'] = (best_bid * ask_size_1 + best_ask * bid_size_1) / total_size
                    
                    # Order book imbalance
                    record['order_book_imbalance'] = (bid_size_1 - ask_size_1) / total_size
                else:
                    record['microprice'] = record['mid_price']
                    record['order_book_imbalance'] = 0
                
                # Calculate aggregated volumes and weighted prices
                total_bid_volume = 0
                total_ask_volume = 0
                weighted_bid_sum = 0
                weighted_ask_sum = 0
                
                for i in range(1, 11):  # Top 10 levels
                    bid_price = record.get(f'bid_price_{i}')
                    bid_size = record.get(f'bid_size_{i}')
                    ask_price = record.get(f'ask_price_{i}')
                    ask_size = record.get(f'ask_size_{i}')
                    
                    if bid_price and bid_size:
                        total_bid_volume += bid_size
                        weighted_bid_sum += bid_price * bid_size
                    
                    if ask_price and ask_size:
                        total_ask_volume += ask_size
                        weighted_ask_sum += ask_price * ask_size
                
                record['total_bid_volume_10'] = total_bid_volume
                record['total_ask_volume_10'] = total_ask_volume
                
                if total_bid_volume > 0:
                    record['weighted_bid_price'] = weighted_bid_sum / total_bid_volume
                    # Price impact for buying (moving up the ask side)
                    record['price_impact_bid'] = (record['weighted_bid_price'] - best_bid) / best_bid * 10000
                else:
                    record['weighted_bid_price'] = best_bid
                    record['price_impact_bid'] = 0
                
                if total_ask_volume > 0:
                    record['weighted_ask_price'] = weighted_ask_sum / total_ask_volume
                    # Price impact for selling (moving down the bid side)
                    record['price_impact_ask'] = (best_ask - record['weighted_ask_price']) / best_ask * 10000
                else:
                    record['weighted_ask_price'] = best_ask
                    record['price_impact_ask'] = 0
                
                # Data quality score (percentage of non-null order book levels)
                non_null_levels = sum(1 for i in range(1, 11) 
                                    if record.get(f'bid_price_{i}') and record.get(f'ask_price_{i}'))
                record['data_quality_score'] = non_null_levels / 10.0
                
            else:
                # Set defaults if no valid bid/ask
                record['mid_price'] = None
                record['spread'] = None
                record['spread_bps'] = None
                record['microprice'] = None
                record['order_book_imbalance'] = None
                record['total_bid_volume_10'] = 0
                record['total_ask_volume_10'] = 0
                record['weighted_bid_price'] = None
                record['weighted_ask_price'] = None
                record['price_impact_bid'] = None
                record['price_impact_ask'] = None
                record['data_quality_score'] = 0
                
        except Exception as e:
            self._log(f"Error calculating microstructure features: {e}")
            # Set safe defaults
            record['data_quality_score'] = 0
        
        return record
    
    def process_l2_file(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Process a single L2 data file and extract order book data.
        
        Args:
            file_path: Path to the compressed L2 data file
            
        Returns:
            List of processed L2 records
        """
        records = []
        
        try:
            with gzip.open(file_path, 'rt', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        data = json.loads(line.strip())
                        
                        # Extract basic information
                        record = {
                            'timestamp': data.get('timestamp_ms', data.get('received_timestamp_ms')),
                            'symbol': data.get('symbol', 'BTC/USDT:USDT'),
                            'exchange': data.get('exchange', 'bybit'),
                            'update_id': data.get('update_id'),
                            'sequence_id': data.get('sequence_id')
                        }
                        
                        # Convert timestamp to datetime if it's in milliseconds
                        if record['timestamp']:
                            if isinstance(record['timestamp'], (int, float)):
                                record['timestamp'] = datetime.fromtimestamp(
                                    record['timestamp'] / 1000, tz=timezone.utc
                                )
                        
                        # Process bids and asks (top 10 levels only)
                        bids = data.get('bids', [])
                        asks = data.get('asks', [])
                        
                        # Extract top 10 levels of order book data
                        for i in range(10):
                            level = i + 1
                            
                            # Bid data
                            if i < len(bids) and len(bids[i]) >= 2:
                                record[f'bid_price_{level}'] = float(bids[i][0])
                                record[f'bid_size_{level}'] = float(bids[i][1])
                            else:
                                record[f'bid_price_{level}'] = None
                                record[f'bid_size_{level}'] = None
                            
                            # Ask data
                            if i < len(asks) and len(asks[i]) >= 2:
                                record[f'ask_price_{level}'] = float(asks[i][0])
                                record[f'ask_size_{level}'] = float(asks[i][1])
                            else:
                                record[f'ask_price_{level}'] = None
                                record[f'ask_size_{level}'] = None
                        
                        # Calculate microstructure features
                        record = self.calculate_microstructure_features(record)
                        
                        records.append(record)
                        
                    except json.JSONDecodeError as e:
                        if line_num <= 5:  # Only log first few errors
                            self._log(f"JSON decode error in {file_path} line {line_num}: {e}")
                        continue
                    except Exception as e:
                        if line_num <= 5:  # Only log first few errors
                            self._log(f"Error processing line {line_num} in {file_path}: {e}")
                        continue
                        
        except Exception as e:
            self._log(f"Error reading file {file_path}: {e}")
            raise
        
        return records
    
    def calculate_targets(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate target variables for training.
        
        Args:
            df: DataFrame with L2 data
            
        Returns:
            DataFrame with target columns added
        """
        if df.empty or 'mid_price' not in df.columns:
            return df
        
        # Sort by timestamp to ensure proper order
        df = df.sort_values('timestamp').copy()
        
        # Use mid_price for target calculations
        price_col = 'mid_price'
        
        # Calculate returns for different horizons (assuming 1-second intervals)
        df['price_shift_1min'] = df[price_col].shift(-60)
        df['price_shift_5min'] = df[price_col].shift(-300)
        
        # Calculate return targets
        df['target_return_1min'] = (df['price_shift_1min'] / df[price_col] - 1) * 100
        df['target_return_5min'] = (df['price_shift_5min'] / df[price_col] - 1) * 100
        
        # Calculate volatility target (rolling standard deviation of returns)
        df['returns'] = df[price_col].pct_change()
        df['target_volatility'] = df['returns'].rolling(window=60, min_periods=10).std() * 100
        
        # Calculate direction target (1 for up, -1 for down, 0 for neutral)
        df['target_direction'] = 0
        df.loc[df['target_return_1min'] > 0.01, 'target_direction'] = 1  # Up if > 1bp
        df.loc[df['target_return_1min'] < -0.01, 'target_direction'] = -1  # Down if < -1bp
        
        # Clean up temporary columns
        df = df.drop(['price_shift_1min', 'price_shift_5min', 'returns'], axis=1, errors='ignore')
        
        return df
    
    def load_to_database(self, records: List[Dict[str, Any]], table_name: str = "l2_training_data_practical") -> None:
        """
        Load processed records to the database using smaller batches.
        
        Args:
            records: List of L2 records to load
            table_name: Target table name
        """
        if not records:
            return
        
        try:
            # Convert to DataFrame for easier manipulation
            df = pd.DataFrame(records)
            
            # Calculate targets
            df = self.calculate_targets(df)
            
            # Connect to database and insert in smaller chunks
            conn = sqlite3.connect(self.db_path)
            
            # Use smaller chunk size to avoid SQL variable limit
            chunk_size = 100  # Much smaller to stay well under 999 variable limit
            
            for i in range(0, len(df), chunk_size):
                chunk = df.iloc[i:i + chunk_size]
                chunk.to_sql(table_name, conn, if_exists='append', index=False, method='multi')
            
            conn.close()
            
            self._log(f"Loaded {len(records)} records to {table_name}")
            
        except Exception as e:
            self._log(f"Error loading records to database: {e}")
            raise
    
    def process_all_files(self) -> Dict[str, Any]:
        """
        Process all L2 data files in the directory.
        
        Returns:
            Summary statistics of the processing
        """
        self._log("Starting L2 ETL processing with practical schema...")
        
        # Create practical database schema first
        self.create_practical_l2_schema()
        
        # Get list of L2 data files
        if not os.path.exists(self.l2_data_dir):
            self._log(f"L2 data directory {self.l2_data_dir} not found")
            return {"error": "L2 data directory not found"}
        
        l2_files = [f for f in os.listdir(self.l2_data_dir) if f.endswith('.jsonl.gz')]
        
        if not l2_files:
            self._log("No L2 data files found")
            return {"error": "No L2 data files found"}
        
        self._log(f"Found {len(l2_files)} L2 data files to process")
        
        total_records = 0
        
        for file_name in l2_files:
            file_path = os.path.join(self.l2_data_dir, file_name)
            
            try:
                self._log(f"Processing {file_name}...")
                
                # Process the file
                records = self.process_l2_file(file_path)
                
                if records:
                    # Load to database in smaller batches
                    batch_size = 500  # Process in smaller batches
                    for i in range(0, len(records), batch_size):
                        batch = records[i:i + batch_size]
                        self.load_to_database(batch)
                    
                    total_records += len(records)
                    self.processed_files.append(file_name)
                    self._log(f"Successfully processed {file_name}: {len(records)} records")
                else:
                    self._log(f"No valid records found in {file_name}")
                    
            except Exception as e:
                self._log(f"Failed to process {file_name}: {e}")
                self.failed_files.append((file_name, str(e)))
                continue
        
        # Generate summary
        summary = {
            "total_files_found": len(l2_files),
            "files_processed": len(self.processed_files),
            "files_failed": len(self.failed_files),
            "total_records_loaded": total_records,
            "processed_files": self.processed_files,
            "failed_files": self.failed_files
        }
        
        self._log("L2 ETL processing completed!")
        self._log(f"Summary: {summary}")
        
        return summary
    
    def validate_data_quality(self) -> Dict[str, Any]:
        """
        Validate the quality of loaded L2 data.
        
        Returns:
            Data quality metrics
        """
        self._log("Validating L2 data quality...")
        
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Basic statistics
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM l2_training_data_practical")
            total_rows = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(DISTINCT symbol) FROM l2_training_data_practical")
            unique_symbols = cursor.fetchone()[0]
            
            cursor.execute("SELECT MIN(timestamp), MAX(timestamp) FROM l2_training_data_practical")
            date_range = cursor.fetchone()
            
            # Data quality checks
            cursor.execute("SELECT COUNT(*) FROM l2_training_data_practical WHERE bid_price_1 IS NULL")
            null_bids = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM l2_training_data_practical WHERE ask_price_1 IS NULL")
            null_asks = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM l2_training_data_practical WHERE mid_price IS NULL")
            null_mid = cursor.fetchone()[0]
            
            # Target availability
            cursor.execute("SELECT COUNT(*) FROM l2_training_data_practical WHERE target_return_1min IS NOT NULL")
            valid_targets = cursor.fetchone()[0]
            
            # Average data quality score
            cursor.execute("SELECT AVG(data_quality_score) FROM l2_training_data_practical")
            avg_quality_score = cursor.fetchone()[0] or 0
            
            conn.close()
            
            quality_metrics = {
                "total_rows": total_rows,
                "unique_symbols": unique_symbols,
                "date_range": date_range,
                "null_bid_price_1": null_bids,
                "null_ask_price_1": null_asks,
                "null_mid_price": null_mid,
                "valid_targets": valid_targets,
                "avg_data_quality_score": avg_quality_score,
                "overall_quality_score": (total_rows - null_bids - null_asks) / total_rows if total_rows > 0 else 0
            }
            
            self._log(f"Data quality metrics: {quality_metrics}")
            return quality_metrics
            
        except Exception as e:
            self._log(f"Error validating data quality: {e}")
            return {"error": str(e)}


def main():
    """Main entry point for the ETL processor."""
    print("="*60)
    print("L2 ETL PROCESSOR (FIXED)")
    print("="*60)
    print(f"Started at: {datetime.now()}")
    
    # Initialize processor
    processor = L2ETLProcessorFixed()
    
    # Process all files
    summary = processor.process_all_files()
    
    # Validate data quality
    quality_metrics = processor.validate_data_quality()
    
    print("\n" + "="*60)
    print("PROCESSING COMPLETE")
    print("="*60)
    print(f"Files processed: {summary.get('files_processed', 0)}")
    print(f"Total records: {summary.get('total_records_loaded', 0)}")
    print(f"Overall quality score: {quality_metrics.get('overall_quality_score', 0):.2%}")
    print(f"Avg data quality score: {quality_metrics.get('avg_data_quality_score', 0):.2f}")
    
    if summary.get('failed_files'):
        print(f"\nFailed files: {len(summary['failed_files'])}")
        for file_name, error in summary['failed_files']:
            print(f"  - {file_name}: {error}")
    
    print(f"\nCompleted at: {datetime.now()}")


if __name__ == "__main__":
    main()
