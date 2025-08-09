# data_handler.py
# L2-Only Trading Strategy Implementation
# Restructured for L2-only mode with price reconstruction

import os
import time
import json
import gzip
import traceback
import pandas as pd
import numpy as np
import sqlite3
import ccxt  # For specific ccxt exceptions
from typing import Optional

# Removed unnecessary imports - keeping only core functionality


class DataHandler:
    """
    Handles fetching, cleaning, loading, and processing L2 order book data.
    L2-Only Mode: No OHLCV dependencies, L2 data is the primary source.
    """

    def __init__(self, config, exchange_api):
        """
        Initializes the DataHandler for L2-only mode.

        Args:
            config (dict): Configuration dictionary.
            exchange_api: Initialized CCXT exchange object.
        """
        self.config = config
        self.exchange = exchange_api
        self.symbol = config.get('symbol', 'BTC/USDT')
        
        # L2-only mode configuration
        self.l2_only_mode = config.get('l2_only_mode', False)
        self.feature_window = config.get('feature_window', 100)
        self.l2_depth_live_snapshot = config.get('l2_depth', 50)
        
        # L2 sampling configuration
        self.l2_sampling_frequency_ms = config.get(
            'l2_sampling_frequency_ms', 100
        )
        self.l2_buffer_size = config.get('l2_buffer_size', 10000)
        self.l2_max_time_diff_ms = config.get('l2_max_time_diff_ms', 1000)

        self.base_dir = config.get('base_dir', './')
        
        # L2 data paths - Fixed to use actual data location
        self.l2_data_folder = config.get('l2_data_folder', 'l2_data')
        # Direct path to l2_data folder at root level
        self.l2_data_path = os.path.join('./', self.l2_data_folder)
        
        # Price reconstruction will be handled by simple mid-price calculation
        self.price_reconstructor = None
        
        # Database upload will be handled directly
        self.upload_manager = None
        
        # OHLCV paths (deprecated in L2-only mode but kept for compatibility)
        if not self.l2_only_mode:
            safe_symbol = self.symbol.replace('/', '_').replace(':', '')
            self.timeframe = config.get('timeframe', '1h')
            self.ohlcv_data_path = os.path.join(
                self.base_dir,
                f"ohlcv_data_{safe_symbol}_{self.timeframe}.csv"
            )
        else:
            self.timeframe = None
            self.ohlcv_data_path = None

        mode_str = 'L2-only' if self.l2_only_mode else 'Mixed'
        print(f"DataHandler initialized in {mode_str} mode.")
        if self.l2_only_mode:
            print(f"L2 data path: {self.l2_data_path}")
            print(f"L2 sampling: {self.l2_sampling_frequency_ms}ms")
        else:
            print(f"OHLCV path: {self.ohlcv_data_path}")
            print(f"L2 data path: {self.l2_data_path}")

    def fetch_l2_order_book_snapshot(self, limit=None, max_retries=3, 
                                   delay_seconds=2):
        """
        Fetches a live snapshot of the current Level 2 order book via REST API.
        Enhanced for L2-only mode with better error handling and validation.
        """
        if not self.exchange or not self.exchange.has.get('fetchL2OrderBook'):
            print("Warning (DataHandler): Exchange does not support "
                  "fetchL2OrderBook or API not available.")
            return None

        fetch_limit = limit if limit is not None else self.l2_depth_live_snapshot

        for attempt in range(max_retries):
            try:
                order_book = self.exchange.fetch_l2_order_book(
                    self.symbol, limit=fetch_limit
                )
                
                if (order_book and isinstance(order_book, dict) and
                    'bids' in order_book and isinstance(order_book['bids'], list) and
                    'asks' in order_book and isinstance(order_book['asks'], list)):

                    # Validate bid/ask structure
                    if (order_book['bids'] and 
                        (not isinstance(order_book['bids'][0], list) or 
                         len(order_book['bids'][0]) != 2)):
                        print("Warning (DataHandler): Invalid L2 bids structure.")
                        return None
                    if (order_book['asks'] and 
                        (not isinstance(order_book['asks'][0], list) or 
                         len(order_book['asks'][0]) != 2)):
                        print("Warning (DataHandler): Invalid L2 asks structure.")
                        return None

                    # Add timestamp for L2-only mode
                    order_book['fetch_timestamp_ms'] = (
                        self.exchange.milliseconds() 
                        if hasattr(self.exchange, 'milliseconds') 
                        else int(time.time() * 1000)
                    )
                    
                    # Enhanced validation for L2-only mode
                    if self.l2_only_mode:
                        order_book = self._validate_l2_snapshot(order_book)
                        if order_book is None:
                            continue
                    
                    return order_book

                print(f"Warning (DataHandler): Invalid L2 data structure "
                      f"received (Attempt {attempt + 1}).")
                if attempt < max_retries - 1:
                    time.sleep(delay_seconds * (2**attempt))
                continue
                
            except (ccxt.NetworkError, ccxt.ExchangeError, 
                    ccxt.RequestTimeout, ccxt.RateLimitExceeded) as e:
                print(f"Warning (DataHandler): CCXT error fetching L2 snapshot "
                      f"(Attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(delay_seconds * (2**attempt))
            except Exception as e:
                print(f"Error (DataHandler): Unexpected error fetching "
                      f"L2 snapshot: {e}")
                traceback.print_exc()
                return None
                
        print("Error (DataHandler): Max retries reached for fetching L2 snapshot.")
        return None

    def _validate_l2_snapshot(self, order_book):
        """
        Enhanced L2 snapshot validation for L2-only mode.
        
        Args:
            order_book: Raw order book data from exchange
            
        Returns:
            Validated order book or None if invalid
        """
        try:
            # Check minimum depth requirements
            min_levels = self.config.get('l2_data_quality', {}).get('min_bid_levels', 1)
            if len(order_book.get('bids', [])) < min_levels:
                print(f"Warning: Insufficient bid levels: "
                      f"{len(order_book.get('bids', []))} < {min_levels}")
                return None
                
            if len(order_book.get('asks', [])) < min_levels:
                print(f"Warning: Insufficient ask levels: "
                      f"{len(order_book.get('asks', []))} < {min_levels}")
                return None
            
            # Check spread validity
            if order_book['bids'] and order_book['asks']:
                best_bid = order_book['bids'][0][0]
                best_ask = order_book['asks'][0][0]
                spread_bps = ((best_ask - best_bid) / best_bid) * 10000
                
                max_spread_bps = self.config.get('l2_data_quality', {}).get(
                    'max_spread_bps', 1000
                )
                if spread_bps > max_spread_bps:
                    print(f"Warning: Spread too wide: {spread_bps:.2f} bps > "
                          f"{max_spread_bps} bps")
                    return None
            
            # Check volume thresholds
            min_volume = self.config.get('l2_data_quality', {}).get(
                'min_volume_threshold', 0.001
            )
            for bid in order_book['bids'][:3]:  # Check top 3 levels
                if bid[1] < min_volume:
                    print(f"Warning: Bid volume too low: {bid[1]} < {min_volume}")
                    return None
                    
            for ask in order_book['asks'][:3]:  # Check top 3 levels
                if ask[1] < min_volume:
                    print(f"Warning: Ask volume too low: {ask[1]} < {min_volume}")
                    return None
            
            return order_book
            
        except Exception as e:
            print(f"Error validating L2 snapshot: {e}")
            return None

    def _process_l2_snapshot(self, order_book):
        """
        Process L2 order book snapshot into feature format.
        
        Args:
            order_book: Order book dict from exchange
            
        Returns:
            Dict with processed L2 features
        """
        try:
            # Extract bids and asks
            bids = order_book.get('bids', [])
            asks = order_book.get('asks', [])
            
            # Calculate basic features
            best_bid = bids[0][0] if bids else 0
            best_ask = asks[0][0] if asks else 0
            mid_price = (best_bid + best_ask) / 2 if best_bid and best_ask else 0
            spread = best_ask - best_bid if best_bid and best_ask else 0
            spread_bps = (spread / mid_price * 10000) if mid_price > 0 else 0
            
            # Calculate volume features
            bid_volume_5 = sum(level[1] for level in bids[:5])
            ask_volume_5 = sum(level[1] for level in asks[:5])
            imbalance = (bid_volume_5 - ask_volume_5) / (bid_volume_5 + ask_volume_5) if (bid_volume_5 + ask_volume_5) > 0 else 0
            
            # Build feature dict
            features = {
                'timestamp': pd.Timestamp.now(tz='UTC'),
                'mid_price': mid_price,
                'spread': spread,
                'spread_bps': spread_bps,
                'bid_volume_5': bid_volume_5,
                'ask_volume_5': ask_volume_5,
                'order_book_imbalance': imbalance,
                'best_bid': best_bid,
                'best_ask': best_ask
            }
            
            # Add individual price/size levels
            # Use the last available price for missing levels instead of 0
            last_bid_price = bids[-1][0] if bids else best_bid
            last_ask_price = asks[-1][0] if asks else best_ask
            
            for i in range(10):
                if i < len(bids):
                    features[f'bid_price_{i+1}'] = bids[i][0]
                    features[f'bid_size_{i+1}'] = bids[i][1]
                else:
                    # Use last bid price minus a small spread for missing levels
                    features[f'bid_price_{i+1}'] = last_bid_price - (i - len(bids) + 1) * spread * 0.1 if last_bid_price > 0 else mid_price * 0.99
                    features[f'bid_size_{i+1}'] = 0
                    
                if i < len(asks):
                    features[f'ask_price_{i+1}'] = asks[i][0]
                    features[f'ask_size_{i+1}'] = asks[i][1]
                else:
                    # Use last ask price plus a small spread for missing levels
                    features[f'ask_price_{i+1}'] = last_ask_price + (i - len(asks) + 1) * spread * 0.1 if last_ask_price > 0 else mid_price * 1.01
                    features[f'ask_size_{i+1}'] = 0
            
            return features
            
        except Exception as e:
            print(f"Error processing L2 snapshot: {e}")
            return {}

    def load_l2_training_data_from_database(self, limit=None, 
                                          start_date=None, end_date=None):
        """
        Load L2 training data from database for L2-only strategy.
        
        Args:
            limit: Maximum number of rows to load
            start_date: Start date filter (datetime or string)
            end_date: End date filter (datetime or string)
            
        Returns:
            DataFrame with L2 training data
        """
        try:
            import sqlite3
            
            # Use direct database path if upload_manager not available
            db_path = self.config.get('database_path', './trading_bot.db')
            
            # Build query - use the practical table which has 519k records
            query = "SELECT * FROM l2_training_data_practical"
            params = []
            conditions = []
            
            if start_date:
                conditions.append("timestamp >= ?")
                params.append(pd.to_datetime(start_date))
                
            if end_date:
                conditions.append("timestamp <= ?")
                params.append(pd.to_datetime(end_date))
            
            if conditions:
                query += " WHERE " + " AND ".join(conditions)
                
            query += " ORDER BY timestamp"
            
            if limit:
                query += f" LIMIT {limit}"
            
            # Execute query
            with sqlite3.connect(db_path) as conn:
                df = pd.read_sql_query(query, conn, params=params)
            
            if df.empty:
                print("Warning: No L2 training data found in database")
                return pd.DataFrame()
            
            # Convert timestamp
            df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
            
            print(f"Loaded {len(df)} L2 training records from database")
            return df
            
        except Exception as e:
            print(f"Error loading L2 training data from database: {e}")
            traceback.print_exc()
            return pd.DataFrame()

    def _load_historical_l2_data(self):
        """
        Loads historical L2 data from .jsonl.gz files in l2_data directory.
        Enhanced for L2-only mode with better processing.
        """
        if not os.path.exists(self.l2_data_path):
            print(f"Warning (DataHandler): L2 data directory not found at "
                  f"{self.l2_data_path}")
            return pd.DataFrame()

        # Find all L2 data files
        l2_files = [f for f in os.listdir(self.l2_data_path) 
                    if f.endswith('.jsonl.gz') or f.endswith('.jsonl')]
        
        if not l2_files:
            print(f"Warning (DataHandler): No L2 data files found in {self.l2_data_path}")
            return pd.DataFrame()
        
        # Sort files by modification time (newest first)
        l2_files.sort(key=lambda x: os.path.getmtime(os.path.join(self.l2_data_path, x)), 
                      reverse=True)
        
        print(f"Found {len(l2_files)} L2 data files. Loading most recent...")
        
        l2_data_list = []
        files_loaded = 0
        
        for filename in l2_files[:5]:  # Load up to 5 most recent files
            file_path = os.path.join(self.l2_data_path, filename)
            try:
                # Check if file is gzipped
                is_gzipped = filename.endswith('.gz')
                open_func = gzip.open if is_gzipped else open
                read_mode = 'rt' if is_gzipped else 'r'

                with open_func(file_path, read_mode, encoding='utf-8') as f:
                    file_records = 0
                    for line_num, line in enumerate(f, 1):
                        try:
                            record = json.loads(line)
                            # Handle both 'b'/'a' and 'bids'/'asks' formats
                            if 'bids' in record and 'b' not in record:
                                record['b'] = record['bids']
                                record['a'] = record['asks']
                            l2_data_list.append(record)
                            file_records += 1
                        except json.JSONDecodeError:
                            if line_num < 10:  # Only warn for first few lines
                                print(f"Warning: Skipping invalid JSON line {line_num} in {filename}")
                    
                    print(f"Loaded {file_records} records from {filename}")
                    files_loaded += 1
                    
                    # Stop if we have enough data
                    if len(l2_data_list) > 100000:
                        print(f"Loaded sufficient data ({len(l2_data_list)} records)")
                        break
                        
            except Exception as e:
                print(f"Error loading {filename}: {e}")
                continue

        if not l2_data_list:
            print("Warning (DataHandler): No data loaded from any L2 files")
            return pd.DataFrame()

        try:
            df_l2 = pd.DataFrame(l2_data_list)
            
            # Handle timestamp columns
            timestamp_col = None
            for col in ['timestamp_ms', 'received_timestamp_ms', 'timestamp']:
                if col in df_l2.columns:
                    timestamp_col = col
                    break
            
            if timestamp_col is None:
                print("Warning (DataHandler): No timestamp column found in L2 data")
                return pd.DataFrame()
            
            # Convert timestamp
            if timestamp_col.endswith('_ms'):
                df_l2['timestamp'] = pd.to_datetime(
                    df_l2[timestamp_col], unit='ms', utc=True, errors='coerce'
                )
            else:
                df_l2['timestamp'] = pd.to_datetime(
                    df_l2[timestamp_col], utc=True, errors='coerce'
                )
            
            df_l2.dropna(subset=['timestamp'], inplace=True)
            df_l2.sort_values('timestamp', inplace=True)
            df_l2.reset_index(drop=True, inplace=True)
            
            print(f"Loaded {len(df_l2)} L2 records from {files_loaded} files")
            
            # Ensure required columns exist
            if 'b' not in df_l2.columns: 
                df_l2['b'] = None
            if 'a' not in df_l2.columns: 
                df_l2['a'] = None
                
            return df_l2[['timestamp', 'b', 'a']]

        except Exception as e:
            print(f"Error (DataHandler): Failed to process L2 data: {e}")
            traceback.print_exc()
            return pd.DataFrame()

    def process_l2_data_for_features(self, df_l2):
        """
        Process L2 data to extract features for L2-only strategy.
        
        Args:
            df_l2: DataFrame with L2 order book data
            
        Returns:
            DataFrame with L2-derived features
        """
        if df_l2.empty:
            print("Warning: Empty L2 data provided for feature processing")
            return pd.DataFrame()
        
        try:
            # Convert L2 order book data to structured format
            l2_features = []
            
            for idx, row in df_l2.iterrows():
                try:
                    timestamp = row['timestamp']
                    bids = row['b'] if row['b'] is not None else []
                    asks = row['a'] if row['a'] is not None else []
                    
                    if not bids or not asks:
                        continue
                    
                    # Extract basic L2 features
                    feature_row = {
                        'timestamp': timestamp,
                        'bid_price_1': bids[0][0] if len(bids) > 0 else np.nan,
                        'bid_size_1': bids[0][1] if len(bids) > 0 else np.nan,
                        'ask_price_1': asks[0][0] if len(asks) > 0 else np.nan,
                        'ask_size_1': asks[0][1] if len(asks) > 0 else np.nan,
                    }
                    
                    # Add additional levels if available
                    for level in range(2, min(6, len(bids) + 1)):
                        if level - 1 < len(bids):
                            feature_row[f'bid_price_{level}'] = bids[level-1][0]
                            feature_row[f'bid_size_{level}'] = bids[level-1][1]
                    
                    for level in range(2, min(6, len(asks) + 1)):
                        if level - 1 < len(asks):
                            feature_row[f'ask_price_{level}'] = asks[level-1][0]
                            feature_row[f'ask_size_{level}'] = asks[level-1][1]
                    
                    # Calculate basic derived features
                    if (not np.isnan(feature_row['bid_price_1']) and 
                        not np.isnan(feature_row['ask_price_1'])):
                        
                        bid_price = feature_row['bid_price_1']
                        ask_price = feature_row['ask_price_1']
                        bid_volume = feature_row['bid_size_1']
                        ask_volume = feature_row['ask_size_1']
                        
                        # Basic microstructure features
                        feature_row['bid_ask_spread'] = ask_price - bid_price
                        feature_row['bid_ask_spread_pct'] = (
                            (ask_price - bid_price) / bid_price * 100
                        )
                        feature_row['mid_price'] = (bid_price + ask_price) / 2
                        
                        # Weighted mid price
                        total_volume = bid_volume + ask_volume
                        if total_volume > 0:
                            feature_row['weighted_mid_price'] = (
                                (bid_price * ask_volume + ask_price * bid_volume) 
                                / total_volume
                            )
                        else:
                            feature_row['weighted_mid_price'] = feature_row['mid_price']
                        
                        # Microprice
                        if total_volume > 0:
                            feature_row['microprice'] = (
                                (bid_price * ask_volume + ask_price * bid_volume) 
                                / total_volume
                            )
                        else:
                            feature_row['microprice'] = feature_row['mid_price']
                    
                    l2_features.append(feature_row)
                    
                except Exception as e:
                    print(f"Warning: Error processing L2 row {idx}: {e}")
                    continue
            
            if not l2_features:
                print("Warning: No valid L2 features extracted")
                return pd.DataFrame()
            
            df_features = pd.DataFrame(l2_features)
            df_features.sort_values('timestamp', inplace=True)
            df_features.reset_index(drop=True, inplace=True)
            
            print(f"Processed {len(df_features)} L2 feature rows")
            return df_features
            
        except Exception as e:
            print(f"Error processing L2 data for features: {e}")
            traceback.print_exc()
            return pd.DataFrame()

    def load_and_prepare_l2_only_data(self, use_database=True, 
                                     use_historical_l2=True, 
                                     limit=None):
        """
        Main method for loading and preparing L2-only data.
        
        Args:
            use_database: Whether to load from database first
            use_historical_l2: Whether to use historical L2 files as fallback
            limit: Maximum number of records to load
            
        Returns:
            DataFrame with processed L2 data ready for feature engineering
        """
        if not self.l2_only_mode:
            print("Warning: load_and_prepare_l2_only_data called but "
                  "l2_only_mode is False")
            return self.load_and_prepare_historical_data()
        
        print("Loading and preparing L2-only data...")
        
        df_l2_processed = pd.DataFrame()
        
        # Try loading from database first
        if use_database:
            print("Attempting to load L2 training data from database...")
            df_l2_processed = self.load_l2_training_data_from_database(limit=limit)
            
            if not df_l2_processed.empty:
                print(f"Successfully loaded {len(df_l2_processed)} records "
                      "from database")
                
                # Reconstruct price series if needed
                if self.price_reconstructor and 'reconstructed_price' not in df_l2_processed.columns:
                    print("Reconstructing price series from L2 data...")
                    df_l2_processed = self.price_reconstructor.reconstruct_price_series(
                        df_l2_processed
                    )
                
                return df_l2_processed
        
        # Fallback to historical L2 files
        if use_historical_l2:
            print("Loading historical L2 data from files...")
            df_l2_raw = self._load_historical_l2_data()
            
            if df_l2_raw.empty:
                print("Error: No L2 data available from any source")
                return pd.DataFrame()
            
            # Process L2 data to extract features
            df_l2_features = self.process_l2_data_for_features(df_l2_raw)
            
            if df_l2_features.empty:
                print("Error: Failed to extract features from L2 data")
                return pd.DataFrame()
            
            # Apply limit if specified
            if limit and len(df_l2_features) > limit:
                df_l2_features = df_l2_features.tail(limit).reset_index(drop=True)
                print(f"Limited to most recent {limit} records")
            
            # Reconstruct price series
            print("Reconstructing price series from L2 features...")
            df_l2_processed = self.price_reconstructor.reconstruct_price_series(
                df_l2_features
            )
            
            print(f"L2-only data preparation complete. "
                  f"Final DataFrame shape: {df_l2_processed.shape}")
            return df_l2_processed
        
        print("Error: No L2 data sources available")
        return pd.DataFrame()
    
    def load_l2_historical_data(
        self,
        limit: Optional[int] = None,
        table_name: str = 'l2_training_data'
    ):
        """
        Load historical L2 data from the database.
        
        Args:
            limit (int, optional): The number of records to load. 
                Defaults to None.
            table_name (str): The name of the table to load from.
            
        Returns:
            pd.DataFrame: DataFrame with L2 data.
        """
        print(f"Loading L2 data from table: {table_name}")
        try:
            db_path = self.config.get('database_path', './trading_bot.db')
            conn = sqlite3.connect(db_path)
            
            query = f"SELECT * FROM {table_name}"
            if limit:
                query += f" LIMIT {limit}"
            
            df = pd.read_sql_query(
                query,
                conn,
                index_col='timestamp',
                parse_dates=['timestamp']
            )
            
            conn.close()
            return df
            
        except Exception as e:
            print(f"Error loading L2 data from database: {e}")
            return None

    def get_latest_l2_data(self, limit: int = 100):
        """
        Alias for load_and_prepare_l2_only_data for orchestrator compatibility.
        
        Returns:
            DataFrame with L2 historical data
        """
        return self.load_and_prepare_l2_only_data(use_database=True, use_historical_l2=True)

    # Legacy methods for backward compatibility (deprecated in L2-only mode)
    def fetch_ohlcv(self, limit=None, since=None, max_retries=3, delay_seconds=5, timeframe=None):
        """
        Fetches historical OHLCV data from the exchange.
        """
        if not self.exchange:
            print("Exchange API not available for fetching OHLCV data.")
            return pd.DataFrame()

        timeframe_to_use = timeframe if timeframe is not None else self.timeframe

        for attempt in range(max_retries):
            try:
                print(f"Fetching OHLCV data for {self.symbol} with timeframe {timeframe_to_use}...")
                ohlcv = self.exchange.fetch_ohlcv(self.symbol, timeframe_to_use, since=since, limit=limit)
                if ohlcv:
                    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
                    return df
            except Exception as e:
                print(f"Error fetching OHLCV data (attempt {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(delay_seconds)
        return pd.DataFrame()

    def clean_ohlcv_data(self, df):
        """
        DEPRECATED: OHLCV cleaning not used in L2-only mode.
        """
        if self.l2_only_mode:
            print("Warning: clean_ohlcv_data() called in L2-only mode. "
                  "Use L2 data methods instead.")
            return pd.DataFrame()
        
        # Original OHLCV cleaning would go here for backward compatibility
        print("Error: OHLCV cleaning not implemented in L2-only mode")
        return df

    def load_and_prepare_historical_data(self, fetch_ohlcv_limit=None, 
                                       use_historical_l2=False, save_ohlcv=True):
        """
        Legacy method - redirects to L2-only method when in L2-only mode.
        """
        if self.l2_only_mode:
            print("Redirecting to L2-only data loading...")
            return self.load_and_prepare_l2_only_data(
                use_database=True,
                use_historical_l2=use_historical_l2,
                limit=fetch_ohlcv_limit
            )
        
        # Original mixed OHLCV+L2 implementation would go here
        print("Error: Mixed OHLCV+L2 mode not implemented in this version")
        return pd.DataFrame()