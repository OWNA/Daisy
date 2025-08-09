#!/usr/bin/env python3
"""
Database module for trading bot data storage using SQLite
"""

import sqlite3
import pandas as pd
import json
import os
from datetime import datetime
from contextlib import contextmanager
from typing import Optional, Dict, List, Any


class TradingDatabase:
    """
    SQLite database handler for trading bot data
    """
    
    def __init__(self, db_path: str = "trading_bot.db"):
        """
        Initialize database connection
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self._create_tables()
    
    @contextmanager
    def get_connection(self):
        """Context manager for database connections"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
    
    def _create_tables(self):
        """Create database tables if they don't exist"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # OHLCV data table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS ohlcv (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    timestamp DATETIME NOT NULL,
                    open REAL NOT NULL,
                    high REAL NOT NULL,
                    low REAL NOT NULL,
                    close REAL NOT NULL,
                    volume REAL NOT NULL,
                    UNIQUE(symbol, timeframe, timestamp)
                )
            """)
            
            # Create indexes for faster queries
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_ohlcv_symbol_timeframe 
                ON ohlcv(symbol, timeframe)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_ohlcv_timestamp 
                ON ohlcv(timestamp)
            """)
            
            # Features table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS features (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ohlcv_id INTEGER NOT NULL,
                    feature_name TEXT NOT NULL,
                    feature_value REAL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (ohlcv_id) REFERENCES ohlcv(id),
                    UNIQUE(ohlcv_id, feature_name)
                )
            """)
            
            # Models table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS models (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_name TEXT NOT NULL,
                    model_type TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    model_data BLOB NOT NULL,
                    feature_list TEXT NOT NULL,
                    metrics TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    is_active BOOLEAN DEFAULT 1
                )
            """)
            
            # Predictions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_id INTEGER NOT NULL,
                    timestamp DATETIME NOT NULL,
                    prediction REAL NOT NULL,
                    signal INTEGER,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (model_id) REFERENCES models(id)
                )
            """)
            
            # Trades table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_id INTEGER,
                    entry_timestamp DATETIME NOT NULL,
                    exit_timestamp DATETIME,
                    direction TEXT NOT NULL,
                    entry_price REAL NOT NULL,
                    exit_price REAL,
                    size REAL NOT NULL,
                    pnl_gross REAL,
                    pnl_net REAL,
                    commission REAL,
                    status TEXT DEFAULT 'open',
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (model_id) REFERENCES models(id)
                )
            """)
            
            # Backtest results table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS backtest_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_id INTEGER NOT NULL,
                    start_date DATETIME NOT NULL,
                    end_date DATETIME NOT NULL,
                    initial_balance REAL NOT NULL,
                    final_balance REAL NOT NULL,
                    total_return REAL NOT NULL,
                    max_drawdown REAL,
                    sharpe_ratio REAL,
                    win_rate REAL,
                    total_trades INTEGER,
                    profit_factor REAL,
                    metrics TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (model_id) REFERENCES models(id)
                )
            """)
    
    # OHLCV Data Methods
    def save_ohlcv_data(self, df: pd.DataFrame, symbol: str, timeframe: str):
        """
        Save OHLCV data to database
        
        Args:
            df: DataFrame with columns [timestamp, open, high, low, close, volume]
            symbol: Trading symbol
            timeframe: Timeframe (e.g., '1m', '1h')
        """
        with self.get_connection() as conn:
            df_to_save = df.copy()
            df_to_save['symbol'] = symbol
            df_to_save['timeframe'] = timeframe
            
            # SQLite has a limit on variables per query (default 999)
            # Batch inserts to avoid "too many SQL variables" error
            batch_size = 100  # Safe batch size
            
            for i in range(0, len(df_to_save), batch_size):
                batch = df_to_save.iloc[i:i + batch_size]
                
                # Use INSERT OR REPLACE to handle duplicates
                for _, row in batch.iterrows():
                    # Convert timestamp to string for SQLite
                    timestamp_str = (row['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
                                     if hasattr(row['timestamp'], 'strftime')
                                     else str(row['timestamp']))
                    
                    conn.execute("""
                        INSERT OR REPLACE INTO ohlcv 
                        (symbol, timeframe, timestamp, open, high, low, close, volume)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        row['symbol'], row['timeframe'], timestamp_str,
                        row['open'], row['high'], row['low'], 
                        row['close'], row['volume']
                    ))
    
    def load_ohlcv_data(self, symbol: str, timeframe: str, 
                        start_date: Optional[str] = None,
                        end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Load OHLCV data from database
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe
            start_date: Optional start date filter
            end_date: Optional end date filter
            
        Returns:
            DataFrame with OHLCV data
        """
        query = """
            SELECT timestamp, open, high, low, close, volume
            FROM ohlcv
            WHERE symbol = ? AND timeframe = ?
        """
        params = [symbol, timeframe]
        
        if start_date:
            query += " AND timestamp >= ?"
            params.append(start_date)
        if end_date:
            query += " AND timestamp <= ?"
            params.append(end_date)
            
        query += " ORDER BY timestamp"
        
        with self.get_connection() as conn:
            df = pd.read_sql_query(query, conn, params=params,
                                   parse_dates=['timestamp'])
        return df
    
    # Model Methods
    def save_model(self, model_name: str, model_type: str,
                   symbol: str, timeframe: str,
                   model_data: bytes, feature_list: List[str],
                   metrics: Optional[Dict] = None) -> int:
        """
        Save a trained model to database
        
        Returns:
            Model ID
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Deactivate previous models for this symbol/timeframe
            cursor.execute("""
                UPDATE models 
                SET is_active = 0 
                WHERE symbol = ? AND timeframe = ? AND is_active = 1
            """, (symbol, timeframe))
            
            # Insert new model
            cursor.execute("""
                INSERT INTO models 
                (model_name, model_type, symbol, timeframe, 
                 model_data, feature_list, metrics)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (model_name, model_type, symbol, timeframe,
                  model_data, json.dumps(feature_list),
                  json.dumps(metrics) if metrics else None))
            
            return cursor.lastrowid
    
    def load_active_model(self, symbol: str, timeframe: str) -> Optional[Dict]:
        """
        Load the active model for a symbol/timeframe
        
        Returns:
            Dictionary with model data or None
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, model_name, model_type, model_data, 
                       feature_list, metrics, created_at
                FROM models
                WHERE symbol = ? AND timeframe = ? AND is_active = 1
                ORDER BY created_at DESC
                LIMIT 1
            """, (symbol, timeframe))
            
            row = cursor.fetchone()
            if row:
                return {
                    'id': row['id'],
                    'model_name': row['model_name'],
                    'model_type': row['model_type'],
                    'model_data': row['model_data'],
                    'feature_list': json.loads(row['feature_list']),
                    'metrics': json.loads(row['metrics']) if row['metrics'] else None,
                    'created_at': row['created_at']
                }
            return None
    
    # Prediction Methods
    def save_predictions(self, model_id: int, predictions: pd.DataFrame):
        """
        Save model predictions
        
        Args:
            model_id: Model ID
            predictions: DataFrame with columns [timestamp, prediction, signal]
        """
        with self.get_connection() as conn:
            df_to_save = predictions.copy()
            df_to_save['model_id'] = model_id
            df_to_save.to_sql('predictions', conn, if_exists='append',
                              index=False, method='multi')
    
    # Trade Methods
    def save_trade(self, trade_data: Dict) -> int:
        """
        Save a trade to database
        
        Returns:
            Trade ID
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Convert timestamps to strings if they are datetime objects
            processed_data = trade_data.copy()
            timestamp_fields = ['entry_timestamp', 'exit_timestamp', 'created_at']
            
            for field in timestamp_fields:
                if field in processed_data and processed_data[field] is not None:
                    if hasattr(processed_data[field], 'strftime'):
                        processed_data[field] = processed_data[field].strftime('%Y-%m-%d %H:%M:%S')
                    else:
                        processed_data[field] = str(processed_data[field])
            
            columns = ', '.join(processed_data.keys())
            placeholders = ', '.join(['?' for _ in processed_data])
            query = f"INSERT INTO trades ({columns}) VALUES ({placeholders})"
            cursor.execute(query, list(processed_data.values()))
            return cursor.lastrowid
    
    def update_trade(self, trade_id: int, update_data: Dict):
        """Update an existing trade"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            set_clause = ', '.join([f"{k} = ?" for k in update_data.keys()])
            query = f"UPDATE trades SET {set_clause} WHERE id = ?"
            cursor.execute(query, list(update_data.values()) + [trade_id])
    
    # Backtest Methods
    def save_backtest_results(self, model_id: int, results: Dict) -> int:
        """
        Save backtest results
        
        Returns:
            Backtest result ID
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Convert timestamps to strings if they are datetime objects
            start_date = results.get('start_date')
            end_date = results.get('end_date')
            
            if hasattr(start_date, 'strftime'):
                start_date = start_date.strftime('%Y-%m-%d %H:%M:%S')
            elif start_date is not None:
                start_date = str(start_date)
                
            if hasattr(end_date, 'strftime'):
                end_date = end_date.strftime('%Y-%m-%d %H:%M:%S')
            elif end_date is not None:
                end_date = str(end_date)
            
            cursor.execute("""
                INSERT INTO backtest_results
                (model_id, start_date, end_date, initial_balance,
                 final_balance, total_return, max_drawdown, sharpe_ratio,
                 win_rate, total_trades, profit_factor, metrics)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                model_id,
                start_date,
                end_date,
                results.get('initial_balance'),
                results.get('final_balance'),
                results.get('total_return'),
                results.get('max_drawdown'),
                results.get('sharpe_ratio'),
                results.get('win_rate'),
                results.get('total_trades'),
                results.get('profit_factor'),
                json.dumps(results.get('additional_metrics', {}))
            ))
            return cursor.lastrowid
    
    def get_backtest_history(self, symbol: str, timeframe: str,
                             limit: int = 10) -> pd.DataFrame:
        """Get backtest history for a symbol/timeframe"""
        query = """
            SELECT br.*, m.model_name, m.created_at as model_created
            FROM backtest_results br
            JOIN models m ON br.model_id = m.id
            WHERE m.symbol = ? AND m.timeframe = ?
            ORDER BY br.created_at DESC
            LIMIT ?
        """
        with self.get_connection() as conn:
            return pd.read_sql_query(query, conn, params=[symbol, timeframe, limit])
    
    # Utility Methods
    def get_database_stats(self) -> Dict:
        """Get database statistics"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            stats = {}
            
            tables = ['ohlcv', 'features', 'models', 'predictions', 
                      'trades', 'backtest_results']
            
            for table in tables:
                cursor.execute(f"SELECT COUNT(*) as count FROM {table}")
                stats[f"{table}_count"] = cursor.fetchone()['count']
            
            # Get database file size
            if os.path.exists(self.db_path):
                stats['database_size_mb'] = os.path.getsize(self.db_path) / 1024 / 1024
            
            return stats
    
    def vacuum(self):
        """Optimize database by running VACUUM"""
        with self.get_connection() as conn:
            conn.execute("VACUUM") 