#!/usr/bin/env python3
"""
data_ingestor.py - Live WebSocket Data Ingestion

This module provides reliable, real-time L2 order book data ingestion from Bybit
using WebSocket connections. It continuously receives market data and writes
it to the database for the live trading system.

Key Features:
- Robust WebSocket connection management with proper URL handling
- Simplified data processing pipeline
- Clean separation of async/sync components
- Comprehensive error handling and recovery
- Database schema validation and auto-migration
"""

import os
import sys
import time
import logging
import sqlite3
import threading
import asyncio
import json
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Callable
from contextlib import contextmanager
from dataclasses import dataclass, asdict
from collections import deque
import statistics
import ccxt.pro as ccxtpro
from dotenv import load_dotenv

logger = logging.getLogger(__name__)


@dataclass
class L2Update:
    """Enhanced data class for L2 order book updates with execution-focused metadata."""
    timestamp: datetime
    symbol: str
    bids: List[tuple]  # [(price, size), ...]
    asks: List[tuple]  # [(price, size), ...]
    exchange_timestamp: Optional[int] = None
    sequence: Optional[int] = None
    
    # Execution quality metadata
    receive_timestamp: Optional[datetime] = None  # When we received the data
    processing_latency_us: Optional[int] = None   # Processing time in microseconds
    data_quality_score: Optional[float] = None
    is_stale: Optional[bool] = None
    market_session: Optional[str] = None  # 'active', 'quiet', 'volatile'
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for database storage."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'symbol': self.symbol,
            'bids': self.bids,
            'asks': self.asks,
            'exchange_timestamp': self.exchange_timestamp,
            'sequence': self.sequence,
            'receive_timestamp': self.receive_timestamp.isoformat() if self.receive_timestamp else None,
            'processing_latency_us': self.processing_latency_us,
            'data_quality_score': self.data_quality_score,
            'is_stale': self.is_stale,
            'market_session': self.market_session
        }
    
    def calculate_total_latency_us(self) -> Optional[int]:
        """Calculate total latency from exchange to processing completion."""
        if not (self.exchange_timestamp and self.receive_timestamp):
            return None
        
        exchange_dt = datetime.fromtimestamp(self.exchange_timestamp / 1000, timezone.utc)
        latency_seconds = (self.receive_timestamp - exchange_dt).total_seconds()
        return int(latency_seconds * 1_000_000)  # Convert to microseconds


class DataIngestorConfig:
    """Configuration for the DataIngestor."""
    
    def __init__(self, config_dict: Optional[Dict[str, Any]] = None):
        """Initialize configuration from dictionary or defaults."""
        config = config_dict or {}
        
        # Exchange configuration
        self.exchange_name = config.get('exchange', 'bybit')
        self.symbol = config.get('symbol', 'BTC/USDT:USDT')
        self.sandbox = config.get('sandbox', False)  # Demo Trading
        
        # Database configuration
        self.db_path = config.get('db_path', './trading_bot_live.db')
        self.table_name = config.get(
            'table_name', 'l2_training_data_practical'
        )
        
        # WebSocket configuration
        self.max_reconnect_attempts = config.get(
            'max_reconnect_attempts', 10
        )
        self.reconnect_delay = config.get('reconnect_delay', 5.0)
        self.heartbeat_interval = config.get('heartbeat_interval', 30.0)
        self.orderbook_depth = config.get('orderbook_depth', 10)
        
        # Performance configuration
        self.buffer_size = config.get('buffer_size', 100)
        self.write_interval = config.get('write_interval', 1.0)
        self.data_retention_hours = config.get('data_retention_hours', 24)
        
        # Execution quality configuration
        self.max_processing_latency_us = config.get('max_processing_latency_us', 10000)  # 10ms
        self.stale_data_threshold_ms = config.get('stale_data_threshold_ms', 500)  # 500ms
        self.sequence_gap_tolerance = config.get('sequence_gap_tolerance', 5)
        self.quality_score_threshold = config.get('quality_score_threshold', 0.8)
        
        # Logging configuration
        self.log_level = config.get('log_level', 'INFO')
        self.log_updates = config.get('log_updates', False)
        
        logger.info(
            f"DataIngestor configured for {self.exchange_name} {self.symbol}"
        )


class DataIngestor:
    """
    Live data ingestor that connects to Bybit WebSocket and continuously
    ingests L2 order book data into the database.
    """
    
    def __init__(self, config: DataIngestorConfig):
        """Initialize the DataIngestor."""
        self.config = config
        self.exchange: Optional[ccxtpro.Exchange] = None
        self.running = False
        self.stop_event = threading.Event()
        self.ingestion_thread: Optional[threading.Thread] = None
        self.writer_thread: Optional[threading.Thread] = None
        
        # Adaptive buffer management with priority lanes
        self.high_priority_buffer: deque = deque(maxlen=self.config.buffer_size // 2)
        self.normal_priority_buffer: deque = deque(maxlen=self.config.buffer_size)
        self.buffer_lock = threading.Lock()
        
        # Buffer management statistics
        self.buffer_stats = {
            'high_priority_count': 0,
            'normal_priority_count': 0,
            'buffer_overflows': 0,
            'adaptive_resizes': 0,
            'last_resize_time': datetime.now()
        }
        
        # Adaptive buffer parameters
        self.min_buffer_size = 50
        self.max_buffer_size = 1000
        self.resize_threshold_seconds = 60  # Resize evaluation interval
        self.volatility_buffer = deque(maxlen=100)  # Track market volatility
        
        # Statistics
        self.stats: Dict[str, Any] = {
            'total_updates': 0,
            'successful_writes': 0,
            'failed_writes': 0,
            'reconnections': 0,
            'last_update_time': None,
            'start_time': datetime.now(),
            'bytes_received': 0,
            'sequence_gaps': 0,
            'stale_data_count': 0,
            'avg_processing_latency_us': 0,
            'high_quality_updates': 0,
            'low_quality_updates': 0
        }
        
        # Performance tracking
        self.latency_buffer = deque(maxlen=1000)  # Track recent latencies
        self.last_sequence = None
        self.sequence_gap_count = 0
        
        # Bybit-specific optimization state
        self.cached_orderbook = None  # Full orderbook cache for delta processing
        self.last_full_snapshot_time = None
        self.delta_updates_count = 0
        self.full_snapshots_count = 0
        self.websocket_reconnect_count = 0
        
        # Callbacks
        self.on_update_callback: Optional[Callable[[L2Update], None]] = None
        self.on_error_callback: Optional[Callable[[Exception], None]] = None
        
        logger.info("DataIngestor initialized")
    
    def set_update_callback(self, callback: Callable[[L2Update], None]):
        """Set callback function to be called on each L2 update."""
        self.on_update_callback = callback
    
    def set_error_callback(self, callback: Callable[[Exception], None]):
        """Set callback function to be called on errors."""
        self.on_error_callback = callback
    
    @contextmanager
    def get_db_connection(self):
        """Context manager for database connections."""
        conn = sqlite3.connect(self.config.db_path)
        try:
            yield conn
        except sqlite3.Error as e:
            logger.error(f"Database error: {e}")
            raise
        finally:
            conn.close()
    
    def _initialize_exchange(self) -> bool:
        """Initialize the CCXT Pro exchange connection with proper URL handling."""
        try:
            logger.info(f"Initializing {self.config.exchange_name} WebSocket connection...")
            logger.info(f"Configuration: sandbox={self.config.sandbox}, symbol={self.config.symbol}")
            
            # Load environment variables for API credentials
            load_dotenv()
            
            # Enhanced Bybit-specific exchange configuration
            exchange_config = {
                'enableRateLimit': True,
                'sandbox': self.config.sandbox,
                'timeout': 30000,  # 30 seconds
                'options': {
                    'defaultType': 'linear',  # USDT perpetual futures
                    'recvWindow': 10000,      # 10 second receive window
                    # Bybit-specific WebSocket optimizations
                    'ws': {
                        'keepAlive': True,
                        'ping_interval': 20,     # Ping every 20 seconds
                        'ping_timeout': 10,      # Wait 10 seconds for pong
                        'max_reconnects': 10,    # Maximum reconnection attempts
                        'reconnect_delay': 2,    # Initial reconnect delay
                        'binary_data': False,    # Use text mode for easier debugging
                        'compress': True,        # Enable compression for lower latency
                    },
                    # Bybit market data specific settings
                    'orderbook': {
                        'depth': self.config.orderbook_depth,
                        'frequency': '100ms',    # Request 100ms updates for better granularity
                        'merge_levels': False,   # Don't merge price levels
                    },
                    # Performance optimizations
                    'performance': {
                        'enable_delta_updates': True,     # Use delta updates when available
                        'cache_orderbook': True,          # Cache full orderbook state
                        'validate_sequence': True,        # Validate sequence numbers
                    }
                }
            }
            
            # Add API credentials based on sandbox mode
            if self.config.sandbox:
                # Testnet credentials
                api_key = os.getenv('BYBIT_API_KEY_TESTNET')
                secret = os.getenv('BYBIT_API_SECRET_TESTNET')
                logger.info("Using testnet credentials")
            else:
                # Mainnet credentials (for demo trading)
                api_key = os.getenv('BYBIT_API_KEY_MAIN')
                secret = os.getenv('BYBIT_API_SECRET_MAIN')
                logger.info("Using mainnet credentials for demo trading")
            
            if api_key and secret:
                exchange_config['apiKey'] = api_key
                exchange_config['secret'] = secret
                logger.info("✓ API credentials loaded")
            else:
                logger.info("⚠ No API credentials - using public connection only")
            
            # Initialize CCXT Pro exchange
            if self.config.exchange_name.lower() != 'bybit':
                raise ValueError(f"Unsupported exchange: {self.config.exchange_name}")
            
            exchange_class = getattr(ccxtpro, self.config.exchange_name)
            self.exchange = exchange_class(exchange_config)
            
            # FIXED: Correct URL configuration logic
            if self.config.sandbox:
                # Testnet environment
                self.exchange.urls['api'] = 'https://api-testnet.bybit.com'
                self.exchange.urls['ws'] = 'wss://stream-testnet.bybit.com/v5/public/linear'
                logger.info("✓ Configured for testnet environment")
            else:
                # Demo trading on mainnet - keep default production URLs
                logger.info("✓ Configured for demo trading on mainnet")
                logger.info(f"  API URL: {self.exchange.urls.get('api', 'default')}")
                # WebSocket URLs are nested, just log that it's configured
                logger.info("  WebSocket URL: Configured for linear futures")
            
            logger.info("✓ Exchange connection initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize exchange: {e}")
            if self.on_error_callback:
                self.on_error_callback(e)
            return False
    
    def _setup_database_schema(self):
        """Ensure database schema is set up for live data ingestion with robust validation."""
        try:
            with self.get_db_connection() as conn:
                cursor = conn.cursor()
                
                # Check if table exists
                cursor.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                    (self.config.table_name,)
                )
                table_exists = cursor.fetchone() is not None
                
                if not table_exists:
                    logger.warning(f"Table {self.config.table_name} does not exist")
                    logger.info("Creating table with basic L2 schema...")
                    
                    # Create table with essential L2 columns
                    create_sql = f"""
                    CREATE TABLE {self.config.table_name} (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        symbol TEXT NOT NULL,
                        exchange TEXT DEFAULT 'bybit',
                        bid_price_1 REAL,
                        bid_size_1 REAL,
                        ask_price_1 REAL,
                        ask_size_1 REAL,
                        mid_price REAL,
                        spread REAL,
                        spread_bps REAL,
                        microprice REAL,
                        order_book_imbalance REAL,
                        total_bid_volume_10 REAL,
                        total_ask_volume_10 REAL,
                        sequence INTEGER,
                        exchange_timestamp INTEGER,
                        data_source TEXT DEFAULT 'live_trading',
                        data_quality_score REAL DEFAULT 1.0,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                    """
                    cursor.execute(create_sql)
                    
                    # Create indexes for performance
                    cursor.execute(f"CREATE INDEX idx_{self.config.table_name}_timestamp ON {self.config.table_name}(timestamp)")
                    cursor.execute(f"CREATE INDEX idx_{self.config.table_name}_symbol ON {self.config.table_name}(symbol)")
                    
                    conn.commit()
                    logger.info("✓ Database table created successfully")
                else:
                    # Validate existing schema has essential columns
                    cursor.execute(f"PRAGMA table_info({self.config.table_name})")
                    existing_columns = {row[1] for row in cursor.fetchall()}
                    
                    essential_columns = {
                        'timestamp', 'symbol', 'bid_price_1', 'bid_size_1',
                        'ask_price_1', 'ask_size_1', 'mid_price', 'spread'
                    }
                    
                    missing_columns = essential_columns - existing_columns
                    if missing_columns:
                        logger.error(f"Table {self.config.table_name} missing essential columns: {missing_columns}")
                        raise ValueError(f"Database schema validation failed - missing columns: {missing_columns}")
                    
                    logger.info("✓ Database schema validation passed")
                
        except Exception as e:
            logger.error(f"Failed to setup database schema: {e}")
            raise
    
    def _process_orderbook_update(self, orderbook_data: Dict[str, Any]) -> Optional[L2Update]:
        """Process raw orderbook data from WebSocket into L2Update with execution quality analysis."""
        processing_start = time.time_ns()
        receive_timestamp = datetime.now(timezone.utc)
        
        try:
            # Extract exchange timestamp with priority
            exchange_timestamp = orderbook_data.get('timestamp')
            if exchange_timestamp:
                timestamp = datetime.fromtimestamp(exchange_timestamp / 1000, timezone.utc)
            else:
                timestamp = receive_timestamp
                logger.debug("No exchange timestamp - using receive time")
            
            # Extract symbol
            symbol = orderbook_data.get('symbol', self.config.symbol)
            
            # Extract bids and asks
            bids = orderbook_data.get('bids', [])
            asks = orderbook_data.get('asks', [])
            
            # Limit to configured depth
            bids = bids[:self.config.orderbook_depth]
            asks = asks[:self.config.orderbook_depth]
            
            # Validate data
            if not bids or not asks:
                logger.warning("Empty bids or asks in orderbook update")
                return None
            
            # Extract sequence number for gap detection
            sequence = orderbook_data.get('sequence')
            
            # Calculate processing latency
            processing_end = time.time_ns()
            processing_latency_us = (processing_end - processing_start) // 1000
            
            # Check for sequence gaps
            if sequence and self.last_sequence:
                if sequence != self.last_sequence + 1:
                    gap_size = sequence - self.last_sequence - 1
                    if gap_size > 0 and gap_size <= self.config.sequence_gap_tolerance:
                        self.sequence_gap_count += gap_size
                        self.stats['sequence_gaps'] += gap_size
                        logger.warning(f"Sequence gap detected: {gap_size} missing updates")
            
            self.last_sequence = sequence
            
            # Detect stale data
            is_stale = False
            if exchange_timestamp:
                age_ms = (receive_timestamp.timestamp() * 1000) - exchange_timestamp
                is_stale = age_ms > self.config.stale_data_threshold_ms
                if is_stale:
                    self.stats['stale_data_count'] += 1
            
            # Calculate data quality score
            data_quality_score = self._calculate_enhanced_data_quality_score(
                bids, asks, processing_latency_us, is_stale, sequence is not None
            )
            
            # Determine market session
            market_session = self._classify_market_session(bids, asks)
            
            # Track latency statistics
            self.latency_buffer.append(processing_latency_us)
            if self.latency_buffer:
                self.stats['avg_processing_latency_us'] = statistics.mean(self.latency_buffer)
            
            # Update quality statistics
            if data_quality_score >= self.config.quality_score_threshold:
                self.stats['high_quality_updates'] += 1
            else:
                self.stats['low_quality_updates'] += 1
            
            # Create enhanced L2Update object
            update = L2Update(
                timestamp=timestamp,
                symbol=symbol,
                bids=bids,
                asks=asks,
                exchange_timestamp=exchange_timestamp,
                sequence=sequence,
                receive_timestamp=receive_timestamp,
                processing_latency_us=processing_latency_us,
                data_quality_score=data_quality_score,
                is_stale=is_stale,
                market_session=market_session
            )
            
            return update
            
        except Exception as e:
            logger.error(f"Error processing orderbook update: {e}")
            return None
    
    def _l2_update_to_database_row(self, update: L2Update) -> Optional[Dict[str, Any]]:
        """Convert L2Update to simplified database row format."""
        try:
            # Validate input data
            if not update.bids or not update.asks:
                logger.warning("Empty bids or asks in L2Update")
                return None
            
            # Extract best bid/ask (Level 1)
            best_bid_price, best_bid_size = update.bids[0] if update.bids else (None, None)
            best_ask_price, best_ask_size = update.asks[0] if update.asks else (None, None)
            
            if not (best_bid_price and best_ask_price and best_bid_size and best_ask_size):
                logger.warning("Invalid best bid/ask data")
                return None
            
            # Calculate enhanced microstructure features for execution optimization
            mid_price = (best_bid_price + best_ask_price) / 2
            spread = best_ask_price - best_bid_price
            spread_bps = (spread / mid_price) * 10000 if mid_price > 0 else 0
            
            # Size-weighted microprice
            total_size = best_bid_size + best_ask_size
            microprice = (best_bid_price * best_ask_size + best_ask_price * best_bid_size) / total_size
            
            # Enhanced order book analytics
            enhanced_features = self._calculate_enhanced_microstructure_features(update.bids, update.asks, mid_price)
            
            # Calculate total volume across top 10 levels
            total_bid_volume = sum(size for _, size in update.bids[:10] if size is not None)
            total_ask_volume = sum(size for _, size in update.asks[:10] if size is not None)
            
            # Use enhanced data quality score if available, otherwise calculate basic score
            if hasattr(update, 'data_quality_score') and update.data_quality_score is not None:
                data_quality_score = update.data_quality_score
            else:
                data_quality_score = self._calculate_data_quality_score(update.bids, update.asks)
            
            # Create enhanced database row with advanced microstructure features
            row = {
                'timestamp': update.timestamp.isoformat(),
                'symbol': update.symbol.split(':')[0].replace('/', ''),  # BTC/USDT:USDT -> BTCUSDT
                'exchange': 'bybit',
                'bid_price_1': best_bid_price,
                'bid_size_1': best_bid_size,
                'ask_price_1': best_ask_price,
                'ask_size_1': best_ask_size,
                'mid_price': mid_price,
                'spread': spread,
                'spread_bps': spread_bps,
                'microprice': microprice,
                'order_book_imbalance': enhanced_features['order_book_imbalance'],
                'total_bid_volume_10': total_bid_volume,
                'total_ask_volume_10': total_ask_volume,
                'sequence': update.sequence,
                'exchange_timestamp': update.exchange_timestamp,
                'data_source': 'live_trading',
                'data_quality_score': data_quality_score
            }
            
            # Add enhanced microstructure features if they exist
            row.update({
                'volume_weighted_price_pressure': enhanced_features.get('volume_weighted_price_pressure', 0.0),
                'order_book_slope_bid': enhanced_features.get('order_book_slope_bid', 0.0),
                'order_book_slope_ask': enhanced_features.get('order_book_slope_ask', 0.0),
                'latency_adjusted_imbalance': enhanced_features.get('latency_adjusted_imbalance', 0.0),
                'price_impact_5_levels': enhanced_features.get('price_impact_5_levels', 0.0),
                'liquidity_imbalance_ratio': enhanced_features.get('liquidity_imbalance_ratio', 0.0),
                'depth_imbalance_weighted': enhanced_features.get('depth_imbalance_weighted', 0.0),
                'execution_urgency_score': enhanced_features.get('execution_urgency_score', 0.0)
            })
            
            return row
            
        except Exception as e:
            logger.error(f"Error converting L2Update to database row: {e}")
            return None
    
    def _calculate_data_quality_score(self, bids: List, asks: List) -> float:
        """Calculate basic data quality score based on order book completeness."""
        try:
            # Count non-null levels
            valid_bid_levels = sum(1 for price, size in bids if price and size)
            valid_ask_levels = sum(1 for price, size in asks if price and size)
            
            # Score based on percentage of valid levels (max 10 each side)
            total_valid = valid_bid_levels + valid_ask_levels
            max_possible = 20  # 10 bids + 10 asks
            
            return total_valid / max_possible
            
        except Exception:
            return 0.0
    
    def _calculate_enhanced_data_quality_score(self, bids: List, asks: List, 
                                             processing_latency_us: int, is_stale: bool, 
                                             has_sequence: bool) -> float:
        """Calculate comprehensive data quality score for execution optimization."""
        try:
            score_components = []
            
            # 1. Order book completeness (40% weight)
            valid_bid_levels = sum(1 for price, size in bids if price and size and size > 0)
            valid_ask_levels = sum(1 for price, size in asks if price and size and size > 0)
            completeness_score = (valid_bid_levels + valid_ask_levels) / 20  # max 10 each side
            score_components.append(('completeness', completeness_score, 0.4))
            
            # 2. Processing latency (20% weight)
            latency_score = max(0, 1 - (processing_latency_us / self.config.max_processing_latency_us))
            score_components.append(('latency', latency_score, 0.2))
            
            # 3. Data freshness (20% weight)
            freshness_score = 0.0 if is_stale else 1.0
            score_components.append(('freshness', freshness_score, 0.2))
            
            # 4. Sequence continuity (10% weight)
            sequence_score = 1.0 if has_sequence else 0.5
            score_components.append(('sequence', sequence_score, 0.1))
            
            # 5. Price consistency (10% weight)
            consistency_score = self._check_price_consistency(bids, asks)
            score_components.append(('consistency', consistency_score, 0.1))
            
            # Calculate weighted score
            total_score = sum(score * weight for _, score, weight in score_components)
            
            return min(1.0, max(0.0, total_score))
            
        except Exception as e:
            logger.debug(f"Error calculating data quality score: {e}")
            return 0.0
    
    def _check_price_consistency(self, bids: List, asks: List) -> float:
        """Check for reasonable price consistency in order book."""
        try:
            if not bids or not asks:
                return 0.0
            
            best_bid = float(bids[0][0]) if bids[0][0] else 0
            best_ask = float(asks[0][0]) if asks[0][0] else 0
            
            if best_bid <= 0 or best_ask <= 0:
                return 0.0
            
            # Check spread reasonableness (should be positive and < 1%)
            spread_pct = (best_ask - best_bid) / best_bid
            if spread_pct <= 0 or spread_pct > 0.01:  # Negative spread or > 1%
                return 0.0
            
            # Check price ordering within each side
            bid_ordering_ok = all(float(bids[i][0]) >= float(bids[i+1][0]) 
                                for i in range(min(len(bids)-1, 4)))
            ask_ordering_ok = all(float(asks[i][0]) <= float(asks[i+1][0]) 
                                for i in range(min(len(asks)-1, 4)))
            
            if not (bid_ordering_ok and ask_ordering_ok):
                return 0.3  # Partial score for ordering issues
            
            return 1.0
            
        except Exception:
            return 0.5  # Default score on error
    
    def _classify_market_session(self, bids: List, asks: List) -> str:
        """Classify current market session based on order book characteristics."""
        try:
            if not bids or not asks:
                return 'unknown'
            
            # Calculate spread and depth
            best_bid = float(bids[0][0])
            best_ask = float(asks[0][0])
            spread_bps = ((best_ask - best_bid) / best_bid) * 10000
            
            # Calculate total depth (top 5 levels)
            total_bid_volume = sum(float(size) for _, size in bids[:5] if size)
            total_ask_volume = sum(float(size) for _, size in asks[:5] if size)
            avg_depth = (total_bid_volume + total_ask_volume) / 2
            
            # Classification based on spread and depth
            if spread_bps <= 1.0 and avg_depth >= 50:  # Tight spread, good depth
                return 'active'
            elif spread_bps > 3.0 or avg_depth < 10:  # Wide spread or thin depth
                return 'quiet'
            elif spread_bps > 1.5:  # Moderate spread
                return 'volatile'
            else:
                return 'normal'
                
        except Exception:
            return 'unknown'
    
    def _calculate_enhanced_microstructure_features(self, bids: List, asks: List, mid_price: float) -> Dict[str, float]:
        """Calculate advanced microstructure features for execution optimization."""
        try:
            features = {}
            
            # Ensure we have sufficient data
            min_levels = min(len(bids), len(asks), 5)
            if min_levels < 2:
                return self._get_default_microstructure_features()
            
            # Convert to float arrays for calculations
            bid_prices = [float(price) for price, _ in bids[:min_levels] if price]
            bid_sizes = [float(size) for _, size in bids[:min_levels] if size]
            ask_prices = [float(price) for price, _ in asks[:min_levels] if price]
            ask_sizes = [float(size) for _, size in asks[:min_levels] if size]
            
            if len(bid_prices) < 2 or len(ask_prices) < 2:
                return self._get_default_microstructure_features()
            
            # 1. Volume-Weighted Price Pressure
            # Measures the tendency of prices to move based on order flow
            total_bid_value = sum(p * s for p, s in zip(bid_prices, bid_sizes))
            total_ask_value = sum(p * s for p, s in zip(ask_prices, ask_sizes))
            total_value = total_bid_value + total_ask_value
            
            if total_value > 0:
                features['volume_weighted_price_pressure'] = (total_bid_value - total_ask_value) / total_value
            else:
                features['volume_weighted_price_pressure'] = 0.0
            
            # 2. Order Book Slope Analysis
            # Linear regression slope of price vs cumulative volume
            features['order_book_slope_bid'] = self._calculate_order_book_slope(bid_prices, bid_sizes, 'bid')
            features['order_book_slope_ask'] = self._calculate_order_book_slope(ask_prices, ask_sizes, 'ask')
            
            # 3. Enhanced Order Book Imbalance
            # Traditional imbalance but considering multiple levels
            total_bid_size = sum(bid_sizes)
            total_ask_size = sum(ask_sizes)
            total_size = total_bid_size + total_ask_size
            
            if total_size > 0:
                features['order_book_imbalance'] = (total_bid_size - total_ask_size) / total_size
            else:
                features['order_book_imbalance'] = 0.0
            
            # 4. Latency-Adjusted Imbalance
            # Weights closer levels more heavily for execution decisions
            weighted_bid_size = sum(size / (i + 1) for i, size in enumerate(bid_sizes))
            weighted_ask_size = sum(size / (i + 1) for i, size in enumerate(ask_sizes))
            weighted_total = weighted_bid_size + weighted_ask_size
            
            if weighted_total > 0:
                features['latency_adjusted_imbalance'] = (weighted_bid_size - weighted_ask_size) / weighted_total
            else:
                features['latency_adjusted_imbalance'] = 0.0
            
            # 5. Price Impact Estimation (5 levels)
            # Estimates the price impact of a typical trade
            features['price_impact_5_levels'] = self._estimate_price_impact(
                bid_prices, bid_sizes, ask_prices, ask_sizes, mid_price
            )
            
            # 6. Liquidity Imbalance Ratio
            # Ratio of bid to ask liquidity at different price levels
            if total_ask_size > 0:
                features['liquidity_imbalance_ratio'] = total_bid_size / total_ask_size
            else:
                features['liquidity_imbalance_ratio'] = float('inf') if total_bid_size > 0 else 1.0
            
            # 7. Depth-Imbalanced Weighted Features
            # Considers both price and size in a unified metric
            features['depth_imbalance_weighted'] = self._calculate_depth_weighted_imbalance(
                bid_prices, bid_sizes, ask_prices, ask_sizes, mid_price
            )
            
            # 8. Execution Urgency Score
            # Combines multiple factors to determine optimal execution timing
            features['execution_urgency_score'] = self._calculate_execution_urgency_score(
                bid_prices, bid_sizes, ask_prices, ask_sizes, mid_price, features
            )
            
            return features
            
        except Exception as e:
            logger.debug(f"Error calculating enhanced microstructure features: {e}")
            return self._get_default_microstructure_features()
    
    def _get_default_microstructure_features(self) -> Dict[str, float]:
        """Return default values for microstructure features."""
        return {
            'volume_weighted_price_pressure': 0.0,
            'order_book_slope_bid': 0.0,
            'order_book_slope_ask': 0.0,
            'order_book_imbalance': 0.0,
            'latency_adjusted_imbalance': 0.0,
            'price_impact_5_levels': 0.0,
            'liquidity_imbalance_ratio': 1.0,
            'depth_imbalance_weighted': 0.0,
            'execution_urgency_score': 0.5
        }
    
    def _calculate_order_book_slope(self, prices: List[float], sizes: List[float], side: str) -> float:
        """Calculate the slope of the order book (price vs cumulative volume)."""
        try:
            if len(prices) < 2:
                return 0.0
            
            # Calculate cumulative volumes
            cumulative_volumes = []
            cumsum = 0
            for size in sizes:
                cumsum += size
                cumulative_volumes.append(cumsum)
            
            # Simple linear regression slope calculation
            n = len(prices)
            sum_x = sum(cumulative_volumes)
            sum_y = sum(prices)
            sum_xy = sum(x * y for x, y in zip(cumulative_volumes, prices))
            sum_x2 = sum(x * x for x in cumulative_volumes)
            
            denominator = n * sum_x2 - sum_x * sum_x
            if abs(denominator) < 1e-10:
                return 0.0
            
            slope = (n * sum_xy - sum_x * sum_y) / denominator
            
            # Normalize slope based on side
            if side == 'bid':
                return -slope  # Negative slope is normal for bids
            else:
                return slope   # Positive slope is normal for asks
                
        except Exception:
            return 0.0
    
    def _estimate_price_impact(self, bid_prices: List[float], bid_sizes: List[float],
                             ask_prices: List[float], ask_sizes: List[float], mid_price: float) -> float:
        """Estimate price impact of a typical trade size."""
        try:
            # Use a typical trade size (e.g., 1 BTC equivalent)
            typical_trade_usd = 70000  # $70k ~ 1 BTC
            typical_trade_size = typical_trade_usd / mid_price
            
            # Simulate market buy (walking the ask side)
            remaining_size = typical_trade_size
            weighted_price = 0.0
            total_filled = 0.0
            
            for price, size in zip(ask_prices, ask_sizes):
                if remaining_size <= 0:
                    break
                    
                fill_size = min(remaining_size, size)
                weighted_price += price * fill_size
                total_filled += fill_size
                remaining_size -= fill_size
            
            if total_filled == 0:
                return 0.0
            
            avg_fill_price = weighted_price / total_filled
            price_impact = (avg_fill_price - mid_price) / mid_price
            
            return price_impact
            
        except Exception:
            return 0.0
    
    def _calculate_depth_weighted_imbalance(self, bid_prices: List[float], bid_sizes: List[float],
                                          ask_prices: List[float], ask_sizes: List[float], 
                                          mid_price: float) -> float:
        """Calculate depth-weighted imbalance considering price levels."""
        try:
            weighted_bid_value = 0.0
            weighted_ask_value = 0.0
            
            # Weight by inverse of price distance from mid
            for price, size in zip(bid_prices, bid_sizes):
                distance = abs(price - mid_price) / mid_price
                weight = 1.0 / (1.0 + distance * 10)  # Exponential decay
                weighted_bid_value += price * size * weight
            
            for price, size in zip(ask_prices, ask_sizes):
                distance = abs(price - mid_price) / mid_price
                weight = 1.0 / (1.0 + distance * 10)  # Exponential decay
                weighted_ask_value += price * size * weight
            
            total_weighted_value = weighted_bid_value + weighted_ask_value
            if total_weighted_value > 0:
                return (weighted_bid_value - weighted_ask_value) / total_weighted_value
            else:
                return 0.0
                
        except Exception:
            return 0.0
    
    def _calculate_execution_urgency_score(self, bid_prices: List[float], bid_sizes: List[float],
                                         ask_prices: List[float], ask_sizes: List[float],
                                         mid_price: float, features: Dict[str, float]) -> float:
        """Calculate execution urgency score for optimal timing."""
        try:
            urgency_factors = []
            
            # Factor 1: Spread tightness (tight spread = higher urgency)
            spread = ask_prices[0] - bid_prices[0]
            spread_pct = spread / mid_price
            spread_score = max(0, 1 - spread_pct * 1000)  # Normalize spread
            urgency_factors.append(('spread', spread_score, 0.3))
            
            # Factor 2: Liquidity availability (more liquidity = lower urgency)
            total_liquidity = sum(bid_sizes) + sum(ask_sizes)
            liquidity_score = min(1.0, total_liquidity / 100.0)  # Normalize to reasonable levels
            urgency_factors.append(('liquidity', 1 - liquidity_score, 0.2))
            
            # Factor 3: Order book imbalance (higher imbalance = higher urgency)
            imbalance_score = abs(features.get('order_book_imbalance', 0))
            urgency_factors.append(('imbalance', imbalance_score, 0.2))
            
            # Factor 4: Price impact (higher impact = higher urgency)
            impact_score = abs(features.get('price_impact_5_levels', 0)) * 100
            urgency_factors.append(('impact', min(1.0, impact_score), 0.2))
            
            # Factor 5: Slope steepness (steeper slope = higher urgency)
            bid_slope = abs(features.get('order_book_slope_bid', 0))
            ask_slope = abs(features.get('order_book_slope_ask', 0))
            slope_score = min(1.0, (bid_slope + ask_slope) / 2)
            urgency_factors.append(('slope', slope_score, 0.1))
            
            # Calculate weighted urgency score
            total_score = sum(score * weight for _, score, weight in urgency_factors)
            
            # Ensure score is between 0 and 1
            return max(0.0, min(1.0, total_score))
            
        except Exception:
            return 0.5  # Default neutral urgency
    
    def _process_bybit_orderbook_update(self, raw_data: Dict[str, Any]) -> Optional[L2Update]:
        """Enhanced processing for Bybit-specific orderbook updates with delta optimization."""
        try:
            # Check if this is a delta update or full snapshot
            is_delta = raw_data.get('type') == 'delta'
            is_snapshot = raw_data.get('type') == 'snapshot'
            
            if is_delta and self.cached_orderbook:
                # Process delta update efficiently
                return self._process_delta_update(raw_data)
            elif is_snapshot or not self.cached_orderbook:
                # Process full snapshot and update cache
                return self._process_full_snapshot(raw_data)
            else:
                # Fallback to standard processing
                return self._process_orderbook_update(raw_data)
                
        except Exception as e:
            logger.error(f"Error in Bybit-specific orderbook processing: {e}")
            return self._process_orderbook_update(raw_data)  # Fallback
    
    def _process_delta_update(self, delta_data: Dict[str, Any]) -> Optional[L2Update]:
        """Process Bybit delta update efficiently using cached orderbook state."""
        try:
            if not self.cached_orderbook:
                logger.warning("Received delta update without cached orderbook - requesting snapshot")
                return None
            
            # Apply delta changes to cached orderbook
            bids_changes = delta_data.get('data', {}).get('b', [])  # Bybit uses 'b' for bids
            asks_changes = delta_data.get('data', {}).get('a', [])  # Bybit uses 'a' for asks
            
            # Update cached bids
            if bids_changes:
                self._apply_orderbook_changes(self.cached_orderbook['bids'], bids_changes, 'bid')
            
            # Update cached asks
            if asks_changes:
                self._apply_orderbook_changes(self.cached_orderbook['asks'], asks_changes, 'ask')
            
            # Create L2Update from updated cache
            orderbook_data = {
                'symbol': delta_data.get('data', {}).get('s', self.config.symbol),
                'timestamp': delta_data.get('ts'),
                'sequence': delta_data.get('data', {}).get('u'),  # Bybit uses 'u' for update_id
                'bids': self.cached_orderbook['bids'][:self.config.orderbook_depth],
                'asks': self.cached_orderbook['asks'][:self.config.orderbook_depth]
            }
            
            self.delta_updates_count += 1
            return self._process_orderbook_update(orderbook_data)
            
        except Exception as e:
            logger.error(f"Error processing delta update: {e}")
            # Invalidate cache and request new snapshot
            self.cached_orderbook = None
            return None
    
    def _process_full_snapshot(self, snapshot_data: Dict[str, Any]) -> Optional[L2Update]:
        """Process full orderbook snapshot and update cache."""
        try:
            data = snapshot_data.get('data', {})
            
            # Extract orderbook data
            bids = data.get('b', [])  # Bybit format
            asks = data.get('a', [])  # Bybit format
            
            if not bids or not asks:
                # Try standard format as fallback
                bids = snapshot_data.get('bids', [])
                asks = snapshot_data.get('asks', [])
            
            # Convert to standard format and sort
            formatted_bids = [(float(price), float(size)) for price, size in bids if price and size]
            formatted_asks = [(float(price), float(size)) for price, size in asks if price and size]
            
            # Sort bids (highest to lowest) and asks (lowest to highest)
            formatted_bids.sort(key=lambda x: x[0], reverse=True)
            formatted_asks.sort(key=lambda x: x[0])
            
            # Update cache
            self.cached_orderbook = {
                'bids': formatted_bids,
                'asks': formatted_asks,
                'timestamp': snapshot_data.get('ts', int(time.time() * 1000)),
                'sequence': data.get('u')
            }
            
            self.last_full_snapshot_time = datetime.now()
            self.full_snapshots_count += 1
            
            # Create orderbook data for processing
            orderbook_data = {
                'symbol': data.get('s', self.config.symbol),
                'timestamp': snapshot_data.get('ts'),
                'sequence': data.get('u'),
                'bids': formatted_bids[:self.config.orderbook_depth],
                'asks': formatted_asks[:self.config.orderbook_depth]
            }
            
            return self._process_orderbook_update(orderbook_data)
            
        except Exception as e:
            logger.error(f"Error processing full snapshot: {e}")
            return None
    
    def _apply_orderbook_changes(self, current_side: List[tuple], changes: List, side_type: str) -> None:
        """Apply delta changes to one side of the orderbook."""
        try:
            # Convert current side to dict for easier manipulation
            price_size_dict = {float(price): float(size) for price, size in current_side}
            
            # Apply changes
            for change in changes:
                if len(change) >= 2:
                    price = float(change[0])
                    size = float(change[1])
                    
                    if size == 0:
                        # Remove price level
                        price_size_dict.pop(price, None)
                    else:
                        # Update price level
                        price_size_dict[price] = size
            
            # Convert back to list and sort
            if side_type == 'bid':
                # Bids: highest price first
                updated_side = sorted(price_size_dict.items(), key=lambda x: x[0], reverse=True)
            else:
                # Asks: lowest price first
                updated_side = sorted(price_size_dict.items(), key=lambda x: x[0])
            
            # Update the original list
            current_side.clear()
            current_side.extend(updated_side)
            
        except Exception as e:
            logger.error(f"Error applying orderbook changes: {e}")
    
    def _should_request_snapshot(self) -> bool:
        """Determine if we should request a fresh orderbook snapshot."""
        if not self.cached_orderbook:
            return True
        
        # Request snapshot if too much time has passed
        if self.last_full_snapshot_time:
            time_since_snapshot = (datetime.now() - self.last_full_snapshot_time).total_seconds()
            if time_since_snapshot > 300:  # 5 minutes
                return True
        
        # Request snapshot if we've had too many sequence gaps
        if self.sequence_gap_count > 10:
            self.sequence_gap_count = 0  # Reset counter
            return True
        
        return False
    
    def _write_updates_to_database(self, updates: List[L2Update]):
        """Write a batch of L2 updates to the database."""
        if not updates:
            return
        
        try:
            with self.get_db_connection() as conn:
                cursor = conn.cursor()
                
                # Convert updates to database rows
                rows = []
                for update in updates:
                    row = self._l2_update_to_database_row(update)
                    if row:
                        rows.append(row)
                
                if not rows:
                    logger.warning("No valid rows to write to database")
                    return
                
                # Prepare INSERT statement
                columns = list(rows[0].keys())
                placeholders = ', '.join(['?' for _ in columns])
                column_names = ', '.join(columns)
                
                insert_sql = f"""
                    INSERT INTO {self.config.table_name} ({column_names})
                    VALUES ({placeholders})
                """
                
                # Execute batch insert
                for row in rows:
                    values = [row[col] for col in columns]
                    cursor.execute(insert_sql, values)
                
                conn.commit()
                
                self.stats['successful_writes'] += len(rows)
                
                if self.config.log_updates:
                    logger.debug(f"Wrote {len(rows)} L2 updates to database")
                
        except Exception as e:
            logger.error(f"Error writing updates to database: {e}")
            self.stats['failed_writes'] += len(updates)
            if self.on_error_callback:
                self.on_error_callback(e)
    
    def _cleanup_old_data(self):
        """Remove old data beyond retention period."""
        try:
            if self.config.data_retention_hours <= 0:
                return  # No cleanup if retention is disabled
            
            cutoff_time = datetime.now(timezone.utc) - timedelta(hours=self.config.data_retention_hours)
            
            with self.get_db_connection() as conn:
                cursor = conn.cursor()
                
                delete_sql = f"""
                    DELETE FROM {self.config.table_name}
                    WHERE timestamp < ? AND data_source = 'demo_trading'
                """
                
                cursor.execute(delete_sql, (cutoff_time.isoformat(),))
                deleted_rows = cursor.rowcount
                conn.commit()
                
                if deleted_rows > 0:
                    logger.info(f"Cleaned up {deleted_rows} old live data rows")
                
        except Exception as e:
            logger.error(f"Error during data cleanup: {e}")
    
    async def _websocket_loop(self):
        """Simplified and robust WebSocket loop for receiving L2 updates."""
        reconnect_count = 0
        max_consecutive_errors = 5
        consecutive_errors = 0
        
        logger.info(f"Starting WebSocket loop for {self.config.symbol}")
        
        while self.running and not self.stop_event.is_set():
            try:
                logger.info(f"Subscribing to orderbook stream: {self.config.symbol}")
                
                # Reset consecutive error count on successful connection
                consecutive_errors = 0
                
                # Main data receiving loop
                while self.running and not self.stop_event.is_set():
                    try:
                        # Get orderbook update with timeout
                        orderbook = await asyncio.wait_for(
                            self.exchange.watch_order_book(self.config.symbol),
                            timeout=30.0  # 30 second timeout
                        )
                        
                        # Validate orderbook data
                        if not self._is_valid_orderbook(orderbook):
                            continue
                        
                        # Process orderbook update with Bybit-specific optimization
                        if self.config.exchange_name.lower() == 'bybit':
                            update = self._process_bybit_orderbook_update(orderbook)
                        else:
                            update = self._process_orderbook_update(orderbook)
                        
                        if not update:
                            # Check if we should request a fresh snapshot
                            if self._should_request_snapshot():
                                logger.info("Requesting fresh orderbook snapshot")
                            continue
                        
                        # Add to appropriate priority buffer (thread-safe)
                        with self.buffer_lock:
                            # Determine priority based on data quality and market conditions
                            is_high_priority = self._is_high_priority_update(update)
                            
                            if is_high_priority:
                                self.high_priority_buffer.append(update)
                                self.buffer_stats['high_priority_count'] += 1
                                
                                # Check for high priority buffer overflow
                                if len(self.high_priority_buffer) >= self.high_priority_buffer.maxlen:
                                    self.buffer_stats['buffer_overflows'] += 1
                                    logger.warning("High priority buffer near capacity")
                            else:
                                self.normal_priority_buffer.append(update)
                                self.buffer_stats['normal_priority_count'] += 1
                            
                            # Adaptive buffer resizing check
                            self._check_adaptive_buffer_resize()
                            
                            # Track market volatility for buffer optimization
                            self._track_market_volatility(update)
                        
                        # Update statistics
                        self.stats['total_updates'] += 1
                        self.stats['last_update_time'] = datetime.now()
                        
                        # Call update callback if set
                        if self.on_update_callback:
                            try:
                                self.on_update_callback(update)
                            except Exception as callback_error:
                                logger.error(f"Error in update callback: {callback_error}")
                        
                        # Log periodic updates with enhanced buffer information
                        if self.stats['total_updates'] % 100 == 0:
                            total_buffer_size = len(self.high_priority_buffer) + len(self.normal_priority_buffer)
                            logger.debug(
                                f"Processed {self.stats['total_updates']} updates, "
                                f"buffers: high={len(self.high_priority_buffer)}, "
                                f"normal={len(self.normal_priority_buffer)}, "
                                f"quality_ratio={self.stats['high_quality_updates']/(self.stats['total_updates'] or 1):.2f}"
                            )
                    
                    except asyncio.TimeoutError:
                        logger.warning("WebSocket timeout - no data received in 30 seconds")
                        consecutive_errors += 1
                        if consecutive_errors >= max_consecutive_errors:
                            raise Exception("Too many consecutive timeouts")
                        continue
                    
                    except Exception as inner_e:
                        logger.error(f"Error in WebSocket data loop: {inner_e}")
                        consecutive_errors += 1
                        if consecutive_errors >= max_consecutive_errors:
                            raise inner_e
                        await asyncio.sleep(1)  # Brief pause before retry
                        continue
                        
            except Exception as e:
                reconnect_count += 1
                self.stats['reconnections'] += 1
                consecutive_errors = 0  # Reset on major error
                
                logger.error(f"WebSocket connection error (attempt {reconnect_count}/{self.config.max_reconnect_attempts}): {e}")
                
                if self.on_error_callback:
                    try:
                        self.on_error_callback(e)
                    except Exception as callback_error:
                        logger.error(f"Error in error callback: {callback_error}")
                
                # Check if we should stop trying
                if reconnect_count >= self.config.max_reconnect_attempts:
                    logger.error("Max reconnection attempts exceeded - stopping WebSocket loop")
                    break
                
                # Wait before reconnecting
                wait_time = min(self.config.reconnect_delay * reconnect_count, 60)  # Exponential backoff, max 60s
                logger.info(f"Reconnecting in {wait_time} seconds...")
                await asyncio.sleep(wait_time)
                
                # Reinitialize exchange connection
                try:
                    if not self._initialize_exchange():
                        logger.error("Failed to reinitialize exchange connection")
                        break
                except Exception as init_error:
                    logger.error(f"Error reinitializing exchange: {init_error}")
                    break
        
        logger.info("WebSocket loop ended")
    
    def _is_valid_orderbook(self, orderbook: Any) -> bool:
        """Validate orderbook data format."""
        try:
            # Must be a dictionary
            if not isinstance(orderbook, dict):
                logger.debug(f"Skipping non-dict orderbook: {type(orderbook)}")
                return False
            
            # Must have bids and asks
            if 'bids' not in orderbook or 'asks' not in orderbook:
                logger.debug(f"Skipping incomplete orderbook: {list(orderbook.keys())}")
                return False
            
            # Bids and asks must be lists with data
            bids = orderbook.get('bids', [])
            asks = orderbook.get('asks', [])
            
            if not isinstance(bids, list) or not isinstance(asks, list):
                logger.debug("Skipping orderbook with invalid bids/asks format")
                return False
            
            if len(bids) == 0 or len(asks) == 0:
                logger.debug("Skipping orderbook with empty bids/asks")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating orderbook: {e}")
            return False
    
    def _writer_loop(self):
        """Improved background thread loop for writing buffered updates to database."""
        last_cleanup = time.time()
        write_failures = 0
        max_write_failures = 10
        
        logger.info("Writer thread started")
        
        while self.running and not self.stop_event.wait(self.config.write_interval):
            try:
                # Get buffered updates with priority handling (thread-safe)
                updates_to_write = []
                with self.buffer_lock:
                    # Process high priority updates first
                    high_priority_batch_size = min(len(self.high_priority_buffer), 
                                                  self.config.buffer_size // 4)
                    if high_priority_batch_size > 0:
                        for _ in range(high_priority_batch_size):
                            if self.high_priority_buffer:
                                updates_to_write.append(self.high_priority_buffer.popleft())
                    
                    # Fill remaining batch with normal priority updates
                    remaining_capacity = self.config.buffer_size - len(updates_to_write)
                    normal_batch_size = min(len(self.normal_priority_buffer), remaining_capacity)
                    if normal_batch_size > 0:
                        for _ in range(normal_batch_size):
                            if self.normal_priority_buffer:
                                updates_to_write.append(self.normal_priority_buffer.popleft())
                
                # Write to database if we have updates
                if updates_to_write:
                    try:
                        self._write_updates_to_database(updates_to_write)
                        write_failures = 0  # Reset failure count on success
                        
                        if len(updates_to_write) >= 10:  # Log larger batches
                            logger.info(f"Successfully wrote {len(updates_to_write)} updates to database")
                        
                    except Exception as write_error:
                        write_failures += 1
                        logger.error(f"Database write failed (attempt {write_failures}/{max_write_failures}): {write_error}")
                        
                        if write_failures >= max_write_failures:
                            logger.critical("Too many database write failures - stopping writer thread")
                            break
                        
                        # Put updates back in appropriate buffers for retry
                        with self.buffer_lock:
                            for update in reversed(updates_to_write):
                                if self._is_high_priority_update(update):
                                    self.high_priority_buffer.appendleft(update)
                                else:
                                    self.normal_priority_buffer.appendleft(update)
                
                # Periodic cleanup (every hour)
                now = time.time()
                if now - last_cleanup > 3600:
                    try:
                        self._cleanup_old_data()
                        last_cleanup = now
                    except Exception as cleanup_error:
                        logger.error(f"Data cleanup failed: {cleanup_error}")
                
                # Log enhanced buffer status periodically
                total_buffer_size = len(self.high_priority_buffer) + len(self.normal_priority_buffer)
                if total_buffer_size > 0 and total_buffer_size % 50 == 0:
                    logger.debug(
                        f"Writer buffers - High: {len(self.high_priority_buffer)}, "
                        f"Normal: {len(self.normal_priority_buffer)}, "
                        f"Overflows: {self.buffer_stats['buffer_overflows']}"
                    )
                
            except Exception as e:
                logger.error(f"Unexpected error in writer loop: {e}")
                time.sleep(1)  # Brief pause on unexpected errors
        
        # Final cleanup - write any remaining buffered updates
        try:
            with self.buffer_lock:
                remaining_updates = list(self.high_priority_buffer) + list(self.normal_priority_buffer)
                if remaining_updates:
                    logger.info(f"Writing {len(remaining_updates)} remaining buffered updates...")
                    self._write_updates_to_database(remaining_updates)
                    self.high_priority_buffer.clear()
                    self.normal_priority_buffer.clear()
        except Exception as final_error:
            logger.error(f"Error writing final buffered updates: {final_error}")
        
        logger.info("Writer thread ended")
    
    def start(self) -> bool:
        """Start the live data ingestion with improved error handling."""
        try:
            logger.info("🚀 Starting live data ingestion...")
            
            # Validate configuration
            if not self.config.symbol:
                logger.error("No symbol specified in configuration")
                return False
            
            # Initialize exchange connection
            logger.info("Initializing exchange connection...")
            if not self._initialize_exchange():
                logger.error("Failed to initialize exchange connection")
                return False
            
            # Setup database schema
            logger.info("Setting up database schema...")
            self._setup_database_schema()
            
            # Clear any existing state
            self.running = False
            self.stop_event.clear()
            
            # Initialize statistics
            self.stats.update({
                'start_time': datetime.now(),
                'total_updates': 0,
                'successful_writes': 0,
                'failed_writes': 0,
                'reconnections': 0,
                'last_update_time': None
            })
            
            # Set running flag
            self.running = True
            
            # Start writer thread first
            logger.info("Starting database writer thread...")
            self.writer_thread = threading.Thread(
                target=self._writer_loop, 
                name="DataIngestor-Writer",
                daemon=True
            )
            self.writer_thread.start()
            logger.info("✓ Writer thread started")
            
            # Start WebSocket ingestion thread with dedicated event loop
            logger.info("Starting WebSocket ingestion thread...")
            
            def run_websocket_loop():
                """Run WebSocket loop in dedicated thread with its own event loop."""
                try:
                    # Create new event loop for this thread
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    
                    # Run the WebSocket loop
                    loop.run_until_complete(self._websocket_loop())
                    
                except Exception as e:
                    logger.error(f"WebSocket thread error: {e}")
                    if self.on_error_callback:
                        try:
                            self.on_error_callback(e)
                        except Exception:
                            pass
                finally:
                    # Clean up event loop
                    try:
                        loop.close()
                    except Exception:
                        pass
                    logger.info("WebSocket thread event loop closed")
            
            self.ingestion_thread = threading.Thread(
                target=run_websocket_loop,
                name="DataIngestor-WebSocket", 
                daemon=True
            )
            self.ingestion_thread.start()
            logger.info("✓ WebSocket ingestion thread started")
            
            # Verify threads are running
            time.sleep(1)
            if not self.writer_thread.is_alive():
                logger.error("Writer thread failed to start")
                self.stop()
                return False
            
            if not self.ingestion_thread.is_alive():
                logger.error("WebSocket thread failed to start")
                self.stop()
                return False
            
            logger.info("✅ Live data ingestion started successfully")
            logger.info(f"   Symbol: {self.config.symbol}")
            logger.info(f"   Database: {self.config.db_path}")
            logger.info(f"   Table: {self.config.table_name}")
            logger.info(f"   Sandbox: {self.config.sandbox}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to start live data ingestion: {e}")
            self.stop()
            return False
    
    def stop(self):
        """Stop the live data ingestion gracefully."""
        logger.info("Stopping live data ingestion...")
        
        # Set stop flags
        self.running = False
        self.stop_event.set()
        
        # Close exchange connection gracefully
        if self.exchange:
            try:
                logger.info("Closing exchange connection...")
                
                # Handle async close method properly
                if hasattr(self.exchange, 'close'):
                    if asyncio.iscoroutinefunction(self.exchange.close):
                        # Try to close async connection
                        try:
                            # Check if there's a running event loop
                            loop = asyncio.get_running_loop()
                            # If there's a running loop, create a task
                            asyncio.create_task(self.exchange.close())
                        except RuntimeError:
                            # No running loop, create new one to close
                            try:
                                loop = asyncio.new_event_loop()
                                asyncio.set_event_loop(loop)
                                loop.run_until_complete(self.exchange.close())
                                loop.close()
                            except Exception as close_error:
                                logger.warning(f"Error in async close: {close_error}")
                    else:
                        # Synchronous close
                        self.exchange.close()
                
                self.exchange = None
                logger.info("✓ Exchange connection closed")
                
            except Exception as e:
                logger.warning(f"Error closing exchange connection: {e}")
        
        # Wait for threads to finish gracefully
        threads_to_stop = [
            ("WebSocket ingestion", self.ingestion_thread),
            ("Database writer", self.writer_thread)
        ]
        
        for thread_name, thread in threads_to_stop:
            if thread and thread.is_alive():
                logger.info(f"Waiting for {thread_name} thread to stop...")
                thread.join(timeout=10)  # Increased timeout
                
                if thread.is_alive():
                    logger.warning(f"{thread_name} thread did not stop gracefully within timeout")
                else:
                    logger.info(f"✓ {thread_name} thread stopped")
        
        # Final buffer flush
        try:
            with self.buffer_lock:
                if self.update_buffer:
                    logger.info(f"Final flush: {len(self.update_buffer)} updates remaining")
                    # Don't try to write if writer thread has stopped
                    self.update_buffer.clear()
        except Exception as e:
            logger.warning(f"Error during final buffer cleanup: {e}")
        
        # Log final statistics
        if self.stats.get('start_time'):
            uptime = datetime.now() - self.stats['start_time']
            logger.info(f"Final stats: {self.stats['total_updates']} updates processed, "
                       f"{self.stats['successful_writes']} successful writes, "
                       f"uptime: {uptime}")
        
        logger.info("✅ Live data ingestion stopped")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive ingestion statistics with execution analytics."""
        stats = self.stats.copy()
        stats['is_running'] = self.running
        stats['high_priority_buffer_size'] = len(self.high_priority_buffer)
        stats['normal_priority_buffer_size'] = len(self.normal_priority_buffer)
        stats['total_buffer_size'] = len(self.high_priority_buffer) + len(self.normal_priority_buffer)
        stats['buffer_stats'] = self.buffer_stats.copy()
        
        if stats['start_time']:
            uptime_seconds = (datetime.now() - stats['start_time']).total_seconds()
            stats['uptime_seconds'] = uptime_seconds
            
            # Calculate throughput metrics
            if uptime_seconds > 0:
                stats['updates_per_second'] = stats['total_updates'] / uptime_seconds
                stats['successful_writes_per_second'] = stats['successful_writes'] / uptime_seconds
                
                # Data quality metrics
                total_quality_updates = stats['high_quality_updates'] + stats['low_quality_updates']
                if total_quality_updates > 0:
                    stats['high_quality_ratio'] = stats['high_quality_updates'] / total_quality_updates
                else:
                    stats['high_quality_ratio'] = 0.0
                
                # Latency percentiles
                if self.latency_buffer:
                    latencies = list(self.latency_buffer)
                    latencies.sort()
                    n = len(latencies)
                    stats['latency_percentiles'] = {
                        'p50': latencies[n // 2] if n > 0 else 0,
                        'p95': latencies[int(n * 0.95)] if n > 0 else 0,
                        'p99': latencies[int(n * 0.99)] if n > 0 else 0
                    }
                
                # Error rates
                total_operations = stats['successful_writes'] + stats['failed_writes']
                if total_operations > 0:
                    stats['error_rate'] = stats['failed_writes'] / total_operations
                else:
                    stats['error_rate'] = 0.0
                
                # Market volatility insights
                if self.volatility_buffer:
                    volatilities = list(self.volatility_buffer)
                    stats['market_volatility'] = {
                        'current_spread_bps': volatilities[-1] if volatilities else 0,
                        'avg_spread_bps': statistics.mean(volatilities),
                        'max_spread_bps': max(volatilities),
                        'volatility_trend': self._calculate_volatility_trend()
                    }
                
                # Bybit-specific optimization statistics
                if self.config.exchange_name.lower() == 'bybit':
                    total_updates = self.delta_updates_count + self.full_snapshots_count
                    stats['bybit_optimizations'] = {
                        'delta_updates_count': self.delta_updates_count,
                        'full_snapshots_count': self.full_snapshots_count,
                        'delta_efficiency_ratio': (self.delta_updates_count / total_updates) if total_updates > 0 else 0,
                        'cached_orderbook_active': self.cached_orderbook is not None,
                        'last_snapshot_age_seconds': (
                            (datetime.now() - self.last_full_snapshot_time).total_seconds()
                            if self.last_full_snapshot_time else None
                        ),
                        'websocket_reconnects': self.websocket_reconnect_count
                    }
        
        return stats
    
    def _calculate_volatility_trend(self) -> str:
        """Calculate current volatility trend."""
        try:
            if len(self.volatility_buffer) < 20:
                return 'insufficient_data'
            
            recent_volatility = list(self.volatility_buffer)[-10:]
            older_volatility = list(self.volatility_buffer)[-20:-10]
            
            recent_avg = statistics.mean(recent_volatility)
            older_avg = statistics.mean(older_volatility)
            
            if recent_avg > older_avg * 1.2:
                return 'increasing'
            elif recent_avg < older_avg * 0.8:
                return 'decreasing'
            else:
                return 'stable'
                
        except Exception:
            return 'unknown'
    
    def get_execution_health_report(self) -> Dict[str, Any]:
        """Generate comprehensive execution health report."""
        try:
            stats = self.get_stats()
            health_score = 0.0
            health_factors = []
            
            # Factor 1: Data processing performance (25%)
            if stats.get('avg_processing_latency_us', 0) < 5000:  # < 5ms
                processing_score = 1.0
            elif stats.get('avg_processing_latency_us', 0) < 10000:  # < 10ms
                processing_score = 0.7
            else:
                processing_score = 0.3
            health_factors.append(('processing_speed', processing_score, 0.25))
            
            # Factor 2: Data quality (30%)
            quality_ratio = stats.get('high_quality_ratio', 0)
            health_factors.append(('data_quality', quality_ratio, 0.30))
            
            # Factor 3: System stability (20%)
            error_rate = stats.get('error_rate', 1.0)
            stability_score = max(0, 1 - error_rate * 10)  # Penalize high error rates
            health_factors.append(('stability', stability_score, 0.20))
            
            # Factor 4: Throughput performance (15%)
            updates_per_sec = stats.get('updates_per_second', 0)
            if updates_per_sec > 10:  # Good throughput
                throughput_score = 1.0
            elif updates_per_sec > 5:  # Moderate throughput
                throughput_score = 0.7
            else:
                throughput_score = 0.3
            health_factors.append(('throughput', throughput_score, 0.15))
            
            # Factor 5: Buffer efficiency (10%)
            total_buffer = stats.get('total_buffer_size', 0)
            if total_buffer < 50:  # Low buffer utilization
                buffer_score = 1.0
            elif total_buffer < 100:  # Moderate utilization
                buffer_score = 0.7
            else:
                buffer_score = 0.3  # High utilization indicates potential issues
            health_factors.append(('buffer_efficiency', buffer_score, 0.10))
            
            # Calculate weighted health score
            health_score = sum(score * weight for _, score, weight in health_factors)
            
            # Generate recommendations
            recommendations = []
            if processing_score < 0.7:
                recommendations.append("Consider optimizing data processing pipeline")
            if quality_ratio < 0.8:
                recommendations.append("Investigate data quality issues - check network connectivity")
            if stability_score < 0.8:
                recommendations.append("System stability concerns - check error logs")
            if throughput_score < 0.7:
                recommendations.append("Low throughput detected - check WebSocket connection")
            if buffer_score < 0.7:
                recommendations.append("High buffer utilization - consider increasing buffer size")
            
            # Overall health assessment
            if health_score >= 0.9:
                health_status = 'excellent'
            elif health_score >= 0.8:
                health_status = 'good'
            elif health_score >= 0.6:
                health_status = 'fair'
            else:
                health_status = 'poor'
            
            return {
                'timestamp': datetime.now().isoformat(),
                'overall_health_score': health_score,
                'health_status': health_status,
                'health_factors': {name: score for name, score, _ in health_factors},
                'recommendations': recommendations,
                'detailed_stats': stats,
                'execution_readiness': health_score > 0.7  # Ready for live execution
            }
            
        except Exception as e:
            logger.error(f"Error generating health report: {e}")
            return {
                'timestamp': datetime.now().isoformat(),
                'overall_health_score': 0.0,
                'health_status': 'error',
                'error': str(e)
            }
    
    def is_healthy(self) -> bool:
        """Check if the data ingestor is healthy."""
        if not self.running:
            return False
        
        # Check if we've received updates recently
        if self.stats['last_update_time']:
            time_since_update = (datetime.now() - self.stats['last_update_time']).total_seconds()
            if time_since_update > 60:  # No updates in last minute
                return False
        
        # Check if threads are alive
        if self.ingestion_thread and not self.ingestion_thread.is_alive():
            return False
        
        if self.writer_thread and not self.writer_thread.is_alive():
            return False
        
        return True
    
    def _is_high_priority_update(self, update: L2Update) -> bool:
        """Determine if an update should be processed with high priority."""
        try:
            # High priority criteria:
            # 1. High data quality score
            # 2. Active market session
            # 3. Low processing latency
            # 4. Fresh data (not stale)
            
            high_quality = (hasattr(update, 'data_quality_score') and 
                          update.data_quality_score and 
                          update.data_quality_score >= 0.9)
            
            active_market = (hasattr(update, 'market_session') and 
                           update.market_session in ['active', 'volatile'])
            
            low_latency = (hasattr(update, 'processing_latency_us') and 
                         update.processing_latency_us and 
                         update.processing_latency_us < 5000)  # < 5ms
            
            fresh_data = not (hasattr(update, 'is_stale') and update.is_stale)
            
            # Consider high priority if meets at least 2 criteria
            priority_score = sum([high_quality, active_market, low_latency, fresh_data])
            return priority_score >= 2
            
        except Exception as e:
            logger.debug(f"Error determining update priority: {e}")
            return False
    
    def get_execution_signals(self) -> Dict[str, Any]:
        """
        Generate real-time execution signals for the trading engine.
        
        Returns comprehensive market microstructure analysis optimized for execution decisions.
        """
        try:
            if not self.cached_orderbook:
                return {
                    'status': 'no_data',
                    'timestamp': datetime.now().isoformat(),
                    'message': 'No cached orderbook available'
                }
            
            # Get current market data
            bids = self.cached_orderbook['bids'][:10]
            asks = self.cached_orderbook['asks'][:10]
            
            if not bids or not asks:
                return {
                    'status': 'insufficient_data',
                    'timestamp': datetime.now().isoformat()
                }
            
            # Calculate enhanced execution metrics
            best_bid_price, best_bid_size = bids[0]
            best_ask_price, best_ask_size = asks[0]
            mid_price = (best_bid_price + best_ask_price) / 2
            
            # Enhanced microstructure analysis
            enhanced_features = self._calculate_enhanced_microstructure_features(bids, asks, mid_price)
            
            # Real-time liquidity scoring
            liquidity_score = self._calculate_real_time_liquidity_score(bids, asks)
            
            # Execution timing optimization
            timing_signals = self._generate_execution_timing_signals(enhanced_features, liquidity_score)
            
            # Market impact estimation for different order sizes
            impact_analysis = self._calculate_execution_impact_analysis(bids, asks, mid_price)
            
            # Optimal execution recommendations
            execution_recommendations = self._generate_execution_recommendations(
                enhanced_features, liquidity_score, timing_signals, impact_analysis
            )
            
            return {
                'status': 'active',
                'timestamp': datetime.now().isoformat(),
                'market_data': {
                    'mid_price': mid_price,
                    'best_bid': best_bid_price,
                    'best_ask': best_ask_price,
                    'spread_bps': ((best_ask_price - best_bid_price) / mid_price) * 10000,
                    'market_session': enhanced_features.get('market_session', 'unknown')
                },
                'liquidity_analysis': {
                    'overall_score': liquidity_score,
                    'depth_analysis': self._analyze_market_depth(bids, asks),
                    'imbalance_metrics': {
                        'order_book_imbalance': enhanced_features.get('order_book_imbalance', 0),
                        'latency_adjusted_imbalance': enhanced_features.get('latency_adjusted_imbalance', 0),
                        'volume_weighted_pressure': enhanced_features.get('volume_weighted_price_pressure', 0)
                    }
                },
                'execution_timing': timing_signals,
                'impact_analysis': impact_analysis,
                'recommendations': execution_recommendations,
                'data_quality': {
                    'score': enhanced_features.get('data_quality_score', 0.5),
                    'age_seconds': (
                        (datetime.now().timestamp() * 1000 - self.cached_orderbook['timestamp']) / 1000
                        if self.cached_orderbook.get('timestamp') else None
                    ),
                    'sequence': self.cached_orderbook.get('sequence')
                }
            }
            
        except Exception as e:
            logger.error(f"Error generating execution signals: {e}")
            return {
                'status': 'error',
                'timestamp': datetime.now().isoformat(),
                'error': str(e)
            }
    
    def _calculate_real_time_liquidity_score(self, bids: List[tuple], asks: List[tuple]) -> float:
        """Calculate real-time liquidity score for execution optimization."""
        try:
            score_factors = []
            
            # Factor 1: Depth availability (40%)
            total_bid_size = sum(size for _, size in bids[:5])
            total_ask_size = sum(size for _, size in asks[:5])
            avg_depth = (total_bid_size + total_ask_size) / 2
            depth_score = min(1.0, avg_depth / 100)  # Normalize to reasonable depth
            score_factors.append(('depth', depth_score, 0.4))
            
            # Factor 2: Spread tightness (30%)
            if bids and asks:
                spread = asks[0][0] - bids[0][0]
                mid_price = (bids[0][0] + asks[0][0]) / 2
                spread_pct = spread / mid_price
                spread_score = max(0, 1 - spread_pct * 1000)  # Tighter spread = higher score
                score_factors.append(('spread', spread_score, 0.3))
            
            # Factor 3: Size distribution (20%)
            # Better liquidity if sizes are well distributed across levels
            bid_sizes = [size for _, size in bids[:5]]
            ask_sizes = [size for _, size in asks[:5]]
            
            if len(bid_sizes) > 1 and len(ask_sizes) > 1:
                bid_coeff_var = statistics.stdev(bid_sizes) / statistics.mean(bid_sizes)
                ask_coeff_var = statistics.stdev(ask_sizes) / statistics.mean(ask_sizes)
                avg_coeff_var = (bid_coeff_var + ask_coeff_var) / 2
                distribution_score = max(0, 1 - avg_coeff_var)  # Lower variance = better distribution
                score_factors.append(('distribution', distribution_score, 0.2))
            
            # Factor 4: Level count (10%)
            level_count = min(len(bids), len(asks))
            level_score = min(1.0, level_count / 10)
            score_factors.append(('levels', level_score, 0.1))
            
            # Calculate weighted score
            total_score = sum(score * weight for _, score, weight in score_factors)
            return max(0.0, min(1.0, total_score))
            
        except Exception:
            return 0.5
    
    def _generate_execution_timing_signals(self, features: Dict[str, float], liquidity_score: float) -> Dict[str, Any]:
        """Generate optimal execution timing signals."""
        try:
            signals = {}
            
            # Overall execution urgency
            urgency_score = features.get('execution_urgency_score', 0.5)
            signals['urgency_level'] = urgency_score
            
            # Timing recommendation
            if urgency_score > 0.8:
                signals['timing_recommendation'] = 'execute_immediately'
                signals['reasoning'] = 'High urgency - tight spread and good liquidity'
            elif urgency_score > 0.6:
                signals['timing_recommendation'] = 'execute_soon'
                signals['reasoning'] = 'Moderate urgency - decent execution conditions'
            elif liquidity_score > 0.7:
                signals['timing_recommendation'] = 'wait_for_better_timing'
                signals['reasoning'] = 'Good liquidity available - can wait for optimal timing'
            else:
                signals['timing_recommendation'] = 'execute_with_caution'
                signals['reasoning'] = 'Limited liquidity - execute carefully'
            
            # Market session analysis
            market_session = features.get('market_session', 'unknown')
            if market_session == 'active':
                signals['session_bias'] = 'favorable'
                signals['expected_execution_quality'] = 'high'
            elif market_session == 'volatile':
                signals['session_bias'] = 'caution'
                signals['expected_execution_quality'] = 'variable'
            else:
                signals['session_bias'] = 'neutral'
                signals['expected_execution_quality'] = 'standard'
            
            # Optimal execution window
            imbalance = abs(features.get('order_book_imbalance', 0))
            if imbalance < 0.1:
                signals['execution_window'] = 'excellent'
            elif imbalance < 0.3:
                signals['execution_window'] = 'good'
            else:
                signals['execution_window'] = 'challenging'
            
            return signals
            
        except Exception:
            return {'timing_recommendation': 'execute_with_caution', 'urgency_level': 0.5}
    
    def _calculate_execution_impact_analysis(self, bids: List[tuple], asks: List[tuple], mid_price: float) -> Dict[str, Any]:
        """Calculate execution impact analysis for different order sizes."""
        try:
            analysis = {}
            
            # Test different order sizes (in USD)
            test_sizes_usd = [1000, 5000, 10000, 25000, 50000]  # Different order sizes to test
            
            for size_usd in test_sizes_usd:
                size_btc = size_usd / mid_price
                
                # Calculate impact for market buy (walking asks)
                buy_impact = self._simulate_market_impact(asks, size_btc, 'buy', mid_price)
                
                # Calculate impact for market sell (walking bids)
                sell_impact = self._simulate_market_impact(bids, size_btc, 'sell', mid_price)
                
                analysis[f'size_{size_usd}_usd'] = {
                    'size_btc': size_btc,
                    'buy_impact_bps': buy_impact * 10000,
                    'sell_impact_bps': sell_impact * 10000,
                    'avg_impact_bps': (buy_impact + sell_impact) / 2 * 10000
                }
            
            return analysis
            
        except Exception:
            return {}
    
    def _simulate_market_impact(self, side_levels: List[tuple], target_size: float, direction: str, mid_price: float) -> float:
        """Simulate market impact for a given order size."""
        try:
            remaining_size = target_size
            total_cost = 0.0
            total_filled = 0.0
            
            for price, size in side_levels:
                if remaining_size <= 0:
                    break
                
                fill_size = min(remaining_size, size)
                total_cost += price * fill_size
                total_filled += fill_size
                remaining_size -= fill_size
            
            if total_filled == 0:
                return 0.5  # High impact if can't fill
            
            avg_fill_price = total_cost / total_filled
            impact = abs(avg_fill_price - mid_price) / mid_price
            
            return impact
            
        except Exception:
            return 0.1  # Default moderate impact
    
    def _analyze_market_depth(self, bids: List[tuple], asks: List[tuple]) -> Dict[str, Any]:
        """Analyze market depth characteristics."""
        try:
            # Calculate cumulative depth at different levels
            bid_cumulative = []
            ask_cumulative = []
            
            bid_sum = 0
            for price, size in bids[:10]:
                bid_sum += size
                bid_cumulative.append(bid_sum)
            
            ask_sum = 0
            for price, size in asks[:10]:
                ask_sum += size
                ask_cumulative.append(ask_sum)
            
            return {
                'total_bid_depth_10': bid_sum,
                'total_ask_depth_10': ask_sum,
                'depth_ratio': bid_sum / ask_sum if ask_sum > 0 else 1.0,
                'depth_at_level_5': {
                    'bids': bid_cumulative[4] if len(bid_cumulative) > 4 else 0,
                    'asks': ask_cumulative[4] if len(ask_cumulative) > 4 else 0
                },
                'average_order_size': {
                    'bids': statistics.mean([size for _, size in bids[:5]]) if bids else 0,
                    'asks': statistics.mean([size for _, size in asks[:5]]) if asks else 0
                }
            }
            
        except Exception:
            return {}
    
    def _generate_execution_recommendations(self, features: Dict[str, float], liquidity_score: float, 
                                          timing_signals: Dict[str, Any], impact_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate specific execution recommendations."""
        recommendations = []
        
        try:
            # Recommendation 1: Order type selection
            spread_bps = ((features.get('order_book_imbalance', 0) * 2) + 1) * 10  # Approximate spread
            if spread_bps < 2:
                recommendations.append({
                    'type': 'order_type',
                    'recommendation': 'use_limit_orders',
                    'reasoning': f'Estimated tight spread ({spread_bps:.1f} bps) favors limit orders',
                    'confidence': 0.9
                })
            else:
                recommendations.append({
                    'type': 'order_type',
                    'recommendation': 'consider_market_orders',
                    'reasoning': f'Estimated wide spread ({spread_bps:.1f} bps) may require market orders',
                    'confidence': 0.7
                })
            
            # Recommendation 2: Order sizing
            liquidity_imbalance = abs(features.get('liquidity_imbalance_ratio', 1.0) - 1.0)
            if liquidity_imbalance > 0.5:
                recommendations.append({
                    'type': 'sizing',
                    'recommendation': 'reduce_order_size',
                    'reasoning': 'High liquidity imbalance suggests reducing order size',
                    'confidence': 0.8
                })
            
            # Recommendation 3: Execution timing
            timing_rec = timing_signals.get('timing_recommendation', 'execute_with_caution')
            if timing_rec == 'execute_immediately':
                recommendations.append({
                    'type': 'timing',
                    'recommendation': 'execute_now',
                    'reasoning': 'Optimal execution conditions detected',
                    'confidence': 0.9
                })
            elif timing_rec == 'wait_for_better_timing':
                recommendations.append({
                    'type': 'timing',
                    'recommendation': 'wait_and_monitor',
                    'reasoning': 'Better execution conditions may emerge',
                    'confidence': 0.7
                })
            
            # Recommendation 4: Slippage management
            urgency = features.get('execution_urgency_score', 0.5)
            if urgency < 0.3:
                recommendations.append({
                    'type': 'slippage',
                    'recommendation': 'increase_slippage_tolerance',
                    'reasoning': 'Low urgency allows for higher slippage tolerance',
                    'confidence': 0.6
                })
            
            return recommendations
            
        except Exception:
            return [{'type': 'error', 'recommendation': 'use_conservative_approach', 'confidence': 0.5}]
    
    def _track_market_volatility(self, update: L2Update) -> None:
        """Track market volatility for adaptive buffer sizing."""
        try:
            if not update.bids or not update.asks:
                return
            
            # Calculate spread as volatility proxy
            best_bid = float(update.bids[0][0])
            best_ask = float(update.asks[0][0])
            spread_bps = ((best_ask - best_bid) / best_bid) * 10000
            
            self.volatility_buffer.append(spread_bps)
            
        except Exception as e:
            logger.debug(f"Error tracking volatility: {e}")
    
    def _check_adaptive_buffer_resize(self) -> None:
        """Check if buffer sizes should be adapted based on market conditions."""
        try:
            now = datetime.now()
            time_since_resize = (now - self.buffer_stats['last_resize_time']).total_seconds()
            
            if time_since_resize < self.resize_threshold_seconds:
                return
            
            # Calculate average volatility
            if len(self.volatility_buffer) < 10:
                return
            
            avg_volatility = statistics.mean(self.volatility_buffer)
            volatility_std = statistics.stdev(self.volatility_buffer) if len(self.volatility_buffer) > 1 else 0
            
            # Current buffer utilization
            high_utilization = len(self.high_priority_buffer) / max(self.high_priority_buffer.maxlen, 1)
            normal_utilization = len(self.normal_priority_buffer) / max(self.normal_priority_buffer.maxlen, 1)
            
            # Determine if resize is needed
            should_increase = (
                (high_utilization > 0.8 or normal_utilization > 0.8) or
                (avg_volatility > 2.0 and volatility_std > 1.0)  # High volatility market
            )
            
            should_decrease = (
                (high_utilization < 0.3 and normal_utilization < 0.3) and
                (avg_volatility < 1.0 and volatility_std < 0.5)  # Low volatility market
            )
            
            if should_increase and self.config.buffer_size < self.max_buffer_size:
                new_size = min(int(self.config.buffer_size * 1.2), self.max_buffer_size)
                self._resize_buffers(new_size)
                logger.info(f"Increased buffer size to {new_size} due to high utilization/volatility")
                
            elif should_decrease and self.config.buffer_size > self.min_buffer_size:
                new_size = max(int(self.config.buffer_size * 0.8), self.min_buffer_size)
                self._resize_buffers(new_size)
                logger.info(f"Decreased buffer size to {new_size} due to low utilization/volatility")
            
            self.buffer_stats['last_resize_time'] = now
            
        except Exception as e:
            logger.debug(f"Error in adaptive buffer resize: {e}")
    
    def _resize_buffers(self, new_size: int) -> None:
        """Resize buffers while preserving existing data."""
        try:
            # Update config
            self.config.buffer_size = new_size
            
            # Create new buffers with updated sizes
            high_priority_data = list(self.high_priority_buffer)
            normal_priority_data = list(self.normal_priority_buffer)
            
            self.high_priority_buffer = deque(high_priority_data, maxlen=new_size // 2)
            self.normal_priority_buffer = deque(normal_priority_data, maxlen=new_size)
            
            self.buffer_stats['adaptive_resizes'] += 1
            
        except Exception as e:
            logger.error(f"Error resizing buffers: {e}")


def create_data_ingestor(config_dict: Optional[Dict[str, Any]] = None) -> DataIngestor:
    """Factory function to create a DataIngestor instance."""
    config = DataIngestorConfig(config_dict)
    return DataIngestor(config)


# Standalone execution for live data ingestion
if __name__ == "__main__":
    # Re-import for standalone execution
    import signal
    
    print("LIVE WEBSOCKET DATA INGESTION")
    print("=" * 50)
    print("Bybit Demo Trading -> trading_bot_live.db")
    print("Complete 63-column L2 schema with microstructure features")
    print("=" * 50)
    
    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create configuration for Demo Trading and live database
    config_dict = {
        'exchange': 'bybit',
        'symbol': 'BTC/USDT:USDT',
        'sandbox': False,  # FIXED: Use mainnet for demo trading (not testnet)
        'db_path': './trading_bot_live.db',
        'table_name': 'l2_training_data_practical',
        'log_updates': True,
        'buffer_size': 100,
        'write_interval': 1.0,
        'orderbook_depth': 10,
        'data_retention_hours': 24,
        'max_reconnect_attempts': 10,
        'reconnect_delay': 5.0
    }
    
    # Create and start data ingestor
    ingestor = create_data_ingestor(config_dict)
    
    def on_update(update: L2Update):
        """Example update callback."""
        print(
            f"L2 Update: {update.symbol} - "
            f"Bid: {update.bids[0][0]:.2f}, "
            f"Ask: {update.asks[0][0]:.2f}"
        )
    
    def on_error(error: Exception):
        """Example error callback."""
        print(f"Error: {error}")
    
    ingestor.set_update_callback(on_update)
    ingestor.set_error_callback(on_error)
    
    # Handle shutdown gracefully
    def signal_handler(signum, frame):
        print(f"\\nShutdown signal received ({signum})")
        ingestor.stop()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Start ingestion
    if ingestor.start():
        print("Data ingestion started. Press Ctrl+C to stop.")
        
        try:
            # Keep main thread alive and print stats periodically
            while ingestor.is_healthy():
                time.sleep(30)
                stats = ingestor.get_stats()
                print(
                    f"Stats: {stats['total_updates']} updates, "
                    f"{stats['successful_writes']} writes, "
                    f"Buffer: {stats['buffer_size']}"
                )
        except KeyboardInterrupt:
            pass
        finally:
            ingestor.stop()
    else:
        print("Failed to start data ingestion")
        sys.exit(1)