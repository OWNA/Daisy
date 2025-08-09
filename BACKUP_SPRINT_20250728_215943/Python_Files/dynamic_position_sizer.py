"""
Dynamic Position Sizing Module
Adjusts position sizes based on market volatility, signal strength, and risk parameters
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class DynamicPositionSizer:
    """
    Advanced position sizing that considers:
    - Real-time volatility from L2 data
    - Signal strength and confidence
    - Market regime (trending/ranging)
    - Funding rates for perpetuals
    - Correlation with other positions
    """
    
    def __init__(self, config: dict):
        """
        Initialize the position sizer
        
        Args:
            config: Configuration dictionary
        """
        # Risk parameters
        self.base_risk_pct = config.get('base_risk_pct', 0.01)  # 1% base risk
        self.max_risk_pct = config.get('max_risk_pct', 0.03)   # 3% max risk
        self.volatility_lookback_minutes = config.get('volatility_lookback_minutes', 60)
        
        # Volatility scaling
        self.target_volatility = config.get('target_volatility', 0.02)  # 2% daily
        self.volatility_scalar = config.get('volatility_scalar', 1.0)
        
        # Signal strength scaling
        self.signal_scale_min = config.get('signal_scale_min', 0.5)
        self.signal_scale_max = config.get('signal_scale_max', 2.0)
        
        # Market regime adjustments
        self.trend_boost_factor = config.get('trend_boost_factor', 1.2)
        self.range_reduce_factor = config.get('range_reduce_factor', 0.8)
        
        # Funding rate considerations
        self.funding_threshold = config.get('funding_threshold', 0.0001)  # 0.01% per 8h
        self.funding_penalty_factor = config.get('funding_penalty_factor', 0.5)
        
        # Portfolio constraints
        self.max_position_value = config.get('max_position_value', 50000)  # $50k max
        self.min_position_value = config.get('min_position_value', 100)    # $100 min
        
        # State tracking
        self.volatility_cache = {}
        self.last_calculation_time = None
        
    def calculate_l2_volatility(self, l2_snapshots: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate volatility metrics from L2 order book data
        
        Args:
            l2_snapshots: DataFrame with L2 snapshots (must have mid_price, timestamp)
            
        Returns:
            Dictionary with volatility metrics
        """
        try:
            if l2_snapshots.empty or 'mid_price' not in l2_snapshots.columns:
                return {'realized_vol': 0.02, 'high_freq_vol': 0.02}  # Default 2%
            
            # Calculate returns at different frequencies
            l2_snapshots = l2_snapshots.sort_values('timestamp')
            
            # 1-second returns for high-frequency volatility
            returns_1s = l2_snapshots['mid_price'].pct_change()
            
            # 1-minute returns for realized volatility
            resampled = l2_snapshots.set_index('timestamp')['mid_price'].resample('1T').last()
            returns_1m = resampled.pct_change()
            
            # Calculate volatilities (annualized)
            # High-frequency vol (1-second returns annualized)
            hf_vol = returns_1s.std() * np.sqrt(86400)  # seconds per day
            
            # Realized vol (1-minute returns annualized)
            realized_vol = returns_1m.std() * np.sqrt(1440)  # minutes per day
            
            # Calculate other microstructure indicators
            # Bid-ask spread volatility
            if 'bid_ask_spread_pct' in l2_snapshots.columns:
                spread_vol = l2_snapshots['bid_ask_spread_pct'].std()
            else:
                spread_vol = 0.0001  # Default 1bp
            
            # Order book imbalance volatility (indicates directional pressure)
            if 'order_book_imbalance_5' in l2_snapshots.columns:
                imbalance_vol = l2_snapshots['order_book_imbalance_5'].std()
            else:
                imbalance_vol = 0.1
            
            return {
                'realized_vol': float(realized_vol) if not np.isnan(realized_vol) else 0.02,
                'high_freq_vol': float(hf_vol) if not np.isnan(hf_vol) else 0.02,
                'spread_vol': float(spread_vol),
                'imbalance_vol': float(imbalance_vol),
                'vol_of_vol': float(returns_1m.std()) if len(returns_1m) > 10 else 0.001
            }
            
        except Exception as e:
            logger.error(f"Error calculating L2 volatility: {e}")
            return {'realized_vol': 0.02, 'high_freq_vol': 0.02}
    
    def detect_market_regime(self, price_data: pd.DataFrame, 
                           l2_features: pd.DataFrame) -> str:
        """
        Detect current market regime using price and L2 data
        
        Args:
            price_data: DataFrame with OHLCV data
            l2_features: DataFrame with L2 features
            
        Returns:
            Market regime: 'trending_up', 'trending_down', 'ranging', 'volatile'
        """
        try:
            if price_data.empty or len(price_data) < 20:
                return 'ranging'
            
            # Calculate trend indicators
            prices = price_data['close'].values
            sma_short = pd.Series(prices).rolling(10).mean().iloc[-1]
            sma_long = pd.Series(prices).rolling(20).mean().iloc[-1]
            
            # Calculate momentum
            momentum = (prices[-1] - prices[-10]) / prices[-10]
            
            # Check L2 features for directional bias
            if not l2_features.empty and 'order_flow_imbalance' in l2_features.columns:
                flow_imbalance = l2_features['order_flow_imbalance'].mean()
            else:
                flow_imbalance = 0
            
            # Determine regime
            if abs(momentum) > 0.02:  # 2% move
                if momentum > 0 and sma_short > sma_long and flow_imbalance > 0.1:
                    return 'trending_up'
                elif momentum < 0 and sma_short < sma_long and flow_imbalance < -0.1:
                    return 'trending_down'
                else:
                    return 'volatile'
            else:
                return 'ranging'
                
        except Exception as e:
            logger.error(f"Error detecting market regime: {e}")
            return 'ranging'
    
    def adjust_for_funding_rate(self, base_size: float, funding_rate: float, 
                              side: str) -> float:
        """
        Adjust position size based on funding rate
        
        Args:
            base_size: Base position size
            funding_rate: Current funding rate (8-hourly)
            side: 'long' or 'short'
            
        Returns:
            Adjusted position size
        """
        # Convert to daily rate (3 funding periods per day)
        daily_funding = funding_rate * 3
        
        # Penalize positions that pay funding
        if (side == 'long' and funding_rate > self.funding_threshold) or \
           (side == 'short' and funding_rate < -self.funding_threshold):
            # Reduce position size based on funding cost
            penalty = min(abs(daily_funding) / 0.001, 1.0)  # Max 100% penalty at 0.1% daily
            adjustment = 1 - (penalty * self.funding_penalty_factor)
            return base_size * adjustment
        
        # Slightly boost positions that receive funding
        elif (side == 'long' and funding_rate < -self.funding_threshold) or \
             (side == 'short' and funding_rate > self.funding_threshold):
            bonus = min(abs(daily_funding) / 0.001, 0.2)  # Max 20% bonus
            return base_size * (1 + bonus)
        
        return base_size
    
    def calculate_signal_strength_multiplier(self, signal_value: float, 
                                           signal_confidence: Optional[float] = None) -> float:
        """
        Convert signal strength to position size multiplier
        
        Args:
            signal_value: Raw signal value (e.g., -1 to 1)
            signal_confidence: Optional confidence metric (0 to 1)
            
        Returns:
            Position size multiplier
        """
        # Normalize signal to 0-1 range
        normalized_signal = min(max(abs(signal_value), 0), 1)
        
        # Apply confidence if provided
        if signal_confidence is not None:
            normalized_signal *= signal_confidence
        
        # Non-linear scaling (more conservative for weak signals)
        scaled_signal = normalized_signal ** 1.5
        
        # Map to multiplier range
        multiplier = self.signal_scale_min + (self.signal_scale_max - self.signal_scale_min) * scaled_signal
        
        return multiplier
    
    def calculate_dynamic_position_size(self, 
                                      account_balance: float,
                                      current_price: float,
                                      side: str,
                                      signal_strength: float,
                                      l2_snapshots: Optional[pd.DataFrame] = None,
                                      price_history: Optional[pd.DataFrame] = None,
                                      funding_rate: Optional[float] = None,
                                      existing_positions: Optional[Dict] = None) -> Dict:
        """
        Calculate optimal position size considering all factors
        
        Args:
            account_balance: Account balance in USD
            current_price: Current asset price
            side: 'buy' or 'sell'
            signal_strength: Signal strength/confidence
            l2_snapshots: Recent L2 order book snapshots
            price_history: Recent price history
            funding_rate: Current funding rate
            existing_positions: Dictionary of existing positions
            
        Returns:
            Dictionary with position sizing details
        """
        try:
            # Start with base risk amount
            base_risk_amount = account_balance * self.base_risk_pct
            
            # 1. Adjust for volatility
            if l2_snapshots is not None and not l2_snapshots.empty:
                vol_metrics = self.calculate_l2_volatility(l2_snapshots)
                current_vol = vol_metrics['realized_vol']
                
                # Inverse volatility scaling
                vol_adjustment = min(self.target_volatility / current_vol, 2.0) if current_vol > 0 else 1.0
                vol_adjustment = max(vol_adjustment, 0.5)  # Don't reduce by more than 50%
            else:
                vol_adjustment = 1.0
                current_vol = self.target_volatility
            
            # 2. Adjust for signal strength
            signal_multiplier = self.calculate_signal_strength_multiplier(signal_strength)
            
            # 3. Adjust for market regime
            if price_history is not None and l2_snapshots is not None:
                regime = self.detect_market_regime(price_history, l2_snapshots)
                
                if regime in ['trending_up', 'trending_down']:
                    regime_adjustment = self.trend_boost_factor
                elif regime == 'volatile':
                    regime_adjustment = 0.9  # Slightly reduce in volatile markets
                else:  # ranging
                    regime_adjustment = self.range_reduce_factor
            else:
                regime_adjustment = 1.0
                regime = 'unknown'
            
            # 4. Adjust for funding rate
            if funding_rate is not None:
                funding_adjustment = self.adjust_for_funding_rate(1.0, funding_rate, side)
            else:
                funding_adjustment = 1.0
            
            # 5. Consider existing positions (correlation/concentration)
            if existing_positions:
                total_exposure = sum(abs(pos.get('value', 0)) for pos in existing_positions.values())
                concentration_limit = account_balance * 0.5  # Max 50% total exposure
                
                if total_exposure > concentration_limit * 0.8:  # Getting close to limit
                    concentration_adjustment = 0.5
                else:
                    concentration_adjustment = 1.0
            else:
                concentration_adjustment = 1.0
            
            # Combine all adjustments
            total_adjustment = (vol_adjustment * signal_multiplier * regime_adjustment * 
                              funding_adjustment * concentration_adjustment)
            
            # Calculate final position size
            risk_amount = base_risk_amount * total_adjustment
            risk_amount = min(risk_amount, account_balance * self.max_risk_pct)
            
            # Convert risk amount to position size
            # Using 2x volatility for stop loss distance
            stop_distance = current_vol * 2 * self.volatility_scalar
            position_value = risk_amount / stop_distance
            
            # Apply min/max constraints
            position_value = max(min(position_value, self.max_position_value), self.min_position_value)
            
            # Calculate position size in base currency
            position_size = position_value / current_price
            
            # Prepare detailed result
            result = {
                'position_size': position_size,
                'position_value': position_value,
                'risk_amount': risk_amount,
                'risk_pct': risk_amount / account_balance,
                'adjustments': {
                    'volatility': vol_adjustment,
                    'signal': signal_multiplier,
                    'regime': regime_adjustment,
                    'funding': funding_adjustment,
                    'concentration': concentration_adjustment,
                    'total': total_adjustment
                },
                'metrics': {
                    'current_volatility': current_vol,
                    'market_regime': regime,
                    'stop_distance': stop_distance,
                    'position_pct_of_balance': position_value / account_balance
                },
                'constraints_applied': {
                    'hit_max_position': position_value == self.max_position_value,
                    'hit_min_position': position_value == self.min_position_value,
                    'hit_max_risk': risk_amount == account_balance * self.max_risk_pct
                }
            }
            
            # Log the calculation
            logger.info(f"Dynamic position size calculated: ${position_value:.2f} "
                       f"({position_size:.6f} BTC), risk: {result['risk_pct']*100:.2f}%, "
                       f"adjustments: vol={vol_adjustment:.2f}, signal={signal_multiplier:.2f}, "
                       f"regime={regime_adjustment:.2f}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error calculating dynamic position size: {e}")
            # Fallback to safe default
            safe_position_value = min(account_balance * 0.01, self.max_position_value)
            return {
                'position_size': safe_position_value / current_price,
                'position_value': safe_position_value,
                'risk_amount': safe_position_value * 0.02,
                'risk_pct': 0.01,
                'error': str(e)
            }