# advanced_risk_manager.py
# Reformatted from notebook export to standard Python file

import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple


class AdvancedRiskManager:
    """
    Manages risk through dynamic position sizing, volatility-adjusted stops,
    take profit levels, and commission-aware trading decisions.
    
    Key Features:
    - Commission-aware position sizing
    - Minimum profit threshold enforcement
    - Trade frequency limiting
    - Dynamic risk adjustment based on market conditions
    """

    def __init__(self, config):
        """
        Initializes the AdvancedRiskManager.

        Args:
            config (dict): Configuration dictionary with risk parameters.
        """
        self.config = config
        self.max_drawdown = config.get('max_drawdown', 0.2)
        self.position_sizing_mode = config.get(
            'position_sizing_mode', 'fixed_fraction'
        )
        self.fixed_position_pct = config.get('fixed_position_pct', 0.02)
        self.volatility_target_pct = config.get('volatility_target_pct', 0.02)
        self.sl_atr_multiplier = config.get('sl_atr_multiplier', 2.0)
        self.tp_atr_multiplier = config.get('tp_atr_multiplier', 3.0)
        self.max_positions = config.get('max_positions', 3)
        self.max_position_size = config.get('max_position_size', 0.1)
        
        # Additional attributes for compatibility
        self.max_equity_risk_pct = config.get('max_equity_risk_pct', 0.10)
        self.fixed_fraction_pct = config.get('fixed_fraction_pct', 0.05)
        self.volatility_lookback = config.get('volatility_lookback', 14)
        
        # Commission-aware parameters
        self.commission_pct = config.get('commission_pct', 0.0006)  # 0.06%
        self.min_expected_profit_pct = config.get(
            'min_expected_profit_pct', 0.003  # 0.3% minimum
        )
        self.max_trades_per_hour = config.get('max_trades_per_hour', 5)
        self.trade_cooldown_minutes = config.get('trade_cooldown_minutes', 5)
        
        # Trade frequency tracking
        self.recent_trades: List[datetime] = []
        self.last_trade_time: Optional[datetime] = None
        
        # Track open positions
        self.open_positions = {}
        
        # Performance tracking
        self.total_commission_paid = 0.0
        self.trades_rejected_min_profit = 0
        self.trades_rejected_frequency = 0
        
        print("AdvancedRiskManager initialized with commission awareness.")

    def calculate_commission_impact(
        self, position_size: float, entry_price: float, 
        exit_price: float = None
    ) -> Dict[str, float]:
        """
        Calculate commission impact for a trade.
        
        Args:
            position_size: Size of position in USD
            entry_price: Entry price
            exit_price: Exit price (optional, for round-trip calculation)
            
        Returns:
            Dictionary with commission details
        """
        # Entry commission
        entry_commission = position_size * self.commission_pct
        
        # Exit commission (if exit price provided)
        if exit_price:
            # Adjust position value for price change
            exit_value = position_size * (exit_price / entry_price)
            exit_commission = exit_value * self.commission_pct
            total_commission = entry_commission + exit_commission
        else:
            exit_commission = entry_commission  # Estimate
            total_commission = entry_commission * 2
        
        # Calculate break-even price movement
        break_even_pct = (total_commission / position_size) * 100
        
        return {
            'entry_commission': entry_commission,
            'exit_commission': exit_commission,
            'total_commission': total_commission,
            'commission_pct_of_position': (total_commission / position_size) * 100,
            'break_even_price_move_pct': break_even_pct
        }

    def check_minimum_profit_threshold(
        self, entry_price: float, target_price: float, 
        position_size: float, side: str = 'long'
    ) -> Tuple[bool, Dict[str, float]]:
        """
        Check if expected profit exceeds minimum threshold after commissions.
        
        Args:
            entry_price: Entry price
            target_price: Target/expected exit price
            position_size: Position size in USD
            side: 'long' or 'short'
            
        Returns:
            Tuple of (is_profitable, profit_details)
        """
        # Calculate expected gross profit
        if side == 'long':
            price_change_pct = (target_price - entry_price) / entry_price
        else:
            price_change_pct = (entry_price - target_price) / entry_price
        
        gross_profit = position_size * price_change_pct
        
        # Calculate commission impact
        commission_impact = self.calculate_commission_impact(
            position_size, entry_price, target_price
        )
        
        # Net profit after commissions
        net_profit = gross_profit - commission_impact['total_commission']
        net_profit_pct = (net_profit / position_size) * 100
        
        # Check against minimum threshold
        meets_threshold = net_profit_pct >= (self.min_expected_profit_pct * 100)
        
        profit_details = {
            'gross_profit': gross_profit,
            'gross_profit_pct': price_change_pct * 100,
            'commission': commission_impact['total_commission'],
            'net_profit': net_profit,
            'net_profit_pct': net_profit_pct,
            'meets_threshold': meets_threshold,
            'threshold_pct': self.min_expected_profit_pct * 100
        }
        
        if not meets_threshold:
            self.trades_rejected_min_profit += 1
        
        return meets_threshold, profit_details

    def check_trade_frequency_limit(self) -> Tuple[bool, Dict[str, any]]:
        """
        Check if trade frequency limits are respected.
        
        Returns:
            Tuple of (can_trade, frequency_details)
        """
        current_time = datetime.now()
        
        # Clean up old trades (older than 1 hour)
        cutoff_time = current_time - timedelta(hours=1)
        self.recent_trades = [
            t for t in self.recent_trades if t > cutoff_time
        ]
        
        # Check cooldown period
        if self.last_trade_time:
            time_since_last = (current_time - self.last_trade_time).seconds / 60
            if time_since_last < self.trade_cooldown_minutes:
                self.trades_rejected_frequency += 1
                return False, {
                    'reason': 'cooldown',
                    'minutes_remaining': self.trade_cooldown_minutes - time_since_last,
                    'trades_in_last_hour': len(self.recent_trades)
                }
        
        # Check hourly limit
        if len(self.recent_trades) >= self.max_trades_per_hour:
            self.trades_rejected_frequency += 1
            return False, {
                'reason': 'hourly_limit',
                'trades_in_last_hour': len(self.recent_trades),
                'max_allowed': self.max_trades_per_hour
            }
        
        return True, {
            'trades_in_last_hour': len(self.recent_trades),
            'can_trade': True
        }

    def calculate_position_size(
        self, account_equity: float, current_volatility_pct: float,
        entry_price: float = None, target_price: float = None
    ) -> float:
        """
        Calculates position size based on the configured mode and commission impact.

        Args:
            account_equity (float): Current account equity.
            current_volatility_pct (float): Current price volatility percentage
                (e.g., ATR / price).
            entry_price (float): Entry price for commission calculation
            target_price (float): Target price for profit calculation

        Returns:
            float: Calculated position size in USD.
        """
        # Base position size calculation
        if self.position_sizing_mode == 'volatility_target':
            if current_volatility_pct <= 1e-5:
                # Fallback to a small fraction of max risk if volatility is negligible
                base_size = account_equity * self.max_equity_risk_pct * 0.1
            else:
                size_usd = (
                    account_equity * self.volatility_target_pct
                ) / current_volatility_pct
                max_size_usd = account_equity * self.max_equity_risk_pct
                base_size = min(size_usd, max_size_usd)

        elif self.position_sizing_mode == 'fixed_fraction':
            base_size = account_equity * self.fixed_fraction_pct
        else:
            # Default to fixed_fraction if mode is unknown
            print(
                f"Warning (RiskManager): Unknown position_sizing_mode "
                f"'{self.position_sizing_mode}'. Defaulting to 'fixed_fraction'."
            )
            base_size = account_equity * self.fixed_fraction_pct
        
        # Adjust for commission impact if prices provided
        if entry_price and target_price:
            commission_impact = self.calculate_commission_impact(
                base_size, entry_price, target_price
            )
            
            # Reduce position size if commission impact is high
            commission_pct = commission_impact['commission_pct_of_position']
            if commission_pct > 1.0:  # If commission > 1% of position
                # Scale down position size
                adjustment_factor = 1.0 - (commission_pct / 100)
                base_size *= adjustment_factor
        
        # Apply maximum position size limit
        return min(base_size, account_equity * self.max_position_size)

    def calculate_stop_loss(
        self, entry_price: float, atr_value: float, side: str = 'long',
        account_for_commission: bool = True
    ) -> float:
        """
        Calculates stop-loss level, optionally accounting for commission.

        Args:
            entry_price (float): The entry price of the position.
            atr_value (float): The current Average True Range (ATR) value.
            side (str, optional): 'long' or 'short'. Default 'long'.
            account_for_commission (bool): Adjust stop for commission impact

        Returns:
            float or None: Stop-loss price, or None if atr_value is invalid.
        """
        if not pd.notna(atr_value) or atr_value <= 0:
            return None

        base_stop_distance = atr_value * self.sl_atr_multiplier
        
        # Adjust for commission if requested
        if account_for_commission:
            # Add commission buffer (approximate)
            commission_buffer = entry_price * self.commission_pct * 2
            base_stop_distance += commission_buffer

        if side == 'long':
            return entry_price - base_stop_distance
        elif side == 'short':
            return entry_price + base_stop_distance
        return None

    def calculate_take_profit(
        self, entry_price: float, atr_value: float, side: str = 'long',
        account_for_commission: bool = True
    ) -> float:
        """
        Calculates take-profit level, optionally accounting for commission.

        Args:
            entry_price (float): The entry price of the position.
            atr_value (float): The current Average True Range (ATR) value.
            side (str, optional): 'long' or 'short'. Default 'long'.
            account_for_commission (bool): Adjust TP for commission impact

        Returns:
            float or None: Take-profit price, or None if atr_value is invalid.
        """
        if not pd.notna(atr_value) or atr_value <= 0:
            return None

        base_tp_distance = atr_value * self.tp_atr_multiplier
        
        # Adjust for commission if requested
        if account_for_commission:
            # Add commission buffer to ensure net profit
            commission_buffer = entry_price * self.commission_pct * 2
            base_tp_distance += commission_buffer

        if side == 'long':
            return entry_price + base_tp_distance
        elif side == 'short':
            return entry_price - base_tp_distance
        return None

    def check_drawdown_limit(self, initial_capital: float, current_capital: float) -> bool:
        """
        Check if drawdown limit is exceeded.

        Args:
            initial_capital (float): The initial capital amount.
            current_capital (float): The current capital amount.

        Returns:
            bool: True if trading is allowed (drawdown within limits), False otherwise.
        """
        if initial_capital <= 0:
            return False
        
        drawdown = (initial_capital - current_capital) / initial_capital
        return drawdown <= self.max_drawdown

    def can_open_position(
        self, position_id: str, size: float, direction: str,
        entry_price: float, target_price: float
    ) -> Tuple[bool, List[str]]:
        """
        Comprehensive check if a position can be opened.
        
        Args:
            position_id: Unique position identifier
            size: Position size in USD
            direction: 'long' or 'short'
            entry_price: Entry price
            target_price: Target price
            
        Returns:
            Tuple of (can_open, list_of_reasons_if_not)
        """
        reasons = []
        
        # Check max positions
        if len(self.open_positions) >= self.max_positions:
            reasons.append(f"Max positions ({self.max_positions}) reached")
        
        # Check trade frequency
        can_trade_freq, freq_details = self.check_trade_frequency_limit()
        if not can_trade_freq:
            reasons.append(f"Trade frequency limit: {freq_details['reason']}")
        
        # Check minimum profit threshold
        meets_profit, profit_details = self.check_minimum_profit_threshold(
            entry_price, target_price, size, direction
        )
        if not meets_profit:
            reasons.append(
                f"Below min profit threshold: {profit_details['net_profit_pct']:.2f}% "
                f"< {profit_details['threshold_pct']:.2f}%"
            )
        
        return len(reasons) == 0, reasons

    def open_position(
        self, position_id: str, size: float, direction: str,
        entry_price: float = None
    ) -> dict:
        """
        Track open positions with commission tracking.

        Args:
            position_id (str): Unique identifier for the position.
            size (float): Position size in base currency.
            direction (str): 'long' or 'short'.
            entry_price (float): Entry price for commission tracking

        Returns:
            dict: Position details or None if position cannot be opened.
        """
        # Create position
        position = {
            'id': position_id,
            'size': size,
            'direction': direction,
            'status': 'open',
            'entry_price': entry_price,
            'entry_time': datetime.now()
        }
        
        # Calculate and track entry commission
        if entry_price:
            commission_impact = self.calculate_commission_impact(
                size, entry_price
            )
            position['entry_commission'] = commission_impact['entry_commission']
            self.total_commission_paid += commission_impact['entry_commission']
        
        # Track the position
        self.open_positions[position_id] = position
        
        # Update trade tracking
        self.recent_trades.append(datetime.now())
        self.last_trade_time = datetime.now()
        
        print(f"Position opened: {position}")
        return position

    def get_risk_metrics(self) -> Dict[str, any]:
        """
        Get current risk management metrics.
        
        Returns:
            Dictionary of risk metrics
        """
        return {
            'open_positions': len(self.open_positions),
            'max_positions': self.max_positions,
            'total_commission_paid': self.total_commission_paid,
            'trades_rejected_min_profit': self.trades_rejected_min_profit,
            'trades_rejected_frequency': self.trades_rejected_frequency,
            'recent_trades_count': len(self.recent_trades),
            'commission_pct': self.commission_pct * 100,
            'min_profit_threshold_pct': self.min_expected_profit_pct * 100
        }