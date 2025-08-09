# backtest_enhanced.py
# Realistic Backtesting Framework with Transaction Costs and Market Impact

import pandas as pd
import numpy as np
import logging
from typing import Dict, Tuple, Optional
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)


class EnhancedBacktester:
    """
    Realistic backtesting framework that includes:
    - Transaction costs (maker/taker fees)
    - Market impact modeling
    - Slippage estimation
    - Position sizing constraints
    - Risk management rules
    - Performance analytics
    """
    
    def __init__(self, config: dict):
        """Initialize enhanced backtester."""
        self.config = config
        
        # Trading costs (Bybit spot fees as example)
        self.maker_fee = config.get('maker_fee', 0.001)  # 0.1%
        self.taker_fee = config.get('taker_fee', 0.001)  # 0.1%
        
        # Market impact parameters
        self.impact_coefficient = config.get('impact_coefficient', 0.0001)
        self.temporary_impact_decay = config.get('temp_impact_decay', 0.5)
        
        # Risk parameters
        self.max_position_size = config.get('max_position_size_btc', 0.1)
        self.max_drawdown_limit = config.get('max_drawdown', 0.1)
        self.position_size_pct = config.get('position_size_pct', 0.01)
        
        # Execution assumptions
        self.min_time_between_trades = config.get('min_trade_interval_ms', 1000)
        self.use_limit_orders = config.get('use_limit_orders', True)
        
        # Performance tracking
        self.trades = []
        self.equity_curve = []
        self.metrics = {}
        
        logger.info("Enhanced Backtester initialized with realistic assumptions")

    def run_backtest(
        self,
        df_signals: pd.DataFrame,
        initial_capital: float = 10000,
        leverage: float = 1.0
    ) -> Dict:
        """
        Run backtest with realistic execution assumptions.
        
        Args:
            df_signals: DataFrame with signals and predictions
            initial_capital: Starting capital in USD
            leverage: Maximum leverage to use
            
        Returns:
            Dict with performance metrics and trade log
        """
        logger.info(f"Starting backtest with ${initial_capital:,.2f} capital")
        
        # Initialize state
        capital = initial_capital
        position = 0  # BTC position
        entry_price = 0
        last_trade_time = pd.Timestamp('1900-01-01')
        
        trades = []
        equity_curve = []
        
        # Add required columns
        df = df_signals.copy()
        if 'mid_price' not in df.columns:
            logger.error("mid_price column required for backtesting")
            return {}
        
        # Iterate through signals
        for idx, row in df.iterrows():
            current_price = row['mid_price']
            signal = row.get('signal', 0)
            confidence = row.get('confidence', 0.5)
            
            # Record equity
            equity = capital + position * current_price
            equity_curve.append({
                'timestamp': idx,
                'equity': equity,
                'capital': capital,
                'position': position,
                'price': current_price
            })
            
            # Check drawdown
            if len(equity_curve) > 1:
                peak_equity = max(e['equity'] for e in equity_curve)
                current_drawdown = (peak_equity - equity) / peak_equity
                if current_drawdown > self.max_drawdown_limit:
                    logger.warning(f"Max drawdown exceeded: {current_drawdown:.2%}")
                    # Close position
                    if position != 0:
                        signal = 0  # Force close
            
            # Check time constraint
            time_since_last = (idx - last_trade_time).total_seconds() * 1000
            if time_since_last < self.min_time_between_trades:
                continue
            
            # Calculate execution price with slippage and impact
            exec_price = self._calculate_execution_price(
                current_price,
                signal,
                position,
                row
            )
            
            # Execute trades
            if signal != 0 and position == 0:
                # Open position
                position_size = self._calculate_position_size(
                    capital,
                    exec_price,
                    confidence,
                    leverage
                )
                
                if signal > 0:  # Long
                    position = position_size
                    cost = position_size * exec_price * (1 + self.taker_fee)
                else:  # Short
                    position = -position_size
                    cost = abs(position_size) * exec_price * self.taker_fee
                
                capital -= cost
                entry_price = exec_price
                last_trade_time = idx
                
                trades.append({
                    'timestamp': idx,
                    'type': 'open_long' if signal > 0 else 'open_short',
                    'price': exec_price,
                    'size': abs(position),
                    'cost': cost,
                    'confidence': confidence
                })
                
            elif signal == 0 and position != 0:
                # Close position
                if position > 0:  # Close long
                    revenue = position * exec_price * (1 - self.taker_fee)
                    pnl = revenue - (position * entry_price * (1 + self.taker_fee))
                else:  # Close short
                    cost = abs(position) * exec_price * (1 + self.taker_fee)
                    revenue = abs(position) * entry_price * (1 - self.taker_fee)
                    pnl = revenue - cost
                
                capital += revenue if position > 0 else (
                    abs(position) * entry_price * (1 - self.taker_fee) - 
                    abs(position) * exec_price * (1 + self.taker_fee)
                )
                
                trades.append({
                    'timestamp': idx,
                    'type': 'close_long' if position > 0 else 'close_short',
                    'price': exec_price,
                    'size': abs(position),
                    'pnl': pnl,
                    'return': pnl / (abs(position) * entry_price)
                })
                
                position = 0
                entry_price = 0
                last_trade_time = idx
            
            elif signal != 0 and position != 0 and np.sign(signal) != np.sign(position):
                # Reverse position (close then open)
                # First close
                if position > 0:
                    revenue = position * exec_price * (1 - self.taker_fee)
                    pnl = revenue - (position * entry_price * (1 + self.taker_fee))
                else:
                    cost = abs(position) * exec_price * (1 + self.taker_fee)
                    revenue = abs(position) * entry_price * (1 - self.taker_fee)
                    pnl = revenue - cost
                
                capital += revenue if position > 0 else (
                    abs(position) * entry_price * (1 - self.taker_fee) - 
                    abs(position) * exec_price * (1 + self.taker_fee)
                )
                
                trades.append({
                    'timestamp': idx,
                    'type': 'close_long' if position > 0 else 'close_short',
                    'price': exec_price,
                    'size': abs(position),
                    'pnl': pnl,
                    'return': pnl / (abs(position) * entry_price)
                })
                
                # Then open new position
                position_size = self._calculate_position_size(
                    capital,
                    exec_price,
                    confidence,
                    leverage
                )
                
                if signal > 0:
                    position = position_size
                    cost = position_size * exec_price * (1 + self.taker_fee)
                else:
                    position = -position_size
                    cost = abs(position_size) * exec_price * self.taker_fee
                
                capital -= cost
                entry_price = exec_price
                last_trade_time = idx
                
                trades.append({
                    'timestamp': idx,
                    'type': 'open_long' if signal > 0 else 'open_short',
                    'price': exec_price,
                    'size': abs(position),
                    'cost': cost,
                    'confidence': confidence
                })
        
        # Close final position if any
        if position != 0:
            final_price = df.iloc[-1]['mid_price']
            exec_price = self._calculate_execution_price(
                final_price, 0, position, df.iloc[-1]
            )
            
            if position > 0:
                revenue = position * exec_price * (1 - self.taker_fee)
                pnl = revenue - (position * entry_price * (1 + self.taker_fee))
            else:
                cost = abs(position) * exec_price * (1 + self.taker_fee)
                revenue = abs(position) * entry_price * (1 - self.taker_fee)
                pnl = revenue - cost
            
            capital += revenue if position > 0 else (
                abs(position) * entry_price * (1 - self.taker_fee) - 
                abs(position) * exec_price * (1 + self.taker_fee)
            )
            
            trades.append({
                'timestamp': df.index[-1],
                'type': 'close_long' if position > 0 else 'close_short',
                'price': exec_price,
                'size': abs(position),
                'pnl': pnl,
                'return': pnl / (abs(position) * entry_price)
            })
        
        # Calculate metrics
        self.trades = trades
        self.equity_curve = pd.DataFrame(equity_curve)
        self.metrics = self._calculate_metrics(initial_capital)
        
        return {
            'metrics': self.metrics,
            'trades': pd.DataFrame(trades),
            'equity_curve': self.equity_curve
        }

    def _calculate_execution_price(
        self,
        mid_price: float,
        signal: int,
        current_position: float,
        row: pd.Series
    ) -> float:
        """Calculate execution price including slippage and market impact."""
        
        # Base slippage from spread
        spread = row.get('spread', mid_price * 0.0001)  # 1 bp default
        
        if self.use_limit_orders:
            # Limit orders: execute at favorable side but may not fill
            if signal > 0 or (signal == 0 and current_position < 0):  # Buying
                base_price = mid_price + spread * 0.1  # 10% into spread
            else:  # Selling
                base_price = mid_price - spread * 0.1
        else:
            # Market orders: immediate execution at unfavorable price
            if signal > 0 or (signal == 0 and current_position < 0):  # Buying
                base_price = mid_price + spread / 2
            else:  # Selling
                base_price = mid_price - spread / 2
        
        # Add market impact
        if 'total_ask_volume_5' in row and 'total_bid_volume_5' in row:
            if signal > 0 or (signal == 0 and current_position < 0):  # Buying
                book_depth = row['total_ask_volume_5']
                impact = self.impact_coefficient * (1 / (book_depth + 1))
            else:  # Selling
                book_depth = row['total_bid_volume_5']
                impact = self.impact_coefficient * (1 / (book_depth + 1))
            
            impact_price = base_price * (1 + impact * np.sign(signal if signal != 0 else -current_position))
        else:
            impact_price = base_price
        
        return impact_price

    def _calculate_position_size(
        self,
        capital: float,
        price: float,
        confidence: float,
        leverage: float
    ) -> float:
        """Calculate position size based on Kelly criterion and constraints."""
        
        # Base size from config
        base_size = capital * self.position_size_pct / price
        
        # Adjust by confidence (simplified Kelly)
        kelly_factor = min(confidence, 0.25)  # Cap at 25% for safety
        size = base_size * kelly_factor * 4  # Scale to reasonable range
        
        # Apply leverage
        size = size * min(leverage, 3.0)  # Cap leverage for safety
        
        # Apply maximum position constraint
        size = min(size, self.max_position_size)
        
        return size

    def _calculate_metrics(self, initial_capital: float) -> Dict:
        """Calculate comprehensive performance metrics."""
        
        if not self.trades:
            return {
                'total_return': 0,
                'num_trades': 0,
                'win_rate': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0
            }
        
        trades_df = pd.DataFrame(self.trades)
        equity = self.equity_curve['equity'].values
        
        # Basic metrics
        final_equity = equity[-1]
        total_return = (final_equity - initial_capital) / initial_capital
        
        # Trade statistics
        completed_trades = trades_df[trades_df['type'].str.contains('close')]
        num_trades = len(completed_trades)
        
        if num_trades > 0:
            win_rate = (completed_trades['pnl'] > 0).mean()
            avg_win = completed_trades[completed_trades['pnl'] > 0]['return'].mean()
            avg_loss = completed_trades[completed_trades['pnl'] <= 0]['return'].mean()
            profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else 0
        else:
            win_rate = 0
            avg_win = 0
            avg_loss = 0
            profit_factor = 0
        
        # Risk metrics
        returns = pd.Series(equity).pct_change().dropna()
        if len(returns) > 1:
            sharpe_ratio = np.sqrt(252) * returns.mean() / returns.std()
            sortino_ratio = np.sqrt(252) * returns.mean() / returns[returns < 0].std()
        else:
            sharpe_ratio = 0
            sortino_ratio = 0
        
        # Drawdown
        running_max = pd.Series(equity).expanding().max()
        drawdown = (pd.Series(equity) - running_max) / running_max
        max_drawdown = abs(drawdown.min())
        
        # Compile metrics
        metrics = {
            'total_return': total_return,
            'final_equity': final_equity,
            'num_trades': num_trades,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown': max_drawdown,
            'total_fees': trades_df['cost'].sum() if 'cost' in trades_df else 0,
            'avg_confidence': trades_df['confidence'].mean() if 'confidence' in trades_df else 0
        }
        
        return metrics

    def plot_results(self, save_path: Optional[str] = None):
        """Plot backtest results."""
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        
        # Equity curve
        axes[0].plot(self.equity_curve['equity'], label='Equity')
        axes[0].set_title('Equity Curve')
        axes[0].set_ylabel('Equity ($)')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Drawdown
        equity = self.equity_curve['equity'].values
        running_max = pd.Series(equity).expanding().max()
        drawdown = (pd.Series(equity) - running_max) / running_max * 100
        axes[1].fill_between(range(len(drawdown)), drawdown, alpha=0.3, color='red')
        axes[1].set_title('Drawdown')
        axes[1].set_ylabel('Drawdown (%)')
        axes[1].grid(True, alpha=0.3)
        
        # Trade distribution
        if self.trades:
            trades_df = pd.DataFrame(self.trades)
            completed = trades_df[trades_df['type'].str.contains('close')]
            if not completed.empty:
                returns = completed['return'] * 100
                axes[2].hist(returns, bins=50, alpha=0.7, edgecolor='black')
                axes[2].axvline(returns.mean(), color='red', linestyle='--', 
                              label=f'Mean: {returns.mean():.2f}%')
                axes[2].set_title('Trade Return Distribution')
                axes[2].set_xlabel('Return (%)')
                axes[2].set_ylabel('Frequency')
                axes[2].legend()
                axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Results plot saved to {save_path}")
        else:
            plt.show()

    def generate_report(self) -> str:
        """Generate text report of backtest results."""
        report = []
        report.append("=" * 60)
        report.append("ENHANCED BACKTEST REPORT")
        report.append("=" * 60)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Performance metrics
        report.append("PERFORMANCE METRICS:")
        report.append("-" * 30)
        for key, value in self.metrics.items():
            if isinstance(value, float):
                if 'return' in key or 'rate' in key or 'ratio' in key:
                    report.append(f"{key:<20}: {value:>10.2%}")
                elif key == 'final_equity' or key == 'total_fees':
                    report.append(f"{key:<20}: ${value:>10,.2f}")
                else:
                    report.append(f"{key:<20}: {value:>10.4f}")
            else:
                report.append(f"{key:<20}: {value:>10}")
        
        # Trade summary
        if self.trades:
            report.append("")
            report.append("TRADE SUMMARY:")
            report.append("-" * 30)
            trades_df = pd.DataFrame(self.trades)
            
            for trade_type in trades_df['type'].unique():
                count = (trades_df['type'] == trade_type).sum()
                report.append(f"{trade_type:<20}: {count:>10}")
        
        # Risk analysis
        report.append("")
        report.append("RISK ANALYSIS:")
        report.append("-" * 30)
        report.append(f"Max position size: {self.max_position_size:.4f} BTC")
        report.append(f"Position sizing: {self.position_size_pct:.1%} of capital")
        report.append(f"Maker fee: {self.maker_fee:.2%}")
        report.append(f"Taker fee: {self.taker_fee:.2%}")
        
        report.append("=" * 60)
        
        return "\n".join(report)