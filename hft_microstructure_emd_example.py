#!/usr/bin/env python3
"""
High-Frequency Trading Microstructure Analysis with EMD
=======================================================

This script demonstrates practical application of EMD decomposition
for analyzing high-frequency trading microstructure patterns.

Features:
- Real-time market microstructure simulation
- Order book imbalance calculation
- EMD-based regime detection
- Trading signal generation
- Performance metrics for HFT strategies

Author: Trading System
Date: 2025-07-30
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import time
import warnings
warnings.filterwarnings('ignore')

# Import EMD with fallback
try:
    from PyEMD import EMD, EEMD, CEEMDAN
    EMD_AVAILABLE = True
except ImportError:
    EMD_AVAILABLE = False
    print("⚠️  PyEMD not available. Install with: pip install PyEMD")


@dataclass
class MarketData:
    """Container for market microstructure data"""
    timestamp: datetime
    bid_price: float
    ask_price: float
    bid_size: float
    ask_size: float
    last_price: float
    last_size: float
    
    @property
    def mid_price(self) -> float:
        return (self.bid_price + self.ask_price) / 2
    
    @property
    def spread(self) -> float:
        return self.ask_price - self.bid_price
    
    @property
    def weighted_mid_price(self) -> float:
        """Volume-weighted mid price"""
        total_size = self.bid_size + self.ask_size
        if total_size == 0:
            return self.mid_price
        return (self.bid_price * self.ask_size + self.ask_price * self.bid_size) / total_size
    
    @property
    def order_imbalance(self) -> float:
        """Order book imbalance (-1 to 1)"""
        total_size = self.bid_size + self.ask_size
        if total_size == 0:
            return 0
        return (self.bid_size - self.ask_size) / total_size


class HFTMicrostructureSimulator:
    """Simulates realistic HFT market microstructure"""
    
    def __init__(self, 
                 base_price: float = 50000.0,
                 tick_size: float = 0.01,
                 sampling_interval_ms: float = 100.0):
        self.base_price = base_price
        self.tick_size = tick_size
        self.dt = sampling_interval_ms / 1000.0
        self.current_price = base_price
        self.current_time = datetime.now()
        
        # Market state
        self.volatility = 0.02
        self.spread_basis_points = 5  # 5 bps typical spread
        
    def generate_microstructure_data(self, num_ticks: int) -> List[MarketData]:
        """Generate realistic microstructure tick data"""
        data = []
        
        for i in range(num_ticks):
            # Update time
            self.current_time += timedelta(milliseconds=100)
            
            # Price evolution with microstructure effects
            price_change = np.random.normal(0, self.volatility * np.sqrt(self.dt))
            self.current_price *= (1 + price_change)
            
            # Calculate spread
            spread = self.current_price * (self.spread_basis_points / 10000)
            spread = max(spread, self.tick_size)  # Minimum 1 tick spread
            
            # Generate bid/ask around current price
            mid_offset = np.random.uniform(-spread/4, spread/4)  # Mid-price uncertainty
            mid_price = self.current_price + mid_offset
            
            bid_price = mid_price - spread/2
            ask_price = mid_price + spread/2
            
            # Round to tick size
            bid_price = round(bid_price / self.tick_size) * self.tick_size
            ask_price = round(ask_price / self.tick_size) * self.tick_size
            
            # Generate realistic order sizes
            # Larger sizes during low volatility, smaller during high volatility
            vol_factor = min(abs(price_change) * 100, 1.0)
            base_size = np.random.exponential(10) * (1 - vol_factor * 0.5)
            
            bid_size = max(0.1, np.random.exponential(base_size))
            ask_size = max(0.1, np.random.exponential(base_size))
            
            # Add some correlation between bid/ask sizes (inventory effects)
            if np.random.random() < 0.3:  # 30% chance of imbalanced book
                if np.random.random() < 0.5:
                    bid_size *= 2  # More buying pressure
                else:
                    ask_size *= 2  # More selling pressure
            
            # Generate last trade
            # Trade more likely on the side with less size (liquidity consumption)
            if bid_size < ask_size:
                last_price = ask_price if np.random.random() < 0.6 else bid_price
            else:
                last_price = bid_price if np.random.random() < 0.6 else ask_price
            
            last_size = np.random.exponential(base_size * 0.5)
            
            data.append(MarketData(
                timestamp=self.current_time,
                bid_price=bid_price,
                ask_price=ask_price,
                bid_size=bid_size,
                ask_size=ask_size,
                last_price=last_price,
                last_size=last_size
            ))
        
        return data


class EMDMicrostructureAnalyzer:
    """Analyzes microstructure data using EMD decomposition"""
    
    def __init__(self):
        self.emd = EMD() if EMD_AVAILABLE else None
        self.eemd = EEMD(trials=50) if EMD_AVAILABLE else None
        
    def decompose_price_series(self, prices: np.ndarray, method: str = 'EMD') -> Tuple[np.ndarray, np.ndarray]:
        """Decompose price series into IMFs"""
        if not EMD_AVAILABLE:
            raise RuntimeError("PyEMD not available")
        
        if method == 'EMD':
            imfs = self.emd(prices)
        elif method == 'EEMD':
            imfs = self.eemd(prices)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return imfs, np.sum(imfs, axis=0)
    
    def extract_microstructure_features(self, market_data: List[MarketData]) -> pd.DataFrame:
        """Extract comprehensive microstructure features"""
        df = pd.DataFrame([
            {
                'timestamp': tick.timestamp,
                'mid_price': tick.mid_price,
                'weighted_mid_price': tick.weighted_mid_price,
                'spread': tick.spread,
                'spread_bps': (tick.spread / tick.mid_price) * 10000,
                'order_imbalance': tick.order_imbalance,
                'bid_size': tick.bid_size,
                'ask_size': tick.ask_size,
                'total_size': tick.bid_size + tick.ask_size,
                'last_price': tick.last_price,
                'last_size': tick.last_size
            }
            for tick in market_data
        ])
        
        # Calculate additional features
        df['price_returns'] = df['mid_price'].pct_change()
        df['log_returns'] = np.log(df['mid_price']).diff()
        df['realized_volatility'] = df['log_returns'].rolling(20).std() * np.sqrt(10 * 60)  # 10-min RV
        
        # Order flow features
        df['order_flow_imbalance'] = df['order_imbalance'].diff()
        df['signed_volume'] = df['last_size'] * np.sign(df['last_price'] - df['mid_price'])
        df['volume_imbalance'] = df['signed_volume'].rolling(10).sum()
        
        # Microstructure noise metrics
        df['effective_spread'] = 2 * abs(df['last_price'] - df['mid_price'])
        df['price_impact'] = df['effective_spread'] / df['last_size']
        
        return df
    
    def detect_market_regimes(self, prices: np.ndarray, window_size: int = 200) -> Dict[str, np.ndarray]:
        """Detect market regimes using EMD decomposition"""
        if not EMD_AVAILABLE:
            return {'regime': np.full(len(prices), 'unknown')}
        
        regimes = []
        trend_strength = []
        cycle_dominance = []
        noise_level = []
        
        # Rolling EMD analysis
        for i in range(window_size, len(prices)):
            window_prices = prices[i-window_size:i]
            
            try:
                imfs = self.emd(window_prices)
                
                # Analyze IMF characteristics
                if len(imfs) >= 3:
                    # High-frequency noise (first IMF)
                    noise_power = np.var(imfs[0])
                    
                    # Mid-frequency cycles (middle IMFs)
                    cycle_power = np.sum([np.var(imf) for imf in imfs[1:-1]])
                    
                    # Low-frequency trend (last IMF)
                    trend_power = np.var(imfs[-1])
                    trend_slope = np.polyfit(range(len(imfs[-1])), imfs[-1], 1)[0]
                    
                    total_power = noise_power + cycle_power + trend_power
                    
                    # Normalize
                    noise_ratio = noise_power / total_power if total_power > 0 else 0
                    cycle_ratio = cycle_power / total_power if total_power > 0 else 0
                    trend_ratio = trend_power / total_power if total_power > 0 else 0
                    
                    # Classify regime
                    if trend_ratio > 0.5 and abs(trend_slope) > np.std(window_prices) * 0.01:
                        regime = 'trending_up' if trend_slope > 0 else 'trending_down'
                    elif cycle_ratio > 0.4:
                        regime = 'cyclical'
                    elif noise_ratio > 0.6:
                        regime = 'noisy'
                    else:
                        regime = 'mixed'
                    
                    regimes.append(regime)
                    trend_strength.append(abs(trend_slope))
                    cycle_dominance.append(cycle_ratio)
                    noise_level.append(noise_ratio)
                    
                else:
                    regimes.append('insufficient_data')
                    trend_strength.append(0)
                    cycle_dominance.append(0)
                    noise_level.append(1)
                    
            except Exception as e:
                regimes.append('error')
                trend_strength.append(0)
                cycle_dominance.append(0)
                noise_level.append(1)
        
        # Pad beginning with 'unknown'
        full_regimes = ['unknown'] * window_size + regimes
        full_trend = [0] * window_size + trend_strength
        full_cycle = [0] * window_size + cycle_dominance
        full_noise = [1] * window_size + noise_level
        
        return {
            'regime': np.array(full_regimes),
            'trend_strength': np.array(full_trend),
            'cycle_dominance': np.array(full_cycle),
            'noise_level': np.array(full_noise)
        }
    
    def generate_trading_signals(self, df: pd.DataFrame, regime_data: Dict[str, np.ndarray]) -> pd.DataFrame:
        """Generate trading signals based on microstructure and regime analysis"""
        df = df.copy()
        
        # Add regime information
        df['regime'] = regime_data['regime']
        df['trend_strength'] = regime_data['trend_strength']
        df['cycle_dominance'] = regime_data['cycle_dominance']
        df['noise_level'] = regime_data['noise_level']
        
        signals = []
        
        for i in range(len(df)):
            regime = df.iloc[i]['regime']
            ofi = df.iloc[i]['order_flow_imbalance'] if not pd.isna(df.iloc[i]['order_flow_imbalance']) else 0
            oi = df.iloc[i]['order_imbalance']
            spread = df.iloc[i]['spread_bps']
            volume_imbalance = df.iloc[i]['volume_imbalance'] if not pd.isna(df.iloc[i]['volume_imbalance']) else 0
            
            signal = 0  # Default: no signal
            confidence = 0
            
            # Signal generation logic based on regime
            if regime == 'trending_up':
                if ofi > 0.1 and oi > 0.2 and spread < 10:  # Strong buy pressure, tight spread
                    signal = 1  # Buy signal
                    confidence = min(abs(ofi) + abs(oi), 1.0)
                elif ofi < -0.2:  # Strong sell pressure against trend
                    signal = -1  # Fade signal (counter-trend)
                    confidence = min(abs(ofi) * 0.5, 1.0)
                    
            elif regime == 'trending_down':
                if ofi < -0.1 and oi < -0.2 and spread < 10:  # Strong sell pressure, tight spread
                    signal = -1  # Sell signal
                    confidence = min(abs(ofi) + abs(oi), 1.0)
                elif ofi > 0.2:  # Strong buy pressure against trend
                    signal = 1  # Fade signal (counter-trend)
                    confidence = min(abs(ofi) * 0.5, 1.0)
                    
            elif regime == 'cyclical':
                # Mean reversion logic
                if volume_imbalance > 0.5 and oi > 0.3:  # Accumulated buying at potential top
                    signal = -1  # Sell signal
                    confidence = min(abs(volume_imbalance) * abs(oi), 1.0)
                elif volume_imbalance < -0.5 and oi < -0.3:  # Accumulated selling at potential bottom
                    signal = 1  # Buy signal
                    confidence = min(abs(volume_imbalance) * abs(oi), 1.0)
                    
            elif regime == 'noisy':
                # Reduce position sizing or avoid trading
                signal = 0
                confidence = 0
            
            # Apply spread filter (don't trade if spread too wide)
            if spread > 20:  # 20 bps spread threshold
                signal = 0
                confidence = 0
            
            signals.append({'signal': signal, 'confidence': confidence})
        
        signal_df = pd.DataFrame(signals)
        df['signal'] = signal_df['signal']
        df['confidence'] = signal_df['confidence']
        
        return df


def create_comprehensive_analysis_plot(df: pd.DataFrame, imfs: Optional[np.ndarray] = None):
    """Create comprehensive analysis visualization"""
    fig, axes = plt.subplots(4, 2, figsize=(20, 16))
    fig.suptitle('HFT Microstructure Analysis with EMD', fontsize=16, fontweight='bold')
    
    # 1. Price and EMD decomposition
    ax1 = axes[0, 0]
    ax1.plot(df.index, df['mid_price'], 'b-', linewidth=1, label='Mid Price')
    ax1.plot(df.index, df['weighted_mid_price'], 'r--', linewidth=1, label='Weighted Mid Price')
    ax1.set_title('Price Evolution')
    ax1.set_ylabel('Price')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. EMD decomposition (if available)
    ax2 = axes[0, 1]
    if imfs is not None and len(imfs) > 0:
        # Show first few IMFs
        colors = ['red', 'green', 'blue', 'orange', 'purple']
        for i, imf in enumerate(imfs[:min(5, len(imfs))]):
            ax2.plot(df.index[:len(imf)], imf, color=colors[i % len(colors)], 
                    linewidth=1, label=f'IMF {i+1}', alpha=0.7)
        ax2.set_title('EMD Decomposition (First 5 IMFs)')
        ax2.legend()
    else:
        ax2.text(0.5, 0.5, 'EMD decomposition\nnot available', 
                ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('EMD Decomposition')
    ax2.grid(True, alpha=0.3)
    
    # 3. Order book metrics
    ax3 = axes[1, 0]
    ax3.plot(df.index, df['order_imbalance'], 'g-', linewidth=1, label='Order Imbalance')
    ax3.plot(df.index, df['order_flow_imbalance'], 'r-', linewidth=1, label='Order Flow Imbalance', alpha=0.7)
    ax3.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax3.set_title('Order Book Imbalance Metrics')
    ax3.set_ylabel('Imbalance')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Spread and volatility
    ax4 = axes[1, 1]
    ax4_twin = ax4.twinx()
    
    line1 = ax4.plot(df.index, df['spread_bps'], 'b-', linewidth=1, label='Spread (bps)')
    line2 = ax4_twin.plot(df.index, df['realized_volatility'] * 100, 'r-', linewidth=1, label='Realized Vol (%)')
    
    ax4.set_title('Spread and Volatility')
    ax4.set_ylabel('Spread (bps)', color='b')
    ax4_twin.set_ylabel('Realized Volatility (%)', color='r')
    
    # Combine legends
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax4.legend(lines, labels, loc='upper left')
    ax4.grid(True, alpha=0.3)
    
    # 5. Market regimes
    ax5 = axes[2, 0]
    regime_colors = {
        'trending_up': 'green',
        'trending_down': 'red', 
        'cyclical': 'blue',
        'noisy': 'orange',
        'mixed': 'purple',
        'unknown': 'gray',
        'error': 'black'
    }
    
    for i, regime in enumerate(df['regime']):
        color = regime_colors.get(regime, 'gray')
        ax5.scatter(i, 0, c=color, s=10, alpha=0.6)
    
    ax5.set_title('Market Regime Detection')
    ax5.set_ylabel('Regime')
    ax5.set_ylim(-0.5, 0.5)
    
    # Create legend
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                 markerfacecolor=color, markersize=8, label=regime)
                      for regime, color in regime_colors.items() 
                      if regime in df['regime'].values]
    ax5.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # 6. Regime characteristics
    ax6 = axes[2, 1]
    ax6.plot(df.index, df['trend_strength'], 'g-', linewidth=1, label='Trend Strength', alpha=0.7)
    ax6.plot(df.index, df['cycle_dominance'], 'b-', linewidth=1, label='Cycle Dominance', alpha=0.7)
    ax6.plot(df.index, df['noise_level'], 'r-', linewidth=1, label='Noise Level', alpha=0.7)
    ax6.set_title('Regime Characteristics')
    ax6.set_ylabel('Strength')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    # 7. Trading signals
    ax7 = axes[3, 0]
    buy_signals = df[df['signal'] == 1]
    sell_signals = df[df['signal'] == -1]
    
    ax7.plot(df.index, df['mid_price'], 'k-', linewidth=1, alpha=0.6, label='Price')
    
    if len(buy_signals) > 0:
        ax7.scatter(buy_signals.index, buy_signals['mid_price'], 
                   c='green', marker='^', s=50, label='Buy Signal', alpha=0.8)
    
    if len(sell_signals) > 0:
        ax7.scatter(sell_signals.index, sell_signals['mid_price'], 
                   c='red', marker='v', s=50, label='Sell Signal', alpha=0.8)
    
    ax7.set_title('Trading Signals')
    ax7.set_ylabel('Price')
    ax7.legend()
    ax7.grid(True, alpha=0.3)
    
    # 8. Signal confidence and volume
    ax8 = axes[3, 1]
    ax8_twin = ax8.twinx()
    
    line1 = ax8.plot(df.index, df['confidence'], 'b-', linewidth=1, label='Signal Confidence')
    line2 = ax8_twin.plot(df.index, df['total_size'], 'g-', linewidth=1, alpha=0.6, label='Total Order Size')
    
    ax8.set_title('Signal Confidence and Order Book Depth')
    ax8.set_ylabel('Confidence', color='b')
    ax8_twin.set_ylabel('Total Size', color='g')
    
    # Combine legends
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax8.legend(lines, labels, loc='upper left')
    ax8.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('hft_microstructure_analysis.png', dpi=300, bbox_inches='tight')
    print("📊 Analysis plot saved as 'hft_microstructure_analysis.png'")
    
    # Don't show in automated environment
    # plt.show()


def calculate_strategy_performance(df: pd.DataFrame) -> Dict[str, float]:
    """Calculate strategy performance metrics"""
    # Simple strategy: take positions based on signals
    df = df.copy()
    df['position'] = df['signal'] * df['confidence']
    df['strategy_returns'] = df['position'].shift(1) * df['price_returns']
    
    # Performance metrics
    total_return = df['strategy_returns'].sum()
    volatility = df['strategy_returns'].std() * np.sqrt(10 * 60)  # Annualized
    sharpe_ratio = total_return / volatility if volatility > 0 else 0
    
    max_drawdown = (df['strategy_returns'].cumsum() - df['strategy_returns'].cumsum().expanding().max()).min()
    
    num_trades = (df['signal'] != 0).sum()
    win_rate = (df['strategy_returns'] > 0).sum() / num_trades if num_trades > 0 else 0
    
    return {
        'total_return': total_return,
        'volatility': volatility,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'num_trades': num_trades,
        'win_rate': win_rate
    }


def main():
    """Main demonstration function"""
    print("🚀 HFT Microstructure Analysis with EMD")
    print("=" * 40)
    
    if not EMD_AVAILABLE:
        print("❌ PyEMD not available. Please install with: pip install PyEMD")
        return
    
    # Parameters
    num_ticks = 2000  # About 3.3 minutes at 100ms intervals
    base_price = 50000.0
    
    print(f"📊 Generating {num_ticks} ticks of HFT microstructure data...")
    print(f"💰 Base price: ${base_price:,.2f}")
    print(f"⏱️  Sampling interval: 100ms")
    
    # Generate microstructure data
    simulator = HFTMicrostructureSimulator(base_price=base_price)
    market_data = simulator.generate_microstructure_data(num_ticks)
    
    print(f"✅ Generated {len(market_data)} market data points")
    
    # Initialize analyzer
    analyzer = EMDMicrostructureAnalyzer()
    
    # Extract features
    print("🔍 Extracting microstructure features...")
    df = analyzer.extract_microstructure_features(market_data)
    
    # EMD decomposition
    print("🌊 Performing EMD decomposition...")
    try:
        prices = df['weighted_mid_price'].values
        imfs, reconstruction = analyzer.decompose_price_series(prices, method='EMD')
        print(f"   ✅ Extracted {len(imfs)} IMFs")
        
        # Regime detection
        print("🎯 Detecting market regimes...")
        regime_data = analyzer.detect_market_regimes(prices)
        
        # Generate trading signals  
        print("📈 Generating trading signals...")
        df = analyzer.generate_trading_signals(df, regime_data)
        
        # Calculate performance
        print("📊 Calculating strategy performance...")
        performance = calculate_strategy_performance(df)
        
        # Display results
        print("\n" + "="*50)
        print("ANALYSIS RESULTS")
        print("="*50)
        
        print(f"📈 Total Return: {performance['total_return']*100:.4f}%")
        print(f"📊 Sharpe Ratio: {performance['sharpe_ratio']:.3f}")
        print(f"📉 Max Drawdown: {performance['max_drawdown']*100:.4f}%")
        print(f"🎯 Number of Trades: {performance['num_trades']}")
        print(f"🎲 Win Rate: {performance['win_rate']*100:.1f}%")
        
        # Regime analysis
        regime_counts = pd.Series(df['regime']).value_counts()
        print(f"\n🎨 Market Regime Distribution:")
        for regime, count in regime_counts.items():
            pct = (count / len(df)) * 100
            print(f"   {regime}: {count} ticks ({pct:.1f}%)")
        
        # Signal analysis
        signal_counts = pd.Series(df['signal']).value_counts()
        print(f"\n📡 Signal Distribution:")
        for signal, count in signal_counts.items():
            pct = (count / len(df)) * 100
            if signal == 1:
                print(f"   Buy signals: {count} ({pct:.1f}%)")
            elif signal == -1:
                print(f"   Sell signals: {count} ({pct:.1f}%)")
            else:
                print(f"   No signal: {count} ({pct:.1f}%)")
        
        # Create visualization
        print("\n📈 Creating comprehensive analysis plot...")
        create_comprehensive_analysis_plot(df, imfs)
        
        # Save data
        df.to_csv('hft_microstructure_analysis_data.csv', index=False)
        print("💾 Analysis data saved as 'hft_microstructure_analysis_data.csv'")
        
        print("\n✅ Analysis complete!")
        print("📁 Files generated:")
        print("   • hft_microstructure_analysis.png - Visualization")
        print("   • hft_microstructure_analysis_data.csv - Raw data")
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()