#!/usr/bin/env python3
"""
Pandas DataFrame Integration with EMD for HFT Analysis
=====================================================

This script demonstrates how to integrate EMD analysis with pandas DataFrames
for high-frequency trading applications.

Features:
- Real-time EMD feature calculation
- Rolling window analysis
- DataFrame-based signal processing
- Performance metrics
- Export capabilities

Author: Trading System  
Date: 2025-07-30
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import time
from typing import Optional

try:
    from PyEMD import EMD, EEMD
    EMD_AVAILABLE = True
    print("PyEMD successfully imported")
except ImportError:
    print("PyEMD not available")
    EMD_AVAILABLE = False


def create_hft_dataframe(length: int = 2000, base_price: float = 50000.0) -> pd.DataFrame:
    """Create a realistic HFT DataFrame with OHLCV-like data"""
    
    # Generate timestamps (100ms intervals)
    start_time = datetime(2024, 1, 1, 9, 30, 0)
    timestamps = [start_time + timedelta(milliseconds=i*100) for i in range(length)]
    
    # Generate realistic price data
    dt = 0.1  # 100ms
    t = np.arange(length) * dt
    
    # Price components
    trend = base_price * (1 + 0.0001 * t)
    cycle1 = 100 * np.sin(2 * np.pi * 0.1 * t)
    cycle2 = 50 * np.sin(2 * np.pi * 0.05 * t)
    noise = 20 * np.random.normal(0, 1, length)
    
    prices = trend + cycle1 + cycle2 + noise
    
    # Generate OHLCV-like data
    price_changes = np.random.normal(0, 10, length)
    
    df = pd.DataFrame({
        'timestamp': timestamps,
        'price': prices,
        'high': prices + np.abs(np.random.normal(0, 5, length)),
        'low': prices - np.abs(np.random.normal(0, 5, length)),
        'volume': np.random.exponential(100, length),
        'bid_size': np.random.exponential(10, length),
        'ask_size': np.random.exponential(10, length),
    })
    
    # Calculate derived features
    df['returns'] = df['price'].pct_change()
    df['log_returns'] = np.log(df['price']).diff()
    df['spread'] = df['high'] - df['low']
    df['order_imbalance'] = (df['bid_size'] - df['ask_size']) / (df['bid_size'] + df['ask_size'])
    
    return df


def add_emd_features(df: pd.DataFrame, price_col: str = 'price', window: int = 200) -> pd.DataFrame:
    """Add EMD-based features to the DataFrame using rolling windows"""
    
    if not EMD_AVAILABLE:
        print("EMD not available - adding dummy features")
        df['emd_trend'] = 0
        df['emd_cycle'] = 0
        df['emd_noise'] = 0
        df['market_regime'] = 'unknown'
        return df
    
    print(f"Adding EMD features with {window}-period rolling window...")
    
    emd = EMD()
    
    # Initialize feature columns
    df['emd_trend'] = np.nan
    df['emd_cycle'] = np.nan  
    df['emd_noise'] = np.nan
    df['trend_strength'] = np.nan
    df['cycle_dominance'] = np.nan
    df['noise_level'] = np.nan
    df['market_regime'] = 'unknown'
    
    # Rolling EMD analysis
    for i in range(window, len(df)):
        if i % 100 == 0:  # Progress update
            print(f"  Processing row {i}/{len(df)}")
        
        try:
            # Extract price window
            price_window = df[price_col].iloc[i-window:i].values
            
            # Apply EMD
            imfs = emd(price_window)
            
            if len(imfs) >= 3:
                # Extract components (last values from window)
                df.loc[df.index[i], 'emd_noise'] = imfs[0][-1]  # High frequency
                df.loc[df.index[i], 'emd_cycle'] = imfs[1][-1]  # Mid frequency
                df.loc[df.index[i], 'emd_trend'] = imfs[-1][-1]  # Trend
                
                # Calculate regime indicators
                noise_var = np.var(imfs[0])
                cycle_var = np.mean([np.var(imf) for imf in imfs[1:-1]]) if len(imfs) > 2 else 0
                trend_var = np.var(imfs[-1])
                
                total_var = noise_var + cycle_var + trend_var
                
                if total_var > 0:
                    noise_ratio = noise_var / total_var
                    cycle_ratio = cycle_var / total_var  
                    trend_ratio = trend_var / total_var
                    
                    df.loc[df.index[i], 'noise_level'] = noise_ratio
                    df.loc[df.index[i], 'cycle_dominance'] = cycle_ratio
                    df.loc[df.index[i], 'trend_strength'] = trend_ratio
                    
                    # Determine regime
                    if trend_ratio > 0.5:
                        trend_slope = np.polyfit(range(len(imfs[-1])), imfs[-1], 1)[0]
                        regime = 'trending_up' if trend_slope > 0 else 'trending_down'
                    elif cycle_ratio > 0.4:
                        regime = 'cyclical'
                    elif noise_ratio > 0.6:
                        regime = 'noisy'
                    else:
                        regime = 'mixed'
                    
                    df.loc[df.index[i], 'market_regime'] = regime
            
        except Exception as e:
            # Handle EMD failures gracefully
            df.loc[df.index[i], 'market_regime'] = 'error'
            continue
    
    print("EMD features added successfully")
    return df


def generate_emd_trading_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Generate trading signals based on EMD features"""
    
    df = df.copy()
    df['signal'] = 0
    df['signal_strength'] = 0
    
    if not EMD_AVAILABLE:
        return df
    
    print("Generating EMD-based trading signals...")
    
    for i in range(len(df)):
        regime = df.iloc[i]['market_regime']
        
        if pd.isna(df.iloc[i]['emd_trend']) or regime == 'unknown':
            continue
            
        trend = df.iloc[i]['emd_trend']
        cycle = df.iloc[i]['emd_cycle'] 
        noise = df.iloc[i]['emd_noise']
        oi = df.iloc[i]['order_imbalance']
        
        signal = 0
        strength = 0
        
        if regime == 'trending_up':
            # Buy dips in uptrend
            if noise < -10 and cycle < 0 and oi > 0.1:
                signal = 1
                strength = min(abs(noise/20) + abs(oi), 1.0)
        
        elif regime == 'trending_down':
            # Sell rallies in downtrend  
            if noise > 10 and cycle > 0 and oi < -0.1:
                signal = -1
                strength = min(abs(noise/20) + abs(oi), 1.0)
        
        elif regime == 'cyclical':
            # Mean reversion in cycles
            if cycle > 20 and oi < -0.2:  # Cycle high with selling pressure
                signal = -1
                strength = min(abs(cycle/30) + abs(oi), 1.0)
            elif cycle < -20 and oi > 0.2:  # Cycle low with buying pressure
                signal = 1
                strength = min(abs(cycle/30) + abs(oi), 1.0)
        
        df.loc[df.index[i], 'signal'] = signal
        df.loc[df.index[i], 'signal_strength'] = strength
    
    return df


def calculate_strategy_performance(df: pd.DataFrame) -> dict:
    """Calculate trading strategy performance metrics"""
    
    # Calculate position and returns
    df['position'] = df['signal'] * df['signal_strength']
    df['strategy_returns'] = df['position'].shift(1) * df['returns']
    
    # Performance metrics
    total_return = df['strategy_returns'].sum()
    volatility = df['strategy_returns'].std() * np.sqrt(10 * 60)  # Annualized (rough)
    
    num_trades = (df['signal'] != 0).sum()
    winning_trades = (df['strategy_returns'] > 0).sum()
    win_rate = winning_trades / num_trades if num_trades > 0 else 0
    
    max_drawdown = (df['strategy_returns'].cumsum() - 
                   df['strategy_returns'].cumsum().expanding().max()).min()
    
    sharpe_ratio = total_return / volatility if volatility > 0 else 0
    
    return {
        'total_return': total_return,
        'volatility': volatility,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'num_trades': num_trades,
        'win_rate': win_rate,
        'winning_trades': winning_trades
    }


def create_analysis_dashboard(df: pd.DataFrame):
    """Create comprehensive analysis dashboard"""
    
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    fig.suptitle('EMD-Enhanced HFT Analysis Dashboard', fontsize=14, fontweight='bold')
    
    # 1. Price and EMD trend
    ax1 = axes[0, 0]
    ax1.plot(df.index, df['price'], 'b-', linewidth=1, alpha=0.7, label='Price')
    if 'emd_trend' in df.columns and df['emd_trend'].notna().any():
        valid_trend = df.dropna(subset=['emd_trend'])
        ax1.plot(valid_trend.index, valid_trend['emd_trend'], 'r-', linewidth=2, label='EMD Trend')
    ax1.set_title('Price vs EMD Trend')
    ax1.set_ylabel('Price')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. EMD components
    ax2 = axes[0, 1]
    if all(col in df.columns for col in ['emd_noise', 'emd_cycle']):
        valid_data = df.dropna(subset=['emd_noise', 'emd_cycle'])
        if not valid_data.empty:
            ax2.plot(valid_data.index, valid_data['emd_noise'], 'g-', linewidth=1, 
                    alpha=0.7, label='Noise')
            ax2.plot(valid_data.index, valid_data['emd_cycle'], 'orange', linewidth=1, 
                    alpha=0.7, label='Cycle')
    ax2.set_title('EMD Components')
    ax2.set_ylabel('Amplitude')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Market regimes
    ax3 = axes[1, 0]
    if 'market_regime' in df.columns:
        regime_colors = {
            'trending_up': 'green', 'trending_down': 'red',
            'cyclical': 'blue', 'noisy': 'orange', 
            'mixed': 'purple', 'unknown': 'gray', 'error': 'black'
        }
        
        for i, regime in enumerate(df['market_regime']):
            color = regime_colors.get(regime, 'gray')
            ax3.scatter(i, 0, c=color, s=5, alpha=0.6)
    
    ax3.set_title('Market Regime Detection')
    ax3.set_ylabel('Regime')
    ax3.set_ylim(-0.5, 0.5)
    
    # 4. Order book metrics
    ax4 = axes[1, 1]
    ax4.plot(df.index, df['order_imbalance'], 'purple', linewidth=1, 
            alpha=0.7, label='Order Imbalance')
    ax4.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax4.set_title('Order Book Imbalance')
    ax4.set_ylabel('Imbalance')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. Trading signals
    ax5 = axes[2, 0]
    ax5.plot(df.index, df['price'], 'k-', linewidth=1, alpha=0.6, label='Price')
    
    buy_signals = df[df['signal'] == 1]
    sell_signals = df[df['signal'] == -1]
    
    if len(buy_signals) > 0:
        ax5.scatter(buy_signals.index, buy_signals['price'], 
                   c='green', marker='^', s=30, alpha=0.8, label='Buy')
    if len(sell_signals) > 0:
        ax5.scatter(sell_signals.index, sell_signals['price'],
                   c='red', marker='v', s=30, alpha=0.8, label='Sell')
    
    ax5.set_title('Trading Signals')
    ax5.set_ylabel('Price')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. Cumulative returns
    ax6 = axes[2, 1]
    if 'strategy_returns' in df.columns:
        cumulative_returns = df['strategy_returns'].cumsum()
        ax6.plot(df.index, cumulative_returns, 'g-', linewidth=2, label='Strategy')
        
        # Buy and hold comparison
        buy_hold_returns = (df['price'] / df['price'].iloc[0] - 1) * df['price'].iloc[0]
        ax6.plot(df.index, buy_hold_returns, 'b--', linewidth=1, alpha=0.7, label='Buy & Hold')
    
    ax6.set_title('Cumulative Returns')
    ax6.set_xlabel('Time')
    ax6.set_ylabel('Cumulative Return')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('pandas_emd_dashboard.png', dpi=300, bbox_inches='tight')
    print("Dashboard saved as 'pandas_emd_dashboard.png'")


def main():
    """Main demonstration function"""
    print("Pandas EMD Integration Example for HFT")
    print("=" * 40)
    
    if not EMD_AVAILABLE:
        print("WARNING: PyEMD not available - will create example without EMD features")
    
    # 1. Create HFT DataFrame
    print("\n1. Creating HFT DataFrame...")
    df = create_hft_dataframe(length=1000)  # Shorter for demo
    print(f"   Created DataFrame with {len(df)} rows")
    print(f"   Time range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    
    # 2. Add EMD features
    print("\n2. Adding EMD features...")
    start_time = time.time()
    df = add_emd_features(df, window=100)  # Smaller window for demo
    emd_time = time.time() - start_time
    print(f"   EMD analysis completed in {emd_time:.1f} seconds")
    
    # 3. Generate trading signals
    print("\n3. Generating trading signals...")
    df = generate_emd_trading_signals(df)
    
    # 4. Calculate performance
    print("\n4. Calculating strategy performance...")
    performance = calculate_strategy_performance(df)
    
    # 5. Create visualizations
    print("\n5. Creating analysis dashboard...")
    create_analysis_dashboard(df)
    
    # 6. Save results
    print("\n6. Saving results...")
    df.to_csv('hft_emd_analysis.csv', index=False)
    
    # Display results
    print("\n" + "=" * 50)
    print("ANALYSIS RESULTS")
    print("=" * 50)
    
    print(f"DataFrame Shape: {df.shape}")
    print(f"EMD Features Available: {EMD_AVAILABLE}")
    
    if EMD_AVAILABLE:
        regime_counts = df['market_regime'].value_counts()
        print(f"\nMarket Regime Distribution:")
        for regime, count in regime_counts.items():
            pct = (count / len(df)) * 100
            print(f"  {regime}: {count} ({pct:.1f}%)")
    
    print(f"\nTrading Performance:")
    print(f"  Total Return: {performance['total_return']:.2f}")
    print(f"  Sharpe Ratio: {performance['sharpe_ratio']:.3f}")
    print(f"  Max Drawdown: {performance['max_drawdown']:.2f}")
    print(f"  Number of Trades: {performance['num_trades']}")
    print(f"  Win Rate: {performance['win_rate']:.1%}")
    
    print(f"\nFiles Generated:")
    print(f"  - hft_emd_analysis.csv (DataFrame with all features)")
    print(f"  - pandas_emd_dashboard.png (Analysis dashboard)")
    
    # Display sample of enhanced DataFrame
    print(f"\nSample of Enhanced DataFrame:")
    sample_cols = ['timestamp', 'price', 'returns', 'market_regime', 'signal']
    if EMD_AVAILABLE:
        sample_cols.extend(['emd_trend', 'emd_cycle', 'emd_noise'])
    
    available_cols = [col for col in sample_cols if col in df.columns]
    print(df[available_cols].head(10).to_string(index=False))
    
    print(f"\nIntegration example completed successfully!")


if __name__ == "__main__":
    main()