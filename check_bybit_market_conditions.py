"""
Bybit Market Conditions Analyzer
Real-time analysis of BTC/USDT perpetual market conditions
Helps identify optimal testing windows for execution strategies
"""

import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import asyncio
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns


class BybitMarketAnalyzer:
    """Analyzes Bybit market conditions for execution optimization"""
    
    def __init__(self, testnet: bool = False):
        """Initialize Bybit connection"""
        self.exchange = ccxt.bybit({
            'enableRateLimit': True,
            'options': {
                'defaultType': 'linear',  # USDT perpetual
            }
        })
        
        if testnet:
            self.exchange.set_sandbox_mode(True)
            
        self.symbol = 'BTC/USDT:USDT'
        self.analysis_results = []
        
    def fetch_current_market_state(self) -> Dict[str, any]:
        """Fetch comprehensive market state"""
        try:
            # Fetch multiple data points
            ticker = self.exchange.fetch_ticker(self.symbol)
            order_book = self.exchange.fetch_order_book(self.symbol, limit=50)
            trades = self.exchange.fetch_trades(self.symbol, limit=100)
            ohlcv = self.exchange.fetch_ohlcv(self.symbol, '1m', limit=60)
            
            # Calculate metrics
            best_bid = order_book['bids'][0][0]
            best_ask = order_book['asks'][0][0]
            mid_price = (best_bid + best_ask) / 2
            spread_bps = ((best_ask - best_bid) / mid_price) * 10000
            
            # Order book metrics
            bid_depth_10 = sum([level[1] for level in order_book['bids'][:10]])
            ask_depth_10 = sum([level[1] for level in order_book['asks'][:10]])
            book_imbalance = (bid_depth_10 - ask_depth_10) / (bid_depth_10 + ask_depth_10)
            
            # Trade flow metrics
            recent_trades = trades[-20:]  # Last 20 trades
            buy_volume = sum([t['amount'] for t in recent_trades if t['side'] == 'buy'])
            sell_volume = sum([t['amount'] for t in recent_trades if t['side'] == 'sell'])
            trade_imbalance = (buy_volume - sell_volume) / (buy_volume + sell_volume) if (buy_volume + sell_volume) > 0 else 0
            
            # Volatility metrics
            df_ohlcv = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            returns = df_ohlcv['close'].pct_change().dropna()
            volatility_1h = returns.std() * np.sqrt(60)  # Hourly volatility
            
            # Price momentum
            price_1h_ago = df_ohlcv['close'].iloc[0]
            price_change_1h = ((mid_price - price_1h_ago) / price_1h_ago) * 100
            
            # Volume analysis
            avg_volume_1h = df_ohlcv['volume'].mean()
            current_volume = ticker.get('quoteVolume', 0)
            
            # Funding rate (important for perps)
            funding_rate = ticker.get('info', {}).get('fundingRate', 0)
            
            return {
                'timestamp': datetime.now(),
                'price': mid_price,
                'spread_bps': spread_bps,
                'book_imbalance': book_imbalance,
                'trade_imbalance': trade_imbalance,
                'volatility_1h': volatility_1h,
                'price_change_1h_pct': price_change_1h,
                'bid_depth_10': bid_depth_10,
                'ask_depth_10': ask_depth_10,
                'avg_volume_1h': avg_volume_1h,
                'current_volume_24h': current_volume,
                'funding_rate': funding_rate,
                'bid_count': len(order_book['bids']),
                'ask_count': len(order_book['asks'])
            }
            
        except Exception as e:
            print(f"Error fetching market state: {e}")
            return {}
            
    def analyze_execution_conditions(self, market_state: Dict) -> Dict[str, any]:
        """Analyze market state for execution suitability"""
        
        conditions = {
            'timestamp': market_state['timestamp'],
            'price': market_state['price'],
            'market_quality': 'unknown',
            'recommended_strategies': [],
            'warnings': [],
            'opportunities': []
        }
        
        # Spread analysis
        spread = market_state['spread_bps']
        if spread < 2:
            conditions['opportunities'].append("Tight spread - good for aggressive execution")
            conditions['recommended_strategies'].append('aggressive')
        elif spread < 5:
            conditions['opportunities'].append("Normal spread - balanced execution suitable")
            conditions['recommended_strategies'].append('balanced')
        else:
            conditions['warnings'].append(f"Wide spread ({spread:.1f} bps) - use passive execution")
            conditions['recommended_strategies'].append('passive')
            
        # Volatility analysis
        vol = market_state['volatility_1h']
        if vol < 0.01:  # Less than 1% hourly vol
            conditions['market_quality'] = 'stable'
            conditions['opportunities'].append("Low volatility - ideal for passive strategies")
            conditions['recommended_strategies'].append('passive')
        elif vol < 0.02:
            conditions['market_quality'] = 'normal'
        else:
            conditions['market_quality'] = 'volatile'
            conditions['warnings'].append(f"High volatility ({vol*100:.1f}% hourly) - use careful position sizing")
            
        # Book imbalance analysis
        imbalance = abs(market_state['book_imbalance'])
        if imbalance > 0.3:
            side = "bid" if market_state['book_imbalance'] > 0 else "ask"
            conditions['warnings'].append(f"Significant book imbalance towards {side} side")
            conditions['opportunities'].append("Consider trading with the imbalance direction")
            
        # Depth analysis
        total_depth = market_state['bid_depth_10'] + market_state['ask_depth_10']
        btc_depth = total_depth  # Already in BTC for perps
        
        if btc_depth < 50:
            conditions['warnings'].append("Thin order book - use iceberg orders for size")
        elif btc_depth > 200:
            conditions['opportunities'].append("Deep liquidity - can execute larger sizes")
            
        # Funding rate analysis
        funding = market_state.get('funding_rate', 0)
        if abs(funding) > 0.001:  # 0.1% funding
            direction = "longs" if funding > 0 else "shorts"
            conditions['warnings'].append(f"High funding rate ({funding*100:.3f}%) - {direction} paying")
            
        # Trade flow analysis
        trade_imb = market_state.get('trade_imbalance', 0)
        if abs(trade_imb) > 0.3:
            direction = "buying" if trade_imb > 0 else "selling"
            conditions['opportunities'].append(f"Strong {direction} pressure in recent trades")
            
        # Time-based recommendations
        hour = datetime.now().hour
        if 8 <= hour <= 10:  # Asian morning
            conditions['opportunities'].append("Asian session - typically lower volatility")
        elif 14 <= hour <= 16:  # European open
            conditions['warnings'].append("European open - expect increased volatility")
        elif 21 <= hour <= 23:  # US active
            conditions['warnings'].append("US session - highest volatility period")
            
        return conditions
        
    def monitor_market_windows(self, duration_minutes: int = 60, interval_seconds: int = 30):
        """Monitor market for optimal execution windows"""
        
        print(f"Monitoring Bybit {self.symbol} market for {duration_minutes} minutes...")
        print(f"Sampling every {interval_seconds} seconds")
        print("=" * 80)
        
        start_time = time.time()
        end_time = start_time + (duration_minutes * 60)
        
        results = []
        
        while time.time() < end_time:
            # Fetch and analyze
            market_state = self.fetch_current_market_state()
            if market_state:
                conditions = self.analyze_execution_conditions(market_state)
                results.append({**market_state, **conditions})
                
                # Display current conditions
                self.display_conditions(conditions)
                
            # Wait for next interval
            time.sleep(interval_seconds)
            
        # Generate summary report
        self.generate_window_report(results)
        return results
        
    def display_conditions(self, conditions: Dict):
        """Display current market conditions"""
        print(f"\n[{conditions['timestamp'].strftime('%H:%M:%S')}] "
              f"BTC: ${conditions['price']:,.0f} | "
              f"Quality: {conditions['market_quality']}")
        
        if conditions['recommended_strategies']:
            print(f"Recommended: {', '.join(conditions['recommended_strategies'])}")
            
        for opp in conditions['opportunities']:
            print(f"  ✓ {opp}")
            
        for warn in conditions['warnings']:
            print(f"  ⚠ {warn}")
            
    def generate_window_report(self, results: List[Dict]):
        """Generate summary report of monitoring period"""
        
        if not results:
            print("No results to analyze")
            return
            
        df = pd.DataFrame(results)
        
        print("\n" + "=" * 80)
        print("MARKET WINDOW ANALYSIS REPORT")
        print("=" * 80)
        
        # Price statistics
        print(f"\nPrice Range: ${df['price'].min():,.0f} - ${df['price'].max():,.0f}")
        print(f"Average Price: ${df['price'].mean():,.0f}")
        
        # Spread statistics
        print(f"\nSpread Statistics:")
        print(f"  Average: {df['spread_bps'].mean():.2f} bps")
        print(f"  Min: {df['spread_bps'].min():.2f} bps")
        print(f"  Max: {df['spread_bps'].max():.2f} bps")
        
        # Volatility windows
        print(f"\nVolatility (1h):")
        print(f"  Average: {df['volatility_1h'].mean()*100:.2f}%")
        print(f"  Max: {df['volatility_1h'].max()*100:.2f}%")
        
        # Market quality distribution
        quality_counts = df['market_quality'].value_counts()
        print(f"\nMarket Quality Distribution:")
        for quality, count in quality_counts.items():
            pct = (count / len(df)) * 100
            print(f"  {quality}: {pct:.1f}% of time")
            
        # Optimal windows
        print(f"\nOptimal Execution Windows:")
        
        # Find stable periods
        stable_mask = (df['spread_bps'] < 3) & (df['volatility_1h'] < 0.01)
        if stable_mask.any():
            stable_periods = df[stable_mask]['timestamp'].tolist()
            print(f"  Stable periods (passive execution): {len(stable_periods)} samples")
            
        # Find high activity periods  
        active_mask = (df['volatility_1h'] > 0.015) | (df['spread_bps'] > 5)
        if active_mask.any():
            active_periods = df[active_mask]['timestamp'].tolist()
            print(f"  High activity periods (urgent execution): {len(active_periods)} samples")
            
        # Strategy recommendations
        print(f"\nStrategy Usage Recommendations:")
        all_strategies = []
        for strats in df['recommended_strategies']:
            all_strategies.extend(strats)
            
        from collections import Counter
        strategy_counts = Counter(all_strategies)
        
        for strategy, count in strategy_counts.most_common():
            pct = (count / len(df)) * 100
            print(f"  {strategy}: {pct:.1f}% of time")
            
    def plot_market_conditions(self, results: List[Dict]):
        """Plot market conditions over time"""
        
        if not results:
            return
            
        df = pd.DataFrame(results)
        
        fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
        
        # Price
        axes[0].plot(df['timestamp'], df['price'])
        axes[0].set_ylabel('Price ($)')
        axes[0].set_title(f'Bybit {self.symbol} Market Conditions')
        
        # Spread
        axes[1].plot(df['timestamp'], df['spread_bps'], color='orange')
        axes[1].set_ylabel('Spread (bps)')
        axes[1].axhline(y=5, color='r', linestyle='--', alpha=0.5)
        
        # Volatility
        axes[2].plot(df['timestamp'], df['volatility_1h'] * 100, color='red')
        axes[2].set_ylabel('1h Volatility (%)')
        axes[2].axhline(y=1, color='g', linestyle='--', alpha=0.5)
        
        # Book Imbalance
        axes[3].plot(df['timestamp'], df['book_imbalance'], color='green')
        axes[3].set_ylabel('Book Imbalance')
        axes[3].set_xlabel('Time')
        axes[3].axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'bybit_market_conditions_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
        plt.show()


def main():
    """Main execution"""
    
    # Initialize analyzer
    analyzer = BybitMarketAnalyzer(testnet=False)  # Use mainnet for real conditions
    
    print("Bybit BTC/USDT Perpetual Market Analysis")
    print("=" * 80)
    
    # Get current snapshot
    current = analyzer.fetch_current_market_state()
    if current:
        conditions = analyzer.analyze_execution_conditions(current)
        
        print(f"\nCurrent Market Snapshot:")
        print(f"Price: ${current['price']:,.2f}")
        print(f"Spread: {current['spread_bps']:.2f} bps")
        print(f"1h Volatility: {current['volatility_1h']*100:.2f}%")
        print(f"Book Imbalance: {current['book_imbalance']:.3f}")
        print(f"Funding Rate: {current.get('funding_rate', 0)*100:.3f}%")
        
        analyzer.display_conditions(conditions)
        
    # Optional: Monitor for windows
    user_input = input("\nMonitor market for execution windows? (y/n): ")
    if user_input.lower() == 'y':
        duration = int(input("Duration in minutes (default 60): ") or "60")
        interval = int(input("Sampling interval in seconds (default 30): ") or "30")
        
        results = analyzer.monitor_market_windows(duration, interval)
        
        # Plot results
        if len(results) > 5:
            analyzer.plot_market_conditions(results)
            

if __name__ == "__main__":
    main()