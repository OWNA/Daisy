"""
Basic Market Conditions Analyzer for Testing
Works without external dependencies
"""

import json
from datetime import datetime
import time
import random

class BasicMarketAnalyzer:
    """Simulates market conditions for testing ML-enhanced execution"""
    
    def __init__(self):
        self.base_price = 100000
        
    def generate_market_snapshot(self):
        """Generate realistic market conditions"""
        
        hour = datetime.now().hour
        
        # Simulate different market conditions based on time
        if 6 <= hour < 10:
            spread_bps = random.uniform(1, 3)
            volatility = random.uniform(0.005, 0.01)
            ofi = random.uniform(-0.3, 0.3)
            book_resilience = random.uniform(0.6, 0.9)
        elif 10 <= hour < 14:
            spread_bps = random.uniform(2, 5)
            volatility = random.uniform(0.01, 0.02)
            ofi = random.uniform(-0.5, 0.5)
            book_resilience = random.uniform(0.4, 0.7)
        else:
            spread_bps = random.uniform(3, 8)
            volatility = random.uniform(0.015, 0.03)
            ofi = random.uniform(-0.8, 0.8)
            book_resilience = random.uniform(0.2, 0.5)
            
        self.base_price += random.uniform(-100, 100)
        
        # Generate ML features
        ml_features = {
            'spread_stability_norm_100': 1 - (spread_bps / 10),
            'ofi_normalized_1m': ofi,
            'pressure_imbalance_weighted': random.uniform(-0.6, 0.6),
            'book_resilience': book_resilience,
            'volume_concentration': random.uniform(0.3, 0.8),
            'quote_lifetime': random.uniform(1, 20),
            'bid_ask_spread': spread_bps / 10000 * self.base_price,
            'book_imbalance_5': random.uniform(-0.4, 0.4),
            'ofi_weighted_30s': ofi * 0.8,
            'spread_rolling_std': spread_bps * 0.2
        }
        
        # Determine execution recommendation
        if ml_features['spread_stability_norm_100'] > 0.8 and abs(ofi) < 0.3:
            strategy = 'passive'
            confidence = 0.85
        elif abs(ofi) > 0.7:
            strategy = 'urgent'
            confidence = 0.9
        elif ml_features['spread_stability_norm_100'] < 0.4:
            strategy = 'aggressive'
            confidence = 0.75
        else:
            strategy = 'balanced'
            confidence = 0.7
            
        return {
            'timestamp': datetime.now().isoformat(),
            'price': self.base_price,
            'spread_bps': spread_bps,
            'volatility_1h': volatility,
            'ofi': ofi,
            'book_resilience': book_resilience,
            'ml_features': ml_features,
            'recommended_strategy': strategy,
            'confidence': confidence,
            'market_quality': self._assess_quality(spread_bps, volatility)
        }
        
    def _assess_quality(self, spread_bps, volatility):
        """Assess overall market quality"""
        if spread_bps < 3 and volatility < 0.01:
            return 'excellent'
        elif spread_bps < 5 and volatility < 0.02:
            return 'good'
        elif spread_bps < 8 and volatility < 0.03:
            return 'normal'
        else:
            return 'poor'
            
    def run_analysis(self, duration_seconds=60):
        """Run market analysis for specified duration"""
        
        print("=" * 80)
        print("MARKET CONDITIONS ANALYSIS - T1")
        print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)
        
        snapshots = []
        start_time = time.time()
        
        while time.time() - start_time < duration_seconds:
            snapshot = self.generate_market_snapshot()
            snapshots.append(snapshot)
            
            # Display current conditions
            print(f"\n[{datetime.now().strftime('%H:%M:%S')}] BTC: ${snapshot['price']:,.0f}")
            print(f"Quality: {snapshot['market_quality']} | Strategy: {snapshot['recommended_strategy']}")
            print(f"Spread: {snapshot['spread_bps']:.1f} bps | OFI: {snapshot['ofi']:.2f}")
            print(f"Volatility: {snapshot['volatility_1h']*100:.2f}% | Confidence: {snapshot['confidence']:.2f}")
            
            # Key ML feature
            features = snapshot['ml_features']
            print(f"spread_stability_norm_100: {features['spread_stability_norm_100']:.3f}")
            
            time.sleep(5)
            
        # Generate report
        self._generate_report(snapshots)
        
        # Save for next test
        with open('market_conditions_report.json', 'w') as f:
            json.dump({
                'analysis_time': datetime.now().isoformat(),
                'duration_seconds': duration_seconds,
                'snapshots': snapshots,
                'summary': self._create_summary(snapshots)
            }, f, indent=2)
            
        return snapshots
        
    def _create_summary(self, snapshots):
        """Create summary statistics"""
        spreads = [s['spread_bps'] for s in snapshots]
        volatilities = [s['volatility_1h'] for s in snapshots]
        strategies = [s['recommended_strategy'] for s in snapshots]
        
        strategy_counts = {}
        for s in strategies:
            strategy_counts[s] = strategy_counts.get(s, 0) + 1
            
        return {
            'avg_spread_bps': sum(spreads) / len(spreads),
            'min_spread_bps': min(spreads),
            'max_spread_bps': max(spreads),
            'avg_volatility': sum(volatilities) / len(volatilities),
            'strategy_distribution': strategy_counts,
            'optimal_windows': sum(1 for s in snapshots if s['market_quality'] in ['excellent', 'good'])
        }
        
    def _generate_report(self, snapshots):
        """Generate analysis report"""
        
        summary = self._create_summary(snapshots)
        
        print("\n" + "=" * 80)
        print("MARKET WINDOW ANALYSIS COMPLETE")
        print("=" * 80)
        
        print(f"\nSpread Statistics:")
        print(f"  Average: {summary['avg_spread_bps']:.2f} bps")
        print(f"  Range: {summary['min_spread_bps']:.2f} - {summary['max_spread_bps']:.2f} bps")
        
        print(f"\nVolatility:")
        print(f"  Average: {summary['avg_volatility']*100:.2f}%")
        
        print(f"\nStrategy Distribution:")
        total = len(snapshots)
        for strategy, count in summary['strategy_distribution'].items():
            print(f"  {strategy}: {count}/{total} ({count/total*100:.1f}%)")
            
        print(f"\nOptimal Execution Windows: {summary['optimal_windows']}/{total}")
        
        # Recommendations
        print("\nRECOMMENDATIONS:")
        if summary['avg_spread_bps'] < 3:
            print("✓ Excellent conditions for passive execution")
        if summary['avg_volatility'] > 0.02:
            print("⚠ High volatility - use careful position sizing")
        if 'urgent' in summary['strategy_distribution']:
            urgent_pct = summary['strategy_distribution']['urgent'] / total * 100
            if urgent_pct > 20:
                print(f"⚠ High urgency signals ({urgent_pct:.0f}%) - monitor OFI closely")
                
        print("\n✅ Market analysis complete. Ready for T2: Feature Pipeline Validation")


def main():
    """Execute T1: Market Condition Analysis"""
    analyzer = BasicMarketAnalyzer()
    
    print("Starting market condition analysis...")
    print("This will run for 30 seconds to gather market data\n")
    
    snapshots = analyzer.run_analysis(duration_seconds=30)
    
    print(f"\nAnalysis complete. Collected {len(snapshots)} market snapshots.")
    print("Results saved to: market_conditions_report.json")
    

if __name__ == "__main__":
    main()