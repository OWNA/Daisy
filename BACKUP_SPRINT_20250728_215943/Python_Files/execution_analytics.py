"""
Execution Analytics Module
Tracks execution performance and provides optimization feedback
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import json
import logging
from collections import deque

logger = logging.getLogger(__name__)


class ExecutionAnalytics:
    """
    Comprehensive execution analytics system that:
    - Tracks slippage, fill rates, and costs
    - Identifies patterns in execution quality
    - Provides adaptive parameter recommendations
    - Monitors exchange-specific behaviors
    """
    
    def __init__(self, config: dict):
        """
        Initialize execution analytics
        
        Args:
            config: Configuration dictionary
        """
        # Storage settings
        self.max_history_size = config.get('max_history_size', 1000)
        self.analysis_window = config.get('analysis_window_hours', 24)
        
        # Performance thresholds
        self.target_fill_rate = config.get('target_fill_rate', 0.95)
        self.max_acceptable_slippage = config.get('max_acceptable_slippage', 0.002)  # 20 bps
        self.cost_warning_threshold = config.get('cost_warning_threshold', 0.001)   # 10 bps
        
        # Execution history
        self.execution_history = deque(maxlen=self.max_history_size)
        self.hourly_stats = {}
        self.strategy_performance = {}
        
        # Adaptive parameters
        self.optimization_params = {
            'spread_offset_bps': 1,
            'max_order_pct': 0.2,
            'post_only_ratio': 0.7,
            'slice_count': 3
        }
        
        # Pattern detection
        self.slippage_patterns = {}
        self.liquidity_patterns = {}
        
    def record_execution(self, execution_result: Dict) -> None:
        """
        Record execution result for analysis
        
        Args:
            execution_result: Result from smart order executor
        """
        try:
            # Add metadata
            execution_record = {
                'timestamp': datetime.now(),
                'hour': datetime.now().hour,
                'weekday': datetime.now().weekday(),
                **execution_result
            }
            
            # Store in history
            self.execution_history.append(execution_record)
            
            # Update real-time stats
            self._update_realtime_stats(execution_record)
            
            # Check for immediate issues
            self._check_execution_alerts(execution_record)
            
        except Exception as e:
            logger.error(f"Error recording execution: {e}")
    
    def _update_realtime_stats(self, execution: Dict) -> None:
        """Update real-time statistics"""
        hour = execution['hour']
        
        if hour not in self.hourly_stats:
            self.hourly_stats[hour] = {
                'count': 0,
                'total_volume': 0,
                'avg_slippage': 0,
                'avg_fill_rate': 0,
                'total_costs': 0
            }
        
        stats = self.hourly_stats[hour]
        stats['count'] += 1
        stats['total_volume'] += execution.get('amount_filled_usd', 0)
        
        # Update moving averages
        alpha = 0.1  # Exponential decay factor
        stats['avg_slippage'] = (1 - alpha) * stats['avg_slippage'] + alpha * execution.get('slippage_pct', 0)
        stats['avg_fill_rate'] = (1 - alpha) * stats['avg_fill_rate'] + alpha * execution.get('fill_rate', 0)
        stats['total_costs'] += execution.get('total_fees_usd', 0)
    
    def _check_execution_alerts(self, execution: Dict) -> None:
        """Check for execution issues requiring immediate attention"""
        # High slippage alert
        if abs(execution.get('slippage_pct', 0)) > self.max_acceptable_slippage:
            logger.warning(f"High slippage detected: {execution['slippage_pct']*100:.3f}% "
                         f"on ${execution['amount_filled_usd']:.2f} order")
        
        # Low fill rate alert
        if execution.get('fill_rate', 1) < self.target_fill_rate * 0.8:
            logger.warning(f"Low fill rate: {execution['fill_rate']*100:.1f}% "
                         f"(target: {self.target_fill_rate*100:.1f}%)")
        
        # High cost alert
        cost_ratio = abs(execution.get('total_fees_usd', 0)) / execution.get('amount_filled_usd', 1)
        if cost_ratio > self.cost_warning_threshold:
            logger.warning(f"High execution cost: {cost_ratio*100:.3f}% of order value")
    
    def analyze_execution_performance(self, lookback_hours: Optional[int] = None) -> Dict:
        """
        Analyze execution performance over specified period
        
        Args:
            lookback_hours: Hours to look back (default: analysis_window)
            
        Returns:
            Comprehensive performance analysis
        """
        if not self.execution_history:
            return {'status': 'no_data'}
        
        lookback_hours = lookback_hours or self.analysis_window
        cutoff_time = datetime.now() - timedelta(hours=lookback_hours)
        
        # Filter recent executions
        recent_executions = [
            e for e in self.execution_history 
            if e['timestamp'] > cutoff_time
        ]
        
        if not recent_executions:
            return {'status': 'no_recent_data'}
        
        # Convert to DataFrame for analysis
        df = pd.DataFrame(recent_executions)
        
        # Overall statistics
        total_volume = df['amount_filled_usd'].sum()
        avg_slippage = df['slippage_pct'].mean()
        avg_fill_rate = df['fill_rate'].mean()
        total_fees = df['total_fees_usd'].sum()
        
        # Slippage analysis
        slippage_by_size = self._analyze_slippage_by_size(df)
        slippage_by_time = self._analyze_slippage_by_time(df)
        
        # Strategy performance
        strategy_stats = df.groupby('strategy_used').agg({
            'slippage_pct': ['mean', 'std'],
            'fill_rate': 'mean',
            'execution_time_seconds': 'mean',
            'total_fees_usd': 'sum'
        }).to_dict()
        
        # Cost analysis
        maker_taker_ratio = self._calculate_maker_taker_ratio(df)
        
        # Optimization recommendations
        recommendations = self._generate_optimization_recommendations(df)
        
        return {
            'period_hours': lookback_hours,
            'total_executions': len(recent_executions),
            'total_volume_usd': total_volume,
            'performance': {
                'avg_slippage_pct': avg_slippage * 100,
                'avg_fill_rate': avg_fill_rate,
                'total_fees_usd': total_fees,
                'fee_pct_of_volume': (total_fees / total_volume * 100) if total_volume > 0 else 0
            },
            'slippage_analysis': {
                'by_size': slippage_by_size,
                'by_time': slippage_by_time,
                'worst_executions': self._get_worst_executions(df, 5)
            },
            'strategy_performance': strategy_stats,
            'cost_analysis': {
                'maker_taker_ratio': maker_taker_ratio,
                'avg_fee_per_trade': total_fees / len(recent_executions),
                'fee_optimization_potential': self._estimate_fee_savings(df)
            },
            'recommendations': recommendations
        }
    
    def _analyze_slippage_by_size(self, df: pd.DataFrame) -> Dict:
        """Analyze how slippage varies with order size"""
        # Create size buckets
        df['size_bucket'] = pd.qcut(df['amount_requested_usd'], 
                                   q=[0, 0.25, 0.5, 0.75, 1.0], 
                                   labels=['small', 'medium', 'large', 'xlarge'])
        
        slippage_by_size = df.groupby('size_bucket')['slippage_pct'].agg(['mean', 'std', 'count'])
        
        return slippage_by_size.to_dict()
    
    def _analyze_slippage_by_time(self, df: pd.DataFrame) -> Dict:
        """Analyze slippage patterns by time of day"""
        hourly_slippage = df.groupby('hour')['slippage_pct'].agg(['mean', 'std', 'count'])
        
        # Identify high slippage hours
        high_slippage_hours = hourly_slippage[
            hourly_slippage['mean'] > hourly_slippage['mean'].mean() + hourly_slippage['std'].mean()
        ].index.tolist()
        
        return {
            'hourly_stats': hourly_slippage.to_dict(),
            'high_slippage_hours': high_slippage_hours,
            'best_execution_hours': hourly_slippage.nsmallest(3, 'mean').index.tolist()
        }
    
    def _calculate_maker_taker_ratio(self, df: pd.DataFrame) -> float:
        """Estimate maker vs taker order ratio"""
        # Estimate based on execution time and strategy
        quick_executions = df[df['execution_time_seconds'] < 5]
        maker_ratio = 1 - (len(quick_executions) / len(df))
        return maker_ratio
    
    def _get_worst_executions(self, df: pd.DataFrame, n: int = 5) -> List[Dict]:
        """Get worst performing executions"""
        worst = df.nlargest(n, 'slippage_pct')[
            ['timestamp', 'amount_requested_usd', 'slippage_pct', 'strategy_used']
        ]
        return worst.to_dict('records')
    
    def _estimate_fee_savings(self, df: pd.DataFrame) -> float:
        """Estimate potential fee savings from optimization"""
        current_fee_rate = df['total_fees_usd'].sum() / df['amount_filled_usd'].sum()
        optimal_fee_rate = -0.00025  # Bybit maker rebate
        
        potential_savings = (current_fee_rate - optimal_fee_rate) * df['amount_filled_usd'].sum()
        return max(potential_savings, 0)
    
    def _generate_optimization_recommendations(self, df: pd.DataFrame) -> List[Dict]:
        """Generate specific optimization recommendations"""
        recommendations = []
        
        # Slippage recommendations
        avg_slippage = df['slippage_pct'].mean()
        if avg_slippage > self.max_acceptable_slippage:
            recommendations.append({
                'type': 'slippage',
                'severity': 'high',
                'message': f'Average slippage {avg_slippage*100:.3f}% exceeds target',
                'action': 'Increase passive order ratio and reduce order sizes',
                'params': {
                    'suggested_max_order_pct': self.optimization_params['max_order_pct'] * 0.8,
                    'suggested_post_only_ratio': min(self.optimization_params['post_only_ratio'] * 1.2, 0.9)
                }
            })
        
        # Fill rate recommendations
        avg_fill_rate = df['fill_rate'].mean()
        if avg_fill_rate < self.target_fill_rate:
            recommendations.append({
                'type': 'fill_rate',
                'severity': 'medium',
                'message': f'Fill rate {avg_fill_rate*100:.1f}% below target',
                'action': 'Adjust spread crossing parameters',
                'params': {
                    'suggested_spread_offset': self.optimization_params['spread_offset_bps'] * 1.5
                }
            })
        
        # Time-based recommendations
        hourly_perf = df.groupby('hour')['slippage_pct'].mean()
        if hourly_perf.std() > 0.001:  # Significant variation
            best_hours = hourly_perf.nsmallest(8).index.tolist()
            recommendations.append({
                'type': 'timing',
                'severity': 'low',
                'message': 'Execution quality varies significantly by hour',
                'action': f'Consider concentrating trades during hours: {best_hours}',
                'params': {'best_execution_hours': best_hours}
            })
        
        # Size recommendations
        size_impact = df.groupby(pd.qcut(df['amount_requested_usd'], 4))['slippage_pct'].mean()
        if size_impact.max() - size_impact.min() > 0.001:
            recommendations.append({
                'type': 'sizing',
                'severity': 'medium',
                'message': 'Large orders experiencing higher slippage',
                'action': 'Increase order splitting for large trades',
                'params': {
                    'suggested_slice_count': min(self.optimization_params['slice_count'] + 1, 5)
                }
            })
        
        return recommendations
    
    def get_adaptive_parameters(self) -> Dict:
        """
        Get current adaptive parameters based on recent performance
        
        Returns:
            Dictionary of optimized execution parameters
        """
        if len(self.execution_history) < 10:
            return self.optimization_params
        
        # Analyze recent performance
        recent_analysis = self.analyze_execution_performance(lookback_hours=4)
        
        # Update parameters based on performance
        if recent_analysis.get('status') != 'no_recent_data':
            perf = recent_analysis.get('performance', {})
            
            # Adjust spread offset based on fill rate
            if perf.get('avg_fill_rate', 1) < self.target_fill_rate:
                self.optimization_params['spread_offset_bps'] *= 1.1
            elif perf.get('avg_fill_rate', 0) > 0.98:
                self.optimization_params['spread_offset_bps'] *= 0.95
            
            # Adjust order size based on slippage
            if perf.get('avg_slippage_pct', 0) > self.max_acceptable_slippage * 100:
                self.optimization_params['max_order_pct'] *= 0.9
                self.optimization_params['slice_count'] = min(self.optimization_params['slice_count'] + 1, 5)
            
            # Adjust maker ratio based on fees
            if perf.get('fee_pct_of_volume', 0) > 0.05:  # More than 5 bps
                self.optimization_params['post_only_ratio'] = min(
                    self.optimization_params['post_only_ratio'] * 1.1, 0.9
                )
        
        # Ensure parameters stay within reasonable bounds
        self.optimization_params['spread_offset_bps'] = max(0.5, min(5, self.optimization_params['spread_offset_bps']))
        self.optimization_params['max_order_pct'] = max(0.1, min(0.5, self.optimization_params['max_order_pct']))
        
        return self.optimization_params.copy()
    
    def export_performance_report(self, filepath: str) -> bool:
        """
        Export detailed performance report
        
        Args:
            filepath: Path to save report
            
        Returns:
            Success status
        """
        try:
            # Generate comprehensive analysis
            report = {
                'generated_at': datetime.now().isoformat(),
                'summary': self.analyze_execution_performance(),
                'hourly_patterns': self.hourly_stats,
                'adaptive_parameters': self.get_adaptive_parameters(),
                'recent_executions': [
                    {k: v for k, v in e.items() if k != 'executed_orders'}
                    for e in list(self.execution_history)[-20:]
                ]
            }
            
            # Save report
            with open(filepath, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            logger.info(f"Performance report exported to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting report: {e}")
            return False