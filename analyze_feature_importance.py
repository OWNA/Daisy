#!/usr/bin/env python3
"""
Feature Importance Analysis for L2-Only Trading Model
Analyzes the 118 features and provides insights on:
1. Most predictive new features (OFI, book pressure, stability)
2. Feature importance rankings
3. Recommendations for optimization
"""

import json
import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple

class FeatureImportanceAnalyzer:
    def __init__(self, model_path: str, features_path: str):
        """Initialize the analyzer with model and feature paths."""
        self.model_path = model_path
        self.features_path = features_path
        self.model = None
        self.features = None
        self.importance_df = None
        
    def load_model_and_features(self):
        """Load the trained model and feature list."""
        # Load model
        self.model = lgb.Booster(model_file=self.model_path)
        
        # Load features
        with open(self.features_path, 'r') as f:
            features_data = json.load(f)
            self.features = features_data['trained_features']
            
        print(f"Loaded model with {len(self.features)} features")
        
    def calculate_feature_importance(self):
        """Calculate and organize feature importance."""
        # Get feature importance (split-based)
        importance_split = self.model.feature_importance(importance_type='split')
        
        # Get feature importance (gain-based)
        importance_gain = self.model.feature_importance(importance_type='gain')
        
        # Create DataFrame
        self.importance_df = pd.DataFrame({
            'feature': self.features,
            'importance_split': importance_split,
            'importance_gain': importance_gain,
            'importance_split_pct': importance_split / importance_split.sum() * 100,
            'importance_gain_pct': importance_gain / importance_gain.sum() * 100
        })
        
        # Add feature categories
        self.importance_df['category'] = self.importance_df['feature'].apply(self._categorize_feature)
        
        # Sort by gain importance
        self.importance_df = self.importance_df.sort_values('importance_gain', ascending=False)
        
    def _categorize_feature(self, feature: str) -> str:
        """Categorize features into groups."""
        if 'ofi_' in feature:
            return 'Order Flow Imbalance'
        elif 'pressure' in feature:
            return 'Book Pressure'
        elif 'stability' in feature or 'lifetime' in feature or 'resilience' in feature:
            return 'Stability Indicators'
        elif 'bid_price_' in feature or 'ask_price_' in feature:
            return 'Raw Prices'
        elif 'bid_size_' in feature or 'ask_size_' in feature:
            return 'Raw Sizes'
        elif 'imbalance' in feature:
            return 'Order Book Imbalance'
        elif 'spread' in feature:
            return 'Spread Features'
        elif 'impact' in feature:
            return 'Price Impact'
        elif 'volatility' in feature:
            return 'Volatility'
        elif 'volume' in feature:
            return 'Volume Features'
        elif 'concentration' in feature:
            return 'Volume Concentration'
        else:
            return 'Other'
    
    def analyze_new_features(self):
        """Analyze the 34 new features added to the model."""
        # Define the new feature categories
        new_feature_categories = [
            'Order Flow Imbalance',
            'Book Pressure', 
            'Stability Indicators',
            'Volume Concentration'
        ]
        
        # Filter new features
        new_features_df = self.importance_df[
            self.importance_df['category'].isin(new_feature_categories)
        ].copy()
        
        print("\n=== NEW FEATURES ANALYSIS (34 features added) ===\n")
        
        # 1. Order Flow Imbalance Features
        ofi_features = new_features_df[new_features_df['category'] == 'Order Flow Imbalance']
        print("1. ORDER FLOW IMBALANCE FEATURES:")
        print(f"   Total OFI features: {len(ofi_features)}")
        print(f"   Average importance (gain): {ofi_features['importance_gain_pct'].mean():.2f}%")
        print("   Top 5 OFI features:")
        for idx, row in ofi_features.head(5).iterrows():
            print(f"   - {row['feature']}: {row['importance_gain_pct']:.2f}% (rank #{self.importance_df.index.get_loc(idx)+1})")
        
        # 2. Book Pressure Features
        pressure_features = new_features_df[new_features_df['category'] == 'Book Pressure']
        print("\n2. BOOK PRESSURE FEATURES:")
        print(f"   Total pressure features: {len(pressure_features)}")
        print(f"   Average importance (gain): {pressure_features['importance_gain_pct'].mean():.2f}%")
        print("   All pressure features:")
        for idx, row in pressure_features.iterrows():
            print(f"   - {row['feature']}: {row['importance_gain_pct']:.2f}% (rank #{self.importance_df.index.get_loc(idx)+1})")
        
        # 3. Stability Indicators
        stability_features = new_features_df[new_features_df['category'] == 'Stability Indicators']
        print("\n3. STABILITY INDICATORS:")
        print(f"   Total stability features: {len(stability_features)}")
        print(f"   Average importance (gain): {stability_features['importance_gain_pct'].mean():.2f}%")
        print("   Top 5 stability features:")
        for idx, row in stability_features.head(5).iterrows():
            print(f"   - {row['feature']}: {row['importance_gain_pct']:.2f}% (rank #{self.importance_df.index.get_loc(idx)+1})")
        
        # 4. Volume Concentration
        concentration_features = new_features_df[new_features_df['category'] == 'Volume Concentration']
        print("\n4. VOLUME CONCENTRATION FEATURES:")
        print(f"   Total concentration features: {len(concentration_features)}")
        if len(concentration_features) > 0:
            print(f"   Average importance (gain): {concentration_features['importance_gain_pct'].mean():.2f}%")
            for idx, row in concentration_features.iterrows():
                print(f"   - {row['feature']}: {row['importance_gain_pct']:.2f}% (rank #{self.importance_df.index.get_loc(idx)+1})")
    
    def analyze_overall_importance(self):
        """Analyze overall feature importance patterns."""
        print("\n=== OVERALL FEATURE IMPORTANCE ANALYSIS ===\n")
        
        # Top 20 features overall
        print("TOP 20 MOST IMPORTANT FEATURES:")
        for i, (idx, row) in enumerate(self.importance_df.head(20).iterrows()):
            print(f"{i+1:2d}. {row['feature']:30s} | {row['importance_gain_pct']:6.2f}% | {row['category']}")
        
        # Category summary
        category_summary = self.importance_df.groupby('category').agg({
            'importance_gain_pct': ['sum', 'mean', 'count']
        }).round(2)
        category_summary.columns = ['Total %', 'Average %', 'Count']
        category_summary = category_summary.sort_values('Total %', ascending=False)
        
        print("\n\nFEATURE CATEGORY IMPORTANCE:")
        print(category_summary)
        
        # Features with very low importance (candidates for removal)
        low_importance = self.importance_df[self.importance_df['importance_gain_pct'] < 0.1]
        print(f"\n\nLOW IMPORTANCE FEATURES (< 0.1% gain): {len(low_importance)} features")
        if len(low_importance) > 0:
            print("Bottom 10 features:")
            for idx, row in low_importance.tail(10).iterrows():
                print(f"   - {row['feature']}: {row['importance_gain_pct']:.3f}%")
    
    def generate_recommendations(self):
        """Generate specific recommendations for model optimization."""
        print("\n=== RECOMMENDATIONS FOR MODEL OPTIMIZATION ===\n")
        
        # 1. Feature reduction analysis
        cumulative_importance = self.importance_df['importance_gain_pct'].cumsum()
        features_for_95 = (cumulative_importance <= 95).sum()
        features_for_90 = (cumulative_importance <= 90).sum()
        features_for_80 = (cumulative_importance <= 80).sum()
        
        print("1. FEATURE REDUCTION OPPORTUNITIES:")
        print(f"   - Top {features_for_80} features capture 80% of importance")
        print(f"   - Top {features_for_90} features capture 90% of importance")
        print(f"   - Top {features_for_95} features capture 95% of importance")
        print(f"   - Potential reduction: {len(self.features) - features_for_95} features (keeping 95% importance)")
        
        # 2. OFI effectiveness analysis
        ofi_features = self.importance_df[self.importance_df['category'] == 'Order Flow Imbalance']
        ofi_total_importance = ofi_features['importance_gain_pct'].sum()
        print(f"\n2. ORDER FLOW IMBALANCE EFFECTIVENESS:")
        print(f"   - Total OFI contribution: {ofi_total_importance:.2f}%")
        print(f"   - Average OFI feature importance: {ofi_features['importance_gain_pct'].mean():.2f}%")
        
        # Check which time windows are most effective
        ofi_by_window = {}
        for window in ['10s', '30s', '1m', '5m']:
            window_features = ofi_features[ofi_features['feature'].str.contains(f'_{window}')]
            if len(window_features) > 0:
                ofi_by_window[window] = window_features['importance_gain_pct'].sum()
        
        if ofi_by_window:
            best_window = max(ofi_by_window, key=ofi_by_window.get)
            print(f"   - Most effective OFI window: {best_window} ({ofi_by_window[best_window]:.2f}% total importance)")
            print("   - OFI importance by window:")
            for window, importance in sorted(ofi_by_window.items(), key=lambda x: x[1], reverse=True):
                print(f"     * {window}: {importance:.2f}%")
        
        # 3. Optimal feature subset for real-time trading
        print("\n3. OPTIMAL FEATURE SUBSET FOR REAL-TIME TRADING:")
        
        # Select features that together capture 90% importance
        optimal_features = self.importance_df.head(features_for_90)
        
        # Group by category
        optimal_by_category = optimal_features.groupby('category').size()
        print(f"   - Recommended feature count: {features_for_90} (90% importance)")
        print("   - Feature distribution:")
        for category, count in optimal_by_category.items():
            print(f"     * {category}: {count} features")
        
        # 4. Specific feature engineering improvements
        print("\n4. FEATURE ENGINEERING IMPROVEMENTS:")
        
        # Check if certain feature types are consistently important
        if ofi_total_importance > 5:
            print("   - OFI features are effective (>5% importance) - consider:")
            print("     * Adding trade-flow toxicity indicators (VPIN)")
            print("     * Implementing Kyle's lambda for price impact")
            print("     * Adding order flow persistence metrics")
        
        stability_features = self.importance_df[self.importance_df['category'] == 'Stability Indicators']
        if stability_features['importance_gain_pct'].sum() > 5:
            print("   - Stability features are effective - consider:")
            print("     * Adding regime detection features")
            print("     * Implementing book shape change detection")
            print("     * Adding microstructure noise estimation")
        
        # 5. Features to drop
        print("\n5. FEATURES TO CONSIDER DROPPING:")
        drop_candidates = self.importance_df[self.importance_df['importance_gain_pct'] < 0.05]
        if len(drop_candidates) > 0:
            print(f"   - {len(drop_candidates)} features with <0.05% importance")
            # Group by category
            drop_by_category = drop_candidates.groupby('category').size()
            for category, count in drop_by_category.items():
                print(f"     * {category}: {count} features")
    
    def save_importance_report(self, output_path: str):
        """Save detailed importance report to CSV."""
        self.importance_df.to_csv(output_path, index=False)
        print(f"\nDetailed feature importance saved to: {output_path}")
    
    def plot_importance_analysis(self):
        """Create visualization plots for feature importance."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Top 20 features bar plot
        ax1 = axes[0, 0]
        top_20 = self.importance_df.head(20)
        ax1.barh(range(len(top_20)), top_20['importance_gain_pct'])
        ax1.set_yticks(range(len(top_20)))
        ax1.set_yticklabels(top_20['feature'], fontsize=8)
        ax1.set_xlabel('Importance (%)')
        ax1.set_title('Top 20 Most Important Features')
        ax1.invert_yaxis()
        
        # 2. Category importance pie chart
        ax2 = axes[0, 1]
        category_importance = self.importance_df.groupby('category')['importance_gain_pct'].sum()
        ax2.pie(category_importance.values, labels=category_importance.index, autopct='%1.1f%%')
        ax2.set_title('Feature Importance by Category')
        
        # 3. Cumulative importance plot
        ax3 = axes[1, 0]
        cumulative = self.importance_df['importance_gain_pct'].cumsum()
        ax3.plot(range(len(cumulative)), cumulative)
        ax3.axhline(y=80, color='r', linestyle='--', label='80%')
        ax3.axhline(y=90, color='g', linestyle='--', label='90%')
        ax3.axhline(y=95, color='b', linestyle='--', label='95%')
        ax3.set_xlabel('Number of Features')
        ax3.set_ylabel('Cumulative Importance (%)')
        ax3.set_title('Cumulative Feature Importance')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. New features importance distribution
        ax4 = axes[1, 1]
        new_categories = ['Order Flow Imbalance', 'Book Pressure', 'Stability Indicators', 'Volume Concentration']
        new_features = self.importance_df[self.importance_df['category'].isin(new_categories)]
        
        if len(new_features) > 0:
            new_category_avg = new_features.groupby('category')['importance_gain_pct'].mean()
            ax4.bar(range(len(new_category_avg)), new_category_avg.values)
            ax4.set_xticks(range(len(new_category_avg)))
            ax4.set_xticklabels(new_category_avg.index, rotation=45, ha='right')
            ax4.set_ylabel('Average Importance (%)')
            ax4.set_title('Average Importance of New Feature Categories')
        
        plt.tight_layout()
        plt.savefig('feature_importance_analysis.png', dpi=300, bbox_inches='tight')
        print("\nFeature importance plots saved to: feature_importance_analysis.png")
        plt.close()


def main():
    # Paths
    base_dir = Path('./trading_bot_data')
    model_path = base_dir / 'lgbm_model_BTC_USDTUSDT_l2_only.txt'
    features_path = base_dir / 'model_features_BTC_USDTUSDT_l2_only.json'
    
    # Initialize analyzer
    analyzer = FeatureImportanceAnalyzer(str(model_path), str(features_path))
    
    # Load model and features
    analyzer.load_model_and_features()
    
    # Calculate importance
    analyzer.calculate_feature_importance()
    
    # Run analyses
    analyzer.analyze_new_features()
    analyzer.analyze_overall_importance()
    analyzer.generate_recommendations()
    
    # Save report
    analyzer.save_importance_report('feature_importance_report.csv')
    
    # Create visualizations
    analyzer.plot_importance_analysis()
    
    print("\n=== ANALYSIS COMPLETE ===")


if __name__ == "__main__":
    main()