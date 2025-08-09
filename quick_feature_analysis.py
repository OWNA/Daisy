#!/usr/bin/env python3
"""Quick feature importance extraction using only LightGBM"""

import json
import sys
import os

# Add the current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import lightgbm as lgb
    
    # Load model
    model_path = './trading_bot_data/lgbm_model_BTC_USDTUSDT_l2_only.txt'
    features_path = './trading_bot_data/model_features_BTC_USDTUSDT_l2_only.json'
    
    print("Loading model...")
    model = lgb.Booster(model_file=model_path)
    
    print("Loading features...")
    with open(features_path, 'r') as f:
        features_data = json.load(f)
        features = features_data['trained_features']
    
    print(f"\nModel has {len(features)} features")
    
    # Get feature importance
    importance_split = model.feature_importance(importance_type='split')
    importance_gain = model.feature_importance(importance_type='gain')
    
    # Create importance dictionary
    feature_importance = []
    for i, feature in enumerate(features):
        feature_importance.append({
            'feature': feature,
            'importance_split': int(importance_split[i]),
            'importance_gain': float(importance_gain[i]),
            'importance_gain_pct': float(importance_gain[i] / importance_gain.sum() * 100)
        })
    
    # Sort by gain importance
    feature_importance.sort(key=lambda x: x['importance_gain'], reverse=True)
    
    # Categorize features
    def categorize_feature(feature):
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
    
    # Add categories
    for item in feature_importance:
        item['category'] = categorize_feature(item['feature'])
    
    # Print top 20 features
    print("\n=== TOP 20 MOST IMPORTANT FEATURES ===")
    for i, feat in enumerate(feature_importance[:20]):
        print(f"{i+1:2d}. {feat['feature']:30s} | {feat['importance_gain_pct']:6.2f}% | {feat['category']}")
    
    # Analyze new features
    print("\n=== NEW FEATURES ANALYSIS ===")
    
    # OFI features
    ofi_features = [f for f in feature_importance if f['category'] == 'Order Flow Imbalance']
    print(f"\n1. ORDER FLOW IMBALANCE ({len(ofi_features)} features):")
    total_ofi_importance = sum(f['importance_gain_pct'] for f in ofi_features)
    print(f"   Total importance: {total_ofi_importance:.2f}%")
    print("   Top 5 OFI features:")
    for i, feat in enumerate(ofi_features[:5]):
        rank = feature_importance.index(feat) + 1
        print(f"   - {feat['feature']}: {feat['importance_gain_pct']:.2f}% (rank #{rank})")
    
    # Book pressure features
    pressure_features = [f for f in feature_importance if f['category'] == 'Book Pressure']
    print(f"\n2. BOOK PRESSURE ({len(pressure_features)} features):")
    total_pressure_importance = sum(f['importance_gain_pct'] for f in pressure_features)
    print(f"   Total importance: {total_pressure_importance:.2f}%")
    for feat in pressure_features:
        rank = feature_importance.index(feat) + 1
        print(f"   - {feat['feature']}: {feat['importance_gain_pct']:.2f}% (rank #{rank})")
    
    # Stability features
    stability_features = [f for f in feature_importance if f['category'] == 'Stability Indicators']
    print(f"\n3. STABILITY INDICATORS ({len(stability_features)} features):")
    total_stability_importance = sum(f['importance_gain_pct'] for f in stability_features)
    print(f"   Total importance: {total_stability_importance:.2f}%")
    print("   Top 5 stability features:")
    for i, feat in enumerate(stability_features[:5]):
        rank = feature_importance.index(feat) + 1
        print(f"   - {feat['feature']}: {feat['importance_gain_pct']:.2f}% (rank #{rank})")
    
    # Volume concentration
    concentration_features = [f for f in feature_importance if f['category'] == 'Volume Concentration']
    print(f"\n4. VOLUME CONCENTRATION ({len(concentration_features)} features):")
    if concentration_features:
        total_conc_importance = sum(f['importance_gain_pct'] for f in concentration_features)
        print(f"   Total importance: {total_conc_importance:.2f}%")
        for feat in concentration_features:
            rank = feature_importance.index(feat) + 1
            print(f"   - {feat['feature']}: {feat['importance_gain_pct']:.2f}% (rank #{rank})")
    
    # Category summary
    print("\n=== FEATURE IMPORTANCE BY CATEGORY ===")
    category_importance = {}
    for feat in feature_importance:
        cat = feat['category']
        if cat not in category_importance:
            category_importance[cat] = {'total': 0, 'count': 0}
        category_importance[cat]['total'] += feat['importance_gain_pct']
        category_importance[cat]['count'] += 1
    
    # Sort categories by total importance
    sorted_categories = sorted(category_importance.items(), key=lambda x: x[1]['total'], reverse=True)
    for cat, data in sorted_categories:
        avg = data['total'] / data['count']
        print(f"{cat:25s} | Total: {data['total']:6.2f}% | Count: {data['count']:3d} | Avg: {avg:5.2f}%")
    
    # Feature reduction analysis
    print("\n=== FEATURE REDUCTION ANALYSIS ===")
    cumulative = 0
    features_for_80 = 0
    features_for_90 = 0
    features_for_95 = 0
    
    for i, feat in enumerate(feature_importance):
        cumulative += feat['importance_gain_pct']
        if cumulative <= 80 and features_for_80 == 0:
            features_for_80 = i + 1
        if cumulative <= 90 and features_for_90 == 0:
            features_for_90 = i + 1
        if cumulative <= 95 and features_for_95 == 0:
            features_for_95 = i + 1
    
    print(f"Features needed to capture:")
    print(f"  80% importance: {features_for_80} features (reduce by {len(features) - features_for_80})")
    print(f"  90% importance: {features_for_90} features (reduce by {len(features) - features_for_90})")
    print(f"  95% importance: {features_for_95} features (reduce by {len(features) - features_for_95})")
    
    # Low importance features
    low_importance = [f for f in feature_importance if f['importance_gain_pct'] < 0.1]
    print(f"\nFeatures with <0.1% importance: {len(low_importance)}")
    if low_importance:
        print("Bottom 5 features:")
        for feat in feature_importance[-5:]:
            print(f"  - {feat['feature']}: {feat['importance_gain_pct']:.3f}%")
    
    # Save to JSON for further analysis
    with open('feature_importance_data.json', 'w') as f:
        json.dump({
            'features': feature_importance,
            'summary': {
                'total_features': len(features),
                'features_for_80pct': features_for_80,
                'features_for_90pct': features_for_90,
                'features_for_95pct': features_for_95,
                'new_features_importance': {
                    'ofi': total_ofi_importance,
                    'pressure': total_pressure_importance,
                    'stability': total_stability_importance,
                    'concentration': sum(f['importance_gain_pct'] for f in concentration_features)
                }
            }
        }, f, indent=2)
    
    print("\nFeature importance data saved to: feature_importance_data.json")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()