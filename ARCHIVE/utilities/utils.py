#!/usr/bin/env python3
"""
L2-Only Utility functions for the trading bot project
Updated for L2-only strategy with Level 2 order book data
"""

import os
import json
import pandas as pd
import sqlite3
import lightgbm as lgb
from featureengineer import FeatureEngineer


def load_model_and_features(model_path=None, features_path=None, symbol='BTC/USDT:USDT'):
    """
    Load trained L2-only model and feature configuration
    
    Args:
        model_path: Path to model file (auto-detected if None)
        features_path: Path to features file (auto-detected if None)
        symbol: Trading symbol for auto-detection
    
    Returns:
        tuple: (model, trained_features)
    """
    # Auto-detect L2-only model paths if not provided
    if model_path is None or features_path is None:
        safe_symbol = symbol.replace('/', '_').replace(':', '')
        
        # Try L2-only model first
        l2_model_path = f'lgbm_model_{safe_symbol}_l2_only.txt'
        l2_features_path = f'model_features_{safe_symbol}_l2_only.json'
        
        if os.path.exists(l2_model_path) and os.path.exists(l2_features_path):
            model_path = l2_model_path
            features_path = l2_features_path
        else:
            # Fallback to original model
            model_path = f'lgbm_model_{safe_symbol}_1m.txt'
            features_path = f'model_features_{safe_symbol}_1m.json'
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    if not os.path.exists(features_path):
        raise FileNotFoundError(f"Features config not found: {features_path}")
    
    # Load model
    model = lgb.Booster(model_file=model_path)
    
    # Load features
    with open(features_path, 'r') as f:
        feature_info = json.load(f)
        trained_features = feature_info.get('trained_features', [])
    
    return model, trained_features


def load_l2_data_from_db(db_path='./trading_bot.db', limit=50000, table='l2_training_data'):
    """
    Load L2 data from database
    
    Args:
        db_path: Path to SQLite database
        limit: Maximum number of rows to load
        table: Database table name
    
    Returns:
        pd.DataFrame: L2 data with timestamp as datetime
    """
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database not found: {db_path}")
    
    with sqlite3.connect(db_path) as conn:
        query = f"""
        SELECT * FROM {table} 
        ORDER BY timestamp 
        LIMIT {limit}
        """
        df = pd.read_sql_query(query, conn)
    
    if df.empty:
        raise ValueError(f"No data found in table: {table}")
    
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df


def generate_l2_features(df, config=None):
    """
    Generate L2-only features using FeatureEngineer
    
    Args:
        df: Input L2 dataframe
        config: L2-only feature engineering config (optional)
    
    Returns:
        pd.DataFrame: Dataframe with L2 features
    """
    if config is None:
        config = {
            'l2_only_mode': True,
            'use_l2_features': True,
            'use_l2_features_for_training': True,
            'feature_window': 100,
            'use_hht_features': True,
            'hht_emd_noise_width': 0.01,
            'ohlcv_base_features': [],  # Disabled for L2-only
            'ta_features': [],          # Disabled for L2-only
            'l2_features': [
                'bid_ask_spread', 'bid_ask_spread_pct', 'weighted_mid_price',
                'microprice', 'order_book_imbalance_2', 'order_book_imbalance_3',
                'order_book_imbalance_5', 'total_bid_volume_2', 'total_ask_volume_2',
                'total_bid_volume_3', 'total_ask_volume_3', 'price_impact_buy',
                'price_impact_sell', 'price_impact_1', 'price_impact_5',
                'price_impact_10', 'bid_slope', 'ask_slope', 'l2_volatility_1min',
                'l2_volatility_5min', 'realized_volatility', 'order_flow_imbalance',
                'trade_intensity', 'effective_spread'
            ]
        }
    
    fe = FeatureEngineer(config)
    return fe.generate_all_features(df)


def make_l2_predictions(df_features, model, trained_features, threshold=0.15):
    """
    Generate L2-based predictions and trading signals
    
    Args:
        df_features: Dataframe with L2 features
        model: Trained LightGBM model
        trained_features: List of feature names used in training
        threshold: L2-adjusted signal threshold (default: 0.15)
    
    Returns:
        tuple: (predictions, signals)
    """
    predictions = []
    signals = []
    feature_window = 100  # L2-specific window
    
    for i in range(len(df_features)):
        if i < feature_window:  # Skip initial rows for L2 data
            predictions.append(0)
            signals.append(0)
        else:
            try:
                # Get available features (handle missing features gracefully)
                available_features = [f for f in trained_features 
                                    if f in df_features.columns]
                
                if len(available_features) < len(trained_features) * 0.8:
                    # Need at least 80% of features for reliable prediction
                    predictions.append(0)
                    signals.append(0)
                    continue
                
                X = df_features.iloc[i][available_features].values.reshape(1, -1)
                pred = model.predict(X, num_iteration=model.best_iteration)[0]
                predictions.append(pred)
                
                # Generate L2-aware signal
                if pred > threshold:
                    signals.append(1)
                elif pred < -threshold:
                    signals.append(-1)
                else:
                    signals.append(0)
            except Exception:
                predictions.append(0)
                signals.append(0)
    
    return predictions, signals


def calculate_l2_backtest_metrics(trades_df, initial_balance=10000, l2_metrics_df=None):
    """
    Calculate L2-specific backtest performance metrics
    
    Args:
        trades_df: DataFrame with trade results
        initial_balance: Starting balance
        l2_metrics_df: DataFrame with L2-specific metrics (optional)
    
    Returns:
        dict: L2-enhanced performance metrics
    """
    if trades_df is None or trades_df.empty:
        return None
    
    metrics = {}
    
    # Basic counts
    metrics['total_trades'] = len(trades_df)
    metrics['winning_trades'] = len(trades_df[trades_df['pnl'] > 0])
    metrics['losing_trades'] = len(trades_df[trades_df['pnl'] < 0])
    
    # Win rate
    if metrics['total_trades'] > 0:
        metrics['win_rate'] = metrics['winning_trades'] / metrics['total_trades']
    else:
        metrics['win_rate'] = 0
    
    # Average win/loss
    win_df = trades_df[trades_df['pnl'] > 0]
    loss_df = trades_df[trades_df['pnl'] < 0]
    metrics['avg_win'] = win_df['pnl'].mean() if len(win_df) > 0 else 0
    metrics['avg_loss'] = loss_df['pnl'].mean() if len(loss_df) > 0 else 0
    
    # Returns
    if not trades_df.empty:
        if 'equity_after_trade' in trades_df.columns:
            final_balance = trades_df.iloc[-1]['equity_after_trade']
        else:
            final_balance = initial_balance + trades_df['pnl'].sum()
    else:
        final_balance = initial_balance
    
    metrics['initial_balance'] = initial_balance
    metrics['final_balance'] = final_balance
    metrics['total_return'] = (final_balance - initial_balance) / initial_balance
    
    # Profit factor
    total_wins = win_df['pnl'].sum() if len(win_df) > 0 else 0
    total_losses = abs(loss_df['pnl'].sum()) if len(loss_df) > 0 else 0
    metrics['profit_factor'] = total_wins / total_losses if total_losses > 0 else 0
    
    # L2-specific metrics
    if l2_metrics_df is not None and not l2_metrics_df.empty:
        metrics['avg_spread_pct'] = l2_metrics_df['spread_pct'].mean()
        metrics['max_spread_pct'] = l2_metrics_df['spread_pct'].max()
        metrics['avg_prediction'] = l2_metrics_df['prediction'].mean()
        metrics['prediction_std'] = l2_metrics_df['prediction'].std()
    
    # L2-specific trade metrics
    if 'spread_at_exit' in trades_df.columns:
        metrics['avg_exit_spread'] = trades_df['spread_at_exit'].mean()
        metrics['max_exit_spread'] = trades_df['spread_at_exit'].max()
    
    # Sharpe ratio (annualized)
    if len(trades_df) > 1:
        returns = trades_df['pnl'] / initial_balance
        if returns.std() > 0:
            metrics['sharpe_ratio'] = returns.mean() / returns.std() * (252**0.5)
        else:
            metrics['sharpe_ratio'] = 0
    else:
        metrics['sharpe_ratio'] = 0
    
    return metrics


def print_l2_backtest_results(metrics):
    """
    Print formatted L2-specific backtest results
    
    Args:
        metrics: Dictionary of L2 performance metrics
    """
    if not metrics:
        print("\n⚠️ No trades executed in L2-only simulation")
        return
    
    print("\n📊 L2-Only Backtest Results:")
    print("="*60)
    print(f"Total Trades: {metrics['total_trades']}")
    print(f"Win Rate: {metrics['win_rate']:.2%}")
    print(f"Winning Trades: {metrics['winning_trades']}")
    print(f"Losing Trades: {metrics['losing_trades']}")
    print(f"Average Win: ${metrics['avg_win']:.2f}")
    print(f"Average Loss: ${metrics['avg_loss']:.2f}")
    print(f"Profit Factor: {metrics['profit_factor']:.2f}")
    print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    
    print(f"\nInitial Balance: ${metrics['initial_balance']:.2f}")
    print(f"Final Balance: ${metrics['final_balance']:.2f}")
    print(f"Total Return: {metrics['total_return']:.2%}")
    
    # L2-specific metrics
    if 'avg_spread_pct' in metrics:
        print(f"\nL2-Specific Metrics:")
        print(f"Average Spread: {metrics['avg_spread_pct']:.4f}%")
        print(f"Max Spread: {metrics['max_spread_pct']:.4f}%")
        print(f"Average Prediction: {metrics['avg_prediction']:.4f}")
        print(f"Prediction Std: {metrics['prediction_std']:.4f}")
    
    if 'avg_exit_spread' in metrics:
        print(f"Average Exit Spread: {metrics['avg_exit_spread']:.4f}%")
        print(f"Max Exit Spread: {metrics['max_exit_spread']:.4f}%")


def validate_l2_features(df_features, min_l2_features=10):
    """
    Validate that dataframe contains sufficient L2 features
    
    Args:
        df_features: DataFrame with features
        min_l2_features: Minimum number of L2 features required
    
    Returns:
        tuple: (is_valid, l2_feature_count, l2_features_list)
    """
    l2_terms = ['l2_', 'bid_ask', 'weighted_mid', 'microprice', 
                'order_book', 'price_impact', 'slope', 'volatility']
    
    l2_features = [col for col in df_features.columns 
                   if any(l2_term in col.lower() for l2_term in l2_terms)]
    
    l2_feature_count = len(l2_features)
    is_valid = l2_feature_count >= min_l2_features
    
    return is_valid, l2_feature_count, l2_features


def get_l2_price_column(df, preferred_order=None):
    """
    Get the best available L2 price column from dataframe
    
    Args:
        df: DataFrame with L2 data
        preferred_order: List of preferred price column names
    
    Returns:
        str: Name of the best available price column
    """
    if preferred_order is None:
        preferred_order = [
            'weighted_mid_price',
            'microprice', 
            'mid_price',
            'close',
            'price'
        ]
    
    for col in preferred_order:
        if col in df.columns:
            return col
    
    # Fallback to first numeric column
    numeric_cols = df.select_dtypes(include=['number']).columns
    if len(numeric_cols) > 0:
        return numeric_cols[0]
    
    raise ValueError("No suitable price column found in L2 data") 