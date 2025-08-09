#!/usr/bin/env python3
"""
Database-enabled utility functions for the trading bot project
"""

import os
import json
import pandas as pd
import lightgbm as lgb
from database import TradingDatabase
from featureengineer import FeatureEngineer


def load_model_and_features_from_db(db: TradingDatabase, 
                                    symbol: str, 
                                    timeframe: str):
    """
    Load trained model and features from database
    
    Args:
        db: TradingDatabase instance
        symbol: Trading symbol
        timeframe: Timeframe
    
    Returns:
        tuple: (model, feature_info, trained_features)
    """
    model_data = db.load_active_model(symbol, timeframe)
    
    if not model_data:
        raise ValueError(f"No active model found for {symbol} {timeframe}")
    
    # Save model data to temp file and load with LightGBM
    import tempfile
    with tempfile.NamedTemporaryFile(delete=False, suffix='.txt') as tmp:
        tmp.write(model_data['model_data'])
        tmp_path = tmp.name
    
    try:
        model = lgb.Booster(model_file=tmp_path)
    finally:
        os.unlink(tmp_path)
    
    feature_info = model_data.get('metrics', {})
    trained_features = model_data['feature_list']
    
    return model, feature_info, trained_features


def load_and_prepare_data_from_db(db: TradingDatabase,
                                  symbol: str,
                                  timeframe: str,
                                  start_date=None,
                                  end_date=None):
    """
    Load and prepare OHLCV data from database
    
    Args:
        db: TradingDatabase instance
        symbol: Trading symbol
        timeframe: Timeframe
        start_date: Optional start date
        end_date: Optional end date
    
    Returns:
        pd.DataFrame: Prepared dataframe
    """
    df = db.load_ohlcv_data(symbol, timeframe, start_date, end_date)
    
    if df.empty:
        raise ValueError(f"No data found for {symbol} {timeframe}")
    
    return df


def save_backtest_to_db(db: TradingDatabase,
                        model_id: int,
                        trades_df: pd.DataFrame,
                        results_df: pd.DataFrame,
                        config: dict):
    """
    Save backtest results to database
    
    Args:
        db: TradingDatabase instance
        model_id: Model ID
        trades_df: DataFrame with trade results
        results_df: DataFrame with equity curve
        config: Configuration dict
    """
    if trades_df is None or trades_df.empty:
        return None
    
    # Calculate metrics
    metrics = calculate_backtest_metrics(trades_df, 
                                         config.get('initial_balance', 10000))
    
    # Add additional metrics
    if results_df is not None and 'equity' in results_df.columns:
        import numpy as np
        equity = results_df['equity'].values
        peak = np.maximum.accumulate(equity)
        drawdown = (equity - peak) / peak
        metrics['max_drawdown'] = drawdown.min()
        
        # Calculate Sharpe ratio (simplified)
        returns = results_df['equity'].pct_change().dropna()
        if len(returns) > 0:
            sharpe = returns.mean() / returns.std() * np.sqrt(252)
            metrics['sharpe_ratio'] = sharpe
    
    # Prepare results for database
    backtest_results = {
        'start_date': results_df['timestamp'].min() if results_df is not None else None,
        'end_date': results_df['timestamp'].max() if results_df is not None else None,
        'initial_balance': metrics['initial_balance'],
        'final_balance': metrics['final_balance'],
        'total_return': metrics['total_return'],
        'max_drawdown': metrics.get('max_drawdown'),
        'sharpe_ratio': metrics.get('sharpe_ratio'),
        'win_rate': metrics['win_rate'],
        'total_trades': metrics['total_trades'],
        'profit_factor': metrics['profit_factor'],
        'additional_metrics': {
            'avg_win': metrics['avg_win'],
            'avg_loss': metrics['avg_loss'],
            'winning_trades': metrics['winning_trades'],
            'losing_trades': metrics['losing_trades']
        }
    }
    
    # Save to database
    return db.save_backtest_results(model_id, backtest_results)


def calculate_backtest_metrics(trades_df, initial_balance=10000):
    """
    Calculate backtest performance metrics
    
    Args:
        trades_df: DataFrame with trade results
        initial_balance: Starting balance
    
    Returns:
        dict: Performance metrics
    """
    if trades_df is None or trades_df.empty:
        return None
    
    metrics = {}
    
    # Basic counts
    metrics['total_trades'] = len(trades_df)
    metrics['winning_trades'] = len(trades_df[trades_df['pnl_net'] > 0])
    metrics['losing_trades'] = len(trades_df[trades_df['pnl_net'] < 0])
    
    # Win rate
    if metrics['total_trades'] > 0:
        metrics['win_rate'] = (metrics['winning_trades'] / 
                               metrics['total_trades'])
    else:
        metrics['win_rate'] = 0
    
    # Average win/loss
    win_df = trades_df[trades_df['pnl_net'] > 0]
    loss_df = trades_df[trades_df['pnl_net'] < 0]
    metrics['avg_win'] = win_df['pnl_net'].mean() if len(win_df) > 0 else 0
    metrics['avg_loss'] = loss_df['pnl_net'].mean() if len(loss_df) > 0 else 0
    
    # Returns
    if not trades_df.empty:
        final_balance = trades_df.iloc[-1]['equity_after_trade']
    else:
        final_balance = initial_balance
    
    metrics['initial_balance'] = initial_balance
    metrics['final_balance'] = final_balance
    metrics['total_return'] = ((final_balance - initial_balance) / 
                               initial_balance)
    
    # Profit factor
    total_wins = win_df['pnl_net'].sum() if len(win_df) > 0 else 0
    total_losses = abs(loss_df['pnl_net'].sum()) if len(loss_df) > 0 else 0
    metrics['profit_factor'] = (total_wins / total_losses 
                                if total_losses > 0 else 0)
    
    return metrics


# Keep original functions for backward compatibility
def load_model_and_features(model_path='lgbm_model_BTC_USDTUSDT_1m.txt',
                            features_path='model_features_BTC_USDTUSDT_1m.json'):
    """Legacy function - loads from files"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    if not os.path.exists(features_path):
        raise FileNotFoundError(f"Features config not found: {features_path}")
    
    model = lgb.Booster(model_file=model_path)
    
    with open(features_path, 'r') as f:
        feature_info = json.load(f)
        trained_features = feature_info['trained_features']
    
    return model, feature_info, trained_features


def load_and_prepare_data(data_file='ohlcv_data_BTC_USDTUSDT_1m.csv'):
    """Legacy function - loads from CSV file"""
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"Data file not found: {data_file}")
    
    df = pd.read_csv(data_file)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df


def generate_features(df, config=None):
    """Generate features using FeatureEngineer"""
    if config is None:
        config = {
            'feature_window': 24,
            'use_hht_features': False,
            'use_l2_features': False,
            'ta_features': ['rsi', 'macd', 'atr'],
            'ta_indicator_params': {}
        }
    
    fe = FeatureEngineer(config, {'HAS_PANDAS_TA': True}, {'ta': None})
    return fe.generate_all_features(df)


def make_predictions(df_features, model, trained_features, threshold=0.25):
    """Generate predictions and trading signals"""
    predictions = []
    signals = []
    
    for i in range(len(df_features)):
        if i < 24:  # Skip initial rows
            predictions.append(0)
            signals.append(0)
        else:
            try:
                X = df_features.iloc[i][trained_features].values.reshape(1, -1)
                pred = model.predict(X, num_iteration=model.best_iteration)[0]
                predictions.append(pred)
                
                # Generate signal
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


def print_backtest_results(metrics):
    """Print formatted backtest results"""
    if not metrics:
        print("\n⚠️ No trades executed in simulation")
        return
    
    print("\n📊 Backtest Results:")
    print("="*60)
    print(f"Total Trades: {metrics['total_trades']}")
    print(f"Win Rate: {metrics['win_rate']:.2%}")
    print(f"Winning Trades: {metrics['winning_trades']}")
    print(f"Losing Trades: {metrics['losing_trades']}")
    print(f"Average Win: ${metrics['avg_win']:.2f}")
    print(f"Average Loss: ${metrics['avg_loss']:.2f}")
    print(f"Profit Factor: {metrics['profit_factor']:.2f}")
    print(f"\nInitial Balance: ${metrics['initial_balance']:.2f}")
    print(f"Final Balance: ${metrics['final_balance']:.2f}")
    print(f"Total Return: {metrics['total_return']:.2%}") 