"""
Module: visualizer_l2
Description: L2-only visualization system for trading bot analysis with microstructure features.
Author: project
Date: 2025-01-07
"""

import os
import pandas as pd
import numpy as np
import sqlite3
import traceback
from typing import Any, Dict, List, Optional


class L2Visualizer:
    """
    Handles the generation and saving of L2 microstructure plots for trading bot analysis.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        has_matplotlib: bool = False,
        plt_module: Any = None,
        has_shap: bool = False,
        shap_module: Any = None,
        db_path: str = "trading_bot.db"
    ) -> None:
        """
        Initializes the L2 Visualizer.

        Args:
            config: Configuration dictionary.
            has_matplotlib: Flag if matplotlib is available.
            plt_module: The imported matplotlib.pyplot module.
            has_shap: Flag if SHAP is available.
            shap_module: The imported SHAP module.
            db_path: Path to the SQLite database.
        """
        self.config = config
        self.symbol = config.get('symbol', 'BTC/USDT:USDT')
        self.timeframe = config.get('timeframe', '1m')
        self.base_dir = config.get('base_dir', './trading_bot_data')
        self.db_path = db_path

        self.HAS_MATPLOTLIB = has_matplotlib
        self.plt = plt_module
        self.HAS_SHAP = has_shap
        self.shap = shap_module

        safe_symbol = self.symbol.replace('/', '_').replace(':', '')
        plot_dir = os.path.join(self.base_dir, 'plots')
        os.makedirs(plot_dir, exist_ok=True)

        # L2-specific plot paths
        self.l2_orderbook_heatmap_path = os.path.join(
            plot_dir,
            f"l2_orderbook_heatmap_{safe_symbol}_{self.timeframe}.png"
        )
        self.l2_microstructure_path = os.path.join(
            plot_dir,
            f"l2_microstructure_{safe_symbol}_{self.timeframe}.png"
        )
        self.l2_spread_evolution_path = os.path.join(
            plot_dir,
            f"l2_spread_evolution_{safe_symbol}_{self.timeframe}.png"
        )
        self.l2_data_quality_path = os.path.join(
            plot_dir,
            f"l2_data_quality_{safe_symbol}_{self.timeframe}.png"
        )
        self.l2_feature_importance_path = os.path.join(
            plot_dir,
            f"l2_feature_importance_{safe_symbol}_{self.timeframe}.png"
        )
        self.l2_price_impact_path = os.path.join(
            plot_dir,
            f"l2_price_impact_{safe_symbol}_{self.timeframe}.png"
        )
        self.backtest_equity_plot_path = os.path.join(
            plot_dir,
            f"backtest_equity_curve_{safe_symbol}_{self.timeframe}.png"
        )
        self.simulation_equity_plot_path = os.path.join(
            plot_dir,
            f"simulation_equity_curve_{safe_symbol}_{self.timeframe}.png"
        )

        if self.HAS_MATPLOTLIB and self.plt:
            # Set style for L2 plots
            try:
                self.plt.style.use('seaborn-v0_8-darkgrid')
            except:
                pass  # Fallback to default style
        print("L2 Visualizer initialized.")

    def load_l2_data(self, limit: int = 10000, start_time: Optional[str] = None, 
                     end_time: Optional[str] = None) -> pd.DataFrame:
        """
        Load L2 data from the database.
        
        Args:
            limit: Maximum number of records to load
            start_time: Optional start time filter
            end_time: Optional end time filter
            
        Returns:
            DataFrame with L2 data
        """
        try:
            conn = sqlite3.connect(self.db_path)
            
            query = """
                SELECT * FROM l2_training_data_practical 
                WHERE symbol = ?
            """
            params = [self.symbol]
            
            if start_time:
                query += " AND timestamp >= ?"
                params.append(start_time)
            if end_time:
                query += " AND timestamp <= ?"
                params.append(end_time)
                
            query += " ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)
            
            df = pd.read_sql_query(query, conn, params=params)
            conn.close()
            
            if not df.empty:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df = df.sort_values('timestamp')
            
            print(f"Loaded {len(df)} L2 records from database")
            return df
            
        except Exception as e:
            print(f"Error loading L2 data: {e}")
            return pd.DataFrame()

    def plot_orderbook_heatmap(self, df: Optional[pd.DataFrame] = None, 
                               time_window_minutes: int = 60) -> None:
        """
        Plot order book depth heatmap for top 10 levels.
        """
        if not (self.HAS_MATPLOTLIB and self.plt):
            print("Warning (L2Visualizer): Matplotlib not available. Cannot plot order book heatmap.")
            return

        if df is None or df.empty:
            df = self.load_l2_data(limit=5000)
        
        if df.empty:
            print("Warning (L2Visualizer): No L2 data available for order book heatmap.")
            return

        try:
            # Filter to recent time window
            if time_window_minutes > 0:
                cutoff_time = df['timestamp'].max() - pd.Timedelta(minutes=time_window_minutes)
                df_plot = df[df['timestamp'] >= cutoff_time].copy()
            else:
                df_plot = df.copy()

            if df_plot.empty:
                print("Warning (L2Visualizer): No data in specified time window.")
                return

            # Sample data for visualization (every 10th record to avoid overcrowding)
            df_plot = df_plot.iloc[::10]

            fig, (ax1, ax2) = self.plt.subplots(2, 1, figsize=(14, 10), sharex=True)

            # Plot bid levels
            bid_data = []
            ask_data = []
            timestamps = df_plot['timestamp'].values

            for level in range(1, 11):
                bid_prices = df_plot[f'bid_price_{level}'].values
                bid_sizes = df_plot[f'bid_size_{level}'].values
                ask_prices = df_plot[f'ask_price_{level}'].values
                ask_sizes = df_plot[f'ask_size_{level}'].values
                
                # Create heatmap data (price levels vs time)
                for i, (timestamp, bid_price, bid_size, ask_price, ask_size) in enumerate(
                    zip(timestamps, bid_prices, bid_sizes, ask_prices, ask_sizes)
                ):
                    if pd.notna(bid_price) and pd.notna(bid_size):
                        bid_data.append([i, level, bid_size])
                    if pd.notna(ask_price) and pd.notna(ask_size):
                        ask_data.append([i, level, ask_size])

            if bid_data:
                bid_array = np.array(bid_data)
                scatter1 = ax1.scatter(bid_array[:, 0], bid_array[:, 1], 
                                     c=bid_array[:, 2], cmap='Greens', 
                                     alpha=0.7, s=20)
                ax1.set_ylabel('Bid Levels')
                ax1.set_title(f'Order Book Depth Heatmap - {self.symbol} (Last {time_window_minutes}min)')
                self.plt.colorbar(scatter1, ax=ax1, label='Bid Size')

            if ask_data:
                ask_array = np.array(ask_data)
                scatter2 = ax2.scatter(ask_array[:, 0], ask_array[:, 1], 
                                     c=ask_array[:, 2], cmap='Reds', 
                                     alpha=0.7, s=20)
                ax2.set_ylabel('Ask Levels')
                ax2.set_xlabel('Time Index')
                self.plt.colorbar(scatter2, ax=ax2, label='Ask Size')

            # Set time labels
            time_indices = range(0, len(timestamps), max(1, len(timestamps)//10))
            time_labels = []
            for i in time_indices:
                if hasattr(timestamps[i], 'strftime'):
                    time_labels.append(timestamps[i].strftime('%H:%M'))
                else:
                    # Convert numpy datetime64 to pandas timestamp
                    time_labels.append(pd.Timestamp(timestamps[i]).strftime('%H:%M'))
            ax2.set_xticks(time_indices)
            ax2.set_xticklabels(time_labels, rotation=45)

            self.plt.tight_layout()
            self.plt.savefig(self.l2_orderbook_heatmap_path, dpi=150)
            print(f"Order book heatmap saved to {self.l2_orderbook_heatmap_path}")
            
            if self.config.get('show_plots', True):
                self.plt.show()
            self.plt.close()

        except Exception as e:
            print(f"Warning (L2Visualizer): Error plotting order book heatmap: {e}")
            traceback.print_exc()

    def plot_microstructure_features(self, df: Optional[pd.DataFrame] = None) -> None:
        """
        Plot key microstructure features over time.
        """
        if not (self.HAS_MATPLOTLIB and self.plt):
            print("Warning (L2Visualizer): Matplotlib not available. Cannot plot microstructure features.")
            return

        if df is None or df.empty:
            df = self.load_l2_data(limit=5000)
        
        if df.empty:
            print("Warning (L2Visualizer): No L2 data available for microstructure plot.")
            return

        try:
            # Sample data for cleaner visualization
            df_plot = df.iloc[::5].copy()  # Every 5th record

            fig, axes = self.plt.subplots(4, 1, figsize=(14, 12), sharex=True)

            # Plot 1: Price evolution (mid price vs microprice)
            axes[0].plot(df_plot['timestamp'], df_plot['mid_price'], 
                        label='Mid Price', color='blue', alpha=0.8)
            if 'microprice' in df_plot.columns:
                axes[0].plot(df_plot['timestamp'], df_plot['microprice'], 
                            label='Microprice', color='red', alpha=0.8)
            axes[0].set_ylabel('Price (USD)')
            axes[0].set_title(f'L2 Microstructure Features - {self.symbol}')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)

            # Plot 2: Spread evolution
            if 'spread_bps' in df_plot.columns:
                axes[1].plot(df_plot['timestamp'], df_plot['spread_bps'], 
                            color='orange', alpha=0.8)
                axes[1].set_ylabel('Spread (bps)')
                axes[1].set_title('Bid-Ask Spread Evolution')
                axes[1].grid(True, alpha=0.3)

            # Plot 3: Order book imbalance
            if 'order_book_imbalance' in df_plot.columns:
                axes[2].plot(df_plot['timestamp'], df_plot['order_book_imbalance'], 
                            color='purple', alpha=0.8)
                axes[2].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
                axes[2].set_ylabel('OB Imbalance')
                axes[2].set_title('Order Book Imbalance (Positive = Bid Heavy)')
                axes[2].grid(True, alpha=0.3)

            # Plot 4: Volume evolution
            if 'total_bid_volume_10' in df_plot.columns and 'total_ask_volume_10' in df_plot.columns:
                axes[3].plot(df_plot['timestamp'], df_plot['total_bid_volume_10'], 
                            label='Total Bid Volume', color='green', alpha=0.8)
                axes[3].plot(df_plot['timestamp'], df_plot['total_ask_volume_10'], 
                            label='Total Ask Volume', color='red', alpha=0.8)
                axes[3].set_ylabel('Volume')
                axes[3].set_xlabel('Time')
                axes[3].set_title('Order Book Volume (Top 10 Levels)')
                axes[3].legend()
                axes[3].grid(True, alpha=0.3)

            # Format x-axis
            for ax in axes:
                ax.tick_params(axis='x', rotation=45)

            self.plt.tight_layout()
            self.plt.savefig(self.l2_microstructure_path, dpi=150)
            print(f"Microstructure features plot saved to {self.l2_microstructure_path}")
            
            if self.config.get('show_plots', True):
                self.plt.show()
            self.plt.close()

        except Exception as e:
            print(f"Warning (L2Visualizer): Error plotting microstructure features: {e}")
            traceback.print_exc()

    def plot_data_quality_dashboard(self, df: Optional[pd.DataFrame] = None) -> None:
        """
        Plot data quality metrics dashboard.
        """
        if not (self.HAS_MATPLOTLIB and self.plt):
            print("Warning (L2Visualizer): Matplotlib not available. Cannot plot data quality dashboard.")
            return

        if df is None or df.empty:
            df = self.load_l2_data(limit=10000)
        
        if df.empty:
            print("Warning (L2Visualizer): No L2 data available for data quality dashboard.")
            return

        try:
            fig, axes = self.plt.subplots(2, 2, figsize=(14, 10))

            # Plot 1: Data quality score over time
            if 'data_quality_score' in df.columns:
                df_sampled = df.iloc[::20]  # Sample for cleaner plot
                axes[0, 0].plot(df_sampled['timestamp'], df_sampled['data_quality_score'], 
                               color='blue', alpha=0.7)
                axes[0, 0].set_ylabel('Quality Score')
                axes[0, 0].set_title('Data Quality Score Over Time')
                axes[0, 0].grid(True, alpha=0.3)
                axes[0, 0].tick_params(axis='x', rotation=45)

            # Plot 2: Spread distribution
            if 'spread_bps' in df.columns:
                spread_data = df['spread_bps'].dropna()
                if not spread_data.empty:
                    # Remove outliers for better visualization
                    q99 = spread_data.quantile(0.99)
                    spread_filtered = spread_data[spread_data <= q99]
                    axes[0, 1].hist(spread_filtered, bins=50, alpha=0.7, color='orange')
                    axes[0, 1].set_xlabel('Spread (bps)')
                    axes[0, 1].set_ylabel('Frequency')
                    axes[0, 1].set_title('Spread Distribution')
                    axes[0, 1].grid(True, alpha=0.3)

            # Plot 3: Missing data heatmap
            missing_data = []
            for level in range(1, 11):
                bid_missing = df[f'bid_price_{level}'].isna().sum()
                ask_missing = df[f'ask_price_{level}'].isna().sum()
                missing_data.append([level, bid_missing, ask_missing])
            
            if missing_data:
                missing_array = np.array(missing_data)
                x_pos = np.arange(len(missing_array))
                width = 0.35
                
                axes[1, 0].bar(x_pos - width/2, missing_array[:, 1], width, 
                              label='Bid Missing', alpha=0.7, color='green')
                axes[1, 0].bar(x_pos + width/2, missing_array[:, 2], width, 
                              label='Ask Missing', alpha=0.7, color='red')
                axes[1, 0].set_xlabel('Order Book Level')
                axes[1, 0].set_ylabel('Missing Count')
                axes[1, 0].set_title('Missing Data by Level')
                axes[1, 0].set_xticks(x_pos)
                axes[1, 0].set_xticklabels([f'L{i}' for i in range(1, 11)])
                axes[1, 0].legend()
                axes[1, 0].grid(True, alpha=0.3)

            # Plot 4: Data quality statistics
            if 'data_quality_score' in df.columns:
                quality_stats = df['data_quality_score'].describe()
                stats_text = f"""
Data Quality Statistics:
Mean: {quality_stats['mean']:.3f}
Std:  {quality_stats['std']:.3f}
Min:  {quality_stats['min']:.3f}
Max:  {quality_stats['max']:.3f}

Total Records: {len(df):,}
Valid Mid Prices: {df['mid_price'].notna().sum():,}
Valid Targets: {df['target_return_1min'].notna().sum():,}
                """
                axes[1, 1].text(0.1, 0.5, stats_text, fontsize=10, 
                               verticalalignment='center', fontfamily='monospace')
                axes[1, 1].set_xlim(0, 1)
                axes[1, 1].set_ylim(0, 1)
                axes[1, 1].set_title('Data Quality Summary')
                axes[1, 1].axis('off')

            self.plt.tight_layout()
            self.plt.savefig(self.l2_data_quality_path, dpi=150)
            print(f"Data quality dashboard saved to {self.l2_data_quality_path}")
            
            if self.config.get('show_plots', True):
                self.plt.show()
            self.plt.close()

        except Exception as e:
            print(f"Warning (L2Visualizer): Error plotting data quality dashboard: {e}")
            traceback.print_exc()

    def plot_l2_feature_importance(self, model_booster: Any, trained_features: List[str]) -> None:
        """
        Plot feature importance for L2 features.
        """
        if not (self.HAS_MATPLOTLIB and self.plt):
            print("Warning (L2Visualizer): Matplotlib not available. Cannot plot feature importance.")
            return

        if model_booster is None or not trained_features:
            print("Warning (L2Visualizer): Model or trained features not provided for importance plot.")
            return

        try:
            self.plt.figure(figsize=(12, 8))
            
            # Get feature importance
            if hasattr(model_booster, 'feature_importances_'):
                importances = model_booster.feature_importances_
            elif hasattr(model_booster, 'feature_importance'):
                importances = model_booster.feature_importance(importance_type='gain')
            else:
                print("Error (L2Visualizer): Model type not supported for importance plot.")
                return

            # Create feature importance DataFrame
            feature_df = pd.DataFrame({
                'feature': trained_features,
                'importance': importances
            }).sort_values('importance', ascending=True)

            # Plot top 20 features
            top_features = feature_df.tail(20)
            
            # Color code by feature type
            colors = []
            for feature in top_features['feature']:
                if 'bid' in feature or 'ask' in feature:
                    colors.append('lightblue')
                elif 'spread' in feature or 'microprice' in feature:
                    colors.append('orange')
                elif 'imbalance' in feature or 'volume' in feature:
                    colors.append('lightgreen')
                elif 'target' in feature:
                    colors.append('red')
                else:
                    colors.append('gray')

            bars = self.plt.barh(range(len(top_features)), top_features['importance'], 
                                color=colors, alpha=0.7)
            
            self.plt.yticks(range(len(top_features)), top_features['feature'])
            self.plt.xlabel('Feature Importance (Gain)')
            self.plt.title(f'L2 Feature Importance - {self.symbol}')
            self.plt.grid(True, alpha=0.3, axis='x')
            
            # Add legend
            legend_elements = [
                self.plt.Rectangle((0,0),1,1, facecolor='lightblue', alpha=0.7, label='Price/Size'),
                self.plt.Rectangle((0,0),1,1, facecolor='orange', alpha=0.7, label='Spread/Microprice'),
                self.plt.Rectangle((0,0),1,1, facecolor='lightgreen', alpha=0.7, label='Imbalance/Volume'),
                self.plt.Rectangle((0,0),1,1, facecolor='red', alpha=0.7, label='Targets'),
                self.plt.Rectangle((0,0),1,1, facecolor='gray', alpha=0.7, label='Other')
            ]
            self.plt.legend(handles=legend_elements, loc='lower right')

            self.plt.tight_layout()
            self.plt.savefig(self.l2_feature_importance_path, dpi=150)
            print(f"L2 feature importance plot saved to {self.l2_feature_importance_path}")
            
            if self.config.get('show_plots', True):
                self.plt.show()
            self.plt.close()

        except Exception as e:
            print(f"Warning (L2Visualizer): Error plotting L2 feature importance: {e}")
            traceback.print_exc()

    def plot_equity_curve(self, equity_df: pd.DataFrame, initial_balance: float,
                         final_balance: float, total_return_pct: float,
                         plot_type: str = "Backtest") -> None:
        """
        Generates and displays/saves a plot of the equity curve.
        """
        if not (self.HAS_MATPLOTLIB and self.plt):
            print("Warning (L2Visualizer): Matplotlib not available. Cannot plot equity curve.")
            return

        if (equity_df is None or equity_df.empty or
            'timestamp' not in equity_df.columns or
            'equity' not in equity_df.columns):
            print("Warning (L2Visualizer): Equity DataFrame is invalid or missing required columns for plotting.")
            return

        try:
            self.plt.figure(figsize=self.config.get('plot_figsize_equity', (14, 7)))
            equity_df_plot = equity_df.copy()
            equity_df_plot['timestamp'] = pd.to_datetime(equity_df_plot['timestamp'])
            
            self.plt.plot(equity_df_plot["timestamp"], equity_df_plot["equity"],
                         label="Equity Curve", color='dodgerblue', lw=1.5)
            
            if len(equity_df_plot) > 1:
                self.plt.fill_between(equity_df_plot["timestamp"], initial_balance,
                                     equity_df_plot["equity"],
                                     where=equity_df_plot["equity"] >= initial_balance,
                                     color='palegreen', alpha=0.5, interpolate=True)
                self.plt.fill_between(equity_df_plot["timestamp"], initial_balance,
                                     equity_df_plot["equity"],
                                     where=equity_df_plot["equity"] < initial_balance,
                                     color='lightcoral', alpha=0.5, interpolate=True)
            
            self.plt.axhline(initial_balance, color='grey', linestyle='--',
                           label=f'Initial Balance (${initial_balance:,.2f})')
            
            title = (f"L2 {plot_type} Equity: {self.symbol} | "
                    f"Final: ${final_balance:,.2f} ({total_return_pct:.2f}%)")
            self.plt.title(title, fontsize=14)
            self.plt.xlabel("Time", fontsize=12)
            self.plt.ylabel("Equity (USD)", fontsize=12)
            self.plt.grid(True, linestyle=':', alpha=0.7)
            self.plt.legend(fontsize=10)
            self.plt.xticks(rotation=45, ha='right')
            self.plt.tight_layout()
            
            save_path = (self.backtest_equity_plot_path if plot_type == "Backtest"
                        else self.simulation_equity_plot_path)
            self.plt.savefig(save_path, dpi=150)
            print(f"L2 {plot_type} equity curve saved to {save_path}")
            
            if self.config.get('show_plots', True):
                self.plt.show()
            self.plt.close()

        except Exception as e:
            print(f"Warning (L2Visualizer): Error plotting equity curve: {e}")
            traceback.print_exc()

    def generate_all_l2_plots(self, limit: int = 5000) -> None:
        """
        Generate all L2-specific plots.
        
        Args:
            limit: Maximum number of records to load for plotting
        """
        print("Generating all L2 visualization plots...")
        
        # Load L2 data once
        df = self.load_l2_data(limit=limit)
        
        if df.empty:
            print("Warning (L2Visualizer): No L2 data available for plotting.")
            return
        
        # Generate all plots
        self.plot_orderbook_heatmap(df)
        self.plot_microstructure_features(df)
        self.plot_data_quality_dashboard(df)
        
        print("All L2 visualization plots generated successfully!")
