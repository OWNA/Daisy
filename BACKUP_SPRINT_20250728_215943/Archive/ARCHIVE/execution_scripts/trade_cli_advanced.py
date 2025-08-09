#!/usr/bin/env python3
"""
Advanced Trading Bot CLI - Full flexibility for workflows, data management, and visualization
"""

import click
import os
import sys
import yaml
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import subprocess
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import track
import plotly.graph_objects as go
import plotly.subplots as sp
from plotly.offline import plot
import seaborn as sns
import matplotlib.pyplot as plt

console = Console()

class AdvancedTradingCLI:
    def __init__(self):
        self.config_file = 'config.yaml'
        self.load_config()
        
    def load_config(self):
        """Load configuration"""
        if os.path.exists(self.config_file):
            with open(self.config_file, 'r') as f:
                self.config = yaml.safe_load(f)
        else:
            console.print("[red]Config file not found![/red]")
            self.config = {}

@click.group()
@click.pass_context
def cli(ctx):
    """Advanced Trading Bot CLI with full flexibility"""
    ctx.obj = AdvancedTradingCLI()

# ============ DATA COLLECTION ============
@cli.group()
def data():
    """L2 data collection and management"""
    pass

@data.command()
@click.option('--duration', '-d', default=300, help='Collection duration in seconds')
@click.option('--symbol', '-s', default='BTC/USDT:USDT', help='Trading symbol')
@click.option('--append', is_flag=True, help='Append to existing data')
@click.pass_obj
def collect(cli_obj, duration, symbol, append):
    """Collect fresh L2 order book data"""
    console.print(Panel(f"[cyan]Collecting L2 data for {duration} seconds[/cyan]"))
    
    # Update config
    cli_obj.config['symbol'] = symbol
    cli_obj.config['l2_collection_duration_seconds'] = duration
    
    cmd = f"python run_l2_collector.py --continuous --interval {duration}"
    if append:
        cmd += " --append"
    
    subprocess.run(cmd, shell=True)

@data.command(name='list')
@click.option('--last', '-l', default=5, help='Show last N files')
@click.pass_obj
def list_data(cli_obj, last):
    """List collected L2 data files"""
    l2_dir = Path('l2_data')
    if not l2_dir.exists():
        console.print("[red]No L2 data directory found[/red]")
        return
        
    files = sorted(l2_dir.glob('*.jsonl.gz'), key=lambda x: x.stat().st_mtime, reverse=True)
    
    table = Table(title="L2 Data Files")
    table.add_column("File", style="cyan")
    table.add_column("Size (MB)", style="green")
    table.add_column("Modified", style="yellow")
    
    for file in files[:last]:
        size_mb = file.stat().st_size / 1024 / 1024
        modified = datetime.fromtimestamp(file.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S')
        table.add_row(file.name, f"{size_mb:.2f}", modified)
    
    console.print(table)

@data.command()
@click.argument('filename')
@click.option('--rows', '-r', default=100, help='Number of rows to show')
@click.pass_obj
def inspect(cli_obj, filename, rows):
    """Inspect L2 data file contents"""
    import gzip
    
    file_path = Path('l2_data') / filename
    if not file_path.exists():
        console.print(f"[red]File not found: {filename}[/red]")
        return
    
    records = []
    with gzip.open(file_path, 'rt') as f:
        for i, line in enumerate(f):
            if i >= rows:
                break
            records.append(json.loads(line))
    
    if records:
        first = records[0]
        last = records[-1]
        
        console.print(f"\n[cyan]File: {filename}[/cyan]")
        console.print(f"Records shown: {len(records)}")
        console.print(f"Time range: {first['timestamp']} to {last['timestamp']}")
        
        # Show sample orderbook
        if 'b' in first and 'a' in first:
            console.print(f"\nSample orderbook (first record):")
            console.print(f"Best bid: ${float(first['b'][0][0]):,.2f} ({first['b'][0][1]})")
            console.print(f"Best ask: ${float(first['a'][0][0]):,.2f} ({first['a'][0][1]})")

# ============ MODEL TRAINING ============
@cli.group()
def model():
    """Model training and management"""
    pass

@model.command()
@click.option('--data-file', '-d', help='Specific L2 data file to use')
@click.option('--features', '-f', default='all', help='Features to use: all/l2/hht')
@click.option('--optuna-trials', '-o', default=50, help='Number of Optuna trials')
@click.option('--remove-features', '-r', multiple=True, help='Features to remove')
@click.pass_obj
def train(cli_obj, data_file, features, optuna_trials, remove_features):
    """Train model with flexibility to select features and data"""
    console.print(Panel("[cyan]Training Model[/cyan]"))
    
    # Use the robust training script
    cmd = f"python train_model_robust.py --features {features} --trials {optuna_trials}"
    
    if data_file:
        cmd += f" --data {data_file}"
    
    # Run training
    result = subprocess.run(cmd, shell=True)
    
    if result.returncode != 0:
        console.print("[red]Training failed! Check error messages above.[/red]")

@model.command(name='list')
@click.pass_obj
def list_models(cli_obj):
    """List available models"""
    models = list(Path('.').glob('lgbm_model_*.txt'))
    
    table = Table(title="Available Models")
    table.add_column("Model File", style="cyan")
    table.add_column("Features File", style="green")
    table.add_column("Modified", style="yellow")
    
    for model in models:
        features_file = model.name.replace('lgbm_model_', 'model_features_').replace('.txt', '.json')
        if os.path.exists(features_file):
            with open(features_file, 'r') as f:
                info = json.load(f)
            feature_count = len(info.get('features', []))
            features_str = f"{features_file} ({feature_count} features)"
        else:
            features_str = "Not found"
            
        modified = datetime.fromtimestamp(model.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S')
        table.add_row(model.name, features_str, modified)
    
    console.print(table)

@model.command()
@click.argument('model_file')
@click.option('--top', '-t', default=20, help='Show top N features')
@click.pass_obj
def inspect(cli_obj, model_file, top):
    """Inspect model features and importance"""
    import lightgbm as lgb
    
    if not os.path.exists(model_file):
        console.print(f"[red]Model not found: {model_file}[/red]")
        return
    
    # Load model
    model = lgb.Booster(model_file=model_file)
    
    # Get feature importance
    importance = model.feature_importance(importance_type='gain')
    features = model.feature_name()
    
    # Create importance df
    df = pd.DataFrame({
        'feature': features,
        'importance': importance
    }).sort_values('importance', ascending=False)
    
    # Show top features
    console.print(f"\n[cyan]Model: {model_file}[/cyan]")
    console.print(f"Total features: {len(features)}")
    console.print(f"\nTop {top} features by importance:")
    
    for idx, row in df.head(top).iterrows():
        if row['importance'] > 0:
            console.print(f"  {row['feature']:<40} {row['importance']:>10.2f}")

# ============ BACKTESTING ============
@cli.group()
def backtest():
    """Backtesting operations"""
    pass

@backtest.command()
@click.option('--model', '-m', help='Model file to use')
@click.option('--start-date', '-s', help='Start date (YYYY-MM-DD)')
@click.option('--end-date', '-e', help='End date (YYYY-MM-DD)')
@click.option('--data-rows', '-r', default=10000, help='Number of data rows to use')
@click.pass_obj
def run(cli_obj, model, start_date, end_date, data_rows):
    """Run backtest with specific parameters"""
    console.print(Panel("[cyan]Running Backtest[/cyan]"))
    
    # Use fixed backtest script
    cmd = "python run_backtest_fixed.py"
    if model:
        cmd += f" --model {model}"
    if data_rows:
        cmd += f" --rows {data_rows}"
    
    result = subprocess.run(cmd, shell=True)
    
    if result.returncode == 0:
        # Show latest results
        results_dir = Path('backtest_results')
        if results_dir.exists():
            results = sorted(results_dir.glob('*.csv'), key=lambda x: x.stat().st_mtime, reverse=True)
            if results:
                console.print(f"[green]Results saved: {results[0].name}[/green]")

@backtest.command()
@click.argument('results_file')
@click.option('--plot-type', '-p', default='plotly', help='Plot type: plotly/seaborn')
@click.pass_obj
def visualize(cli_obj, results_file, plot_type):
    """Visualize backtest results on price chart"""
    if not os.path.exists(results_file):
        console.print(f"[red]Results file not found: {results_file}[/red]")
        return
    
    df = pd.read_csv(results_file)
    
    if plot_type == 'plotly':
        # Create interactive plot
        fig = sp.make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=('Price & Trades', 'Position', 'Equity Curve'),
            row_heights=[0.5, 0.2, 0.3]
        )
        
        # Price chart
        fig.add_trace(
            go.Scatter(x=df['timestamp'], y=df['price'], 
                      name='Price', line=dict(color='gray')),
            row=1, col=1
        )
        
        # Buy/Sell markers
        buys = df[df['action'] == 'buy']
        sells = df[df['action'] == 'sell']
        
        fig.add_trace(
            go.Scatter(x=buys['timestamp'], y=buys['price'],
                      mode='markers', name='Buy',
                      marker=dict(color='green', size=10, symbol='triangle-up')),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=sells['timestamp'], y=sells['price'],
                      mode='markers', name='Sell',
                      marker=dict(color='red', size=10, symbol='triangle-down')),
            row=1, col=1
        )
        
        # Position
        fig.add_trace(
            go.Scatter(x=df['timestamp'], y=df['position'],
                      name='Position', fill='tozeroy'),
            row=2, col=1
        )
        
        # Equity curve
        fig.add_trace(
            go.Scatter(x=df['timestamp'], y=df['equity'],
                      name='Equity', line=dict(color='blue')),
            row=3, col=1
        )
        
        fig.update_layout(height=900, title_text=f"Backtest Results: {results_file}")
        fig.write_html('backtest_visualization.html')
        console.print("[green]Visualization saved to backtest_visualization.html[/green]")
        
    else:  # seaborn
        fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
        
        # Price & trades
        axes[0].plot(df.index, df['price'], color='gray', alpha=0.7)
        buys = df[df['action'] == 'buy']
        sells = df[df['action'] == 'sell']
        axes[0].scatter(buys.index, buys['price'], color='green', marker='^', s=100, label='Buy')
        axes[0].scatter(sells.index, sells['price'], color='red', marker='v', s=100, label='Sell')
        axes[0].set_ylabel('Price')
        axes[0].legend()
        
        # Position
        axes[1].fill_between(df.index, df['position'], alpha=0.3)
        axes[1].set_ylabel('Position')
        axes[1].axhline(y=0, color='black', linestyle='--', alpha=0.3)
        
        # Equity
        axes[2].plot(df.index, df['equity'], color='blue')
        axes[2].set_ylabel('Equity')
        axes[2].set_xlabel('Time')
        
        plt.tight_layout()
        plt.savefig('backtest_visualization.png', dpi=150)
        console.print("[green]Visualization saved to backtest_visualization.png[/green]")

# ============ SIMULATION ============
@cli.group()
def simulate():
    """Live simulation operations"""
    pass

@simulate.command()
@click.option('--duration', '-d', default=300, help='Simulation duration in seconds')
@click.option('--model', '-m', help='Model file to use')
@click.option('--paper', is_flag=True, help='Run in paper trading mode')
@click.option('--output', '-o', help='Output file for live monitoring')
@click.pass_obj
def run(cli_obj, duration, model, paper, output):
    """Run live simulation"""
    console.print(Panel(f"[cyan]Running {'Paper Trading' if paper else 'Simulation'} for {duration}s[/cyan]"))
    
    # Update config
    cli_obj.config['simulation_duration_seconds'] = duration
    
    if model:
        # Update model in config
        safe_symbol = cli_obj.config.get('symbol', 'BTC/USDT:USDT').replace('/', '_').replace(':', '')
        cli_obj.config['model_path'] = model
    
    with open('config.yaml', 'w') as f:
        yaml.dump(cli_obj.config, f)
    
    if output:
        # Use wrapper for output monitoring
        cmd = f"python run_live_simulation_with_output.py --duration {duration} --model {model or 'lgbm_model_BTC_USDTUSDT_optuna_l2_hht.txt'} --output {output}"
    else:
        if paper:
            cmd = "python run_trading_bot.py --workflow l2_live_trading"
        else:
            cmd = "python start_live_simulation.py"
    
    subprocess.run(cmd, shell=True)

# ============ ANALYSIS ============
@cli.group()
def analyze():
    """Analysis and visualization tools"""
    pass

@analyze.command()
@click.option('--model', '-m', required=True, help='Model file')
@click.option('--data', '-d', help='Data file for SHAP analysis')
@click.pass_obj
def shap(cli_obj, model, data):
    """Generate SHAP analysis for model interpretation"""
    console.print(Panel("[cyan]Generating SHAP Analysis[/cyan]"))
    
    # Create SHAP script
    shap_code = f"""
import shap
import lightgbm as lgb
import pandas as pd
import matplotlib.pyplot as plt

# Load model
model = lgb.Booster(model_file='{model}')

# Load data
if '{data}':
    df = pd.read_csv('{data}')
else:
    # Use recent prepared data
    df = pd.read_csv('prepared_data_l2_only_BTC_USDTUSDT.csv')

# Get features
feature_cols = [c for c in df.columns if c not in ['timestamp', 'target', 'label']]
X = df[feature_cols].head(1000)  # Use subset for speed

# SHAP analysis
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

# Plots
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X, show=False)
plt.savefig('shap_summary.png', dpi=150, bbox_inches='tight')

plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X, plot_type="bar", show=False)
plt.savefig('shap_importance.png', dpi=150, bbox_inches='tight')

print("SHAP analysis complete!")
"""
    
    with open('_temp_shap.py', 'w') as f:
        f.write(shap_code)
    
    subprocess.run("python _temp_shap.py", shell=True)
    os.remove('_temp_shap.py')
    
    console.print("[green]SHAP plots saved: shap_summary.png, shap_importance.png[/green]")

@analyze.command()
@click.option('--results-dir', '-r', default='backtest_results', help='Results directory')
@click.pass_obj
def performance(cli_obj, results_dir):
    """Analyze performance metrics across multiple runs"""
    results_path = Path(results_dir)
    if not results_path.exists():
        console.print(f"[red]Results directory not found: {results_dir}[/red]")
        return
    
    # Collect all results
    all_metrics = []
    for file in results_path.glob('*.csv'):
        df = pd.read_csv(file)
        if len(df) > 0:
            metrics = {
                'file': file.name,
                'total_trades': len(df),
                'final_equity': df.iloc[-1]['equity'] if 'equity' in df else 0,
                'return': ((df.iloc[-1]['equity'] - 10000) / 10000 * 100) if 'equity' in df else 0,
                'win_rate': (df['pnl'] > 0).sum() / len(df) * 100 if 'pnl' in df else 0
            }
            all_metrics.append(metrics)
    
    if all_metrics:
        df_metrics = pd.DataFrame(all_metrics)
        
        table = Table(title="Performance Summary")
        table.add_column("File", style="cyan")
        table.add_column("Trades", style="green")
        table.add_column("Return %", style="yellow")
        table.add_column("Win Rate %", style="magenta")
        
        for _, row in df_metrics.iterrows():
            table.add_row(
                row['file'][:30],
                str(row['total_trades']),
                f"{row['return']:.2f}",
                f"{row['win_rate']:.1f}"
            )
        
        console.print(table)

# ============ CONFIGURATION ============
@cli.group()
def config():
    """Configuration management"""
    pass

@config.command()
@click.pass_obj
def show(cli_obj):
    """Show current configuration"""
    console.print(Panel("[cyan]Current Configuration[/cyan]"))
    
    for key, value in cli_obj.config.items():
        if isinstance(value, dict):
            console.print(f"\n[yellow]{key}:[/yellow]")
            for k, v in value.items():
                console.print(f"  {k}: {v}")
        else:
            console.print(f"{key}: {value}")

@config.command()
@click.argument('key')
@click.argument('value')
@click.pass_obj
def set(cli_obj, key, value):
    """Set configuration value"""
    # Try to parse value
    try:
        value = json.loads(value)
    except:
        pass
    
    # Handle nested keys
    if '.' in key:
        keys = key.split('.')
        current = cli_obj.config
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]
        current[keys[-1]] = value
    else:
        cli_obj.config[key] = value
    
    # Save config
    with open('config.yaml', 'w') as f:
        yaml.dump(cli_obj.config, f)
    
    console.print(f"[green]Set {key} = {value}[/green]")

# ============ WORKFLOW ============
@cli.group()
def workflow():
    """Complete workflow automation"""
    pass

@workflow.command()
@click.option('--collect-duration', '-c', default=300, help='Data collection duration')
@click.option('--train-optuna', '-o', default=50, help='Optuna trials for training')
@click.option('--backtest-rows', '-b', default=10000, help='Rows for backtesting')
@click.option('--simulate-duration', '-s', default=300, help='Simulation duration')
@click.pass_obj
def full(cli_obj, collect_duration, train_optuna, backtest_rows, simulate_duration):
    """Run complete workflow: collect → train → backtest → simulate"""
    console.print(Panel("[cyan]Running Full Workflow[/cyan]"))
    
    # 1. Collect data
    console.print("\n[yellow]Step 1: Collecting L2 data[/yellow]")
    ctx = click.get_current_context()
    ctx.invoke(collect, duration=collect_duration)
    
    # 2. Train model
    console.print("\n[yellow]Step 2: Training model[/yellow]")
    ctx.invoke(train, optuna_trials=train_optuna)
    
    # 3. Run backtest
    console.print("\n[yellow]Step 3: Running backtest[/yellow]")
    ctx.invoke(run, data_rows=backtest_rows)
    
    # 4. Run simulation
    console.print("\n[yellow]Step 4: Running live simulation[/yellow]")
    ctx.invoke(run, duration=simulate_duration)
    
    console.print("\n[green]✅ Full workflow complete![/green]")

# ============ MAIN ============
if __name__ == '__main__':
    # Clean up old files on first run
    if '--cleanup' in sys.argv:
        console.print("[yellow]Running cleanup first...[/yellow]")
        subprocess.run("python cleanup_project.py", shell=True)
        sys.argv.remove('--cleanup')
    
    cli()