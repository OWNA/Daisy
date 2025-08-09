#!/usr/bin/env python3
"""
Trading Bot CLI - Easy navigation and control for the trading system
"""
import click
import os
import sys
import subprocess
import yaml
from datetime import datetime
from pathlib import Path
import pandas as pd
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import track
from rich import print as rprint

console = Console()

# Ensure we're in the right directory
TRADE_DIR = Path(__file__).parent.absolute()
os.chdir(TRADE_DIR)

# Add current directory to Python path
sys.path.insert(0, str(TRADE_DIR))

@click.group()
@click.version_option(version='1.0.0')
def cli():
    """Trading Bot CLI - Manage your crypto trading bot with ease"""
    pass

@cli.command()
def status():
    """Check system status and configuration"""
    console.print(Panel.fit("🔍 System Status Check", style="bold blue"))
    
    # Check virtual environment
    venv_active = os.environ.get('VIRTUAL_ENV') is not None
    if venv_active:
        console.print("✅ Virtual environment: [green]Active[/green]")
    else:
        console.print("❌ Virtual environment: [red]Not active[/red]")
        console.print("   Run: [yellow]source venv/bin/activate[/yellow]")
    
    # Check database
    try:
        import sqlite3
        conn = sqlite3.connect('trading_bot.db')
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM L2_data")
        l2_count = cursor.fetchone()[0]
        conn.close()
        console.print(f"✅ Database: [green]Connected[/green] ({l2_count:,} L2 records)")
    except Exception as e:
        console.print(f"❌ Database: [red]Error - {str(e)}[/red]")
    
    # Check configurations
    config_files = ['config.yaml', 'config_l2.yaml', 'config_l2_only.yaml', 
                   'config_live_sim.yaml', 'config_wfo.yaml']
    
    console.print("\n📁 Configuration Files:")
    for config_file in config_files:
        if os.path.exists(config_file):
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
                exchange = config.get('exchange', 'Not configured')
                console.print(f"  ✅ {config_file}: [green]Found[/green] (Exchange: {exchange})")
        else:
            console.print(f"  ❌ {config_file}: [red]Missing[/red]")
    
    # Check models
    console.print("\n🤖 Trained Models:")
    model_files = list(Path('.').glob('lgbm_model_*.txt'))
    if model_files:
        for model in model_files:
            console.print(f"  ✅ {model.name}: [green]Available[/green]")
    else:
        console.print("  ❌ No trained models found")

@cli.command()
@click.option('--config', '-c', default='config_l2.yaml', help='Configuration file to use')
@click.option('--continuous', is_flag=True, help='Run continuously')
@click.option('--interval', '-i', default=60, help='Collection interval in seconds')
def collect_data(config, continuous, interval):
    """Collect L2 order book data"""
    console.print(Panel.fit(f"📊 Collecting L2 Data", style="bold blue"))
    console.print(f"Config: {config}")
    console.print(f"Mode: {'Continuous' if continuous else 'One-time'}")
    if continuous:
        console.print(f"Interval: {interval} seconds")
    
    if click.confirm('Start data collection?'):
        cmd = f"python run_l2_collector.py --config {config}"
        if continuous:
            cmd += " --continuous --interval {interval}"
        subprocess.run(cmd, shell=True)

@cli.command()
@click.option('--config', '-c', default='config_l2.yaml', help='Configuration file')
@click.option('--data-source', '-d', default='L2', help='Data source (L2 or OHLCV)')
def train_model(config, data_source):
    """Train a new trading model"""
    console.print(Panel.fit("🧠 Training Model", style="bold blue"))
    console.print(f"Config: {config}")
    console.print(f"Data source: {data_source}")
    
    if click.confirm('Start model training?'):
        if data_source == 'L2':
            cmd = f"python run_l2_training_pipeline.py --config {config}"
        else:
            cmd = f"python train_model.py --config {config}"
        subprocess.run(cmd, shell=True)

@cli.command()
@click.option('--config', '-c', default='config_l2_only.yaml', help='Configuration file')
@click.option('--l2-only', is_flag=True, default=True, help='Use L2-only mode')
def backtest(config, l2_only):
    """Run backtesting simulation"""
    console.print(Panel.fit("📈 Running Backtest", style="bold blue"))
    console.print(f"Config: {config}")
    console.print(f"Mode: {'L2-only' if l2_only else 'OHLCV'}")
    
    # Load config to get initial balance
    try:
        with open(config, 'r') as f:
            cfg = yaml.safe_load(f)
            initial_balance = cfg.get('initial_balance', 10000)
            console.print(f"Initial balance: ${initial_balance:,}")
    except:
        initial_balance = 10000
        console.print(f"Initial balance: ${initial_balance:,} (default)")
    
    cmd = f"python run_backtest.py --config {config}"
    if l2_only:
        cmd += " --l2-only"
    
    if click.confirm('Start backtest?'):
        subprocess.run(cmd, shell=True)

@cli.command()
@click.option('--config', '-c', default='config_live_sim.yaml', help='Configuration file')
@click.option('--mode', '-m', type=click.Choice(['simulation', 'paper', 'live']), 
              default='simulation', help='Trading mode')
def start_bot(config, mode):
    """Start the trading bot"""
    console.print(Panel.fit(f"🤖 Starting Trading Bot ({mode} mode)", style="bold blue"))
    console.print(f"Config: {config}")
    
    warnings = []
    if mode == 'live':
        warnings.append("⚠️  LIVE MODE - Real money will be used!")
        warnings.append("⚠️  Make sure you have tested thoroughly!")
    elif mode == 'paper':
        warnings.append("📝 Paper trading mode - No real money")
    else:
        warnings.append("🧪 Simulation mode - Testing only")
    
    for warning in warnings:
        console.print(warning, style="yellow")
    
    if click.confirm(f'Start bot in {mode} mode?'):
        if mode == 'simulation':
            cmd = f"python start_live_simulation.py --config {config}"
        else:
            cmd = f"python run_trading_bot.py --config {config} --mode {mode}"
        subprocess.run(cmd, shell=True)

@cli.command()
def results():
    """View recent trading results"""
    console.print(Panel.fit("📊 Recent Trading Results", style="bold blue"))
    
    # Check for backtest results
    results_dir = Path('backtest_results')
    if results_dir.exists():
        result_files = sorted(results_dir.glob('backtest_log_*.csv'), 
                            key=lambda x: x.stat().st_mtime, reverse=True)[:5]
        
        if result_files:
            console.print("\n🔍 Recent Backtests:")
            for file in result_files:
                console.print(f"  • {file.name}")
                try:
                    df = pd.read_csv(file)
                    if len(df) > 0:
                        last_balance = df.iloc[-1]['balance']
                        total_trades = len(df)
                        console.print(f"    Final balance: ${last_balance:,.2f}")
                        console.print(f"    Total trades: {total_trades}")
                except:
                    pass
    
    # Check for live results
    if os.path.exists('trading_bot.db'):
        try:
            import sqlite3
            conn = sqlite3.connect('trading_bot.db')
            cursor = conn.cursor()
            
            # Recent trades
            cursor.execute("""
                SELECT timestamp, symbol, action, amount, price 
                FROM trades 
                ORDER BY timestamp DESC 
                LIMIT 10
            """)
            trades = cursor.fetchall()
            
            if trades:
                console.print("\n💹 Recent Live Trades:")
                table = Table(show_header=True, header_style="bold magenta")
                table.add_column("Time", style="dim")
                table.add_column("Symbol")
                table.add_column("Action", style="bold")
                table.add_column("Amount", justify="right")
                table.add_column("Price", justify="right")
                
                for trade in trades:
                    time_str = datetime.fromisoformat(trade[0]).strftime("%Y-%m-%d %H:%M")
                    action_style = "green" if trade[2] == "BUY" else "red"
                    table.add_row(
                        time_str, trade[1], 
                        f"[{action_style}]{trade[2]}[/{action_style}]",
                        f"{trade[3]:.6f}", f"${trade[4]:,.2f}"
                    )
                
                console.print(table)
            
            conn.close()
        except Exception as e:
            console.print(f"Could not load live trades: {e}")

@cli.command()
def analyze():
    """Analyze trading performance"""
    console.print(Panel.fit("📊 Performance Analysis", style="bold blue"))
    
    analysis_scripts = [
        ('analyze_backtest_results.py', 'Backtest Analysis'),
        ('analyze_predictions.py', 'Prediction Analysis'),
        ('analyze_l2_spread_frequency.py', 'L2 Spread Analysis'),
        ('l2_performance_benchmarker.py', 'L2 Performance Benchmark')
    ]
    
    console.print("Available analyses:")
    for i, (script, name) in enumerate(analysis_scripts, 1):
        if os.path.exists(script):
            console.print(f"  {i}. {name} ✅")
        else:
            console.print(f"  {i}. {name} ❌ (not found)")
    
    choice = click.prompt('Select analysis to run (0 to cancel)', type=int, default=0)
    
    if 0 < choice <= len(analysis_scripts):
        script, name = analysis_scripts[choice - 1]
        if os.path.exists(script):
            console.print(f"\nRunning {name}...")
            subprocess.run(f"python {script}", shell=True)
        else:
            console.print(f"Script {script} not found!", style="red")

@cli.command()
def setup():
    """Run initial setup and validation"""
    console.print(Panel.fit("🔧 System Setup", style="bold blue"))
    
    steps = [
        ("Create virtual environment", "python3 -m venv venv"),
        ("Activate environment", "source venv/bin/activate"),
        ("Install dependencies", "pip install -r requirements.txt"),
        ("Run setup script", "python setup_local_environment.py"),
        ("Validate installation", "python final_validation_test.py")
    ]
    
    for step_name, cmd in steps:
        console.print(f"\n📌 {step_name}")
        console.print(f"   Command: [yellow]{cmd}[/yellow]")
        
        if click.confirm(f'Run {step_name}?'):
            if "source" in cmd:
                console.print("Please run this manually in your shell")
            else:
                subprocess.run(cmd, shell=True)

@cli.command()
@click.argument('query', nargs=-1)
def db(query):
    """Query the trading database"""
    if not query:
        # Interactive mode
        console.print(Panel.fit("🗄️  Database Query Tool", style="bold blue"))
        console.print("Available tables: L2_data, trades, predictions, performance_metrics")
        console.print("Example: trade_cli db \"SELECT COUNT(*) FROM L2_data\"")
        return
    
    query_str = ' '.join(query)
    try:
        import sqlite3
        conn = sqlite3.connect('trading_bot.db')
        df = pd.read_sql_query(query_str, conn)
        conn.close()
        
        if len(df) > 0:
            console.print(df.to_string())
        else:
            console.print("No results found")
    except Exception as e:
        console.print(f"Error: {e}", style="red")

@cli.command()
def clean():
    """Clean up temporary files and logs"""
    console.print(Panel.fit("🧹 Cleanup Tool", style="bold blue"))
    
    patterns = [
        ('*.pyc', 'Python cache files'),
        ('__pycache__', 'Python cache directories'),
        ('*.log', 'Log files'),
        ('temp_*', 'Temporary files'),
        ('.pytest_cache', 'Pytest cache')
    ]
    
    total_cleaned = 0
    for pattern, description in patterns:
        console.print(f"\n🔍 Searching for {description} ({pattern})...")
        
        if '*' in pattern:
            files = list(Path('.').rglob(pattern))
        else:
            files = list(Path('.').rglob(f'**/{pattern}'))
        
        if files:
            console.print(f"   Found {len(files)} items")
            if click.confirm(f'   Delete these {description}?'):
                for file in track(files, description=f"Deleting {description}"):
                    try:
                        if file.is_dir():
                            import shutil
                            shutil.rmtree(file)
                        else:
                            file.unlink()
                        total_cleaned += 1
                    except:
                        pass
    
    console.print(f"\n✅ Cleaned {total_cleaned} items")

if __name__ == '__main__':
    cli()