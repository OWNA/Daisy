#!/usr/bin/env python3
"""
Consolidated Trading CLI - Unified interface for all trading operations
Combines features from multiple CLI implementations
"""

import os
import sys
import yaml
import json
import logging
import argparse
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Any

# Try to import rich for better console output (optional)
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.progress import track
    console = Console()
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    
# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TradingCLI:
    """Unified CLI for Bitcoin L2 Trading System"""
    
    def __init__(self, config_path: str = 'config.yaml'):
        self.config_path = config_path
        self.config = self.load_config()
        self.venv_python = self._get_venv_python()
        
    def load_config(self) -> Dict[str, Any]:
        """Load configuration file"""
        if os.path.exists(self.config_path):
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        else:
            logger.warning(f"Config file {self.config_path} not found")
            return {}
    
    def _get_venv_python(self) -> str:
        """Get path to Python in virtual environment"""
        if os.name == 'nt':  # Windows
            return os.path.join('venv', 'Scripts', 'python.exe')
        else:  # Unix/Linux
            return os.path.join('venv', 'bin', 'python')
    
    def _run_command(self, cmd: str, show_output: bool = True) -> subprocess.CompletedProcess:
        """Run command with proper Python environment"""
        # Use venv python if available
        if os.path.exists(self.venv_python):
            cmd = cmd.replace('python ', f'{self.venv_python} ')
        
        logger.info(f"Running: {cmd}")
        
        if show_output:
            return subprocess.run(cmd, shell=True)
        else:
            return subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    def print_header(self, title: str):
        """Print formatted header"""
        if RICH_AVAILABLE:
            console.print(Panel(title, expand=False, style="bold blue"))
        else:
            print("\n" + "="*60)
            print(f" {title}")
            print("="*60 + "\n")
    
    def print_status(self, message: str, status: str = "info"):
        """Print status message"""
        if RICH_AVAILABLE:
            color_map = {
                "info": "blue",
                "success": "green",
                "warning": "yellow",
                "error": "red"
            }
            console.print(f"[{color_map.get(status, 'white')}]{message}[/{color_map.get(status, 'white')}]")
        else:
            print(f"[{status.upper()}] {message}")
    
    # ========== COMMANDS ==========
    
    def collect_data(self, duration: int = 5):
        """Collect L2 data from Bybit"""
        self.print_header("L2 Data Collection")
        self.print_status(f"Collecting data for {duration} minutes...", "info")
        
        cmd = f"python main.py collect --duration {duration} --config {self.config_path}"
        result = self._run_command(cmd)
        
        if result.returncode == 0:
            self.print_status("Data collection completed successfully", "success")
        else:
            self.print_status("Data collection failed", "error")
        
        return result.returncode
    
    def train_model(self, data_file: Optional[str] = None, trials: int = 50):
        """Train LightGBM model"""
        self.print_header("Model Training")
        
        if data_file:
            self.print_status(f"Training on specific file: {data_file}", "info")
            cmd = f"python train_on_specific_file.py --data {data_file} --trials {trials}"
        else:
            self.print_status("Training on all available L2 data", "info")
            cmd = f"python main.py train --config {self.config_path}"
        
        result = self._run_command(cmd)
        
        if result.returncode == 0:
            self.print_status("Model training completed successfully", "success")
            self._display_model_info()
        else:
            self.print_status("Model training failed", "error")
        
        return result.returncode
    
    def run_backtest(self):
        """Run backtesting"""
        self.print_header("Backtesting")
        self.print_status("Running backtest on historical data...", "info")
        
        cmd = f"python main.py backtest --config {self.config_path}"
        result = self._run_command(cmd)
        
        if result.returncode == 0:
            self.print_status("Backtesting completed", "success")
        else:
            self.print_status("Backtesting failed", "error")
        
        return result.returncode
    
    def paper_trade(self):
        """Run paper trading"""
        self.print_header("Paper Trading")
        self.print_status("Starting paper trading (press Ctrl+C to stop)...", "info")
        
        cmd = f"python main.py trade --paper --config {self.config_path}"
        
        try:
            result = self._run_command(cmd)
            if result.returncode == 0:
                self.print_status("Paper trading stopped", "success")
            else:
                self.print_status("Paper trading error", "error")
        except KeyboardInterrupt:
            self.print_status("\nPaper trading stopped by user", "warning")
        
        return 0
    
    def live_trade(self):
        """Run live trading (with confirmation)"""
        self.print_header("LIVE TRADING")
        self.print_status("WARNING: This will execute real trades!", "warning")
        
        confirm = input("\nType 'YES' to confirm live trading: ")
        if confirm != 'YES':
            self.print_status("Live trading cancelled", "warning")
            return 1
        
        self.print_status("Starting live trading (press Ctrl+C to stop)...", "info")
        cmd = f"python main.py trade --config {self.config_path}"
        
        try:
            result = self._run_command(cmd)
            if result.returncode == 0:
                self.print_status("Live trading stopped", "success")
            else:
                self.print_status("Live trading error", "error")
        except KeyboardInterrupt:
            self.print_status("\nLive trading stopped by user", "warning")
        
        return 0
    
    def check_status(self):
        """Check system status"""
        self.print_header("System Status")
        
        # Check config
        if os.path.exists(self.config_path):
            self.print_status("✓ Config file found", "success")
        else:
            self.print_status("✗ Config file missing", "error")
        
        # Check model
        model_path = os.path.join(
            self.config.get('base_dir', './trading_bot_data'),
            f"lgbm_model_{self.config.get('symbol', 'SYMBOL').replace('/', '_').replace(':', '')}_l2_only.txt"
        )
        if os.path.exists(model_path):
            self.print_status("✓ Trained model found", "success")
            self._display_model_info()
        else:
            self.print_status("✗ No trained model found", "warning")
        
        # Check L2 data
        l2_path = self.config.get('l2_data_folder', 'l2_data')
        if os.path.exists(l2_path):
            files = [f for f in os.listdir(l2_path) if f.endswith(('.gz', '.jsonl'))]
            self.print_status(f"✓ L2 data: {len(files)} files", "success")
        else:
            self.print_status("✗ No L2 data found", "warning")
        
        # Check database
        db_path = self.config.get('database_path', './trading_bot.db')
        if os.path.exists(db_path):
            size_mb = os.path.getsize(db_path) / 1024 / 1024
            self.print_status(f"✓ Database: {size_mb:.1f} MB", "success")
        else:
            self.print_status("✗ Database not found", "warning")
        
        return 0
    
    def _display_model_info(self):
        """Display information about trained model"""
        try:
            features_path = os.path.join(
                self.config.get('base_dir', './trading_bot_data'),
                f"model_features_{self.config.get('symbol', 'SYMBOL').replace('/', '_').replace(':', '')}_l2_only.json"
            )
            
            if os.path.exists(features_path):
                with open(features_path, 'r') as f:
                    features_data = json.load(f)
                    
                if RICH_AVAILABLE:
                    table = Table(title="Model Information")
                    table.add_column("Property", style="cyan")
                    table.add_column("Value", style="green")
                    
                    table.add_row("Feature Count", str(features_data.get('n_features', 'N/A')))
                    table.add_row("Model Type", features_data.get('model_type', 'N/A'))
                    table.add_row("Training Date", features_data.get('training_date', 'N/A'))
                    
                    console.print(table)
                else:
                    print("\nModel Information:")
                    print(f"  Feature Count: {features_data.get('n_features', 'N/A')}")
                    print(f"  Model Type: {features_data.get('model_type', 'N/A')}")
                    print(f"  Training Date: {features_data.get('training_date', 'N/A')}")
        except Exception as e:
            logger.debug(f"Could not display model info: {e}")
    
    def interactive_menu(self):
        """Run interactive menu mode"""
        while True:
            self.print_header("Bitcoin L2 Trading System")
            
            print("1. Collect L2 Data")
            print("2. Train Model")
            print("3. Run Backtest")
            print("4. Paper Trading")
            print("5. Live Trading")
            print("6. Check Status")
            print("7. Exit\n")
            
            choice = input("Select option (1-7): ")
            
            if choice == '1':
                duration = input("Duration in minutes (default 5): ") or "5"
                self.collect_data(int(duration))
                input("\nPress Enter to continue...")
                
            elif choice == '2':
                self.train_model()
                input("\nPress Enter to continue...")
                
            elif choice == '3':
                self.run_backtest()
                input("\nPress Enter to continue...")
                
            elif choice == '4':
                self.paper_trade()
                input("\nPress Enter to continue...")
                
            elif choice == '5':
                self.live_trade()
                input("\nPress Enter to continue...")
                
            elif choice == '6':
                self.check_status()
                input("\nPress Enter to continue...")
                
            elif choice == '7':
                self.print_status("Exiting...", "info")
                break
                
            else:
                self.print_status("Invalid option", "error")
                input("\nPress Enter to continue...")


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description='Bitcoin L2 Trading System CLI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Commands:
  collect     Collect L2 data from Bybit
  train       Train LightGBM model on collected data
  backtest    Run backtesting on historical data
  trade       Run paper or live trading
  status      Check system status
  menu        Interactive menu mode

Examples:
  %(prog)s collect --duration 10
  %(prog)s train --trials 100
  %(prog)s trade --paper
  %(prog)s status
        """
    )
    
    parser.add_argument('command', nargs='?', 
                       choices=['collect', 'train', 'backtest', 'trade', 'status', 'menu'],
                       help='Command to execute')
    parser.add_argument('--config', default='config.yaml',
                       help='Path to config file (default: config.yaml)')
    parser.add_argument('--duration', type=int, default=5,
                       help='Data collection duration in minutes (default: 5)')
    parser.add_argument('--trials', type=int, default=50,
                       help='Optuna trials for training (default: 50)')
    parser.add_argument('--paper', action='store_true',
                       help='Run in paper trading mode')
    parser.add_argument('--data', type=str,
                       help='Specific data file for training')
    
    args = parser.parse_args()
    
    # Initialize CLI
    cli = TradingCLI(args.config)
    
    # If no command, show menu
    if not args.command:
        cli.interactive_menu()
        return 0
    
    # Execute command
    if args.command == 'collect':
        return cli.collect_data(args.duration)
        
    elif args.command == 'train':
        return cli.train_model(args.data, args.trials)
        
    elif args.command == 'backtest':
        return cli.run_backtest()
        
    elif args.command == 'trade':
        if args.paper:
            return cli.paper_trade()
        else:
            return cli.live_trade()
            
    elif args.command == 'status':
        return cli.check_status()
        
    elif args.command == 'menu':
        cli.interactive_menu()
        return 0
    
    return 0


if __name__ == "__main__":
    sys.exit(main())