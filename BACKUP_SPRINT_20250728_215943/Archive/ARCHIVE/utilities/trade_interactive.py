#!/usr/bin/env python3
"""
Interactive Trading Bot CLI - Easy menu-driven interface
"""

import os
import sys
import subprocess
import json
import yaml
from pathlib import Path
from datetime import datetime
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.prompt import Prompt, IntPrompt, Confirm
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.layout import Layout
from rich.text import Text
import time

console = Console()

class InteractiveTradingCLI:
    def __init__(self):
        self.config_file = 'config.yaml'
        self.load_config()
        
    def load_config(self):
        """Load configuration"""
        if os.path.exists(self.config_file):
            with open(self.config_file, 'r') as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = {}
    
    def save_config(self):
        """Save configuration"""
        with open(self.config_file, 'w') as f:
            yaml.dump(self.config, f)
    
    def clear_screen(self):
        """Clear the console screen"""
        os.system('cls' if os.name == 'nt' else 'clear')
    
    def show_header(self):
        """Display header"""
        self.clear_screen()
        console.print(Panel.fit(
            "[bold cyan]🤖 Advanced Trading Bot - Interactive CLI[/bold cyan]\n" +
            "[dim]Navigate with numbers • Press Enter to confirm[/dim]",
            border_style="cyan"
        ))
        console.print()
    
    def main_menu(self):
        """Main menu"""
        while True:
            self.show_header()
            
            menu_items = [
                ("1", "📊 Data Collection", self.data_menu),
                ("2", "🧠 Model Training", self.model_menu),
                ("3", "📈 Backtesting", self.backtest_menu),
                ("4", "🤖 Live Simulation", self.simulation_menu),
                ("5", "📊 Analysis & Visualization", self.analysis_menu),
                ("6", "⚙️  Configuration", self.config_menu),
                ("7", "🚀 Quick Workflows", self.workflow_menu),
                ("8", "📋 View Status", self.status_menu),
                ("9", "❌ Exit", None)
            ]
            
            table = Table(show_header=False, box=None)
            table.add_column("", style="cyan", width=3)
            table.add_column("", style="white")
            
            for num, label, _ in menu_items:
                table.add_row(num, label)
            
            console.print(Panel(table, title="[bold]Main Menu[/bold]", border_style="green"))
            
            choice = Prompt.ask("\n[bold yellow]Select option[/bold yellow]", 
                              choices=[item[0] for item in menu_items])
            
            if choice == "9":
                console.print("\n[bold green]👋 Goodbye![/bold green]")
                break
            
            for num, _, func in menu_items:
                if choice == num and func:
                    func()
                    break
    
    def data_menu(self):
        """Data collection submenu"""
        while True:
            self.show_header()
            console.print(Panel("[bold]📊 Data Collection[/bold]", border_style="blue"))
            
            menu_items = [
                ("1", "🔄 Collect L2 Data (Live)", self.collect_l2_data),
                ("2", "📁 List Data Files", self.list_data_files),
                ("3", "🔍 Inspect Data File", self.inspect_data_file),
                ("4", "🗑️  Clean Old Data", self.clean_old_data),
                ("5", "⬅️  Back to Main Menu", None)
            ]
            
            table = Table(show_header=False, box=None)
            table.add_column("", style="cyan", width=3)
            table.add_column("", style="white")
            
            for num, label, _ in menu_items:
                table.add_row(num, label)
            
            console.print(table)
            
            choice = Prompt.ask("\n[bold yellow]Select option[/bold yellow]", 
                              choices=[item[0] for item in menu_items])
            
            if choice == "5":
                break
            
            for num, _, func in menu_items:
                if choice == num and func:
                    func()
                    break
    
    def collect_l2_data(self):
        """Collect L2 order book data"""
        console.print("\n[bold cyan]📊 L2 Data Collection[/bold cyan]")
        
        # Get duration
        duration = IntPrompt.ask(
            "Collection duration in seconds",
            default=300
        )
        
        # Get symbol
        symbol = Prompt.ask(
            "Trading symbol",
            default=self.config.get('symbol', 'BTC/USDT:USDT')
        )
        
        # Append option
        append = Confirm.ask("Append to existing data?", default=False)
        
        console.print(f"\n[yellow]Collecting {symbol} data for {duration} seconds...[/yellow]")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task(f"Collecting L2 data...", total=None)
            
            cmd = f"python run_l2_collector.py --continuous --interval {duration}"
            if append:
                cmd += " --append"
            
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            
            if result.returncode == 0:
                console.print("[green]✅ Data collection complete![/green]")
            else:
                console.print(f"[red]❌ Error: {result.stderr}[/red]")
        
        Prompt.ask("\nPress Enter to continue")
    
    def list_data_files(self):
        """List L2 data files"""
        console.print("\n[bold cyan]📁 L2 Data Files[/bold cyan]")
        
        l2_dir = Path('l2_data')
        if not l2_dir.exists():
            console.print("[red]No L2 data directory found[/red]")
            Prompt.ask("\nPress Enter to continue")
            return
        
        files = sorted(l2_dir.glob('*.jsonl.gz'), key=lambda x: x.stat().st_mtime, reverse=True)
        
        if not files:
            console.print("[yellow]No data files found[/yellow]")
        else:
            table = Table(title="Available Data Files")
            table.add_column("File", style="cyan")
            table.add_column("Size (MB)", style="green")
            table.add_column("Modified", style="yellow")
            
            for file in files[:10]:  # Show last 10
                size_mb = file.stat().st_size / 1024 / 1024
                modified = datetime.fromtimestamp(file.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S')
                table.add_row(file.name, f"{size_mb:.2f}", modified)
            
            console.print(table)
        
        Prompt.ask("\nPress Enter to continue")
    
    def inspect_data_file(self):
        """Inspect a data file"""
        # First list files
        l2_dir = Path('l2_data')
        if not l2_dir.exists():
            console.print("[red]No L2 data directory found[/red]")
            Prompt.ask("\nPress Enter to continue")
            return
        
        files = sorted(l2_dir.glob('*.jsonl.gz'), key=lambda x: x.stat().st_mtime, reverse=True)[:10]
        
        if not files:
            console.print("[yellow]No data files found[/yellow]")
            Prompt.ask("\nPress Enter to continue")
            return
        
        console.print("\n[bold cyan]Select file to inspect:[/bold cyan]")
        for i, file in enumerate(files, 1):
            console.print(f"{i}. {file.name}")
        
        choice = IntPrompt.ask("Select file number", default=1)
        if 1 <= choice <= len(files):
            selected_file = files[choice - 1]
            
            console.print(f"\n[yellow]Inspecting {selected_file.name}...[/yellow]")
            cmd = f"python trade_cli_advanced.py data inspect {selected_file.name} -r 50"
            subprocess.run(cmd, shell=True)
        
        Prompt.ask("\nPress Enter to continue")
    
    def clean_old_data(self):
        """Clean old data files"""
        console.print("\n[bold yellow]🗑️  Clean Old Data Files[/bold yellow]")
        
        l2_dir = Path('l2_data')
        if not l2_dir.exists():
            console.print("[red]No L2 data directory found[/red]")
            Prompt.ask("\nPress Enter to continue")
            return
        
        files = list(l2_dir.glob('*.jsonl.gz'))
        if not files:
            console.print("[green]No data files to clean[/green]")
            Prompt.ask("\nPress Enter to continue")
            return
        
        console.print(f"Found {len(files)} data files")
        keep = IntPrompt.ask("How many recent files to keep?", default=5)
        
        if Confirm.ask(f"Delete all but the {keep} most recent files?"):
            files_sorted = sorted(files, key=lambda x: x.stat().st_mtime, reverse=True)
            for file in files_sorted[keep:]:
                file.unlink()
                console.print(f"[red]Deleted: {file.name}[/red]")
            
            console.print(f"\n[green]✅ Kept {keep} most recent files[/green]")
        
        Prompt.ask("\nPress Enter to continue")
    
    def model_menu(self):
        """Model training submenu"""
        while True:
            self.show_header()
            console.print(Panel("[bold]🧠 Model Training[/bold]", border_style="blue"))
            
            menu_items = [
                ("1", "🚀 Quick Train (10 trials)", lambda: self.train_model(10)),
                ("2", "💪 Full Train (50 trials)", lambda: self.train_model(50)),
                ("3", "🔥 Advanced Train (100 trials)", lambda: self.train_model(100)),
                ("4", "📋 List Models", self.list_models),
                ("5", "🔍 Inspect Model", self.inspect_model),
                ("6", "⬅️  Back to Main Menu", None)
            ]
            
            table = Table(show_header=False, box=None)
            table.add_column("", style="cyan", width=3)
            table.add_column("", style="white")
            
            for num, label, _ in menu_items:
                table.add_row(num, label)
            
            console.print(table)
            
            choice = Prompt.ask("\n[bold yellow]Select option[/bold yellow]", 
                              choices=[item[0] for item in menu_items])
            
            if choice == "6":
                break
            
            for num, _, func in menu_items:
                if choice == num and func:
                    func()
                    break
    
    def train_model(self, trials):
        """Train model with specified trials"""
        console.print(f"\n[bold cyan]🧠 Training Model ({trials} Optuna trials)[/bold cyan]")
        
        # Feature selection
        console.print("\nSelect features to use:")
        console.print("1. All features (L2 + HHT)")
        console.print("2. L2 features only")
        console.print("3. HHT features only")
        
        feature_choice = Prompt.ask("Select features", choices=["1", "2", "3"], default="1")
        features = {"1": "all", "2": "l2", "3": "hht"}[feature_choice]
        
        # Optional: specific data file
        use_specific = Confirm.ask("Use specific data file?", default=False)
        data_file = ""
        if use_specific:
            # List all files
            l2_dir = Path('l2_data')
            all_files = sorted(l2_dir.glob('*.jsonl.gz'), key=lambda x: x.stat().st_mtime, reverse=True)
            
            if all_files:
                # Show all files with pagination
                console.print(f"\n[cyan]Found {len(all_files)} data files:[/cyan]")
                
                # Show more files
                show_count = min(20, len(all_files))  # Show up to 20 files
                
                for i, file in enumerate(all_files[:show_count], 1):
                    size_mb = file.stat().st_size / 1024 / 1024
                    # Highlight the file you're looking for
                    if "040413" in file.name:
                        console.print(f"[bold yellow]{i}. {file.name} ({size_mb:.1f} MB) ← YOUR FILE[/bold yellow]")
                    else:
                        console.print(f"{i}. {file.name} ({size_mb:.1f} MB)")
                
                if len(all_files) > show_count:
                    console.print(f"[dim]... and {len(all_files) - show_count} more files[/dim]")
                
                # Allow entering filename directly
                console.print("\n[cyan]Enter number (1-{}) or type filename directly:[/cyan]".format(show_count))
                choice_str = Prompt.ask("Selection", default="1")
                
                try:
                    # Try as number first
                    choice = int(choice_str)
                    if 1 <= choice <= len(all_files):
                        data_file = all_files[choice - 1].name
                        console.print(f"[green]Selected: {data_file}[/green]")
                    else:
                        console.print("[red]Invalid number[/red]")
                        data_file = ""
                except ValueError:
                    # Try as filename
                    if choice_str.endswith('.gz'):
                        # Full filename provided
                        matching_files = [f for f in all_files if f.name == choice_str]
                        if matching_files:
                            data_file = matching_files[0].name
                            console.print(f"[green]Selected: {data_file}[/green]")
                        else:
                            console.print(f"[red]File not found: {choice_str}[/red]")
                    else:
                        # Partial filename - search for it
                        matching_files = [f for f in all_files if choice_str in f.name]
                        if len(matching_files) == 1:
                            data_file = matching_files[0].name
                            console.print(f"[green]Found and selected: {data_file}[/green]")
                        elif len(matching_files) > 1:
                            console.print(f"[yellow]Multiple matches for '{choice_str}':[/yellow]")
                            for i, f in enumerate(matching_files[:5], 1):
                                console.print(f"  {i}. {f.name}")
                            console.print("[red]Please be more specific[/red]")
                        else:
                            console.print(f"[red]No files matching '{choice_str}'[/red]")
            else:
                console.print("[red]No data files found![/red]")
        
        console.print(f"\n[yellow]Training with {features} features, {trials} trials...[/yellow]")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Training model...", total=None)
            
            # Use the WORKING training script
            cmd = f"python train_direct_fixed.py"
            if data_file:
                cmd += f" l2_data/{data_file}"
            cmd += f" {trials}"
            
            # Run in visible mode to see progress
            result = subprocess.run(cmd, shell=True)
            
            if result.returncode == 0:
                console.print("\n[green]✅ Model training complete![/green]")
                
                # Show latest model files (all patterns)
                model_files = sorted(Path('.').glob('lgbm_model_*.txt'), 
                                   key=lambda x: x.stat().st_mtime, reverse=True)
                if model_files:
                    console.print(f"[cyan]Latest model: {model_files[0].name}[/cyan]")
                    # Show R² if available
                    features_file = model_files[0].name.replace('lgbm_model_', 'model_features_').replace('.txt', '.json')
                    if os.path.exists(features_file):
                        with open(features_file, 'r') as f:
                            meta = json.load(f)
                        if 'r2_score' in meta:
                            console.print(f"[green]R² score: {meta['r2_score']:.4f}[/green]")
                        if 'hht_contribution_pct' in meta:
                            console.print(f"[green]HHT contribution: {meta['hht_contribution_pct']:.1f}%[/green]")
            else:
                console.print(f"\n[red]❌ Error during training[/red]")
        
        Prompt.ask("\nPress Enter to continue")
    
    def list_models(self):
        """List available models"""
        console.print("\n[bold cyan]📋 Available Models[/bold cyan]")
        
        cmd = "python trade_cli_advanced.py model list"
        subprocess.run(cmd, shell=True)
        
        Prompt.ask("\nPress Enter to continue")
    
    def inspect_model(self):
        """Inspect model features"""
        models = list(Path('.').glob('lgbm_model_*.txt'))
        
        if not models:
            console.print("[red]No models found[/red]")
            Prompt.ask("\nPress Enter to continue")
            return
        
        console.print("\n[bold cyan]Select model to inspect:[/bold cyan]")
        for i, model in enumerate(models, 1):
            console.print(f"{i}. {model.name}")
        
        choice = IntPrompt.ask("Select model number", default=1)
        if 1 <= choice <= len(models):
            selected_model = models[choice - 1]
            
            top_n = IntPrompt.ask("Show top N features", default=20)
            
            console.print(f"\n[yellow]Inspecting {selected_model.name}...[/yellow]")
            cmd = f"python trade_cli_advanced.py model inspect {selected_model.name} -t {top_n}"
            subprocess.run(cmd, shell=True)
        
        Prompt.ask("\nPress Enter to continue")
    
    def backtest_menu(self):
        """Backtesting submenu"""
        while True:
            self.show_header()
            console.print(Panel("[bold]📈 Backtesting[/bold]", border_style="blue"))
            
            menu_items = [
                ("1", "🏃 Quick Backtest (1k rows)", lambda: self.run_backtest(1000)),
                ("2", "📊 Standard Backtest (10k rows)", lambda: self.run_backtest(10000)),
                ("3", "💪 Full Backtest (all data)", lambda: self.run_backtest(None)),
                ("4", "📈 Visualize Results", self.visualize_backtest),
                ("5", "📋 List Results", self.list_backtest_results),
                ("6", "⬅️  Back to Main Menu", None)
            ]
            
            table = Table(show_header=False, box=None)
            table.add_column("", style="cyan", width=3)
            table.add_column("", style="white")
            
            for num, label, _ in menu_items:
                table.add_row(num, label)
            
            console.print(table)
            
            choice = Prompt.ask("\n[bold yellow]Select option[/bold yellow]", 
                              choices=[item[0] for item in menu_items])
            
            if choice == "6":
                break
            
            for num, _, func in menu_items:
                if choice == num and func:
                    func()
                    break
    
    def run_backtest(self, rows):
        """Run backtest"""
        console.print(f"\n[bold cyan]📈 Running Backtest{f' ({rows} rows)' if rows else ''}[/bold cyan]")
        
        # Select model
        models = list(Path('.').glob('lgbm_model_*.txt'))
        if not models:
            console.print("[red]No models found! Train a model first.[/red]")
            Prompt.ask("\nPress Enter to continue")
            return
        
        console.print("\nAvailable models:")
        for i, model in enumerate(models, 1):
            console.print(f"{i}. {model.name}")
        
        choice = IntPrompt.ask("Select model", default=1)
        if not (1 <= choice <= len(models)):
            return
        
        selected_model = models[choice - 1]
        
        console.print(f"\n[yellow]Running backtest with {selected_model.name}...[/yellow]")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Running backtest...", total=None)
            
            # Use the simple backtest that works
            cmd = f"python backtest_simple.py {selected_model.name}"
            if rows:
                cmd += f" '' {rows}"  # Empty string for data_file to use latest
            
            result = subprocess.run(cmd, shell=True)
            
            if result.returncode == 0:
                console.print("[green]✅ Backtest complete![/green]")
                # Try to extract some stats from output
                if "Final equity:" in result.stdout:
                    console.print(result.stdout.split("Final equity:")[-1].split("\n")[0])
            else:
                console.print(f"[red]❌ Error during backtest[/red]")
        
        Prompt.ask("\nPress Enter to continue")
    
    def visualize_backtest(self):
        """Visualize backtest results"""
        results_dir = Path('backtest_results')
        if not results_dir.exists():
            console.print("[red]No backtest results directory found[/red]")
            Prompt.ask("\nPress Enter to continue")
            return
        
        results = sorted(results_dir.glob('*.csv'), key=lambda x: x.stat().st_mtime, reverse=True)
        
        if not results:
            console.print("[yellow]No backtest results found[/yellow]")
            Prompt.ask("\nPress Enter to continue")
            return
        
        console.print("\n[bold cyan]Select result to visualize:[/bold cyan]")
        for i, result in enumerate(results[:10], 1):
            console.print(f"{i}. {result.name}")
        
        choice = IntPrompt.ask("Select result number", default=1)
        if 1 <= choice <= len(results):
            selected_result = results[choice - 1]
            
            console.print("\nVisualization type:")
            console.print("1. Interactive (Plotly)")
            console.print("2. Static (Seaborn)")
            console.print("3. 🌐 Web Dashboard (Live Server)")
            
            viz_choice = Prompt.ask("Select type", choices=["1", "2", "3"], default="3")
            
            if viz_choice == "3":
                # Launch web visualization server
                console.print(f"\n[yellow]Starting web dashboard...[/yellow]")
                console.print("[dim]Press Ctrl+C in the terminal to stop the server[/dim]")
                
                cmd = f"python trade_visualizer.py --backtest {selected_result}"
                subprocess.run(cmd, shell=True)
            else:
                plot_type = "plotly" if viz_choice == "1" else "seaborn"
                
                console.print(f"\n[yellow]Creating visualization...[/yellow]")
                cmd = f"python trade_cli_advanced.py backtest visualize {selected_result} -p {plot_type}"
                subprocess.run(cmd, shell=True)
                
                if plot_type == "plotly":
                    console.print("[green]✅ Visualization saved to backtest_visualization.html[/green]")
                    if Confirm.ask("Open in browser?"):
                        os.system("start backtest_visualization.html")
                else:
                    console.print("[green]✅ Visualization saved to backtest_visualization.png[/green]")
        
        Prompt.ask("\nPress Enter to continue")
    
    def list_backtest_results(self):
        """List backtest results"""
        console.print("\n[bold cyan]📋 Backtest Results[/bold cyan]")
        
        cmd = "python trade_cli_advanced.py analyze performance"
        subprocess.run(cmd, shell=True)
        
        Prompt.ask("\nPress Enter to continue")
    
    def simulation_menu(self):
        """Live simulation submenu"""
        while True:
            self.show_header()
            console.print(Panel("[bold]🤖 Live Simulation[/bold]", border_style="blue"))
            
            menu_items = [
                ("1", "🚀 Quick Simulation (5 min)", lambda: self.run_simulation(300, False)),
                ("2", "📊 Standard Simulation (15 min)", lambda: self.run_simulation(900, False)),
                ("3", "💼 Paper Trading (10 min)", lambda: self.run_simulation(600, True)),
                ("4", "📋 View Results", self.view_simulation_results),
                ("5", "⬅️  Back to Main Menu", None)
            ]
            
            table = Table(show_header=False, box=None)
            table.add_column("", style="cyan", width=3)
            table.add_column("", style="white")
            
            for num, label, _ in menu_items:
                table.add_row(num, label)
            
            console.print(table)
            
            choice = Prompt.ask("\n[bold yellow]Select option[/bold yellow]", 
                              choices=[item[0] for item in menu_items])
            
            if choice == "5":
                break
            
            for num, _, func in menu_items:
                if choice == num and func:
                    func()
                    break
    
    def run_simulation(self, duration, paper):
        """Run live simulation"""
        mode = "Paper Trading" if paper else "Live Simulation"
        console.print(f"\n[bold cyan]🤖 {mode} ({duration//60} minutes)[/bold cyan]")
        
        # Select model
        models = list(Path('.').glob('lgbm_model_*.txt'))
        if not models:
            console.print("[red]No models found! Train a model first.[/red]")
            Prompt.ask("\nPress Enter to continue")
            return
        
        if len(models) > 1:
            console.print("\nAvailable models:")
            for i, model in enumerate(models, 1):
                console.print(f"{i}. {model.name}")
            
            choice = IntPrompt.ask("Select model", default=1)
            if not (1 <= choice <= len(models)):
                return
            
            selected_model = models[choice - 1]
        else:
            selected_model = models[0]
        
        # Ask about real-time monitoring
        monitor_live = Confirm.ask("\nLaunch real-time monitoring dashboard?", default=True)
        
        if monitor_live:
            # Start monitoring server in background
            console.print(f"\n[green]Starting monitoring dashboard at http://localhost:5000[/green]")
            
            # Generate a temp file name for live results
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            live_file = f"live_results_{timestamp}.csv"
            
            # Start visualizer in background
            import threading
            def run_visualizer():
                subprocess.run(f"python trade_visualizer.py --live {live_file} --no-browser", shell=True)
            
            viz_thread = threading.Thread(target=run_visualizer, daemon=True)
            viz_thread.start()
            
            # Give server time to start
            time.sleep(2)
            
            # Open browser
            import webbrowser
            webbrowser.open('http://localhost:5000')
            
            console.print(f"\n[yellow]Starting {mode} with live monitoring...[/yellow]")
            console.print("[dim]Dashboard will update in real-time[/dim]")
            console.print("[dim]Press Ctrl+C to stop[/dim]")
            
            # Use wrapper script for live output
            cmd = f"python run_live_simulation_with_output.py --duration {duration} --model {selected_model.name} --output {live_file}"
        else:
            console.print(f"\n[yellow]Starting {mode} with {selected_model.name}...[/yellow]")
            console.print("[dim]Press Ctrl+C to stop early[/dim]")
            
            cmd = f"python trade_cli_advanced.py simulate run -d {duration} -m {selected_model.name}"
            if paper:
                cmd += " --paper"
        
        try:
            subprocess.run(cmd, shell=True)
            console.print(f"\n[green]✅ {mode} complete![/green]")
        except KeyboardInterrupt:
            console.print(f"\n[yellow]⚠️  {mode} stopped by user[/yellow]")
        
        Prompt.ask("\nPress Enter to continue")
    
    def view_simulation_results(self):
        """View simulation results"""
        console.print("\n[bold cyan]📋 Simulation Results[/bold cyan]")
        
        # Check both directories
        for dir_name in ['paper_trading_results', 'simulation_results']:
            dir_path = Path(dir_name)
            if dir_path.exists():
                files = list(dir_path.glob('*.json'))
                if files:
                    console.print(f"\n[yellow]{dir_name}:[/yellow]")
                    for file in sorted(files, key=lambda x: x.stat().st_mtime, reverse=True)[:5]:
                        console.print(f"  {file.name}")
        
        Prompt.ask("\nPress Enter to continue")
    
    def analysis_menu(self):
        """Analysis submenu"""
        while True:
            self.show_header()
            console.print(Panel("[bold]📊 Analysis & Visualization[/bold]", border_style="blue"))
            
            menu_items = [
                ("1", "📊 SHAP Analysis", self.shap_analysis),
                ("2", "📈 Performance Summary", self.performance_summary),
                ("3", "🔍 Feature Importance", self.feature_importance),
                ("4", "📤 Export Results", self.export_results),
                ("5", "🌐 Launch Web Dashboard", self.launch_dashboard),
                ("6", "⬅️  Back to Main Menu", None)
            ]
            
            table = Table(show_header=False, box=None)
            table.add_column("", style="cyan", width=3)
            table.add_column("", style="white")
            
            for num, label, _ in menu_items:
                table.add_row(num, label)
            
            console.print(table)
            
            choice = Prompt.ask("\n[bold yellow]Select option[/bold yellow]", 
                              choices=[item[0] for item in menu_items])
            
            if choice == "6":
                break
            
            for num, _, func in menu_items:
                if choice == num and func:
                    func()
                    break
    
    def shap_analysis(self):
        """Generate SHAP analysis"""
        console.print("\n[bold cyan]📊 SHAP Analysis[/bold cyan]")
        
        models = list(Path('.').glob('lgbm_model_*.txt'))
        if not models:
            console.print("[red]No models found![/red]")
            Prompt.ask("\nPress Enter to continue")
            return
        
        console.print("\nSelect model for SHAP analysis:")
        for i, model in enumerate(models, 1):
            console.print(f"{i}. {model.name}")
        
        choice = IntPrompt.ask("Select model", default=1)
        if 1 <= choice <= len(models):
            selected_model = models[choice - 1]
            
            console.print(f"\n[yellow]Generating SHAP analysis for {selected_model.name}...[/yellow]")
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console
            ) as progress:
                task = progress.add_task("Generating SHAP plots...", total=None)
                
                cmd = f"python trade_cli_advanced.py analyze shap -m {selected_model.name}"
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
                
                if "SHAP analysis complete!" in result.stdout or result.returncode == 0:
                    console.print("[green]✅ SHAP plots saved:[/green]")
                    console.print("  - shap_summary.png")
                    console.print("  - shap_importance.png")
                else:
                    console.print(f"[red]❌ Error generating SHAP analysis[/red]")
        
        Prompt.ask("\nPress Enter to continue")
    
    def performance_summary(self):
        """Show performance summary"""
        console.print("\n[bold cyan]📈 Performance Summary[/bold cyan]")
        
        cmd = "python trade_cli_advanced.py analyze performance"
        subprocess.run(cmd, shell=True)
        
        Prompt.ask("\nPress Enter to continue")
    
    def feature_importance(self):
        """Show feature importance"""
        models = list(Path('.').glob('lgbm_model_*.txt'))
        
        if not models:
            console.print("[red]No models found![/red]")
            Prompt.ask("\nPress Enter to continue")
            return
        
        console.print("\n[bold cyan]Select model:[/bold cyan]")
        for i, model in enumerate(models, 1):
            console.print(f"{i}. {model.name}")
        
        choice = IntPrompt.ask("Select model", default=1)
        if 1 <= choice <= len(models):
            selected_model = models[choice - 1]
            
            cmd = f"python trade_cli_advanced.py model inspect {selected_model.name} -t 30"
            subprocess.run(cmd, shell=True)
        
        Prompt.ask("\nPress Enter to continue")
    
    def export_results(self):
        """Export results in various formats"""
        console.print("\n[bold cyan]📤 Export Results[/bold cyan]")
        
        # Choose what to export
        console.print("\nWhat would you like to export?")
        console.print("1. Backtest Results")
        console.print("2. Model Performance Metrics")
        console.print("3. Feature Importance Data")
        console.print("4. Complete Report (All of the above)")
        
        export_choice = Prompt.ask("Select data to export", choices=["1", "2", "3", "4"])
        
        # Choose format
        console.print("\nExport format:")
        console.print("1. HTML Report")
        console.print("2. PDF Report")
        console.print("3. CSV Data")
        console.print("4. Excel Workbook")
        console.print("5. All formats")
        
        format_choice = Prompt.ask("Select format", choices=["1", "2", "3", "4", "5"])
        
        console.print(f"\n[yellow]Exporting data...[/yellow]")
        
        # Create exports directory
        export_dir = Path('exports')
        export_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Export based on choices
        if format_choice in ["1", "5"]:  # HTML
            filename = export_dir / f"trading_report_{timestamp}.html"
            self.create_html_report(filename, export_choice)
            console.print(f"[green]✅ HTML report saved: {filename}[/green]")
        
        if format_choice in ["2", "5"]:  # PDF
            filename = export_dir / f"trading_report_{timestamp}.pdf"
            console.print("[yellow]PDF export requires additional setup[/yellow]")
        
        if format_choice in ["3", "5"]:  # CSV
            if export_choice in ["1", "4"]:
                # Export backtest results
                results_dir = Path('backtest_results')
                if results_dir.exists():
                    for csv_file in results_dir.glob('*.csv'):
                        export_file = export_dir / f"backtest_{csv_file.name}"
                        import shutil
                        shutil.copy2(csv_file, export_file)
                        console.print(f"[green]✅ CSV exported: {export_file.name}[/green]")
        
        if format_choice in ["4", "5"]:  # Excel
            console.print("[yellow]Excel export requires openpyxl package[/yellow]")
        
        if Confirm.ask("\nOpen export directory?"):
            os.system(f"explorer {export_dir.absolute()}")
        
        Prompt.ask("\nPress Enter to continue")
    
    def create_html_report(self, filename, export_type):
        """Create HTML report with results"""
        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Trading Bot Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                h1 { color: #333; }
                .section { margin: 20px 0; padding: 20px; background: #f5f5f5; border-radius: 10px; }
                table { width: 100%; border-collapse: collapse; }
                th, td { padding: 10px; text-align: left; border-bottom: 1px solid #ddd; }
                .metric { font-size: 24px; font-weight: bold; color: #007bff; }
            </style>
        </head>
        <body>
            <h1>Trading Bot Performance Report</h1>
            <p>Generated: {timestamp}</p>
            
            <div class="section">
                <h2>Summary</h2>
                <p>This report contains the analysis results from the trading bot.</p>
            </div>
            
            <div class="section">
                <h2>Performance Metrics</h2>
                <table>
                    <tr><th>Metric</th><th>Value</th></tr>
                    <tr><td>Total Trades</td><td class="metric">-</td></tr>
                    <tr><td>Win Rate</td><td class="metric">-</td></tr>
                    <tr><td>Total Return</td><td class="metric">-</td></tr>
                </table>
            </div>
        </body>
        </html>
        """.format(timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        
        with open(filename, 'w') as f:
            f.write(html_content)
    
    def launch_dashboard(self):
        """Launch web dashboard for analysis"""
        console.print("\n[bold cyan]🌐 Web Dashboard[/bold cyan]")
        
        console.print("\nSelect data source:")
        console.print("1. Latest Backtest Results")
        console.print("2. Specific Result File")
        console.print("3. Empty Dashboard (Add data later)")
        
        source_choice = Prompt.ask("Select option", choices=["1", "2", "3"])
        
        if source_choice == "1":
            # Find latest backtest
            results_dir = Path('backtest_results')
            if results_dir.exists():
                results = sorted(results_dir.glob('*.csv'), key=lambda x: x.stat().st_mtime, reverse=True)
                if results:
                    data_file = results[0]
                else:
                    data_file = None
            else:
                data_file = None
        elif source_choice == "2":
            # List files to choose
            results_dir = Path('backtest_results')
            if results_dir.exists():
                results = sorted(results_dir.glob('*.csv'), key=lambda x: x.stat().st_mtime, reverse=True)[:10]
                if results:
                    console.print("\nAvailable files:")
                    for i, file in enumerate(results, 1):
                        console.print(f"{i}. {file.name}")
                    
                    choice = IntPrompt.ask("Select file", default=1)
                    if 1 <= choice <= len(results):
                        data_file = results[choice - 1]
                    else:
                        data_file = None
                else:
                    data_file = None
            else:
                data_file = None
        else:
            data_file = None
        
        console.print(f"\n[green]Starting web dashboard at http://localhost:5000[/green]")
        console.print("[dim]The dashboard will open in your browser[/dim]")
        console.print("[dim]Press Ctrl+C to stop the server[/dim]")
        
        cmd = "python trade_visualizer.py"
        if data_file:
            cmd += f" --backtest {data_file}"
        
        try:
            subprocess.run(cmd, shell=True)
        except KeyboardInterrupt:
            console.print("\n[yellow]Dashboard stopped[/yellow]")
        
        Prompt.ask("\nPress Enter to continue")
    
    def config_menu(self):
        """Configuration submenu"""
        while True:
            self.show_header()
            console.print(Panel("[bold]⚙️  Configuration[/bold]", border_style="blue"))
            
            menu_items = [
                ("1", "📋 View Config", self.view_config),
                ("2", "🔧 Update Symbol", self.update_symbol),
                ("3", "💰 Update Balance", self.update_balance),
                ("4", "📊 Update Risk Settings", self.update_risk),
                ("5", "⬅️  Back to Main Menu", None)
            ]
            
            table = Table(show_header=False, box=None)
            table.add_column("", style="cyan", width=3)
            table.add_column("", style="white")
            
            for num, label, _ in menu_items:
                table.add_row(num, label)
            
            console.print(table)
            
            choice = Prompt.ask("\n[bold yellow]Select option[/bold yellow]", 
                              choices=[item[0] for item in menu_items])
            
            if choice == "5":
                break
            
            for num, _, func in menu_items:
                if choice == num and func:
                    func()
                    break
    
    def view_config(self):
        """View current configuration"""
        console.print("\n[bold cyan]📋 Current Configuration[/bold cyan]")
        
        cmd = "python trade_cli_advanced.py config show"
        subprocess.run(cmd, shell=True)
        
        Prompt.ask("\nPress Enter to continue")
    
    def update_symbol(self):
        """Update trading symbol"""
        current = self.config.get('symbol', 'BTC/USDT:USDT')
        console.print(f"\n[bold cyan]Current symbol: {current}[/bold cyan]")
        
        new_symbol = Prompt.ask("Enter new symbol", default=current)
        
        cmd = f'python trade_cli_advanced.py config set symbol "{new_symbol}"'
        subprocess.run(cmd, shell=True)
        
        self.load_config()  # Reload config
        console.print(f"[green]✅ Symbol updated to: {new_symbol}[/green]")
        
        Prompt.ask("\nPress Enter to continue")
    
    def update_balance(self):
        """Update initial balance"""
        current = self.config.get('initial_balance', 10000)
        console.print(f"\n[bold cyan]Current balance: ${current:,.2f}[/bold cyan]")
        
        new_balance = IntPrompt.ask("Enter new balance", default=current)
        
        cmd = f"python trade_cli_advanced.py config set initial_balance {new_balance}"
        subprocess.run(cmd, shell=True)
        
        self.load_config()  # Reload config
        console.print(f"[green]✅ Balance updated to: ${new_balance:,.2f}[/green]")
        
        Prompt.ask("\nPress Enter to continue")
    
    def update_risk(self):
        """Update risk settings"""
        console.print("\n[bold cyan]🎯 Risk Settings[/bold cyan]")
        
        # Position size
        current_size = self.config.get('risk_management', {}).get('position_size_pct', 0.1)
        console.print(f"\nCurrent position size: {current_size*100:.1f}%")
        new_size = IntPrompt.ask("New position size %", default=int(current_size*100)) / 100
        
        # Max drawdown
        current_dd = self.config.get('risk_management', {}).get('max_drawdown', 0.2)
        console.print(f"\nCurrent max drawdown: {current_dd*100:.1f}%")
        new_dd = IntPrompt.ask("New max drawdown %", default=int(current_dd*100)) / 100
        
        # Update
        cmd1 = f"python trade_cli_advanced.py config set risk_management.position_size_pct {new_size}"
        cmd2 = f"python trade_cli_advanced.py config set risk_management.max_drawdown {new_dd}"
        
        subprocess.run(cmd1, shell=True)
        subprocess.run(cmd2, shell=True)
        
        console.print("[green]✅ Risk settings updated![/green]")
        
        Prompt.ask("\nPress Enter to continue")
    
    def workflow_menu(self):
        """Workflow submenu"""
        while True:
            self.show_header()
            console.print(Panel("[bold]🚀 Quick Workflows[/bold]", border_style="blue"))
            
            menu_items = [
                ("1", "⚡ Quick Test (5min collect, 10 trials, quick backtest)", 
                 lambda: self.run_workflow(300, 10, 1000, 300)),
                ("2", "📊 Standard Run (15min collect, 50 trials, full test)", 
                 lambda: self.run_workflow(900, 50, 10000, 600)),
                ("3", "💪 Full Production (30min collect, 100 trials, complete test)", 
                 lambda: self.run_workflow(1800, 100, 20000, 900)),
                ("4", "⬅️  Back to Main Menu", None)
            ]
            
            table = Table(show_header=False, box=None)
            table.add_column("", style="cyan", width=3)
            table.add_column("", style="white")
            
            for num, label, _ in menu_items:
                table.add_row(num, label)
            
            console.print(table)
            
            choice = Prompt.ask("\n[bold yellow]Select option[/bold yellow]", 
                              choices=[item[0] for item in menu_items])
            
            if choice == "4":
                break
            
            for num, _, func in menu_items:
                if choice == num and func:
                    func()
                    break
    
    def run_workflow(self, collect_time, trials, backtest_rows, sim_time):
        """Run complete workflow"""
        console.print(f"\n[bold cyan]🚀 Running Complete Workflow[/bold cyan]")
        console.print(f"  • Collect: {collect_time}s")
        console.print(f"  • Train: {trials} trials")
        console.print(f"  • Backtest: {backtest_rows} rows")
        console.print(f"  • Simulate: {sim_time}s")
        
        if not Confirm.ask("\nProceed?", default=True):
            return
        
        console.print("\n[yellow]Starting workflow...[/yellow]")
        
        cmd = (f"python trade_cli_advanced.py workflow full "
               f"-c {collect_time} -o {trials} -b {backtest_rows} -s {sim_time}")
        
        try:
            subprocess.run(cmd, shell=True)
            console.print("\n[green]✅ Workflow complete![/green]")
        except KeyboardInterrupt:
            console.print("\n[yellow]⚠️  Workflow interrupted[/yellow]")
        
        Prompt.ask("\nPress Enter to continue")
    
    def status_menu(self):
        """Status overview"""
        self.show_header()
        console.print(Panel("[bold]📋 System Status[/bold]", border_style="blue"))
        
        # Data files
        l2_dir = Path('l2_data')
        if l2_dir.exists():
            data_files = len(list(l2_dir.glob('*.jsonl.gz')))
        else:
            data_files = 0
        
        # Models
        models = len(list(Path('.').glob('lgbm_model_*.txt')))
        
        # Best model
        best_model = None
        if Path('lgbm_model_BTC_USDTUSDT_optuna_l2_hht.txt').exists():
            best_model = "lgbm_model_BTC_USDTUSDT_optuna_l2_hht.txt"
        
        # Results
        backtest_results = 0
        if Path('backtest_results').exists():
            backtest_results = len(list(Path('backtest_results').glob('*.csv')))
        
        table = Table(title="System Overview", box=None)
        table.add_column("Component", style="cyan")
        table.add_column("Status", style="green")
        
        table.add_row("L2 Data Files", f"{data_files} files")
        table.add_row("Trained Models", f"{models} models")
        table.add_row("Best Model", best_model or "Not trained yet")
        table.add_row("Backtest Results", f"{backtest_results} runs")
        table.add_row("Config", "✅ Ready" if Path('config.yaml').exists() else "❌ Missing")
        
        console.print(table)
        
        # Current config
        if self.config:
            console.print(f"\n[bold]Current Settings:[/bold]")
            console.print(f"  Symbol: {self.config.get('symbol', 'Not set')}")
            console.print(f"  Balance: ${self.config.get('initial_balance', 10000):,.2f}")
        
        Prompt.ask("\nPress Enter to continue")
    
    def run(self):
        """Run the interactive CLI"""
        try:
            self.main_menu()
        except KeyboardInterrupt:
            console.print("\n\n[yellow]Interrupted by user[/yellow]")
        except Exception as e:
            console.print(f"\n[red]Error: {e}[/red]")
            console.print("[dim]Check logs for details[/dim]")

if __name__ == "__main__":
    cli = InteractiveTradingCLI()
    cli.run()