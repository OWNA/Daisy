#!/usr/bin/env python3
"""
Simple Trading Menu - Everything just works
"""

import os
import sys
import subprocess
from pathlib import Path
from datetime import datetime
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, IntPrompt, Confirm
from rich.table import Table

console = Console()

class SimpleTradingMenu:
    def __init__(self):
        self.console = console
        self.data_dir = Path('l2_data')
        self.data_dir.mkdir(exist_ok=True)
        
    def clear_screen(self):
        os.system('cls' if os.name == 'nt' else 'clear')
    
    def show_header(self):
        self.clear_screen()
        console.print(Panel.fit(
            "[bold cyan]🚀 Simple Trading System[/bold cyan]\n" +
            "[dim]Collect → Train → Trade[/dim]",
            border_style="cyan"
        ))
        console.print()
    
    def main_menu(self):
        while True:
            self.show_header()
            
            menu = """[bold]What would you like to do?[/bold]

1. 📊 Collect Fresh L2 Data
2. 🧠 Train Model on Your Data  
3. 🤖 Run Live Trading Simulation
4. ⚡ Quick Test (1min collect → train → simulate)
5. 🚀 Full Workflow (5min each step)
6. 📁 View Your Files
7. ❌ Exit"""
            
            console.print(menu)
            
            choice = Prompt.ask("\n[yellow]Select option[/yellow]", choices=["1","2","3","4","5","6","7"])
            
            if choice == "1":
                self.collect_data()
            elif choice == "2":
                self.train_model()
            elif choice == "3":
                self.run_simulation()
            elif choice == "4":
                self.quick_test()
            elif choice == "5":
                self.full_workflow()
            elif choice == "6":
                self.view_files()
            elif choice == "7":
                console.print("\n[green]Goodbye! 👋[/green]")
                break
    
    def collect_data(self):
        """Collect L2 data"""
        self.show_header()
        console.print("[bold cyan]📊 Collect L2 Data[/bold cyan]\n")
        
        duration = IntPrompt.ask("How many seconds to collect?", default=300)
        
        console.print(f"\n[yellow]Collecting data for {duration} seconds...[/yellow]")
        console.print("[dim]This will connect to Bybit and save real L2 orderbook data[/dim]\n")
        
        cmd = f"python unified_trading_system.py collect --duration {duration}"
        subprocess.run(cmd, shell=True)
        
        Prompt.ask("\nPress Enter to continue")
    
    def train_model(self):
        """Train model on collected data"""
        self.show_header()
        console.print("[bold cyan]🧠 Train Model[/bold cyan]\n")
        
        # List available data files
        data_files = sorted(self.data_dir.glob('*.jsonl.gz'), key=lambda x: x.stat().st_mtime, reverse=True)
        
        if not data_files:
            console.print("[red]No data files found! Collect some data first.[/red]")
            Prompt.ask("\nPress Enter to continue")
            return
        
        # Show files
        console.print("Available data files:\n")
        for i, file in enumerate(data_files[:10], 1):
            size_mb = file.stat().st_size / 1024 / 1024
            console.print(f"{i}. {file.name} ({size_mb:.1f} MB)")
        
        # Select file
        if len(data_files) == 1:
            selected_file = data_files[0]
            console.print(f"\n[green]Using: {selected_file.name}[/green]")
        else:
            choice = IntPrompt.ask("\nSelect file number", default=1)
            if 1 <= choice <= len(data_files):
                selected_file = data_files[choice-1]
            else:
                console.print("[red]Invalid selection[/red]")
                return
        
        # Optuna trials
        trials = IntPrompt.ask("\nHow many optimization trials?", default=20)
        
        console.print(f"\n[yellow]Training model with {trials} trials...[/yellow]")
        console.print("[dim]This will take a few minutes[/dim]\n")
        
        cmd = f"python unified_trading_system.py train --data {selected_file} --trials {trials}"
        subprocess.run(cmd, shell=True)
        
        Prompt.ask("\nPress Enter to continue")
    
    def run_simulation(self):
        """Run live trading simulation"""
        self.show_header()
        console.print("[bold cyan]🤖 Live Trading Simulation[/bold cyan]\n")
        
        # List available models
        model_files = list(Path('.').glob('model_unified_*.txt'))
        
        if not model_files:
            console.print("[red]No models found! Train a model first.[/red]")
            Prompt.ask("\nPress Enter to continue")
            return
        
        # Sort by date
        model_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        
        # Show models
        console.print("Available models:\n")
        for i, file in enumerate(model_files[:5], 1):
            console.print(f"{i}. {file.name}")
        
        # Select model
        if len(model_files) == 1:
            selected_model = model_files[0]
            console.print(f"\n[green]Using: {selected_model.name}[/green]")
        else:
            choice = IntPrompt.ask("\nSelect model number", default=1)
            if 1 <= choice <= len(model_files):
                selected_model = model_files[choice-1]
            else:
                console.print("[red]Invalid selection[/red]")
                return
        
        # Duration
        duration = IntPrompt.ask("\nSimulation duration (seconds)?", default=300)
        
        # Output file for monitoring
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = f"simulation_{timestamp}.csv"
        
        # Launch monitoring?
        if Confirm.ask("\nLaunch real-time monitoring dashboard?", default=True):
            console.print("\n[green]Starting monitoring dashboard...[/green]")
            # Start visualizer in background
            import threading
            def run_viz():
                subprocess.run(f"python trade_visualizer.py --live {output_file}", shell=True)
            viz_thread = threading.Thread(target=run_viz, daemon=True)
            viz_thread.start()
            
            import time
            time.sleep(2)  # Give it time to start
            
            # Open browser
            import webbrowser
            webbrowser.open('http://localhost:5000')
        
        console.print(f"\n[yellow]Running simulation for {duration} seconds...[/yellow]")
        console.print("[dim]Trading with real-time data from Bybit[/dim]\n")
        
        cmd = f"python unified_trading_system.py simulate --model {selected_model} --duration {duration} --output {output_file}"
        subprocess.run(cmd, shell=True)
        
        Prompt.ask("\nPress Enter to continue")
    
    def quick_test(self):
        """Quick test workflow"""
        self.show_header()
        console.print("[bold cyan]⚡ Quick Test Workflow[/bold cyan]\n")
        console.print("This will run a complete test in about 3 minutes:")
        console.print("  1. Collect data for 60 seconds")
        console.print("  2. Train model with 10 trials")  
        console.print("  3. Run simulation for 60 seconds\n")
        
        if Confirm.ask("Proceed?", default=True):
            cmd = "python unified_trading_system.py workflow --duration 60 --trials 10"
            subprocess.run(cmd, shell=True)
        
        Prompt.ask("\nPress Enter to continue")
    
    def full_workflow(self):
        """Full workflow"""
        self.show_header()
        console.print("[bold cyan]🚀 Full Workflow[/bold cyan]\n")
        console.print("This will run the complete workflow:")
        console.print("  1. Collect data for 5 minutes")
        console.print("  2. Train model with 50 trials")
        console.print("  3. Run simulation for 5 minutes\n")
        console.print("[yellow]Total time: ~15-20 minutes[/yellow]\n")
        
        if Confirm.ask("Proceed?", default=True):
            cmd = "python unified_trading_system.py workflow --duration 300 --trials 50"
            subprocess.run(cmd, shell=True)
        
        Prompt.ask("\nPress Enter to continue")
    
    def view_files(self):
        """View collected files"""
        self.show_header()
        console.print("[bold cyan]📁 Your Files[/bold cyan]\n")
        
        # Data files
        data_files = list(self.data_dir.glob('*.jsonl.gz'))
        console.print(f"[yellow]Data files:[/yellow] {len(data_files)}")
        for file in data_files[-5:]:
            size_mb = file.stat().st_size / 1024 / 1024
            console.print(f"  {file.name} ({size_mb:.1f} MB)")
        
        # Models
        model_files = list(Path('.').glob('model_unified_*.txt'))
        console.print(f"\n[yellow]Models:[/yellow] {len(model_files)}")
        for file in model_files[-5:]:
            console.print(f"  {file.name}")
        
        # Results
        result_files = list(Path('.').glob('simulation_*.csv'))
        console.print(f"\n[yellow]Simulation results:[/yellow] {len(result_files)}")
        for file in result_files[-5:]:
            console.print(f"  {file.name}")
        
        Prompt.ask("\nPress Enter to continue")

if __name__ == "__main__":
    menu = SimpleTradingMenu()
    menu.main_menu()