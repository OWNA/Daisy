#!/usr/bin/env python3
"""
Simple menu that uses EXISTING WORKING scripts
"""

import os
import sys
import subprocess
from pathlib import Path

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def run_command(cmd):
    """Run command in venv"""
    if os.name == 'nt':  # Windows
        full_cmd = f"venv\\Scripts\\activate && {cmd}"
    else:
        full_cmd = f"source venv/bin/activate && {cmd}"
    return subprocess.run(full_cmd, shell=True)

def main():
    while True:
        clear_screen()
        print("="*50)
        print("SIMPLE TRADING MENU - Uses Existing Scripts")
        print("="*50)
        print("\n1. Collect L2 Data (run_l2_collector.py)")
        print("2. Train Model (Your file: 040413)")
        print("3. Run Backtest (run_backtest.py)")
        print("4. Exit\n")
        
        choice = input("Select: ")
        
        if choice == "1":
            print("\nCollecting L2 data for 300 seconds...")
            run_command("python run_l2_collector.py --config config.yaml --interval 300")
            input("\nPress Enter to continue...")
            
        elif choice == "2":
            print("\nTraining on your specific file...")
            # Use the direct training script that works
            run_command("python train_direct.py --data l2_data/l2_data_040413_converted.jsonl.gz --trials 50")
            input("\nPress Enter to continue...")
            
        elif choice == "3":
            print("\nRunning backtest...")
            run_command("python run_backtest.py --config config.yaml")
            input("\nPress Enter to continue...")
            
        elif choice == "4":
            break

if __name__ == "__main__":
    main()