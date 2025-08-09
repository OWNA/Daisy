#!/usr/bin/env python3
"""
Complete System Diagnosis and Fix Plan
"""

import os
import sys
import json
import subprocess
from pathlib import Path
from datetime import datetime

def run_command(cmd):
    """Run a command and return result"""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        return result.returncode == 0, result.stdout, result.stderr
    except Exception as e:
        return False, "", str(e)

def check_python_env():
    """Check Python environment"""
    print("\n=== PYTHON ENVIRONMENT ===")
    print(f"Python: {sys.executable}")
    print(f"Version: {sys.version}")
    
    # Check if we're in venv
    in_venv = hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)
    print(f"Virtual Environment: {'Yes' if in_venv else 'No'}")
    
    # Check venv directory
    if Path('venv').exists():
        print("✓ venv directory exists")
        venv_python = Path('venv/bin/python')
        if venv_python.exists():
            print(f"  venv Python: {venv_python}")
        else:
            print("✗ venv Python not found")
    else:
        print("✗ No venv directory")
    
    return in_venv

def check_requirements():
    """Check required packages"""
    print("\n=== REQUIRED PACKAGES ===")
    
    required = [
        'pandas', 'numpy', 'lightgbm', 'ccxt', 'websocket-client',
        'rich', 'PyYAML', 'scikit-learn', 'optuna', 'shap',
        'matplotlib', 'seaborn', 'plotly', 'flask'
    ]
    
    missing = []
    installed = []
    
    for package in required:
        try:
            __import__(package.replace('-', '_'))
            installed.append(package)
        except ImportError:
            missing.append(package)
    
    print(f"Installed: {len(installed)}/{len(required)}")
    if missing:
        print(f"Missing: {', '.join(missing)}")
    
    return missing

def check_config_files():
    """Check configuration files"""
    print("\n=== CONFIG FILES ===")
    
    configs = ['config.yaml', 'config_l2_only.yaml', '.env']
    found = []
    
    for config in configs:
        if Path(config).exists():
            print(f"✓ {config}")
            found.append(config)
        else:
            print(f"✗ {config}")
    
    # Check config content
    if 'config.yaml' in found:
        try:
            import yaml
            with open('config.yaml', 'r') as f:
                config_data = yaml.safe_load(f)
            print(f"  Symbol: {config_data.get('symbol', 'NOT SET')}")
            print(f"  L2 features: {config_data.get('use_l2_features', False)}")
        except Exception as e:
            print(f"  Error reading config: {e}")
    
    return found

def check_data_files():
    """Check data files"""
    print("\n=== DATA FILES ===")
    
    # Check L2 data directory
    l2_dir = Path('l2_data')
    if l2_dir.exists():
        l2_files = list(l2_dir.glob('*.jsonl.gz'))
        print(f"✓ L2 data directory: {len(l2_files)} files")
        
        if l2_files:
            # Show sizes
            total_size = sum(f.stat().st_size for f in l2_files) / 1024 / 1024
            print(f"  Total size: {total_size:.2f} MB")
            
            # Show latest file
            latest = max(l2_files, key=lambda x: x.stat().st_mtime)
            print(f"  Latest: {latest.name} ({latest.stat().st_size / 1024 / 1024:.2f} MB)")
    else:
        print("✗ No L2 data directory")
        l2_files = []
    
    # Check database
    if Path('trading_bot.db').exists():
        print(f"✓ Database: trading_bot.db ({Path('trading_bot.db').stat().st_size / 1024 / 1024:.2f} MB)")
    else:
        print("✗ No database file")
    
    return len(l2_files) > 0

def check_models():
    """Check model files"""
    print("\n=== MODEL FILES ===")
    
    models = list(Path('.').glob('lgbm_model_*.txt'))
    feature_files = list(Path('.').glob('model_features_*.json'))
    
    print(f"Models found: {len(models)}")
    for model in models:
        print(f"  ✓ {model.name} ({model.stat().st_size / 1024:.2f} KB)")
    
    print(f"Feature files: {len(feature_files)}")
    for feat in feature_files:
        print(f"  ✓ {feat.name}")
    
    return len(models) > 0

def check_core_scripts():
    """Check if core scripts are working"""
    print("\n=== CORE SCRIPTS ===")
    
    scripts = {
        'l2_data_collector.py': 'Data collection',
        'train_model_robust.py': 'Model training',
        'run_backtest.py': 'Backtesting',
        'run_live_simulation_with_output.py': 'Live simulation',
        'trade_interactive.py': 'Interactive CLI',
        'trade_visualizer.py': 'Visualization'
    }
    
    working = []
    broken = []
    
    for script, desc in scripts.items():
        if Path(script).exists():
            # Try to import it (basic syntax check)
            success, _, error = run_command(f"{sys.executable} -m py_compile {script}")
            if success:
                print(f"✓ {script} - {desc}")
                working.append(script)
            else:
                print(f"✗ {script} - Syntax error")
                broken.append((script, error))
        else:
            print(f"✗ {script} - Not found")
            broken.append((script, "File not found"))
    
    return working, broken

def check_workflow_connectivity():
    """Check if components can connect"""
    print("\n=== WORKFLOW CONNECTIVITY ===")
    
    checks = []
    
    # Check if we can read L2 data
    l2_files = list(Path('l2_data').glob('*.jsonl.gz'))
    if l2_files:
        print("✓ L2 data files accessible")
        checks.append(True)
    else:
        print("✗ No L2 data to process")
        checks.append(False)
    
    # Check if models match data format
    if Path('lgbm_model_BTC_USDTUSDT_optuna_l2_hht.txt').exists():
        print("✓ L2+HHT model available")
        checks.append(True)
    else:
        print("✗ No L2+HHT model found")
        checks.append(False)
    
    # Check feature consistency
    feature_files = list(Path('.').glob('model_features_*.json'))
    if feature_files:
        print("✓ Feature definitions found")
        # Try to load and check
        try:
            with open(feature_files[0], 'r') as f:
                features = json.load(f)
            print(f"  Features count: {len(features)}")
            checks.append(True)
        except:
            print("  ✗ Error reading features")
            checks.append(False)
    else:
        print("✗ No feature definitions")
        checks.append(False)
    
    return all(checks)

def generate_fix_plan():
    """Generate fix plan based on diagnosis"""
    print("\n=== FIX PLAN ===")
    
    fixes = []
    
    # Check if we need to activate venv
    if Path('venv').exists() and not (hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)):
        fixes.append("1. Activate virtual environment:")
        fixes.append("   source venv/bin/activate")
    
    # Check missing packages
    missing = check_requirements()
    if missing:
        fixes.append(f"2. Install missing packages:")
        fixes.append(f"   pip install {' '.join(missing)}")
    
    # Check config
    if not Path('config.yaml').exists():
        fixes.append("3. Create config.yaml:")
        fixes.append("   python quick_setup.py")
    
    # Check data
    if not list(Path('l2_data').glob('*.jsonl.gz')):
        fixes.append("4. Collect initial data:")
        fixes.append("   python run_l2_collector.py --interval 300")
    
    # Check models
    if not list(Path('.').glob('lgbm_model_*.txt')):
        fixes.append("5. Train initial model:")
        fixes.append("   python train_model_robust.py --trials 10")
    
    if fixes:
        print("Run these commands to fix issues:")
        for fix in fixes:
            print(fix)
    else:
        print("✓ System appears ready!")
    
    return fixes

def main():
    """Run complete diagnosis"""
    print("="*60)
    print("TRADING BOT SYSTEM DIAGNOSIS")
    print("="*60)
    
    # Run all checks
    in_venv = check_python_env()
    missing_packages = check_requirements()
    config_files = check_config_files()
    has_data = check_data_files()
    has_models = check_models()
    working_scripts, broken_scripts = check_core_scripts()
    workflow_ok = check_workflow_connectivity()
    
    # Generate fix plan
    fixes = generate_fix_plan()
    
    # Summary
    print("\n=== SUMMARY ===")
    issues = []
    
    if not in_venv and Path('venv').exists():
        issues.append("Not using virtual environment")
    if missing_packages:
        issues.append(f"{len(missing_packages)} missing packages")
    if not config_files:
        issues.append("No configuration files")
    if not has_data:
        issues.append("No data files")
    if not has_models:
        issues.append("No trained models")
    if broken_scripts:
        issues.append(f"{len(broken_scripts)} broken scripts")
    if not workflow_ok:
        issues.append("Workflow connectivity issues")
    
    if issues:
        print(f"Found {len(issues)} issues:")
        for issue in issues:
            print(f"  - {issue}")
        print(f"\nGenerated {len(fixes)} fix steps")
    else:
        print("✓ System is ready to use!")
        print("\nRun: python trade_interactive.py")

if __name__ == "__main__":
    main()