#!/usr/bin/env python3
"""
Local Environment Setup for Trading Bot
This script prepares your local environment to run the trading bot
that was originally designed for Google Colab.
"""

import os
import sys
import subprocess
import platform

def print_header(text):
    """Print formatted header"""
    print("\n" + "="*60)
    print(f"{text}")
    print("="*60)

def check_python_version():
    """Check if Python version is compatible"""
    print_header("Checking Python Version")
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("Python 3.8+ is required!")
        return False
    
    print("Python version is compatible")
    return True

def create_virtual_environment():
    """Create a virtual environment for the project"""
    print_header("Setting Up Virtual Environment")
    
    venv_name = "trading_bot_env"
    
    if os.path.exists(venv_name):
        print(f"Virtual environment '{venv_name}' already exists")
        return venv_name
    
    try:
        subprocess.run([sys.executable, "-m", "venv", venv_name], check=True)
        print(f"Created virtual environment: {venv_name}")
        
        # Provide activation instructions
        if platform.system() == "Windows":
            activate_cmd = f"{venv_name}\\Scripts\\activate"
        else:
            activate_cmd = f"source {venv_name}/bin/activate"
        
        print(f"\nTo activate the environment, run:")
        print(f"   {activate_cmd}")
        
        return venv_name
    except Exception as e:
        print(f"Failed to create virtual environment: {e}")
        return None

def create_requirements_file():
    """Create requirements.txt with all necessary dependencies"""
    print_header("Creating Requirements File")
    
    requirements = """# Core dependencies
numpy==1.26.4
pandas==2.0.3
scipy==1.11.4
scikit-learn>=1.3.0
lightgbm==4.1.0

# Trading and data
ccxt>=4.0.0
pandas-ta>=0.3.14b

# Machine Learning
optuna>=3.0.0
shap>=0.42.0

# Signal processing
EMD-signal<1.4.0

# Visualization
matplotlib>=3.5.0
seaborn>=0.12.0

# Utilities
PyYAML>=6.0
websocket-client>=1.4.0
dill==0.3.7
joblib>=1.3.0

# Additional for local development
python-dotenv>=1.0.0
jupyter>=1.0.0
ipykernel>=6.25.0
"""
    
    with open("requirements.txt", "w") as f:
        f.write(requirements)
    
    print("Created requirements.txt")
    return True

def create_project_structure():
    """Create necessary project directories"""
    print_header("Setting Up Project Structure")
    
    directories = [
        "data",
        "models",
        "logs",
        "configs",
        "l2_data",
        "backtest_results",
        "plots"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")
    
    return True

def create_env_template():
    """Create .env template for API keys"""
    print_header("Creating Environment Template")
    
    env_template = """# API Keys for Trading Bot
# Copy this file to .env and fill in your actual keys

# Bybit API Keys (Testnet)
BYBIT_API_KEY_MAIN_TEST=your_testnet_api_key_here
BYBIT_API_SECRET_MAIN_TEST=your_testnet_api_secret_here

# Bybit API Keys (Mainnet) - Use with caution!
# BYBIT_API_KEY_MAIN=your_mainnet_api_key_here
# BYBIT_API_SECRET_MAIN=your_mainnet_api_secret_here

# Project paths
BOT_BASE_DIR=./
"""
    
    with open(".env.template", "w") as f:
        f.write(env_template)
    
    print("Created .env.template")
    print("Copy .env.template to .env and add your API keys")
    
    return True

def main():
    """Main setup function"""
    print_header("Trading Bot Local Setup")
    print("Converting from Google Colab to local environment...")
    
    # Check Python version
    if not check_python_version():
        return
    
    # Create virtual environment
    venv_name = create_virtual_environment()
    if not venv_name:
        return
    
    # Create requirements file
    create_requirements_file()
    
    # Create project structure
    create_project_structure()
    
    # Create environment template
    create_env_template()
    
    print_header("Setup Complete!")
    print("\nNext steps:")
    print("1. Activate the virtual environment:")
    if platform.system() == "Windows":
        print(f"   {venv_name}\\Scripts\\activate")
    else:
        print(f"   source {venv_name}/bin/activate")
    print("\n2. Install dependencies:")
    print("   pip install -r requirements.txt")
    print("\n3. Copy .env.template to .env and add your API keys:")
    print("   cp .env.template .env")
    print("\n4. Run the main script:")
    print("   python run_trading_bot.py")
    
    print("\nYour trading bot environment is ready!")

if __name__ == "__main__":
    main()