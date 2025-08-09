#!/usr/bin/env python3
"""
Create a fresh virtual environment and install dependencies
"""

import os
import sys
import subprocess

def create_fresh_venv():
    """Create a new virtual environment"""
    print("Creating fresh virtual environment...\n")
    
    # Remove old venv if it exists
    if os.path.exists('venv'):
        print("⚠️  Old venv exists. Please remove it manually:")
        print("   Windows: rmdir /s venv")
        print("   Linux/Mac: rm -rf venv")
        return False
    
    # Create new venv
    print("Creating new virtual environment...")
    subprocess.run([sys.executable, '-m', 'venv', 'venv'])
    
    # Determine pip path based on OS
    if os.name == 'nt':  # Windows
        pip_path = os.path.join('venv', 'Scripts', 'pip.exe')
        python_path = os.path.join('venv', 'Scripts', 'python.exe')
    else:  # Linux/Mac
        pip_path = os.path.join('venv', 'bin', 'pip')
        python_path = os.path.join('venv', 'bin', 'python')
    
    if not os.path.exists(pip_path):
        print("❌ Failed to create virtual environment")
        return False
    
    print("✅ Virtual environment created successfully")
    
    # Install packages
    print("\nInstalling required packages...")
    packages = [
        'pandas>=1.3.0',
        'numpy>=1.21.0',
        'scikit-learn>=1.0.0',
        'lightgbm>=3.3.0',
        'optuna>=3.0.0',
        'ccxt>=3.0.0',
        'pyyaml>=6.0',
        'websocket-client>=1.3.0',
        'python-dotenv>=0.19.0',
        'matplotlib>=3.5.0',
        'seaborn>=0.11.0',
        'shap>=0.40.0',
        'sqlalchemy>=1.4.0',
        'aiohttp>=3.8.0'
    ]
    
    for package in packages:
        print(f"Installing {package}...")
        result = subprocess.run([pip_path, 'install', package], 
                              capture_output=True, text=True)
        if result.returncode != 0:
            print(f"❌ Failed to install {package}")
            print(result.stderr)
        else:
            print(f"✅ {package} installed")
    
    print("\n✅ Setup complete!")
    print("\nTo activate the virtual environment:")
    print("   Windows: .\\venv\\Scripts\\activate")
    print("   Linux/Mac: source venv/bin/activate")
    
    return True

if __name__ == "__main__":
    create_fresh_venv()