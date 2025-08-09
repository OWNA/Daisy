# Fix Virtual Environment

## Problem
The virtual environment is broken or corrupt, causing import errors with LightGBM and other dependencies.

## Solution
Create a fresh virtual environment with all required dependencies.

## Method 1: PowerShell (Recommended for Windows)

```powershell
# Run the setup script
.\setup_fresh_venv.ps1
```

This script will:
1. Remove old venv (with confirmation)
2. Create fresh virtual environment
3. Install all dependencies from requirements.txt
4. Show instructions for using the system

## Method 2: Command Prompt

```cmd
# Run the batch file
setup_fresh_venv.bat
```

## Method 3: Manual Steps

```powershell
# 1. Remove old venv
rmdir /s venv

# 2. Create new venv
python -m venv venv

# 3. Activate venv
.\venv\Scripts\activate

# 4. Upgrade pip
python -m pip install --upgrade pip

# 5. Install requirements
pip install -r requirements.txt
```

## Required Dependencies

The `requirements.txt` includes:
- pandas, numpy, scikit-learn (data processing)
- lightgbm, optuna (ML modeling)
- ccxt, websocket-client (exchange connectivity)
- pyyaml, python-dotenv (configuration)
- matplotlib, seaborn, shap (visualization/analysis)
- PyEMD (Hilbert-Huang Transform)
- pandas-ta (technical analysis)

## After Setup

Once the fresh venv is created:

```powershell
# Activate virtual environment
.\venv\Scripts\activate

# Train model
python main.py train

# Run paper trading
python main.py trade --paper
```

## Troubleshooting

If you still get errors:

1. **Check Python version**: Should be 3.8-3.10
   ```powershell
   python --version
   ```

2. **Check if in virtual environment**:
   ```powershell
   where python
   # Should show: ...\venv\Scripts\python.exe
   ```

3. **For LightGBM issues on Windows**:
   - Install Visual C++ Redistributable
   - Or use: `pip install lightgbm --install-option=--nomp`

4. **Clear pip cache**:
   ```powershell
   pip cache purge
   ```