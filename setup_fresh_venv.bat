@echo off
echo BTC L2 Trading System - Fresh Virtual Environment Setup
echo ======================================================
echo.

REM Check if old venv exists
if exist "venv\" (
    echo WARNING: Old virtual environment found!
    set /p response=Remove old venv and create fresh one? (y/n): 
    if /i "%response%"=="y" (
        echo Removing old virtual environment...
        rmdir /s /q venv
    ) else (
        echo Aborting. Please manually remove the venv folder first.
        pause
        exit /b
    )
)

REM Create new virtual environment
echo Creating new virtual environment...
python -m venv venv

if not exist "venv\Scripts\python.exe" (
    echo ERROR: Failed to create virtual environment!
    echo Make sure Python is installed and in PATH
    pause
    exit /b
)

echo Virtual environment created successfully!
echo.

REM Activate and upgrade pip
echo Activating virtual environment...
call venv\Scripts\activate.bat

echo Upgrading pip...
python -m pip install --upgrade pip

REM Install requirements
echo.
echo Installing requirements...
pip install -r requirements.txt

if %errorlevel% equ 0 (
    echo.
    echo All packages installed successfully!
    echo.
    echo Virtual environment is ready!
    echo.
    echo To use the system:
    echo 1. Activate venv: venv\Scripts\activate
    echo 2. Train model: python main.py train
    echo 3. Paper trade: python main.py trade --paper
) else (
    echo.
    echo ERROR: Some packages failed to install!
    echo Check the error messages above.
)

echo.
pause