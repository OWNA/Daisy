# PowerShell script to create fresh virtual environment on Windows

Write-Host "BTC L2 Trading System - Fresh Virtual Environment Setup" -ForegroundColor Cyan
Write-Host "======================================================" -ForegroundColor Cyan
Write-Host ""

# Check if old venv exists
if (Test-Path ".\venv") {
    Write-Host "⚠️  Old virtual environment found!" -ForegroundColor Yellow
    $response = Read-Host "Remove old venv and create fresh one? (y/n)"
    
    if ($response -eq 'y') {
        Write-Host "Removing old virtual environment..." -ForegroundColor Yellow
        Remove-Item -Recurse -Force ".\venv"
    } else {
        Write-Host "Aborting. Please manually remove the venv folder first." -ForegroundColor Red
        exit
    }
}

# Create new virtual environment
Write-Host "Creating new virtual environment..." -ForegroundColor Green
python -m venv venv

if (-not (Test-Path ".\venv\Scripts\python.exe")) {
    Write-Host "❌ Failed to create virtual environment!" -ForegroundColor Red
    Write-Host "Make sure Python is installed and in PATH" -ForegroundColor Red
    exit
}

Write-Host "✅ Virtual environment created successfully!" -ForegroundColor Green
Write-Host ""

# Activate and upgrade pip
Write-Host "Activating virtual environment..." -ForegroundColor Green
& .\venv\Scripts\Activate.ps1

Write-Host "Upgrading pip..." -ForegroundColor Green
python -m pip install --upgrade pip

# Install requirements
Write-Host ""
Write-Host "Installing requirements..." -ForegroundColor Green
pip install -r requirements.txt

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "✅ All packages installed successfully!" -ForegroundColor Green
    Write-Host ""
    Write-Host "Virtual environment is ready!" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "To use the system:" -ForegroundColor Yellow
    Write-Host "1. Activate venv: .\venv\Scripts\activate" -ForegroundColor White
    Write-Host "2. Train model: python main.py train" -ForegroundColor White
    Write-Host "3. Paper trade: python main.py trade --paper" -ForegroundColor White
} else {
    Write-Host ""
    Write-Host "❌ Some packages failed to install!" -ForegroundColor Red
    Write-Host "Check the error messages above." -ForegroundColor Red
}

Write-Host ""
Write-Host "Press any key to exit..."
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")