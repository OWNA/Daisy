@echo off
echo Fixing Unicode encoding issue...
echo.

echo Setting UTF-8 encoding for current session...
chcp 65001 >nul

echo Running setup with proper encoding...
python setup_trading_agents.py

if errorlevel 1 (
    echo.
    echo If you still get errors, try:
    echo 1. Make sure Python is using UTF-8 encoding
    echo 2. Run: set PYTHONIOENCODING=utf-8
    echo 3. Then run: python setup_trading_agents.py
    pause
) else (
    echo.
    echo ========================================
    echo Setup completed successfully!
    echo ========================================
    echo.
    echo Next steps:
    echo   1. Run: claude-code team activate a-team
    echo   2. Test: claude-code team status
    echo   3. Start: claude-code team workflow daily-standup
    echo.
)

pause
