@echo off
echo Setting up Trading A-Team...
echo.

echo Step 1: Installing Claude Code CLI (requires Node.js)
npm install -g @anthropic-ai/claude-code
if errorlevel 1 (
    echo ERROR: Failed to install Claude Code CLI. Please install Node.js first.
    echo Download from: https://nodejs.org/
    pause
    exit /b 1
)

echo.
echo Step 2: Setting up Trading Agents
chcp 65001 >nul
python setup_trading_agents.py
if errorlevel 1 (
    echo ERROR: Failed to setup trading agents
    echo Try running: python setup_trading_agents.py
    pause
    exit /b 1
)

echo.
echo Step 3: Installing Agent Dependencies
pip install -r agent_requirements.txt
if errorlevel 1 (
    echo WARNING: Some dependencies may have failed to install
)

echo.
echo Step 4: Activating A-Team
claude-code team activate a-team
if errorlevel 1 (
    echo ERROR: Failed to activate team. Make sure Claude Code CLI is authenticated.
    echo Run: claude-code auth login
    pause
    exit /b 1
)

echo.
echo ========================================
echo Trading A-Team Setup Complete!
echo ========================================
echo.
echo Your 3-agent team is ready:
echo   🏗️  Architect: Clean up fragmented codebase
echo   🧠  ML Specialist: Enhance model accuracy
echo   ⚡  Execution Specialist: Optimize order execution
echo.
echo Next steps:
echo   1. Run: claude-code team status
echo   2. Start with: claude-code team workflow daily-standup
echo   3. Get system analysis: claude-code chat architect "Analyze my current system"
echo.
pause
