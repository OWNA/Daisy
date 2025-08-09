# Quick Unicode Fix and Setup Guide

## ✅ Problem Fixed!

The Unicode encoding error was caused by special arrow characters (→) in the agent files. I've fixed:

1. **Updated `setup_trading_agents.py`** - Now uses UTF-8 encoding
2. **Cleaned Unicode characters** - Replaced → with -> 
3. **Updated batch files** - Better encoding handling

## 🚀 Try Again Now

### Option 1: Use the Fixed Script
```cmd
# Run this - it should work now
fix_and_setup.bat
```

### Option 2: Manual Setup
```cmd
# Set UTF-8 encoding first
chcp 65001

# Then run setup
python setup_trading_agents.py
```

### Option 3: PowerShell (if still having issues)
```powershell
# In PowerShell, run:
$env:PYTHONIOENCODING="utf-8"
python setup_trading_agents.py
```

## 🔧 If You Still Get Errors

### Quick Debug:
```cmd
# Check Python version
python --version

# Check current encoding
python -c "import sys; print(sys.getdefaultencoding())"

# Force UTF-8 and run
set PYTHONIOENCODING=utf-8
python setup_trading_agents.py
```

## ✅ Expected Success Output

You should see:
```
Setting up Focused Trading A-Team (3 Core Agents)...
============================================================
Created directory: .claude
Created directory: .claude/agents
Created directory: .claude/teams
Created directory: .claude/workflows
Created team configuration: .claude/teams/a-team.yaml
Created agent file: .claude/agents/trading-systems-architect.md
Created agent file: .claude/agents/ml-model-enhancement-specialist.md
Created agent file: .claude/agents/execution-optimization-specialist.md
Created config file: .claude/config.yaml
Created README.md
============================================================
Setup complete! Your focused Trading A-Team is ready.
```

## 🎯 After Successful Setup

```bash
# Activate your team
claude-code team activate a-team

# Test it works
claude-code team status

# Start using your agents
claude-code chat architect "Hello, analyze my trading system"
```

The fix is ready - try running `fix_and_setup.bat` now!
