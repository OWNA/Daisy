# Agent File Locations

Your trading system agents have been placed in multiple locations to ensure the `/agents` command can find them:

## Agent Files Created:
1. **btc_system_cleanup_agent.md** - For consolidating the codebase
2. **btc_ml_enhancement_agent.md** - For improving the ML model
3. **btc_execution_optimization_agent.md** - For optimizing trade execution

## Locations:

### 1. Original Location (in your Trade directory):
```
/mnt/c/Users/simon/Trade/agents/
├── btc_system_cleanup_agent.md
├── btc_ml_enhancement_agent.md
└── btc_execution_optimization_agent.md
```

### 2. User Home Directory:
```
/home/simontys/agents/
├── btc_system_cleanup_agent.md
├── btc_ml_enhancement_agent.md
└── btc_execution_optimization_agent.md
```

### 3. Hidden Directory (alternative):
```
/home/simontys/.agents/
├── btc_system_cleanup_agent.md
├── btc_ml_enhancement_agent.md
└── btc_execution_optimization_agent.md
```

## Testing the Agents:

Try the `/agents` command again. If it still doesn't work, you can use the agents directly:

### Direct Usage with Task Tool:
```
Task: You are the btc_system_cleanup_agent. Analyze the current codebase and suggest further improvements.
```

```
Task: You are the btc_ml_enhancement_agent. Review the current model performance and suggest feature improvements.
```

```
Task: You are the btc_execution_optimization_agent. Analyze the current execution logic and suggest optimizations.
```

## If `/agents` Still Doesn't Work:

The command might be looking in:
- `/agents/` (root directory) - requires sudo to create
- A specific configuration directory
- Or it might need a specific file format or naming convention

You can also try:
1. Restarting your terminal/session
2. Checking if there's a configuration file that specifies the agents directory
3. Looking for documentation on where the `/agents` command expects files