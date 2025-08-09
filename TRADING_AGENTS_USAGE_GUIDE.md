# Focused Trading A-Team Usage Guide

## Quick Setup Commands

### 1. Run the Setup Script
```bash
# Make the script executable and run it  
python setup_trading_agents.py
```

### 2. Activate the Team
```bash
# Activate the focused A-Team
claude-code team activate a-team
```

### 3. Verify Setup
```bash
# Check team status
claude-code team status

# List all agents
claude-code agents list

# Test team introduction
claude-code team workflow daily-standup
```

## Core Agent Interactions

### 🏗️ Trading Systems Architect
```bash
# System analysis and cleanup
claude-code chat architect "Analyze the current 100+ file codebase and identify duplicate functionality"
claude-code chat architect "Create a refactoring plan to consolidate into <20 organized files"
claude-code chat architect "Design the data flow: Bybit → L2 processing → Features → Model → Execution"
claude-code chat architect "Fix the inconsistent SQLite database schema"
claude-code chat architect "Implement comprehensive logging and error handling"

# Architecture documentation
claude-code chat architect "Document the proposed system architecture"
claude-code chat architect "Create database migration scripts"
claude-code chat architect "Design unit tests for critical components"
```

### 🧠 ML Model Enhancement Specialist
```bash
# Feature engineering improvements
claude-code chat ml-specialist "Analyze the current 84 L2-derived features and identify weaknesses"
claude-code chat ml-specialist "Create new order flow imbalance features across multiple time windows"
claude-code chat ml-specialist "Develop microstructure stability metrics to reduce false signals"
claude-code chat ml-specialist "Implement ensemble methods for better prediction confidence"
claude-code chat ml-specialist "Optimize feature computation to <10ms latency"

# Model validation
claude-code chat ml-specialist "Create proper walk-forward analysis without future leakage"
claude-code chat ml-specialist "Implement transaction cost-aware backtesting"
claude-code chat ml-specialist "Build model performance monitoring dashboard"
```

### ⚡ Execution Optimization Specialist
```bash
# Smart execution development
claude-code chat execution-specialist "Analyze current simple market order execution and identify improvements"
claude-code chat execution-specialist "Implement passive order placement strategies"
claude-code chat execution-specialist "Create dynamic position sizing based on order book liquidity"
claude-code chat execution-specialist "Develop market impact estimation models"
claude-code chat execution-specialist "Build real-time execution analytics dashboard"

# Risk management
claude-code chat execution-specialist "Implement dynamic position sizing based on volatility"
claude-code chat execution-specialist "Create funding rate optimization logic"
claude-code chat execution-specialist "Design liquidation price management system"
```

## Team Collaboration Workflows

### System Integration Tasks
```bash
# Architecture + ML Integration
claude-code chat architect,ml-specialist "Optimize the ML feature pipeline architecture for real-time processing"

# Architecture + Execution Integration  
claude-code chat architect,execution-specialist "Design the execution system architecture and interfaces"

# ML + Execution Integration
claude-code chat ml-specialist,execution-specialist "Optimize signal-to-execution latency and decision logic"

# Full team coordination
claude-code chat architect,ml-specialist,execution-specialist "Review complete system integration and performance"
```

### Daily Operations
```bash
# Morning standup
claude-code team workflow daily-standup

# Regular code reviews
claude-code team workflow code-review

# System optimization sessions
claude-code team workflow system-optimization

# Model enhancement sessions
claude-code team workflow model-enhancement
```

## Project Phase Commands

### Phase 1: System Analysis & Planning
```bash
# Analyze current system
claude-code chat architect "Perform comprehensive analysis of current 100+ file system"
claude-code chat ml-specialist "Analyze current LightGBM model performance and false signal issues"
claude-code chat execution-specialist "Analyze current execution performance and slippage patterns"

# Create improvement plan
claude-code chat architect,ml-specialist,execution-specialist "Create integrated improvement plan"
```

### Phase 2: Core System Refactoring
```bash
# Architecture cleanup
claude-code chat architect "Begin consolidating duplicate code and organizing file structure"
claude-code chat architect "Implement new database schema and migration scripts"

# Feature engineering improvements
claude-code chat ml-specialist "Implement new order book features to reduce false signals"

# Execution enhancements
claude-code chat execution-specialist "Implement smart order placement algorithms"
```

### Phase 3: Integration & Testing
```bash
# System integration
claude-code chat architect,ml-specialist "Integrate enhanced ML pipeline with system architecture"
claude-code chat architect,execution-specialist "Integrate execution optimization with system architecture"

# End-to-end testing
claude-code chat architect,ml-specialist,execution-specialist "Perform complete system testing and optimization"
```

### Phase 4: Production Deployment
```bash
# Production readiness
claude-code chat architect "Implement production monitoring and alerting"
claude-code chat ml-specialist "Deploy model performance monitoring"
claude-code chat execution-specialist "Deploy execution analytics dashboard"

# Final system review
claude-code chat architect,ml-specialist,execution-specialist "Final production readiness review"
```

## Specific Problem-Solving Commands

### Codebase Cleanup
```bash
claude-code chat architect "Identify and merge redundant WebSocket implementations"
claude-code chat architect "Consolidate CCXT and native API code into unified interface"
claude-code chat architect "Create clear separation between data collection, feature engineering, and execution"
```

### ML Signal Quality
```bash
claude-code chat ml-specialist "Debug why model has false signals after removing target leakage"
claude-code chat ml-specialist "Implement confidence scoring for predictions"
claude-code chat ml-specialist "Create adaptive thresholds based on market regime"
```

### Execution Performance
```bash
claude-code chat execution-specialist "Implement queue position modeling for better fills"
claude-code chat execution-specialist "Create pre-trade impact estimation"
claude-code chat execution-specialist "Optimize order sizing based on market liquidity"
```

## Context Management
```bash
# Set system context
claude-code context share --team a-team "Current BTC price: $43,500, High volatility period, 100+ files to consolidate"

# Agent-specific context
claude-code context set architect "Priority: Database schema fix, 100+ files → <20 files"
claude-code context set ml-specialist "Priority: Reduce false signals, improve confidence scoring"
claude-code context set execution-specialist "Priority: Minimize slippage, implement smart routing"
```

## Performance Monitoring
```bash
# System health checks
claude-code chat architect "Generate system health and performance report"
claude-code chat ml-specialist "Generate model performance and accuracy report"
claude-code chat execution-specialist "Generate execution quality and slippage report"

# Team performance review
claude-code chat architect,ml-specialist,execution-specialist "Weekly team performance review"
```

This focused 3-agent approach is much more practical for real trading system development - each agent has clear, specific responsibilities that address the core challenges of your fragmented BTC trading system.
