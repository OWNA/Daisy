# BTC Perpetual Futures Trading System

## Overview
Focused 3-agent team to transform a fragmented BTC trading system into a production-grade platform.

## Core Team (3 Agents)

### 🏗️ Trading Systems Architect
**Mission**: Clean up 100+ file codebase into maintainable architecture
- Consolidate duplicate code
- Establish clear data flow: Bybit -> L2 -> Features -> Model -> Execution
- Fix database schema and create modular architecture
- Add comprehensive logging and testing

### 🧠 ML Model Enhancement Specialist  
**Mission**: Improve LightGBM model accuracy and reduce false signals
- Advanced L2 order book feature engineering
- Ensemble methods and confidence scoring
- Proper validation without future leakage
- Real-time feature computation (<10ms)

### ⚡ Execution Optimization Specialist
**Mission**: Minimize slippage and maximize fill rates
- Smart order execution algorithms
- Market impact minimization
- Risk-aware position sizing
- Real-time execution analytics

## Current System Issues
- 100+ fragmented Python files
- L2 order book data from Bybit WebSocket
- LightGBM with false signal issues
- Inconsistent SQLite schema
- Simple market orders with fixed sizing

## Usage

### Activate Team
```bash
claude-code team activate a-team
```

### Core Workflows
```bash
# Daily coordination
claude-code team workflow daily-standup

# System architecture
claude-code chat architect "Analyze current codebase and propose refactoring plan"

# Model improvement
claude-code chat ml-specialist "Enhance LightGBM features to reduce false signals"

# Execution optimization
claude-code chat execution-specialist "Implement smart order execution for Bybit"
```

### Collaborative Tasks
```bash
# System integration
claude-code chat architect,ml-specialist "Optimize ML feature pipeline architecture"

# Execution enhancement
claude-code chat architect,execution-specialist "Design execution system architecture"

# Full system optimization
claude-code chat architect,ml-specialist,execution-specialist "Review complete system performance"
```

## Project Goals
1. **<20 clean, organized files** (from 100+)
2. **Improved ML signal quality** (reduce false positives)
3. **Optimized execution** (minimize slippage)
4. **Robust architecture** (maintainable, testable)
5. **Production-ready system** (monitoring, logging)

## Development Priority
1. 🏗️ **Architecture Cleanup** - Consolidate and organize codebase
2. 🧠 **ML Enhancement** - Improve signal quality and validation
3. ⚡ **Execution Optimization** - Smart order management
4. 🔄 **Integration Testing** - End-to-end system validation
