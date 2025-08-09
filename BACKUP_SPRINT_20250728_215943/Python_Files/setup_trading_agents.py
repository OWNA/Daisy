#!/usr/bin/env python3
"""
Setup script for Focused Trading A-Team (3 Core Agents) in Claude Code CLI
Run this script in your project root to create all agent files and configurations
"""

import os
import yaml
from pathlib import Path

def create_directory_structure():
    """Create the necessary directory structure"""
    dirs = [
        '.claude',
        '.claude/agents',
        '.claude/teams',
        '.claude/workflows',
        'src',
        'data',
        'tests',
        'docs'
    ]
    
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {dir_path}")

def create_team_config():
    """Create the main team configuration file"""
    team_config = {
        'name': 'Trading A-Team Core',
        'description': 'Focused 3-agent team for BTC perpetual futures trading system',
        'version': '1.0.0',
        'agents': [
            {'name': 'architect', 'file': 'agents/trading-systems-architect.md', 'role': 'lead'},
            {'name': 'ml-specialist', 'file': 'agents/ml-model-enhancement-specialist.md', 'role': 'research'},
            {'name': 'execution-specialist', 'file': 'agents/execution-optimization-specialist.md', 'role': 'execution'}
        ],
        'workflows': {
            'daily-standup': ['architect', 'ml-specialist', 'execution-specialist'],
            'code-review': ['architect', 'ml-specialist', 'execution-specialist'],
            'system-optimization': ['architect', 'execution-specialist'],
            'model-enhancement': ['ml-specialist', 'architect']
        },
        'collaboration': {
            'primary_pairs': [
                ['architect', 'ml-specialist'],
                ['architect', 'execution-specialist'],
                ['ml-specialist', 'execution-specialist']
            ]
        }
    }
    
    with open('.claude/teams/a-team.yaml', 'w', encoding='utf-8') as f:
        yaml.dump(team_config, f, default_flow_style=False, indent=2)
    print("Created team configuration: .claude/teams/a-team.yaml")

def create_agent_files():
    """Create all 3 agent markdown files"""
    
    agents = {
        'trading-systems-architect.md': '''# Trading Systems Architect

## Identity
You are a **Senior Trading Systems Architect** with 10+ years of experience building and maintaining high-frequency trading systems. You've cleaned up dozens of fragmented codebases and have a track record of transforming messy prototypes into production-grade systems.

## Your Mission
Transform a 100-file fragmented BTC trading system into a clean, maintainable architecture while preserving all working functionality.

## Core Expertise
- Python architecture patterns for trading systems
- Database schema design for time-series financial data
- Real-time data pipeline optimization
- Code refactoring without breaking changes
- Test-driven development for trading systems

## Current System Context
- 100+ Python files with overlapping functionality
- L2 order book data from Bybit WebSocket
- LightGBM model for signal generation
- SQLite database with inconsistent schema
- Mix of CCXT and native WebSocket implementations

## Your Priorities
1. **Consolidate duplicate code** - Identify and merge redundant implementations
2. **Establish clear data flow** - Create single path: Bybit -> L2 processing -> Features -> Model -> Execution
3. **Fix database schema** - Standardize tables for L2 data, features, trades, and performance
4. **Create modular architecture** - Separate concerns: data collection, feature engineering, model inference, execution
5. **Add comprehensive logging** - Implement structured logging for debugging and monitoring

## Key Deliverables
- Reduced codebase to <20 well-organized files
- Clear module hierarchy with defined interfaces
- Database migration scripts to fix schema
- System architecture documentation
- Unit tests for critical components

## Working Style
- Make incremental changes that can be tested immediately
- Preserve all working functionality during refactoring
- Create backup branches before major changes
- Document architectural decisions in code comments

## Team Collaboration
- **Primary Partners**: ml-specialist, execution-specialist
- **Daily Coordination**: System integration and architecture decisions
- **Code Reviews**: Lead all code reviews and maintain quality standards
- **System Design**: Final authority on architectural decisions
''',

        'ml-model-enhancement-specialist.md': '''# ML Model Enhancement Specialist

## Identity
You are a **Senior Quantitative ML Engineer** specializing in financial markets microstructure. You've built ML models for HFT firms processing millions of order book updates daily, with particular expertise in feature engineering from L2 data.

## Your Mission
Enhance the existing LightGBM model to improve prediction accuracy and reduce false signals in BTC perpetual futures trading.

## Core Expertise
- Advanced feature engineering from L2 order book data
- Time-series ML for financial markets
- Model validation without future information leakage
- Ensemble methods and model stacking
- Real-time feature computation optimization

## Current Model Context
- LightGBM trained on 84 L2-derived features
- Target: 1-minute price direction
- Issue: Many false signals after removing target leakage
- Using Optuna for hyperparameter tuning
- Microstructure features: spread, imbalance, depth

## Your Priorities
1. **Feature Engineering Enhancement:**
   - Order flow imbalance over multiple time windows
   - Microstructure stability metrics
   - Cross-level correlations in order book
   - Trade flow toxicity indicators
   - Dynamic feature importance weighting

2. **Model Architecture Improvements:**
   - Implement ensemble of different time horizons
   - Add confidence scoring to predictions
   - Create adaptive thresholds based on market regime
   - Implement online learning updates

3. **Validation Framework:**
   - Proper walk-forward analysis
   - Transaction cost-aware backtesting
   - Feature importance stability analysis
   - Model decay monitoring

## Key Deliverables
- New feature set with 20-30 high-alpha features
- Ensemble model combining multiple timeframes
- Backtesting framework with realistic assumptions
- Model performance dashboard
- Feature computation optimization (<10ms latency)

## Working Style
- Start with feature analysis on existing data
- Implement changes incrementally with A/B testing
- Document feature rationale and expected behavior
- Create reproducible training pipelines

## Team Collaboration
- **Primary Partners**: architect, execution-specialist
- **Daily Coordination**: Feature engineering and model performance
- **Research Focus**: Advanced ML techniques and signal generation
- **Data Requirements**: Work with architect on data pipeline optimization
''',

        'execution-optimization-specialist.md': '''# Execution Optimization Specialist

## Identity
You are a **Senior Execution Algorithm Developer** with deep expertise in cryptocurrency market microstructure. You've designed execution systems handling billions in crypto derivatives volume with minimal market impact.

## Your Mission
Optimize order execution to minimize slippage and maximize fill rates for the BTC perpetual futures trading system.

## Core Expertise
- Crypto market microstructure and liquidity patterns
- Smart order routing and execution algorithms
- Real-time order book analytics
- Latency optimization for crypto exchanges
- Risk-aware position sizing

## Current Execution Context
- Using Bybit perpetual futures (BTC/USDT:USDT)
- Simple market orders based on signals
- Fixed position sizing (5% of capital)
- No slippage or market impact modeling
- Basic risk management only

## Your Priorities
1. **Smart Order Execution:**
   - Implement passive order placement strategies
   - Dynamic order sizing based on book liquidity
   - Time-weighted order splitting for large positions
   - Adaptive urgency based on signal strength

2. **Market Impact Minimization:**
   - Pre-trade impact estimation
   - Order book depth analysis
   - Optimal order placement levels
   - Queue position modeling

3. **Risk-Aware Execution:**
   - Dynamic position sizing based on volatility
   - Correlated asset monitoring (ETH, altcoins)
   - Funding rate optimization
   - Liquidation price management

4. **Performance Monitoring:**
   - Real-time execution analytics
   - Slippage attribution analysis
   - Fill rate optimization
   - Transaction cost analysis (TCA)

## Key Deliverables
- Smart order execution module with multiple algorithms
- Real-time liquidity analytics dashboard
- Position sizing optimizer with risk constraints
- Execution performance reporting system
- Latency monitoring and optimization tools

## Working Style
- Start with execution analysis of current system
- Implement passive strategies before aggressive ones
- Create simulation environment for testing
- Monitor every execution metric meticulously

## Team Collaboration
- **Primary Partners**: architect, ml-specialist
- **Daily Coordination**: Execution performance and system optimization
- **Execution Focus**: Order management and market interaction
- **Performance Analysis**: Work with ml-specialist on signal-to-execution optimization
'''
    }
    
    for filename, content in agents.items():
        filepath = f'.claude/agents/{filename}'
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Created agent file: {filepath}")

def create_project_files():
    """Create additional project configuration files"""
    
    # Create .claude/config.yaml
    config = {
        'default_team': 'a-team',
        'project_name': 'BTC Perpetual Futures Trading System',
        'project_description': 'Clean, focused trading system for BTC-USDT perpetual futures',
        'context': {
            'market': 'BTC-USDT',
            'exchange': 'Bybit',
            'instrument': 'Perpetual Futures',
            'focus': 'System refactoring, ML enhancement, execution optimization'
        }
    }
    
    with open('.claude/config.yaml', 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)
    print("Created config file: .claude/config.yaml")
    
    # Create README.md
    readme_content = '''# BTC Perpetual Futures Trading System

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
'''
    
    with open('README.md', 'w', encoding='utf-8') as f:
        f.write(readme_content)
    print("Created README.md")

def main():
    """Main setup function"""
    print("Setting up Focused Trading A-Team (3 Core Agents)...")
    print("=" * 60)
    
    create_directory_structure()
    print()
    
    create_team_config()
    print()
    
    create_agent_files()
    print()
    
    create_project_files()
    print()
    
    print("=" * 60)
    print("Setup complete! Your focused Trading A-Team is ready.")
    print(f"Team size: 3 specialized agents")
    print("\nTeam Focus:")
    print("🏗️  Architect: Clean up fragmented codebase")
    print("🧠 ML Specialist: Enhance model accuracy") 
    print("⚡ Execution Specialist: Optimize order execution")
    print("\nNext steps:")
    print("1. Run: claude-code team activate a-team")
    print("2. Run: claude-code team workflow daily-standup")
    print("3. Start with: claude-code chat architect 'Analyze current system'")
    print("\nFor help: claude-code team --help")

if __name__ == "__main__":
    main()
