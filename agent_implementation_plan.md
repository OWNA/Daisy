# Agent Implementation Plan for BTC L2 Trading System

## Quick Start: How to Use These Agents

### 1. System Cleanup Agent - Immediate Actions

**First Task:**
```
"You are the System Cleanup & Architecture Specialist. Please analyze the current codebase structure and create a consolidation plan. Start by:
1. Identifying all duplicate functionality across files
2. Mapping the data flow from Bybit → Features → Model → Execution
3. Proposing a simplified architecture with <20 files
Focus on datahandler.py, featureengineer.py, and the 41 archived files."
```

**Expected Output:**
- File consolidation map
- Proposed new architecture
- List of safe-to-delete files
- Migration plan

### 2. ML Model Enhancement Agent - Immediate Actions

**First Task:**
```
"You are the ML Model Enhancement Specialist. The current LightGBM model has 84 features but generates many false signals (threshold had to be lowered to 0.01). Please:
1. Analyze the current feature set in model_features_BTC_USDTUSDT_l2_only.json
2. Propose 10-15 new microstructure features that could improve signal quality
3. Suggest ensemble approach to reduce false positives
Start with feature importance analysis."
```

**Expected Output:**
- Feature importance ranking
- New feature proposals with rationale
- Ensemble architecture design
- Expected improvement metrics

### 3. Execution Optimization Agent - Immediate Actions

**First Task:**
```
"You are the Execution Optimization Specialist. The system currently uses simple market orders with fixed 5% position sizing. Please:
1. Analyze the current execution in main.py and smartorderexecutor.py
2. Design a passive order placement strategy using limit orders
3. Propose dynamic position sizing based on order book depth
Focus on reducing slippage for BTC/USDT:USDT on Bybit."
```

**Expected Output:**
- Current execution analysis
- Passive order algorithm design
- Position sizing formula
- Implementation plan

## Practical Workflow

### Day 1: System Analysis
Run all three agents in parallel with their first tasks:
```bash
# Create three terminal windows or use Task tool

# Terminal 1: System Cleanup
python main.py  # Let agent analyze current structure

# Terminal 2: Model Enhancement  
python check_predictions.py  # Let agent see current predictions

# Terminal 3: Execution Analysis
python main.py trade --paper  # Let agent observe execution
```

### Day 2-3: Implementation Priority

**Priority 1: Fix Critical Issues (System Cleanup Agent)**
- Consolidate L2 data handling (currently split across multiple files)
- Fix database schema for features table
- Create single configuration system

**Priority 2: Quick Wins (Execution Agent)**
- Implement basic limit order placement
- Add order book depth check before trading
- Improve position sizing logic

**Priority 3: Model Improvements (ML Agent)**
- Add order flow imbalance features
- Implement signal confidence scoring
- Create feature importance monitoring

### Week 1 Deliverables

**System Cleanup:**
- ✓ Reduced to 20 organized files
- ✓ Fixed database schema
- ✓ Clear data pipeline

**Model Enhancement:**
- ✓ 10 new microstructure features
- ✓ Signal confidence scoring
- ✓ Improved backtesting

**Execution:**
- ✓ Passive order placement
- ✓ Dynamic position sizing
- ✓ Slippage monitoring

## Integration Points

### Agent Collaboration Examples:

1. **Cleanup + ML Agent:**
   - Cleanup agent creates clean feature pipeline
   - ML agent adds new features to pipeline
   - Both ensure <10ms computation time

2. **ML + Execution Agent:**
   - ML agent provides signal confidence (0-1)
   - Execution agent uses confidence for urgency
   - Higher confidence = more aggressive execution

3. **All Three:**
   - Cleanup provides clean architecture
   - ML generates alpha signals
   - Execution optimizes entry/exit

## Specific File Assignments

### System Cleanup Agent Owns:
- `main.py` - Entry point consolidation
- `datahandler.py` - L2 data processing
- `database.py` - Schema fixes
- `config.yaml` - Configuration management

### ML Enhancement Agent Owns:
- `featureengineer.py` - Feature computation
- `modeltrainer.py` - Training pipeline
- `modelpredictor.py` - Inference optimization
- `labelgenerator.py` - Target creation

### Execution Agent Owns:
- `smartorderexecutor.py` - Order algorithms
- `advancedriskmanager.py` - Position sizing
- `livesimulator.py` - Execution testing
- Bybit API integration

## Success Checklist

### Week 1:
- [ ] Codebase reduced to <20 files
- [ ] Database queries <10ms
- [ ] 10+ new features implemented
- [ ] Limit orders working
- [ ] Position sizing dynamic

### Week 2:
- [ ] Full test coverage
- [ ] Ensemble model deployed
- [ ] Passive execution live
- [ ] Monitoring dashboard

### Month 1:
- [ ] Sharpe ratio improved 30%+
- [ ] Slippage reduced 50%+
- [ ] System running 24/7
- [ ] Profitable in paper trading

## Common Commands for Agents

```bash
# System Cleanup Agent
find . -name "*.py" | wc -l  # Count Python files
grep -r "class DataHandler" . # Find duplicates
python -m pytest  # Run tests

# ML Enhancement Agent  
python modeltrainer.py  # Retrain model
python check_predictions.py  # Analyze predictions
python strategybacktester.py  # Backtest changes

# Execution Agent
python main.py trade --paper  # Test execution
python diagnose_system.py  # Check performance
tail -f logs/trading.log  # Monitor execution
```

## Emergency Fixes

If agents break something:
1. Revert to backup: `git checkout HEAD~1`
2. Use archived files: `cp archive/old_file.py .`
3. Restore config: `cp config_backup_*.yaml config.yaml`
4. Check logs: `tail -n 100 logs/*.log`

Remember: Each agent should make small, testable changes and always backup before major modifications!