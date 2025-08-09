# Archive Directory Structure

This directory contains archived files from the BTC trading system consolidation.
Files are organized by their original purpose to make retrieval easy if needed.

## Directory Structure:

### debug_scripts/
- One-off debugging scripts (diagnose_*.py, check_*.py, debug_*.py)
- Diagnostic tools that were used to fix specific issues

### execution_scripts/
- Various run_*.py scripts that executed different parts of the system
- Replaced by unified CLI interface

### test_scripts/
- Old test files (test_*.py)
- Will be replaced by proper test suite in tests/

### setup_scripts/
- Environment setup scripts
- One-time configuration scripts

### duplicate_systems/
- Complete duplicate implementations of the trading system
- Multiple orchestrators that did the same thing

### utilities/
- Misc utility scripts
- Data upload managers
- Cleanup scripts

### model_versions/
- Different versions of model training scripts
- Fix attempts for model issues

### cli_versions/
- Multiple CLI and menu implementations
- Now consolidated into single cli.py

### l2_processing/
- Various L2 data processing implementations
- Now consolidated into l2_processor.py

## Retrieval:
If you need to reference any archived code, files are named exactly as they were
in the main directory. Check the git history for when they were moved.