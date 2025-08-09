CLAUDE CODE OPERATING RULES - Apply these for ALL remaining phases:

CODING STANDARDS:
1. Maximum 200 lines per file - if longer, justify why or split it
2. One responsibility per function/class
3. No nested functions deeper than 2 levels
4. All functions must have type hints and docstrings
5. Use logging instead of print statements everywhere

CHANGE DISCIPLINE:
1. Before modifying ANY file, state: "Modifying [file] to fix [specific audit issue]"
2. Show only the changed lines, not entire files
3. Never modify more than 3 files per response without asking
4. If you want to refactor something not in the current phase, note it for later

DATA HANDLING RULES:
1. All data validation must be explicit with clear error messages
2. Missing data handling must be consistent across the entire pipeline
3. No silent failures - everything must log success/failure
4. Database operations must be transactional

TRADING LOGIC RULES:
1. Separate signal generation from position sizing from execution
2. All trading decisions must be logged with reasoning
3. Risk management checks before every trade
4. Clear separation between backtesting and live trading modes

TESTING DISCIPLINE:
1. Each phase must have a simple test to prove it works
2. No moving to next phase until current phase executes successfully

Acknowledge these rules before we proceed to restructure.