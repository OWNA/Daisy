# 🚨 Sprint 2 - CRITICAL PIVOT & REFACTOR PLAN

**Objective:** To fix the cascading schema mismatch errors and establish a single, stable, and fully functional live trading script (`run.py`).

---

### Phase 1: Unify the Database Schema

*   **Task 1.1: Create a Unified, 63-Column Database (COMPLETED)**
    *   `[x]` The agent discovered the original `trading_bot.db` was severely corrupted.
    *   `[x]` A new, clean database (`trading_bot_live.db`) was created with a complete and verified 63-column schema that supports all features and metadata.
    *   `[x]` All old database files have been deleted, establishing a clean slate.

*   **Task 1.2: Update All Configurations to Use `trading_bot_live.db` (COMPLETED)**
    *   `[x]` The agent has successfully updated all configuration files (`.yaml`, `.py`) to point to the new `trading_bot_live.db`.
    *   `[x]` A dedicated verification script confirmed that 100% of the system is now configured for the new database.
    
*   **Task 1.3: Point System to Bybit Demo Trading Service (COMPLETED)**
    *   `[x]` All references to `testnet` or `sandbox: True` have been correctly updated to `sandbox: False`.
    *   `[x]` The system is now configured to use the official Bybit Demo Trading URLs for realistic data.

### Phase 2: Refactor Core Components for Robustness

*   **Task 2.1: Make the Data Loader Resilient**
    *   `[ ]` The `@architect` will refactor the `load_raw_l2_data_from_db` method in `modeltrainer_enhanced.py`.
    *   `[ ]` The SQL query within this method must be **dynamically generated**. It should only select the columns that are explicitly defined in our `FeatureRegistry`. This will prevent future `no such column` errors.

*   **Task 2.2: Consolidate and Clean Up Scripts**
    *   `[ ]` The `@architect` will **delete** the following redundant scripts: `run_production_paper_trading.py`, `run_live_trading.py`, and `run_offline_mode.py`.
    *   `[ ]` The `@architect` will ensure that `run.py` is the single, stable entry point for the application and that it correctly uses the new `trading_bot_live.db` and the refactored data loader.

### Phase 3: Final End-to-End Validation

*   **Task 3.1: Re-Run Full System Validation**
    *   `[ ]` The `@architect` will execute the `run.py` script.
    *   **Goal:** To see the system initialize, run its integration tests (which should now pass), and begin the live trading loop using the new database and resilient data loader, all without errors.


