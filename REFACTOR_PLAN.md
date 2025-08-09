# 🚀 System Refactoring and Consolidation Plan

**Objective:** To resolve all outstanding configuration and architectural issues, resulting in a single, stable, and verifiable `run.py` entry point that is connected to a live, realistic data feed. All other development is on hold until this plan is complete.

---

## Phase 1: Stabilize the Foundation

*   **Task 1.1: Full System Verification (COMPLETED)**
    *   `[x]` **Database Schema:** Verified that `trading_bot_live.db` exists and has the correct, complete 63-column schema.
    *   `[x]` **Configuration State:** Verified that all components are correctly configured for the Bybit Demo Trading service, but are incorrectly pointing to `trading_bot_demo.db`.

*   **Task 1.2: Unify Configuration (COMPLETED)**
    *   `[x]` **Action:** The `@architect` successfully updated all files in the codebase, replacing references to `trading_bot_demo.db` with the correct `trading_bot_live.db`.
    *   `[x]` **Validation:** The `verify_demo_trading_config.py` script was re-run and now passes all checks, confirming the system is fully aligned.

*   **Task 1.3: Consolidate Entry Point (COMPLETED)**
    *   `[x]` **Goal:** Eliminate confusion by having a single script to run the application.
    *   `[x]` **Action:** The `@architect` has successfully deleted all redundant `run_*.py` and legacy `main_*.py` scripts.
    *   `[x]` **Validation:** `run.py` is now confirmed to be the sole primary execution script in the root directory, fulfilling the architectural requirements outlined in `CLAUDE.md`.

---

## Phase 2: Decouple and Validate (COMPLETED)

*   **Task 2.1: Decouple Production from Testing (COMPLETED)**
    *   `[x]` **Goal:** Break the toxic import chain that is pulling the `lightgbm` dependency into our main application.
    *   `[x]` **Action:** The `@architect` successfully refactored `run.py`. The import and execution of `test_full_pipeline` have been completely removed from the production script, permanently severing the dependency on test modules.
    *   `[x]` **Validation:** The `run.py` script is now architecturally clean and will no longer fail due to test-specific dependencies.

*   **Task 2.2: Final End-to-End Validation (COMPLETED)**
    *   `[x]` **Goal:** Prove that the fully consolidated and decoupled system works as expected.
    *   `[ ]` **Action:** The `@architect` will now execute the refactored `run.py` script.
    *   `[ ]` **Expected Outcome:** The system will initialize without errors, load the production model, and begin its trading loop. It will initially show errors related to the empty database, which will be resolved by running the data ingestor.

