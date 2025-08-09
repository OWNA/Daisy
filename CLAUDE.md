# Project BTC Trading Bot - Core Meta-Plan

This document serves as the central, high-level guide for all AI agents working on this project.

## Core Objective
To build a robust, ML-driven, live paper trading bot for BTC/USDT on the Bybit exchange.

## Primary Documentation
- **Current Tasks:** `REFACTOR_PLAN.md` contains the immediate, highest-priority tasks for stabilizing the system. **Consult this file first.**
- **Sprint Goals:** `SPRINT_2_PLAN.md` contains the overall goals for the current development sprint.

## Key Architectural Components
- **Primary Execution Script:** The single, definitive entry point for this application is `run.py`. All other `run_*.py` or `main_*.py` scripts are considered deprecated or are for specific testing purposes only.
- **Primary Database:** The unified database for all live and training data is `trading_bot_live.db`.
- **Configuration:** Core configuration is managed in `config.yaml` and loaded via the `ComponentFactory` in `component_factory.py`.

## Agent Protocol
1.  Always consult `REFACTOR_PLAN.md` for current tasks before starting work.
2.  Adhere strictly to the "verify, then trust" principle. Validate all changes.
3.  Do not create new execution scripts. Refactor and improve the existing `run.py`.

