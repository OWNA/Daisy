@echo off
if "%1"=="" (
    python trade_interactive.py
) else (
    python trade_cli_advanced.py %*
)