# PowerShell wrapper for Trading CLI
if ($args.Count -eq 0) {
    python trade_interactive.py
} else {
    python trade_cli_advanced.py $args
}