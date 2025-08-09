# Trading Bot CLI Guide

## Installation

1. Make sure you have Python 3.8+ installed
2. Run the setup command:
   ```bash
   ./trade setup
   ```

## Quick Start

```bash
# Check system status
./trade status

# Collect L2 data
./trade collect-data --symbol BTC/USDT --limit 1000

# Train a model
./trade train-model --config config_l2.yaml

# Run backtest
./trade backtest --initial-balance 10000

# Start bot (simulation mode)
./trade start-bot --mode simulation

# View results
./trade results
```

## Commands

### `trade status`
Check system health, configurations, and available models.

### `trade collect-data`
Collect L2 order book data from the exchange.
- `--config, -c`: Configuration file (default: config_l2.yaml)
- `--symbol, -s`: Trading pair (default: BTC/USDT)
- `--limit, -l`: Number of records (default: 1000)

### `trade train-model`
Train a new trading model using collected data.
- `--config, -c`: Configuration file
- `--data-source, -d`: Data source - L2 or OHLCV (default: L2)

### `trade backtest`
Run backtesting to evaluate strategy performance.
- `--config, -c`: Configuration file
- `--start-date, -s`: Start date (YYYY-MM-DD)
- `--end-date, -e`: End date (YYYY-MM-DD)
- `--initial-balance, -b`: Starting balance (default: 10000)

### `trade start-bot`
Start the trading bot in various modes.
- `--config, -c`: Configuration file
- `--mode, -m`: Trading mode - simulation/paper/live (default: simulation)

### `trade results`
View recent trading results and performance metrics.

### `trade analyze`
Run various performance analysis scripts.

### `trade db`
Query the trading database directly.
Examples:
```bash
./trade db "SELECT COUNT(*) FROM L2_data"
./trade db "SELECT * FROM trades ORDER BY timestamp DESC LIMIT 10"
```

### `trade clean`
Clean up temporary files, caches, and logs.

### `trade setup`
Run initial system setup and validation.

## Trading Modes

1. **Simulation**: Test with historical data, no real trades
2. **Paper**: Live data but simulated trades
3. **Live**: Real trading with actual funds (use with caution!)

## Configuration Files

- `config.yaml`: Default configuration
- `config_l2.yaml`: L2 data collection config
- `config_l2_only.yaml`: L2-only strategy config
- `config_live_sim.yaml`: Live simulation config
- `config_wfo.yaml`: Walk-forward optimization config

## Workflow Examples

### Complete Training Workflow
```bash
# 1. Check status
./trade status

# 2. Collect fresh data
./trade collect-data --symbol BTC/USDT --limit 5000

# 3. Train model
./trade train-model --data-source L2

# 4. Backtest the model
./trade backtest --initial-balance 10000

# 5. Review results
./trade results
./trade analyze
```

### Daily Operations
```bash
# Morning check
./trade status
./trade results

# Run simulation
./trade start-bot --mode simulation

# Evening analysis
./trade analyze
./trade db "SELECT * FROM trades WHERE DATE(timestamp) = DATE('now')"
```

## Tips

1. Always run `./trade status` first to check system health
2. Test thoroughly in simulation mode before paper trading
3. Use paper trading for at least a week before going live
4. Monitor logs in the `logs/` directory for detailed information
5. Back up your database regularly: `cp trading_bot.db trading_bot_backup.db`

## Troubleshooting

If you encounter issues:

1. Check virtual environment is active
2. Run `./trade setup` to reinstall dependencies
3. Check logs for detailed error messages
4. Ensure all config files have the 'exchange' key set
5. Verify database connectivity with `./trade db "SELECT 1"`

## Safety Notes

- **NEVER** share your API keys or configuration files
- Always test strategies thoroughly before live trading
- Start with small amounts when going live
- Set appropriate stop-loss levels
- Monitor the bot regularly, especially in live mode