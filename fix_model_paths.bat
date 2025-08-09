@echo off
echo Fixing model file paths...

REM Create trading_bot_data directory if it doesn't exist
if not exist "trading_bot_data" mkdir trading_bot_data

REM Copy scaling file to correct location
if exist "lgbm_model_BTC_USDTUSDT_l2_only_scaling.json" (
    copy "lgbm_model_BTC_USDTUSDT_l2_only_scaling.json" "trading_bot_data\lgbm_model_BTC_USDTUSDT_l2_only_scaling.json"
    echo Copied scaling file to trading_bot_data
) else (
    echo Scaling file not found in root directory
)

echo.
echo Files fixed! You can now run paper trading:
echo python main.py trade --paper
pause