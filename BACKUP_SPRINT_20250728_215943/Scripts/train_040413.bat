@echo off
echo.
echo ========================================
echo Training on your specific L2 data file
echo ========================================
echo.

REM First check if file exists
if exist "l2_data\l2_data_BTC_USDT_USDT_20250707_040413.jsonl.gz" (
    echo File found: l2_data_BTC_USDT_USDT_20250707_040413.jsonl.gz
    
    REM Check format first
    echo.
    echo Checking file format...
    python convert_l2_format.py --check l2_data\l2_data_BTC_USDT_USDT_20250707_040413.jsonl.gz
    
    echo.
    echo Converting to standard format if needed...
    python convert_l2_format.py --convert l2_data\l2_data_BTC_USDT_USDT_20250707_040413.jsonl.gz --output l2_data\l2_data_040413_converted.jsonl.gz
    
    echo.
    echo Training model with converted data...
    python train_model_robust.py --data l2_data_040413_converted.jsonl.gz --features all --trials 50
) else (
    echo ERROR: File not found!
    echo.
    echo Available files in l2_data:
    dir l2_data\*.gz /b | findstr "040413"
)

echo.
pause