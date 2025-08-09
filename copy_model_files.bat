@echo off
echo Copying model files to correct location...

copy "trading_bot_data\lgbm_model_BTC_USDTUSDT_l2_only.txt" "lgbm_model_BTC_USDTUSDT_l2_only.txt"
copy "trading_bot_data\model_features_BTC_USDTUSDT_l2_only.json" "model_features_BTC_USDTUSDT_l2_only.json"

if exist "lgbm_model_BTC_USDTUSDT_l2_only_scaling.json" (
    echo Scaling file already in root directory
) else (
    echo Creating scaling file in root directory...
)

echo.
echo Files copied successfully!
echo You can now run: python main.py trade --paper
pause