@echo off
echo Training on l2_data_BTC_USDT_USDT_20250707_040413.jsonl.gz
python train_model_robust.py --data l2_data_BTC_USDT_USDT_20250707_040413.jsonl.gz --features all --trials 50
pause