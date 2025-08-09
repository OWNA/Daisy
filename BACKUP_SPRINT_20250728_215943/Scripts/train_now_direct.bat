@echo off
echo Direct L2 Model Training
echo =======================
echo.
echo This will train directly on your converted L2 data
echo without database dependencies or complex configurations
echo.

python train_direct.py --data l2_data\l2_data_040413_converted.jsonl.gz --trials 50

echo.
echo If you want to test with fewer records first:
echo python train_direct.py --data l2_data\l2_data_040413_converted.jsonl.gz --trials 10 --sample 10000
echo.
pause