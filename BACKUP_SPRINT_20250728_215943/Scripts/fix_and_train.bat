@echo off
echo FIXING AND TRAINING YOUR L2 DATA
echo ================================

REM Activate venv
call venv\Scripts\activate

REM Train using the existing model trainer with your config
echo Training with your existing ModelTrainer...
python -c "import yaml; import sys; sys.path.append('.'); from modeltrainer import ModelTrainer; from datahandler import DataHandler; from featureengineer import FeatureEngineer; from labelgenerator import LabelGenerator; import ccxt; config = yaml.safe_load(open('config.yaml')); config['use_l2_features'] = True; config['use_hht_features'] = True; config['optuna_trials'] = 50; exchange = ccxt.bybit(); dh = DataHandler(config, exchange); fe = FeatureEngineer(config); lg = LabelGenerator(config); dh.l2_raw_data_path = 'l2_data/l2_data_040413_converted.jsonl.gz'; df = dh.load_l2_historical_data(); df_features = fe.generate_all_features(df); df_labeled, _, _ = lg.generate_labels(df_features); mt = ModelTrainer(config, list(df_features.columns)); model, features = mt.train_model(df_labeled); print('SUCCESS!')"

pause