import sys
import os
import yaml
import ccxt
import argparse

from datahandler import DataHandler
from featureengineer import FeatureEngineer
from labelgenerator import LabelGenerator
from modeltrainer import ModelTrainer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def load_config():
    """Load configuration from YAML files."""
    for config_file in ['config.yaml', 'config_l2_only.yaml']:
        if os.path.exists(config_file):
            with open(config_file, 'r') as f:
                return yaml.safe_load(f)
    print("Warning: No config file found. Using default settings.")
    return {'symbol': 'BTC/USDT:USDT', 'optuna_trials': 10}


def train_l2_model(features='l2', trials=50):
    """Main function to orchestrate the L2 model training process."""
    print("=" * 60)
    print("STARTING ROBUST L2 MODEL TRAINING")
    print("=" * 60)

    config = load_config()
    config['use_l2_features'] = features in ['all', 'l2']
    config['use_hht_features'] = features in ['all', 'hht']
    config['optuna_trials'] = trials

    print("\n[1/5] Initializing components...")
    try:
        exchange = ccxt.bybit({'enableRateLimit': True})
        data_handler = DataHandler(config, exchange)
        feature_engineer = FeatureEngineer(config)
        label_generator = LabelGenerator(config)
        trainer = ModelTrainer(config)
    except Exception as e:
        print(f"Error initializing components: {e}")
        return False
    print("Components initialized successfully.")

    print("\n[2/5] Loading L2 data from database...")
    try:
        df_l2 = data_handler.load_l2_historical_data(
            table_name='l2_training_data_practical'
        )
        if df_l2 is None or df_l2.empty:
            print("Error: No L2 data loaded from the database.")
            return False
        print(f"Loaded {len(df_l2)} L2 records.")
    except Exception as e:
        print(f"Error loading L2 data: {e}")
        return False

    print("\n[3/5] Generating features...")
    try:
        df_features = feature_engineer.generate_features(df_l2)
        if df_features is None or df_features.empty:
            print("Error: Feature generation failed.")
            return False
        print(f"Generated {len(df_features.columns)} features.")
    except Exception as e:
        print(f"Error generating features: {e}")
        return False

    print("\n[4/5] Generating labels...")
    try:
        df_labeled, _, _ = label_generator.generate_labels(df_features)
        if df_labeled is None or df_labeled.empty:
            print("Error: Label generation failed.")
            return False
        print(f"Generated labels for {len(df_labeled)} samples.")
    except Exception as e:
        print(f"Error generating labels: {e}")
        return False

    print("\n[5/5] Training model...")
    try:
        model, trained_features = trainer.train_model(df_labeled)
        if model is None:
            print("Error: Model training failed.")
            return False
        print(f"Model training complete. "
              f"Trained on {len(trained_features)} features.")
    except Exception as e:
        print(f"Error during model training: {e}")
        return False

    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Robust L2 Model Training')
    parser.add_argument(
        '--features', choices=['l2', 'hht', 'all'], default='l2',
        help='Feature set to use for training.'
    )
    parser.add_argument(
        '--trials', type=int, default=5,
        help='Number of Optuna trials.'
    )
    args = parser.parse_args()

    try:
        is_success = train_l2_model(
            features=args.features,
            trials=args.trials
        )
    except Exception:
        import traceback
        print("\n--- A CRITICAL ERROR OCCURRED ---")
        traceback.print_exc()
        is_success = False

    if is_success:
        print("\nTraining pipeline completed successfully!")
    else:
        print("\nTraining pipeline failed.")
        sys.exit(1) 