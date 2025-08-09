# model_trainer.py
# Reformatted from notebook export to standard Python file

import os
import json
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import traceback
import optuna


class ModelTrainer:
    def __init__(self, config):
        self.config = config
        self.trained_features = []
        self.base_dir = config.get('base_dir', './')
        safe_symbol = config.get(
            'symbol', 'SYMBOL'
        ).replace('/', '_').replace(':', '')
        
        # Use the exact filenames expected by the backtester
        self.model_path = os.path.join(
            self.base_dir, f"lgbm_model_{safe_symbol}_l2_only.txt"
        )
        self.features_json_path = os.path.join(
            self.base_dir, f"model_features_{safe_symbol}_l2_only.json"
        )
        print("ModelTrainer initialized.")

    def _prepare_data(self, df):
        # Exclude ALL non-predictive columns to prevent target leakage
        exclude_columns = [
            'target', 'timestamp', 'symbol', 'exchange', 'id',
            # CRITICAL: Exclude all target-related columns that leak future information
            'target_return_1min', 'target_return_5min', 'target_volatility', 
            'target_direction', 'target_return', 'target_price',
            # Also exclude metadata that's not useful for prediction
            'update_id', 'sequence_id', 'data_quality_score',
            # Close is redundant with mid_price in L2-only mode
            'close'
        ]
        
        features = [
            c for c in df.columns
            if c not in exclude_columns
        ]
        
        print(f"Preparing {len(features)} features for training (excluded {len(df.columns) - len(features) - 1} columns)")
        print(f"Excluded columns: {[c for c in df.columns if c in exclude_columns]}")
        
        X = df[features]
        y = df['target']
        self.trained_features = features
        return X, y

    def train(self, df_labeled):
        X, y = self._prepare_data(df_labeled)
        if X.empty or y.empty:
            print("Error: Data preparation failed.")
            return None, []

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        def objective(trial):
            params = {
                'objective': 'regression_l1',
                'metric': 'mae',
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'learning_rate': trial.suggest_float(
                    'learning_rate', 1e-3, 0.1, log=True
                ),
                'num_leaves': trial.suggest_int('num_leaves', 20, 150),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'lambda_l1': trial.suggest_float(
                    'lambda_l1', 1e-8, 10.0, log=True
                ),
                'lambda_l2': trial.suggest_float(
                    'lambda_l2', 1e-8, 10.0, log=True
                ),
                'feature_fraction': trial.suggest_float(
                    'feature_fraction', 0.5, 1.0
                ),
                'bagging_fraction': trial.suggest_float(
                    'bagging_fraction', 0.5, 1.0
                ),
                'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
                'min_child_samples': trial.suggest_int(
                    'min_child_samples', 5, 100
                ),
                'verbosity': -1,
                'n_jobs': -1,
            }
            model = lgb.LGBMRegressor(**params)
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                eval_metric='mae',
                callbacks=[lgb.early_stopping(25, verbose=False)]
            )
            return mean_absolute_error(y_val, model.predict(X_val))

        study = optuna.create_study(direction='minimize')
        study.optimize(
            objective,
            n_trials=self.config.get('optuna_trials', 50)
        )

        best_params = study.best_params
        print("Best Optuna params:", best_params)
        
        final_model = lgb.LGBMRegressor(**best_params)
        final_model.fit(X, y)

        self.save_model(final_model)
        return final_model.booster_, self.trained_features

    def save_model(self, model):
        try:
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            model.booster_.save_model(self.model_path)
            
            features_info = {
                'trained_features': self.trained_features
            }
            with open(self.features_json_path, 'w') as f:
                json.dump(features_info, f, indent=4)
            
            print(f"Model saved to {self.model_path}")
            print(f"Features saved to {self.features_json_path}")
        except Exception as e:
            print(f"Error saving model: {e}")
            traceback.print_exc()

    def train_model(self, df_labeled_features, save=True):
        return self.train(df_labeled_features)