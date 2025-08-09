# model_predictor.py
# Reformatted from notebook export to standard Python file

import os
import json
import pickle
import pandas as pd
import numpy as np
import lightgbm as lgb  # For loading lgb.Booster
import traceback  # For detailed error logging


class ModelPredictor:
    """
    Loads trained models, manages scaling information, and generates 
    predictions/signals for L2-only trading strategy.
    
    L2-Only Mode Changes:
    - Removes OHLCV dependencies from scaling info recalculation
    - Uses L2-derived price series for scaling calculations
    - Validates L2 feature availability for predictions
    """

    def __init__(self, config, data_handler=None, label_generator=None):
        """
        Initializes the ModelPredictor for L2-only mode.

        Args:
            config (dict): Configuration dictionary.
            data_handler (DataHandler, optional): Instance of DataHandler, 
                needed if scaling info needs to be recalculated.
            label_generator (LabelGenerator, optional): Instance of 
                LabelGenerator, needed if scaling info needs to be recalculated.
        """
        self.config = config
        self.data_handler = data_handler
        self.label_generator = label_generator

        self.model_object = None
        self.trained_features = []

        self.target_mean = None
        self.target_std = None

        self.base_dir = config.get('base_dir', './trading_bot_data')
        safe_symbol = config.get('symbol', 'SYMBOL').replace('/', '_').replace(':', '')
        timeframe = config.get('timeframe', 'TIMEFRAME')

        self.model_path_default = os.path.join(
            self.base_dir,
            f"lgbm_model_{safe_symbol}_{timeframe}.txt"
        )
        self.ensemble_model_path_default = os.path.join(
            self.base_dir,
            f"ensemble_models_{safe_symbol}_{timeframe}.pkl"
        )
        self.features_json_path_default = os.path.join(
            self.base_dir,
            f"model_features_{safe_symbol}_{timeframe}.json"
        )

        print("ModelPredictor initialized in L2-Only Mode.")

    def set_scaling_params(self, target_mean: float, target_std: float) -> None:
        """
        Sets the target scaling parameters externally.
        These should be the mean and std of the *scaled* target the model 
        was trained on.
        """
        self.target_mean = target_mean
        self.target_std = target_std
        print(
            f"ModelPredictor L2-Only: Scaling params set: "
            f"mean={self.target_mean}, std={self.target_std}"
        )

    def _ensure_scaling_info(self) -> bool:
        """
        Ensures target_mean and target_std are available for L2-only mode.
        If not, attempts to recalculate them using L2-derived price data.
        """
        if self.target_mean is not None and self.target_std is not None:
            return True

        print(
            "Warning (ModelPredictor L2-Only): target_mean or target_std not set. "
            "Attempting to recalculate using L2 data..."
        )
        if not self.data_handler or not self.label_generator:
            print(
                "Error (ModelPredictor L2-Only): DataHandler or LabelGenerator "
                "not provided. Cannot recalculate scaling info."
            )
            self.target_mean = (self.target_mean 
                               if self.target_mean is not None else 0.0)
            self.target_std = (self.target_std 
                              if self.target_std is not None else 1.0)
            print(
                f"ModelPredictor L2-Only: Using fallback scaling: "
                f"mean={self.target_mean}, std={self.target_std}"
            )
            return False

        try:
            print(
                "ModelPredictor L2-Only: Attempting to load L2 data for "
                "scaling info recalculation."
            )
            # L2-Only Mode: Load L2 data instead of OHLCV
            df_for_scaling_raw = self.data_handler.load_and_prepare_historical_data(
                fetch_ohlcv_limit=0,  # L2-Only: No OHLCV data needed
                use_historical_l2=True,  # L2-Only: Use L2 data
                save_ohlcv=False
            )

            if (
                df_for_scaling_raw is None
                or df_for_scaling_raw.empty
            ):
                print(
                    "Error (ModelPredictor L2-Only): Failed to get L2 data for "
                    "recalculating scaling info."
                )
                self.target_mean = 0.0
                self.target_std = 1.0
                return False

            # L2-Only Mode: Check for L2-derived price column
            price_column = None
            for col in ['weighted_mid_price', 'microprice', 'mid_price', 'price']:
                if col in df_for_scaling_raw.columns:
                    price_column = col
                    break
            
            if price_column is None:
                print(
                    "Error (ModelPredictor L2-Only): No L2-derived price column "
                    "found for scaling calculation."
                )
                self.target_mean = 0.0
                self.target_std = 1.0
                return False

            # Create a temporary DataFrame with the price column as 'close' 
            # for label generation
            df_temp = df_for_scaling_raw.copy()
            df_temp['close'] = df_temp[price_column]

            _, mean_val, std_val = self.label_generator.generate_labels(df_temp)

            if mean_val is not None and std_val is not None:
                self.target_mean = mean_val
                self.target_std = std_val
                print(
                    f"ModelPredictor L2-Only: Recalculated scaling info using "
                    f"{price_column}: mean={self.target_mean}, "
                    f"std={self.target_std}"
                )
                return True
            else:
                print(
                    "Error (ModelPredictor L2-Only): Failed to recalculate "
                    "scaling info from LabelGenerator."
                )
                self.target_mean = 0.0
                self.target_std = 1.0
                return False

        except Exception as e:
            print(f"Error (ModelPredictor L2-Only): Exception during scaling "
                  f"info recalculation: {e}")
            traceback.print_exc()
            self.target_mean = (self.target_mean 
                               if self.target_mean is not None else 0.0)
            self.target_std = (self.target_std 
                              if self.target_std is not None else 1.0)
            return False

    def load_model_and_features(
        self,
        model_file_path: str = None,
        features_file_path: str = None,
        load_ensemble: bool = False
    ) -> bool:
        """
        Loads a trained model (standard booster or ensemble) and its feature list.
        Also ensures scaling information is available.
        """
        _model_path = model_file_path or (
            self.ensemble_model_path_default if load_ensemble else self.model_path_default
        )
        _features_path = features_file_path

        try:
            if load_ensemble:
                with open(_model_path, 'rb') as f:
                    ensemble_content = pickle.load(f)
                if isinstance(ensemble_content, dict):
                    self.model_object = ensemble_content
                    self.trained_features = ensemble_content.get('trained_features', [])
                    self.target_mean = ensemble_content.get('target_mean', self.target_mean)
                    self.target_std = ensemble_content.get('target_std', self.target_std)

                    if not self.trained_features and _features_path and os.path.exists(_features_path):
                        with open(_features_path, 'r') as f_json:
                            self.trained_features = json.load(f_json)
                else:
                    print(
                        f"Error (ModelPredictor): Ensemble model at {_model_path} is not a dictionary."
                    )
                    return False
                print(f"Ensemble model loaded from {_model_path}")
            else:
                self.model_object = lgb.Booster(model_file=_model_path)
                _features_path_to_load = _features_path or self.features_json_path_default
                if os.path.exists(_features_path_to_load):
                    with open(_features_path_to_load, 'r') as f_json:
                        features_data = json.load(f_json)
                        if isinstance(features_data, dict):
                            self.trained_features = features_data.get('trained_features', [])
                            # Also load scaling params if available
                            if 'target_mean' in features_data and features_data['target_mean'] is not None:
                                self.target_mean = features_data['target_mean']
                            if 'target_std' in features_data and features_data['target_std'] is not None:
                                self.target_std = features_data['target_std']
                        else:
                            self.trained_features = features_data
                else:
                    print(
                        f"Warning (ModelPredictor): Features JSON file not found at "
                        f"{_features_path_to_load} for standard model."
                    )
                print(f"Standard model booster loaded from {_model_path}")

            if not self.trained_features:
                print(
                    f"Warning (ModelPredictor): No trained features loaded from path or ensemble "
                    f"for model '{_model_path}'."
                )

            print(
                f"ModelPredictor: Loaded {len(self.trained_features)} features: "
                f"{self.trained_features[:5]}..."
            )

            if not self._ensure_scaling_info():
                print(
                    "Warning (ModelPredictor): Scaling info could not be confirmed/recalculated. "
                    "Predictions might be unscaled or use defaults."
                )
            return True

        except FileNotFoundError:
            print(
                f"Error (ModelPredictor): Model or features file not found. Model: '{_model_path}', "
                f"Features path used: '{_features_path if _features_path else 'within ensemble or default'}'"
            )
            self.model_object = None
            self.trained_features = []
            return False
        except Exception as e:
            print(f"Error (ModelPredictor): Loading model/features failed: {e}")
            traceback.print_exc()
            self.model_object = None
            self.trained_features = []
            return False

    def predict_signals(
        self,
        df_with_features: pd.DataFrame,
        threshold: float = None,
        use_ensemble: bool = False
    ) -> pd.DataFrame:
        """
        Generates predictions and trading signals on the input DataFrame.
        """
        if self.model_object is None:
            print(
                "Error (ModelPredictor): Model not loaded. Call load_model_and_features() first."
            )
            return None
        if not self.trained_features:
            print(
                "Error (ModelPredictor): Trained feature list not available. "
                "Cannot determine input features for model."
            )
            return None
        if df_with_features is None or df_with_features.empty:
            print("Error (ModelPredictor): Input DataFrame for prediction is empty.")
            return None

        _threshold = threshold if threshold is not None else self.config.get(
            'prediction_threshold', self.config.get('backtest_threshold', 0.5)
        )

        if self.target_mean is None or self.target_std is None:
            print(
                "Warning (ModelPredictor): Scaling parameters (target_mean, target_std) are not set. "
                "Unscaling will not be performed or use defaults."
            )
            if not self._ensure_scaling_info():
                print(
                    "Critical Warning (ModelPredictor): Failed to ensure scaling info. "
                    "Predictions might be inaccurate if model expects scaled target."
                )

        missing_features = [
            f for f in self.trained_features if f not in df_with_features.columns
        ]
        if missing_features:
            print(
                f"Error (ModelPredictor): Input DataFrame is missing required features: "
                f"{missing_features}"
            )
            return None

        X_predict = df_with_features[self.trained_features].copy()

        if X_predict.isnull().values.any():
            nan_cols = X_predict.columns[X_predict.isnull().any()].tolist()
            print(
                f"Warning (ModelPredictor): NaNs found in features for prediction: "
                f"{nan_cols}. Model might handle them if trained accordingly. Consider imputation."
            )

        result_df = df_with_features.copy()

        try:
            if use_ensemble:
                if not isinstance(self.model_object, dict) or 'classifier' not in self.model_object:
                    print(
                        "Error (ModelPredictor): Ensemble model not loaded correctly or 'classifier' missing."
                    )
                    return None

                clf = self.model_object['classifier']
                reg = self.model_object.get('regressor')
                clf_map_to_lgbm = self.model_object.get('clf_target_map_to_lgbm', {-1: 0, 0: 1, 1: 2})
                clf_map_from_lgbm = {v: k for k, v in clf_map_to_lgbm.items()}

                clf_preds_raw = clf.predict(X_predict)
                result_df["signal"] = pd.Series(
                    clf_preds_raw, index=X_predict.index
                ).map(clf_map_from_lgbm).fillna(0).astype(int)

                if reg:
                    pred_reg_scaled = reg.predict(X_predict)
                    result_df["pred_scaled"] = pred_reg_scaled
                    if (
                        self.target_mean is not None
                        and self.target_std is not None
                        and self.target_std > 1e-9
                    ):
                        result_df["pred_unscaled_target"] = (
                            pred_reg_scaled * self.target_std
                        ) + self.target_mean
                    else:
                        result_df["pred_unscaled_target"] = pred_reg_scaled
                else:
                    result_df["pred_scaled"] = result_df["signal"] * _threshold
                    result_df["pred_unscaled_target"] = np.nan
            else:
                pred_scaled = self.model_object.predict(X_predict)
                result_df["pred_scaled"] = pred_scaled

                if (
                    self.target_mean is not None
                    and self.target_std is not None
                    and self.target_std > 1e-9
                ):
                    result_df["pred_unscaled_target"] = (
                        pred_scaled * self.target_std
                    ) + self.target_mean
                else:
                    result_df["pred_unscaled_target"] = pred_scaled

                result_df["signal"] = np.select(
                    [pred_scaled > _threshold, pred_scaled < -_threshold],
                    [1, -1],
                    default=0
                )
            print("Predictions and signals generated.")
        except Exception as e:
            print(f"Error (ModelPredictor): Prediction failed: {e}")
            traceback.print_exc()
            return None

        return result_df

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Simple prediction method for backward compatibility.
        Returns raw predictions without signal generation.
        
        Args:
            X: Feature array of shape (n_samples, n_features)
            
        Returns:
            Array of predictions
        """
        if self.model_object is None:
            raise ValueError("Model not loaded. Call load_model_and_features() first.")
            
        if isinstance(self.model_object, dict):
            # Ensemble model
            if 'regressor' in self.model_object and self.model_object['regressor'] is not None:
                return self.model_object['regressor'].predict(X)
            else:
                # Use classifier predictions as numeric values
                return self.model_object['classifier'].predict(X).astype(float)
        else:
            # Standard LightGBM model
            return self.model_object.predict(X)
            
    def predict_single(self, features_dict):
        """
        Make a single prediction from a dictionary of features.
        
        Args:
            features_dict (dict): Dictionary with feature names as keys
            
        Returns:
            float: Single prediction value
        """
        try:
            # Convert dict to DataFrame row
            import pandas as pd
            df = pd.DataFrame([features_dict])
            
            # Ensure we have all required features
            if hasattr(self, 'trained_features') and self.trained_features:
                # Check if all required features are present
                missing_features = set(self.trained_features) - set(df.columns)
                if missing_features:
                    # Add missing features with 0 values
                    for feat in missing_features:
                        df[feat] = 0.0
                # Reorder columns to match training
                df = df[self.trained_features]
            
            # Convert to numpy array for prediction
            X = df.values
            
            # Make prediction
            predictions = self.predict(X)
            return float(predictions[0]) if len(predictions) > 0 else 0.0
            
        except Exception as e:
            print(f"Error in predict_single: {e}")
            import traceback
            traceback.print_exc()
            return 0.0