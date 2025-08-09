#!/usr/bin/env python3
"""
production_model_predictor.py - Production Model Predictor

This module provides a predictor class that loads and uses the production
ensemble model for real-time predictions in the paper trading system.

Sprint 2 - Priority 1: Train Production ML Model
"""

import os
import logging
import pickle
import json
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)


class ProductionModelPredictor:
    """Production model predictor using the trained ensemble model."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize production model predictor."""
        self.config = config
        self.base_dir = config.get('base_dir', './trading_bot_data')
        self.symbol = config.get('symbol', 'BTCUSDT')
        
        # Model file paths - normalize symbol to match training format
        safe_symbol = self._normalize_symbol(self.symbol)
        self.ensemble_path = os.path.join(self.base_dir, f"enhanced_ensemble_{safe_symbol}.pkl")
        self.features_path = os.path.join(self.base_dir, f"enhanced_features_{safe_symbol}.json")
        self.metrics_path = os.path.join(self.base_dir, f"enhanced_metrics_{safe_symbol}.json")
        
        # Model components
        self.ensemble = None
        self.feature_names = None
        self.horizons = None
        self.ensemble_weights = None
        self.model_metrics = None
        
        # Prediction parameters
        self.confidence_threshold = config.get('confidence_threshold', 0.6)
        self.signal_threshold = 1.0  # Minimum ensemble prediction for signal
        
        # Load production model
        self.load_production_model()
        
        logger.info(f"ProductionModelPredictor initialized")
        if self.horizons:
            logger.info(f"Loaded model with {len(self.horizons)} horizons: {self.horizons}")
        if self.feature_names:
            logger.info(f"Features: {len(self.feature_names)} features")

    def _normalize_symbol(self, symbol: str) -> str:
        """Normalize symbol to match training format."""
        # Convert exchange format to training format
        # BTC/USDT:USDT -> BTCUSDT
        # BTC/USDT -> BTCUSDT
        # BTCUSDT -> BTCUSDT
        
        if '/' in symbol or ':' in symbol:
            # Extract base and quote currency from exchange format
            if ':' in symbol:
                # Perpetual futures format: BTC/USDT:USDT
                base_quote = symbol.split(':')[0]  # BTC/USDT
            else:
                # Spot format: BTC/USDT
                base_quote = symbol
            
            if '/' in base_quote:
                base, quote = base_quote.split('/')
                return f"{base}{quote}"  # BTCUSDT
            else:
                return base_quote
        else:
            # Already in training format
            return symbol

    def load_production_model(self) -> bool:
        """Load the production ensemble model and metadata."""
        try:
            # Check if model files exist
            if not os.path.exists(self.ensemble_path):
                logger.error(f"Production model not found: {self.ensemble_path}")
                return False
            
            if not os.path.exists(self.features_path):
                logger.error(f"Features file not found: {self.features_path}")
                return False
            
            # Load ensemble model
            with open(self.ensemble_path, 'rb') as f:
                self.ensemble = pickle.load(f)
            
            # Load feature metadata
            with open(self.features_path, 'r') as f:
                features_data = json.load(f)
                self.feature_names = features_data['features']
                self.horizons = features_data['horizons']
                self.ensemble_weights = features_data['weights']
            
            # Load metrics if available
            if os.path.exists(self.metrics_path):
                with open(self.metrics_path, 'r') as f:
                    self.model_metrics = json.load(f)
            
            # Validate model components
            if 'models' not in self.ensemble:
                raise ValueError("Invalid ensemble model structure")
            
            # Convert horizon keys if needed
            self.horizons = [int(h) for h in self.horizons]
            
            logger.info(f"✓ Production model loaded successfully")
            logger.info(f"  Model training date: {self.model_metrics.get('training_date', 'Unknown')}")
            logger.info(f"  Model type: {self.model_metrics.get('model_type', 'Unknown')}")
            
            # Log model performance
            if self.model_metrics and 'validation_metrics' in self.model_metrics:
                for horizon, metrics in self.model_metrics['validation_metrics'].items():
                    mae = metrics.get('mae', 'N/A')
                    corr = metrics.get('correlation', 'N/A')
                    profit_ratio = metrics.get('profitable_ratio', 'N/A')
                    logger.info(f"  Horizon {horizon}: MAE={mae:.4f}, Correlation={corr:.4f}, Profit Ratio={profit_ratio:.4f}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load production model: {e}")
            return False

    def predict(self, features_df: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Generate production model prediction from features."""
        try:
            if self.ensemble is None:
                logger.error("Production model not loaded")
                return None
            
            # Get the most recent row of features
            if features_df.empty:
                logger.warning("No features provided for prediction")
                return None
            
            recent_features = features_df.tail(1)
            
            # Prepare feature vector
            feature_vector = self._prepare_feature_vector(recent_features)
            
            if feature_vector is None:
                logger.warning("Could not prepare feature vector")
                return None
            
            # Get predictions from all horizon models
            horizon_predictions = {}
            horizon_confidences = {}
            
            for horizon in self.horizons:
                if horizon in self.ensemble['models']:
                    model = self.ensemble['models'][horizon]
                    
                    try:
                        pred = model.predict(feature_vector.reshape(1, -1))[0]
                        horizon_predictions[horizon] = pred
                        
                        # Calculate confidence based on prediction magnitude and model performance
                        if self.model_metrics and 'validation_metrics' in self.model_metrics:
                            horizon_str = str(horizon)
                            if horizon_str in self.model_metrics['validation_metrics']:
                                mae = self.model_metrics['validation_metrics'][horizon_str]['mae']
                                correlation = self.model_metrics['validation_metrics'][horizon_str]['correlation']
                                
                                # Confidence based on prediction strength and model quality
                                pred_confidence = min(abs(pred) / 2.0, 1.0)  # Scale prediction magnitude
                                model_confidence = correlation  # Use correlation as model confidence
                                confidence = (pred_confidence * model_confidence) * 0.8  # Conservative scaling
                                
                                horizon_confidences[horizon] = max(0.1, min(0.9, confidence))
                            else:
                                horizon_confidences[horizon] = 0.5
                        else:
                            horizon_confidences[horizon] = 0.5
                            
                    except Exception as e:
                        logger.warning(f"Prediction failed for horizon {horizon}: {e}")
                        continue
            
            if not horizon_predictions:
                logger.warning("No valid predictions from any horizon")
                return None
            
            # Calculate ensemble prediction
            ensemble_prediction = self._calculate_ensemble_prediction(horizon_predictions)
            ensemble_confidence = self._calculate_ensemble_confidence(horizon_confidences)
            
            # Generate trading signal
            signal, signal_strength = self._generate_trading_signal(
                ensemble_prediction, ensemble_confidence, horizon_predictions
            )
            
            # Get current price for context
            current_price = None
            if 'mid_price' in recent_features.columns:
                current_price = recent_features['mid_price'].iloc[-1]
            elif 'microprice' in recent_features.columns:
                current_price = recent_features['microprice'].iloc[-1]
            
            prediction_result = {
                'signal': signal,
                'confidence': ensemble_confidence,
                'signal_strength': signal_strength,
                'prediction_value': ensemble_prediction,
                'horizon_predictions': horizon_predictions,
                'horizon_confidences': horizon_confidences,
                'ensemble_weights': self.ensemble_weights,
                'volatility_regime': self._assess_volatility_regime(recent_features),
                'timestamp': datetime.now(),
                'horizons': self.horizons,
                'source': 'production_model',
                'current_price': current_price,
                'model_type': 'sklearn_ensemble'
            }
            
            logger.info(f"Production prediction: Signal={signal}, Confidence={ensemble_confidence:.3f}, "
                       f"Ensemble={ensemble_prediction:.4f}")
            
            return prediction_result
            
        except Exception as e:
            logger.error(f"Error in production model prediction: {e}")
            return None

    def _prepare_feature_vector(self, features_df: pd.DataFrame) -> Optional[np.ndarray]:
        """Prepare feature vector for model prediction."""
        try:
            # Check if all required features are available
            missing_features = [f for f in self.feature_names if f not in features_df.columns]
            if missing_features:
                logger.warning(f"Missing features for prediction: {missing_features}")
                return None
            
            # Extract feature values in correct order
            feature_values = []
            for feature_name in self.feature_names:
                value = features_df[feature_name].iloc[-1]
                
                # Handle NaN values
                if pd.isna(value):
                    value = 0.0
                
                feature_values.append(value)
            
            feature_vector = np.array(feature_values, dtype=np.float32)
            
            # Validate feature vector
            if np.any(np.isnan(feature_vector)) or np.any(np.isinf(feature_vector)):
                logger.warning("Invalid values in feature vector")
                feature_vector = np.nan_to_num(feature_vector, nan=0.0, posinf=1e6, neginf=-1e6)
            
            return feature_vector
            
        except Exception as e:
            logger.error(f"Error preparing feature vector: {e}")
            return None

    def _calculate_ensemble_prediction(self, horizon_predictions: Dict[int, float]) -> float:
        """Calculate weighted ensemble prediction."""
        try:
            weighted_sum = 0.0
            total_weight = 0.0
            
            for horizon, prediction in horizon_predictions.items():
                weight = self.ensemble_weights.get(str(horizon), 0.25)  # Default equal weight
                weighted_sum += prediction * weight
                total_weight += weight
            
            if total_weight > 0:
                return weighted_sum / total_weight
            else:
                return 0.0
                
        except Exception as e:
            logger.error(f"Error calculating ensemble prediction: {e}")
            return 0.0

    def _calculate_ensemble_confidence(self, horizon_confidences: Dict[int, float]) -> float:
        """Calculate ensemble confidence."""
        try:
            if not horizon_confidences:
                return 0.0
            
            # Use weighted average of confidences
            weighted_sum = 0.0
            total_weight = 0.0
            
            for horizon, confidence in horizon_confidences.items():
                weight = self.ensemble_weights.get(str(horizon), 0.25)
                weighted_sum += confidence * weight
                total_weight += weight
            
            if total_weight > 0:
                base_confidence = weighted_sum / total_weight
                
                # Boost confidence if multiple horizons agree
                agreement_boost = min(len(horizon_confidences) / len(self.horizons), 1.0) * 0.1
                
                return min(0.9, base_confidence + agreement_boost)
            else:
                return 0.0
                
        except Exception as e:
            logger.error(f"Error calculating ensemble confidence: {e}")
            return 0.0

    def _generate_trading_signal(self, ensemble_prediction: float, confidence: float, 
                               horizon_predictions: Dict[int, float]) -> tuple:
        """Generate trading signal from ensemble prediction."""
        try:
            # Determine signal direction
            if abs(ensemble_prediction) < self.signal_threshold:
                signal = 0  # No signal
                signal_strength = 0.0
            elif ensemble_prediction > 0:
                signal = 1  # Buy signal
                signal_strength = min(ensemble_prediction / 5.0, 1.0)  # Scale to 0-1
            else:
                signal = -1  # Sell signal
                signal_strength = min(abs(ensemble_prediction) / 5.0, 1.0)  # Scale to 0-1
            
            # Check for horizon agreement
            positive_horizons = sum(1 for pred in horizon_predictions.values() if pred > 0)
            negative_horizons = sum(1 for pred in horizon_predictions.values() if pred < 0)
            total_horizons = len(horizon_predictions)
            
            # Reduce signal strength if horizons disagree
            if total_horizons > 1:
                agreement_ratio = max(positive_horizons, negative_horizons) / total_horizons
                if agreement_ratio < 0.6:  # Less than 60% agreement
                    signal_strength *= 0.5
                    
                    if agreement_ratio < 0.5:  # Less than 50% agreement
                        signal = 0  # Cancel signal
                        signal_strength = 0.0
            
            return signal, signal_strength
            
        except Exception as e:
            logger.error(f"Error generating trading signal: {e}")
            return 0, 0.0

    def _assess_volatility_regime(self, features_df: pd.DataFrame) -> str:
        """Assess current volatility regime."""
        try:
            if 'l2_volatility_10' in features_df.columns:
                vol_10 = features_df['l2_volatility_10'].iloc[-1]
                
                if pd.isna(vol_10):
                    return 'normal'
                
                if vol_10 < 0.01:
                    return 'low'
                elif vol_10 > 0.03:
                    return 'high'
                else:
                    return 'normal'
            else:
                return 'normal'
                
        except Exception as e:
            logger.error(f"Error assessing volatility regime: {e}")
            return 'normal'

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        if self.ensemble is None:
            return {'status': 'not_loaded'}
        
        return {
            'status': 'loaded',
            'horizons': self.horizons,
            'features': self.feature_names,
            'weights': self.ensemble_weights,
            'metrics': self.model_metrics,
            'ensemble_path': self.ensemble_path
        }

    def is_model_available(self) -> bool:
        """Check if production model is available and loaded."""
        return self.ensemble is not None