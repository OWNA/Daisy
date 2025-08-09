# modelpredictor_enhanced.py
# Enhanced Model Predictor with Ensemble Predictions and Dynamic Thresholds

import os
import json
import pickle
import pandas as pd
import numpy as np
import logging
from typing import Dict, Optional, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)


class EnhancedModelPredictor:
    """
    Enhanced predictor that uses ensemble models with:
    - Multi-timeframe predictions
    - Dynamic threshold adjustment based on volatility
    - Confidence scoring
    - Signal quality metrics
    """
    
    def __init__(self, config: dict):
        """Initialize enhanced predictor."""
        self.config = config
        self.base_dir = config.get('base_dir', './trading_bot_data')
        
        safe_symbol = config.get('symbol', 'SYMBOL').replace('/', '_').replace(':', '')
        
        # Model paths
        self.ensemble_path = os.path.join(
            self.base_dir, f"enhanced_ensemble_{safe_symbol}.pkl"
        )
        self.features_path = os.path.join(
            self.base_dir, f"enhanced_features_{safe_symbol}.json"
        )
        
        # Loaded model components
        self.ensemble = None
        self.features = []
        self.horizons = []
        self.weights = {}
        self.thresholds = {}
        
        # Runtime metrics
        self.recent_volatility = None
        self.signal_history = []
        self.prediction_cache = {}
        
        logger.info("Enhanced ModelPredictor initialized")

    def load_models(self) -> bool:
        """Load ensemble models and configuration."""
        try:
            # Load ensemble
            with open(self.ensemble_path, 'rb') as f:
                self.ensemble = pickle.load(f)
            
            # Load features
            with open(self.features_path, 'r') as f:
                features_data = json.load(f)
                self.features = features_data['features']
                self.horizons = features_data['horizons']
                self.weights = features_data['weights']
            
            # Extract components
            self.models = self.ensemble['models']
            self.thresholds = self.ensemble['thresholds']
            
            logger.info(f"Loaded ensemble with {len(self.models)} models")
            logger.info(f"Horizons: {self.horizons}")
            logger.info(f"Weights: {self.weights}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            return False

    def predict_signals(
        self,
        df_features: pd.DataFrame,
        use_dynamic_threshold: bool = True,
        min_confidence: float = 0.6
    ) -> pd.DataFrame:
        """
        Generate ensemble predictions with confidence scores.
        
        Args:
            df_features: DataFrame with L2 features
            use_dynamic_threshold: Whether to adjust thresholds based on volatility
            min_confidence: Minimum confidence required for signal
            
        Returns:
            DataFrame with predictions, signals, and confidence scores
        """
        if self.ensemble is None:
            logger.error("Models not loaded")
            return None
        
        if df_features.empty:
            logger.warning("Empty features DataFrame")
            return None
        
        # Ensure we have required features
        missing_features = set(self.features) - set(df_features.columns)
        if missing_features:
            logger.error(f"Missing features: {missing_features}")
            return None
        
        X = df_features[self.features].copy()
        
        # Calculate current volatility for dynamic thresholds
        if 'l2_volatility_50' in df_features.columns:
            self.recent_volatility = df_features['l2_volatility_50'].iloc[-100:].mean()
        
        # Generate predictions for each horizon
        predictions = {}
        for horizon in self.horizons:
            if horizon in self.models:
                model = self.models[horizon]
                preds = model['model'].predict(X)
                predictions[f'pred_{horizon}'] = preds
        
        # Create results DataFrame
        result_df = pd.DataFrame(index=df_features.index)
        
        # Add individual predictions
        for col, preds in predictions.items():
            result_df[col] = preds
        
        # Calculate ensemble prediction
        ensemble_pred = np.zeros(len(result_df))
        for horizon in self.horizons:
            if f'pred_{horizon}' in result_df.columns:
                weight = self.weights.get(horizon, 0)
                ensemble_pred += weight * result_df[f'pred_{horizon}'].values
        
        result_df['prediction'] = ensemble_pred
        
        # Calculate prediction statistics
        pred_cols = [col for col in result_df.columns if col.startswith('pred_')]
        if len(pred_cols) > 1:
            result_df['pred_std'] = result_df[pred_cols].std(axis=1)
            result_df['pred_agreement'] = (
                (result_df[pred_cols] > 0).sum(axis=1) / len(pred_cols)
            )
        else:
            result_df['pred_std'] = 0
            result_df['pred_agreement'] = 1
        
        # Apply thresholds
        if use_dynamic_threshold:
            threshold = self._get_dynamic_threshold()
        else:
            # Use median of all thresholds
            all_thresholds = []
            for h_thresh in self.thresholds.values():
                all_thresholds.extend(h_thresh.values())
            threshold = np.median(all_thresholds)
        
        # Generate signals with confidence
        result_df['signal'] = 0
        result_df['confidence'] = 0
        
        # Long signals
        long_mask = result_df['prediction'] > threshold
        result_df.loc[long_mask, 'signal'] = 1
        
        # Short signals
        short_mask = result_df['prediction'] < -threshold
        result_df.loc[short_mask, 'signal'] = -1
        
        # Calculate confidence scores
        result_df['confidence'] = self._calculate_confidence(result_df, threshold)
        
        # Filter by minimum confidence
        low_conf_mask = result_df['confidence'] < min_confidence
        result_df.loc[low_conf_mask, 'signal'] = 0
        
        # Add signal quality metrics
        result_df['signal_quality'] = self._assess_signal_quality(result_df)
        
        # Store threshold used
        result_df['threshold_used'] = threshold
        
        # Update signal history
        self._update_signal_history(result_df)
        
        # Log statistics
        signal_rate = (result_df['signal'] != 0).mean()
        if signal_rate > 0:
            avg_confidence = result_df[result_df['signal'] != 0]['confidence'].mean()
            logger.info(
                f"Signal rate: {signal_rate:.2%}, "
                f"Avg confidence: {avg_confidence:.2f}, "
                f"Threshold: {threshold:.4f}"
            )
        
        return result_df

    def _get_dynamic_threshold(self) -> float:
        """Get threshold based on current market volatility."""
        if self.recent_volatility is None:
            # Use overall threshold
            all_overall = []
            for h_thresh in self.thresholds.values():
                if 'overall' in h_thresh:
                    all_overall.append(h_thresh['overall'])
            return np.median(all_overall) if all_overall else 0.01
        
        # Find appropriate volatility regime
        # This is simplified - in production, you'd want more sophisticated logic
        vol_level = 'vol_p50'  # Default to median
        
        # Get thresholds for each horizon and average
        thresholds = []
        for horizon in self.horizons:
            if horizon in self.thresholds:
                h_thresh = self.thresholds[horizon]
                if vol_level in h_thresh:
                    thresholds.append(h_thresh[vol_level])
                elif 'overall' in h_thresh:
                    thresholds.append(h_thresh['overall'])
        
        return np.median(thresholds) if thresholds else 0.01

    def _calculate_confidence(self, df: pd.DataFrame, threshold: float) -> pd.Series:
        """
        Calculate confidence scores for predictions.
        
        Factors:
        - Distance from threshold (farther = more confident)
        - Agreement among ensemble members
        - Prediction stability (low std)
        """
        confidence = pd.Series(0.5, index=df.index)
        
        # Distance factor (sigmoid-like)
        pred_abs = df['prediction'].abs()
        distance_factor = 1 / (1 + np.exp(-5 * (pred_abs / threshold - 1)))
        
        # Agreement factor
        agreement_factor = df['pred_agreement'].apply(
            lambda x: 1.0 if x > 0.8 else (0.5 if x > 0.6 else 0.0)
        )
        
        # Stability factor (lower std = higher confidence)
        if 'pred_std' in df.columns:
            std_normalized = df['pred_std'] / (pred_abs + 1e-8)
            stability_factor = 1 / (1 + std_normalized)
        else:
            stability_factor = 1.0
        
        # Combine factors
        confidence = (
            0.4 * distance_factor +
            0.4 * agreement_factor +
            0.2 * stability_factor
        )
        
        # Ensure [0, 1] range
        confidence = confidence.clip(0, 1)
        
        return confidence

    def _assess_signal_quality(self, df: pd.DataFrame) -> pd.Series:
        """
        Assess signal quality based on multiple factors.
        Returns quality score [0, 1].
        """
        quality = pd.Series(0.5, index=df.index)
        
        # High confidence is good
        quality += 0.3 * df['confidence']
        
        # Low prediction variance is good
        if 'pred_std' in df.columns:
            quality -= 0.2 * (df['pred_std'] / df['pred_std'].max()).fillna(0)
        
        # Strong agreement is good
        if 'pred_agreement' in df.columns:
            quality += 0.2 * df['pred_agreement']
        
        # Clip to [0, 1]
        quality = quality.clip(0, 1)
        
        return quality

    def _update_signal_history(self, df: pd.DataFrame):
        """Update signal history for performance tracking."""
        # Keep last 1000 signals
        new_signals = df[df['signal'] != 0][['signal', 'confidence', 'prediction']].tail(100)
        self.signal_history.extend(new_signals.to_dict('records'))
        self.signal_history = self.signal_history[-1000:]

    def get_feature_importance(self) -> pd.DataFrame:
        """Get aggregated feature importance across ensemble."""
        if not self.ensemble:
            return pd.DataFrame()
        
        # Aggregate importance across models
        importance_sum = pd.Series(0, index=self.features)
        
        for horizon, model_data in self.ensemble['models'].items():
            if 'importance' in model_data:
                imp_df = model_data['importance']
                for _, row in imp_df.iterrows():
                    if row['feature'] in importance_sum.index:
                        importance_sum[row['feature']] += row['importance'] * self.weights.get(horizon, 0)
        
        # Convert to DataFrame
        importance_df = pd.DataFrame({
            'feature': importance_sum.index,
            'importance': importance_sum.values
        }).sort_values('importance', ascending=False)
        
        # Normalize
        importance_df['importance'] = (
            importance_df['importance'] / importance_df['importance'].sum()
        )
        
        return importance_df

    def get_model_diagnostics(self) -> Dict:
        """Get diagnostic information about model performance."""
        diagnostics = {
            'loaded': self.ensemble is not None,
            'num_models': len(self.models) if self.ensemble else 0,
            'horizons': self.horizons,
            'weights': self.weights,
            'recent_volatility': self.recent_volatility,
            'signal_history_size': len(self.signal_history)
        }
        
        if self.signal_history:
            recent_signals = pd.DataFrame(self.signal_history[-100:])
            diagnostics['recent_signal_rate'] = len(recent_signals) / 100
            diagnostics['recent_avg_confidence'] = recent_signals['confidence'].mean()
            diagnostics['recent_long_ratio'] = (recent_signals['signal'] == 1).mean()
        
        return diagnostics