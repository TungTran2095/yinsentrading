"""
Stacking Ensemble for combining multiple models
"""
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
import numpy as np
import logging
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import sys
sys.path.append('..')
from config import TRAIN_TEST_SPLIT_RATIO, RANDOM_SEED
from models import BaseModel
from .base_ensemble import BaseEnsemble

logger = logging.getLogger(__name__)

class StackingEnsemble(BaseEnsemble):
    """
    Stacking Ensemble for combining multiple models
    """
    
    def __init__(self, symbol: str, timeframe: str, prediction_horizon: int = 1, alpha: float = 1.0):
        """
        Initialize the Stacking Ensemble
        
        Args:
            symbol: Trading pair symbol (e.g., "BTC/USDT")
            timeframe: Timeframe for data (e.g., "1h", "1d")
            prediction_horizon: Number of time periods to predict into the future
            alpha: Regularization strength for the meta-model
        """
        super().__init__("stacking", symbol, timeframe, prediction_horizon)
        self.alpha = alpha
        self.meta_model = Ridge(alpha=alpha, random_state=RANDOM_SEED)
        
        logger.info(f"Initialized StackingEnsemble for {symbol} with timeframe {timeframe}")
    
    def add_model(self, model: BaseModel) -> None:
        """
        Add a model to the ensemble
        
        Args:
            model: Model to add
        """
        if not model.is_trained:
            raise ValueError("Model must be trained before adding to ensemble")
        
        self.models.append(model)
        
        logger.info(f"Added {model.__class__.__name__} to StackingEnsemble, total models: {len(self.models)}")
    
    def train(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Train the ensemble by training a meta-model on the predictions of base models
        
        Args:
            df: DataFrame with market data and technical indicators
            
        Returns:
            Dictionary with training metrics
        """
        logger.info(f"Training StackingEnsemble for {self.symbol} with timeframe {self.timeframe}")
        
        if len(self.models) == 0:
            raise ValueError("No models in ensemble")
        
        # Split data for training and validation
        train_df, val_df = train_test_split(df, test_size=1-TRAIN_TEST_SPLIT_RATIO, random_state=RANDOM_SEED)
        
        # Get predictions from each model on validation data
        X_meta = []
        for model in self.models:
            pred_df = model.predict(val_df)
            X_meta.append(pred_df["prediction"].values)
        
        # Stack predictions as features for meta-model
        X_meta = np.column_stack(X_meta)
        
        # Create target
        target_col = "close"
        target_shift = self.prediction_horizon
        y_meta = val_df[target_col].shift(-target_shift).dropna()
        
        # Align features with target
        X_meta = X_meta[:len(y_meta)]
        
        # Train meta-model
        self.meta_model.fit(X_meta, y_meta)
        
        # Make predictions with meta-model
        meta_pred = self.meta_model.predict(X_meta)
        
        # Calculate metrics
        mse = mean_squared_error(y_meta, meta_pred)
        mae = mean_absolute_error(y_meta, meta_pred)
        r2 = r2_score(y_meta, meta_pred)
        
        # Calculate directional accuracy
        direction_correct = np.sum(np.sign(y_meta.values[1:] - y_meta.values[:-1]) == 
                                  np.sign(meta_pred[1:] - meta_pred[:-1]))
        direction_accuracy = direction_correct / (len(y_meta) - 1)
        
        # Set trained flag
        self.is_trained = True
        
        # Return metrics
        metrics = {
            "mse": mse,
            "mae": mae,
            "r2": r2,
            "direction_accuracy": direction_accuracy,
            "meta_model_coefficients": dict(zip([model.__class__.__name__ for model in self.models], 
                                              self.meta_model.coef_)),
            "meta_model_intercept": float(self.meta_model.intercept_)
        }
        
        logger.info(f"Trained StackingEnsemble with metrics: MSE={mse:.6f}, MAE={mae:.6f}, R²={r2:.6f}, Direction Accuracy={direction_accuracy:.6f}")
        logger.info(f"Meta-model coefficients: {metrics['meta_model_coefficients']}")
        
        return metrics
    
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Make predictions with the ensemble
        
        Args:
            df: DataFrame with market data and technical indicators
            
        Returns:
            DataFrame with predictions
        """
        if not self.is_trained:
            raise ValueError("Ensemble must be trained before making predictions")
        
        if len(self.models) == 0:
            raise ValueError("No models in ensemble")
        
        logger.info(f"Making predictions with StackingEnsemble for {self.symbol} with timeframe {self.timeframe}")
        
        # Get predictions from each model
        model_predictions = []
        for model in self.models:
            pred_df = model.predict(df)
            model_predictions.append(pred_df)
        
        # Find common timestamps
        common_timestamps = set(model_predictions[0].index)
        for pred_df in model_predictions[1:]:
            common_timestamps = common_timestamps.intersection(set(pred_df.index))
        
        common_timestamps = sorted(list(common_timestamps))
        
        # Create feature matrix for meta-model
        X_meta = []
        for pred_df in model_predictions:
            # Filter to common timestamps
            filtered_pred = pred_df.loc[common_timestamps]
            X_meta.append(filtered_pred["prediction"].values)
        
        X_meta = np.column_stack(X_meta)
        
        # Make predictions with meta-model
        meta_pred = self.meta_model.predict(X_meta)
        
        # Create DataFrame with predictions
        ensemble_pred = pd.DataFrame(index=common_timestamps)
        ensemble_pred["prediction"] = meta_pred
        ensemble_pred["model"] = self.name
        ensemble_pred["symbol"] = self.symbol
        ensemble_pred["timeframe"] = self.timeframe
        ensemble_pred["prediction_horizon"] = self.prediction_horizon
        
        logger.info(f"Made {len(ensemble_pred)} predictions with StackingEnsemble")
        
        return ensemble_pred
    
    def evaluate(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Evaluate the ensemble
        
        Args:
            df: DataFrame with market data and technical indicators
            
        Returns:
            Dictionary with evaluation metrics
        """
        if not self.is_trained:
            raise ValueError("Ensemble must be trained before evaluation")
        
        logger.info(f"Evaluating StackingEnsemble for {self.symbol} with timeframe {self.timeframe}")
        
        # Make predictions
        pred_df = self.predict(df)
        
        # Create target
        target_col = "close"
        target_shift = self.prediction_horizon
        target = df[target_col].shift(-target_shift).dropna()
        
        # Align predictions with target
        common_timestamps = set(pred_df.index).intersection(set(target.index))
        common_timestamps = sorted(list(common_timestamps))
        
        if len(common_timestamps) == 0:
            raise ValueError("No common timestamps between predictions and target")
        
        # Filter to common timestamps
        pred = pred_df.loc[common_timestamps, "prediction"].values
        target = target.loc[common_timestamps].values
        
        # Calculate metrics
        mse = mean_squared_error(target, pred)
        mae = mean_absolute_error(target, pred)
        r2 = r2_score(target, pred)
        
        # Calculate directional accuracy
        direction_correct = np.sum(np.sign(target[1:] - target[:-1]) == np.sign(pred[1:] - pred[:-1]))
        direction_accuracy = direction_correct / (len(target) - 1)
        
        # Calculate sharpe ratio (simplified)
        returns_actual = np.diff(target) / target[:-1]
        returns_pred = np.diff(pred) / pred[:-1]
        sharpe_ratio = np.mean(returns_pred) / np.std(returns_pred) if np.std(returns_pred) > 0 else 0
        
        # Return metrics
        metrics = {
            "mse": mse,
            "mae": mae,
            "r2": r2,
            "direction_accuracy": direction_accuracy,
            "sharpe_ratio": sharpe_ratio
        }
        
        logger.info(f"Evaluated StackingEnsemble with metrics: MSE={mse:.6f}, MAE={mae:.6f}, R²={r2:.6f}, Direction Accuracy={direction_accuracy:.6f}, Sharpe Ratio={sharpe_ratio:.6f}")
        
        return metrics
