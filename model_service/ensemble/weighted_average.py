"""
Weighted Average Ensemble for combining multiple models
"""
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
import numpy as np
import logging
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import sys
sys.path.append('..')
from models import BaseModel
from .base_ensemble import BaseEnsemble

logger = logging.getLogger(__name__)

class WeightedAverageEnsemble(BaseEnsemble):
    """
    Weighted Average Ensemble for combining multiple models
    """
    
    def __init__(self, symbol: str, timeframe: str, prediction_horizon: int = 1):
        """
        Initialize the Weighted Average Ensemble
        
        Args:
            symbol: Trading pair symbol (e.g., "BTC/USDT")
            timeframe: Timeframe for data (e.g., "1h", "1d")
            prediction_horizon: Number of time periods to predict into the future
        """
        super().__init__("weighted_avg", symbol, timeframe, prediction_horizon)
        self.weights = []
        
        logger.info(f"Initialized WeightedAverageEnsemble for {symbol} with timeframe {timeframe}")
    
    def add_model(self, model: BaseModel) -> None:
        """
        Add a model to the ensemble
        
        Args:
            model: Model to add
        """
        if not model.is_trained:
            raise ValueError("Model must be trained before adding to ensemble")
        
        self.models.append(model)
        # Initialize with equal weights
        self.weights = [1.0 / len(self.models)] * len(self.models)
        
        logger.info(f"Added {model.__class__.__name__} to WeightedAverageEnsemble, total models: {len(self.models)}")
    
    def train(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Train the ensemble by optimizing weights
        
        Args:
            df: DataFrame with market data and technical indicators
            
        Returns:
            Dictionary with training metrics
        """
        logger.info(f"Training WeightedAverageEnsemble for {self.symbol} with timeframe {self.timeframe}")
        
        if len(self.models) == 0:
            raise ValueError("No models in ensemble")
        
        # Get predictions from each model
        predictions = []
        for model in self.models:
            pred_df = model.predict(df)
            predictions.append(pred_df["prediction"].values)
        
        # Create target
        target_col = "close"
        target_shift = self.prediction_horizon
        target = df[target_col].shift(-target_shift).dropna()
        
        # Align predictions with target
        aligned_predictions = []
        for pred in predictions:
            # Trim predictions to match target length
            aligned_pred = pred[:len(target)]
            aligned_predictions.append(aligned_pred)
        
        # Optimize weights using grid search
        best_weights = self._optimize_weights(aligned_predictions, target.values)
        self.weights = best_weights
        
        # Make ensemble predictions
        ensemble_pred = np.zeros_like(target.values)
        for i, pred in enumerate(aligned_predictions):
            ensemble_pred += self.weights[i] * pred
        
        # Calculate metrics
        mse = mean_squared_error(target.values, ensemble_pred)
        mae = mean_absolute_error(target.values, ensemble_pred)
        r2 = r2_score(target.values, ensemble_pred)
        
        # Calculate directional accuracy
        direction_correct = np.sum(np.sign(target.values[1:] - target.values[:-1]) == 
                                  np.sign(ensemble_pred[1:] - ensemble_pred[:-1]))
        direction_accuracy = direction_correct / (len(target.values) - 1)
        
        # Set trained flag
        self.is_trained = True
        
        # Return metrics
        metrics = {
            "mse": mse,
            "mae": mae,
            "r2": r2,
            "direction_accuracy": direction_accuracy,
            "weights": dict(zip([model.__class__.__name__ for model in self.models], self.weights))
        }
        
        logger.info(f"Trained WeightedAverageEnsemble with metrics: MSE={mse:.6f}, MAE={mae:.6f}, R²={r2:.6f}, Direction Accuracy={direction_accuracy:.6f}")
        logger.info(f"Optimized weights: {metrics['weights']}")
        
        return metrics
    
    def _optimize_weights(self, predictions: List[np.ndarray], target: np.ndarray) -> List[float]:
        """
        Optimize weights for ensemble
        
        Args:
            predictions: List of predictions from each model
            target: Target values
            
        Returns:
            Optimized weights
        """
        # Simple grid search for weights
        best_weights = [1.0 / len(predictions)] * len(predictions)
        best_mse = float('inf')
        
        # Generate weight combinations
        weight_options = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        
        # For small number of models, try all combinations
        if len(predictions) <= 3:
            from itertools import product
            weight_combinations = list(product(weight_options, repeat=len(predictions)))
        else:
            # For more models, use a simpler approach
            # Start with equal weights and perturb
            base_weight = 1.0 / len(predictions)
            weight_combinations = []
            for i in range(len(predictions)):
                for w in weight_options:
                    weights = [base_weight] * len(predictions)
                    weights[i] = w
                    # Normalize
                    weights = [w / sum(weights) for w in weights]
                    weight_combinations.append(weights)
        
        # Evaluate each combination
        for weights in weight_combinations:
            # Normalize weights to sum to 1
            weights = np.array(weights) / sum(weights)
            
            # Make ensemble prediction
            ensemble_pred = np.zeros_like(target)
            for i, pred in enumerate(predictions):
                ensemble_pred += weights[i] * pred
            
            # Calculate MSE
            mse = mean_squared_error(target, ensemble_pred)
            
            # Update best weights if better
            if mse < best_mse:
                best_mse = mse
                best_weights = weights
        
        return best_weights.tolist()
    
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
        
        logger.info(f"Making predictions with WeightedAverageEnsemble for {self.symbol} with timeframe {self.timeframe}")
        
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
        
        # Create ensemble predictions
        ensemble_pred = pd.DataFrame(index=common_timestamps)
        ensemble_pred["prediction"] = 0.0
        
        for i, pred_df in enumerate(model_predictions):
            # Filter to common timestamps
            filtered_pred = pred_df.loc[common_timestamps]
            # Add weighted prediction
            ensemble_pred["prediction"] += self.weights[i] * filtered_pred["prediction"]
        
        # Add metadata
        ensemble_pred["model"] = self.name
        ensemble_pred["symbol"] = self.symbol
        ensemble_pred["timeframe"] = self.timeframe
        ensemble_pred["prediction_horizon"] = self.prediction_horizon
        
        logger.info(f"Made {len(ensemble_pred)} predictions with WeightedAverageEnsemble")
        
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
        
        logger.info(f"Evaluating WeightedAverageEnsemble for {self.symbol} with timeframe {self.timeframe}")
        
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
        
        logger.info(f"Evaluated WeightedAverageEnsemble with metrics: MSE={mse:.6f}, MAE={mae:.6f}, R²={r2:.6f}, Direction Accuracy={direction_accuracy:.6f}, Sharpe Ratio={sharpe_ratio:.6f}")
        
        return metrics
