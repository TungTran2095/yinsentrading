"""
XGBoost model for market prediction
"""
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Any, Tuple
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import sys
sys.path.append('..')
from config import TRAIN_TEST_SPLIT_RATIO, RANDOM_SEED
from .base_model import BaseModel

logger = logging.getLogger(__name__)

class XGBoostModel(BaseModel):
    """
    XGBoost model for market prediction
    """
    
    def __init__(self, symbol: str, timeframe: str, prediction_horizon: int = 1, 
                 n_estimators: int = 100, max_depth: int = 6, learning_rate: float = 0.1):
        """
        Initialize the XGBoost model
        
        Args:
            symbol: Trading pair symbol (e.g., "BTC/USDT")
            timeframe: Timeframe for data (e.g., "1h", "1d")
            prediction_horizon: Number of time periods to predict into the future
            n_estimators: Number of boosting rounds
            max_depth: Maximum depth of the trees
            learning_rate: Learning rate
        """
        super().__init__("xgboost", symbol, timeframe, prediction_horizon)
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        
        # Initialize model
        self.model = xgb.XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=RANDOM_SEED,
            n_jobs=-1
        )
        
        # Initialize scaler
        self.scaler = StandardScaler()
        
        logger.info(f"Initialized XGBoostModel with n_estimators={n_estimators}, max_depth={max_depth}, learning_rate={learning_rate}")
    
    def preprocess_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocess data for training or prediction
        
        Args:
            df: DataFrame with market data and technical indicators
            
        Returns:
            Tuple of (X, y) where X is features and y is target
        """
        # Create features and target
        features, target = self._create_features_target(df, self.target_name, self.prediction_horizon)
        
        # Store feature names
        self.feature_names = features.columns.tolist()
        
        # Convert to numpy arrays
        X = features.values
        y = target.values
        
        # Scale features
        if not self.is_trained:
            X = self.scaler.fit_transform(X)
        else:
            X = self.scaler.transform(X)
        
        return X, y
    
    def train(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Train the model
        
        Args:
            df: DataFrame with market data and technical indicators
            
        Returns:
            Dictionary with training metrics
        """
        logger.info(f"Training XGBoostModel for {self.symbol} with timeframe {self.timeframe}")
        
        # Preprocess data
        X, y = self.preprocess_data(df)
        
        # Split data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=1-TRAIN_TEST_SPLIT_RATIO, random_state=RANDOM_SEED
        )
        
        # Train model
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            early_stopping_rounds=10,
            verbose=False
        )
        
        # Make predictions on test set
        y_pred = self.model.predict(X_test)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Calculate directional accuracy
        direction_correct = np.sum(np.sign(y_test[1:] - y_test[:-1]) == np.sign(y_pred[1:] - y_pred[:-1]))
        direction_accuracy = direction_correct / (len(y_test) - 1)
        
        # Set trained flag
        self.is_trained = True
        
        # Get feature importance
        importance = self.model.feature_importances_
        feature_importance = dict(zip(self.feature_names, importance))
        
        # Return metrics
        metrics = {
            "mse": mse,
            "mae": mae,
            "r2": r2,
            "direction_accuracy": direction_accuracy,
            "feature_importance": feature_importance
        }
        
        logger.info(f"Trained XGBoostModel with metrics: MSE={mse:.6f}, MAE={mae:.6f}, R²={r2:.6f}, Direction Accuracy={direction_accuracy:.6f}")
        
        return metrics
    
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Make predictions
        
        Args:
            df: DataFrame with market data and technical indicators
            
        Returns:
            DataFrame with predictions
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        logger.info(f"Making predictions with XGBoostModel for {self.symbol} with timeframe {self.timeframe}")
        
        # Make a copy to avoid modifying the original
        df_copy = df.copy()
        
        # Preprocess data
        X, _ = self.preprocess_data(df_copy)
        
        # Make predictions
        predictions = self.model.predict(X)
        
        # Create DataFrame with predictions
        df_pred = pd.DataFrame({
            "timestamp": df_copy.index if isinstance(df_copy.index, pd.DatetimeIndex) else df_copy.index,
            "prediction": predictions,
            "model": self.name,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "prediction_horizon": self.prediction_horizon
        })
        
        # Set timestamp as index if it's not already
        if not isinstance(df_pred.index, pd.DatetimeIndex) and "timestamp" in df_pred.columns:
            df_pred = df_pred.set_index("timestamp")
        
        logger.info(f"Made {len(df_pred)} predictions with XGBoostModel")
        
        return df_pred
    
    def evaluate(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Evaluate the model
        
        Args:
            df: DataFrame with market data and technical indicators
            
        Returns:
            Dictionary with evaluation metrics
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")
        
        logger.info(f"Evaluating XGBoostModel for {self.symbol} with timeframe {self.timeframe}")
        
        # Preprocess data
        X, y = self.preprocess_data(df)
        
        # Make predictions
        y_pred = self.model.predict(X)
        
        # Calculate metrics
        mse = mean_squared_error(y, y_pred)
        mae = mean_absolute_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        
        # Calculate directional accuracy
        direction_correct = np.sum(np.sign(y[1:] - y[:-1]) == np.sign(y_pred[1:] - y_pred[:-1]))
        direction_accuracy = direction_correct / (len(y) - 1)
        
        # Calculate sharpe ratio (simplified)
        returns_actual = np.diff(y) / y[:-1]
        returns_pred = np.diff(y_pred) / y_pred[:-1]
        sharpe_ratio = np.mean(returns_pred) / np.std(returns_pred) if np.std(returns_pred) > 0 else 0
        
        # Return metrics
        metrics = {
            "mse": mse,
            "mae": mae,
            "r2": r2,
            "direction_accuracy": direction_accuracy,
            "sharpe_ratio": sharpe_ratio
        }
        
        logger.info(f"Evaluated XGBoostModel with metrics: MSE={mse:.6f}, MAE={mae:.6f}, R²={r2:.6f}, Direction Accuracy={direction_accuracy:.6f}, Sharpe Ratio={sharpe_ratio:.6f}")
        
        return metrics
