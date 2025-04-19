"""
Base model class for machine learning models
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
import numpy as np
import logging
import joblib
import os
from datetime import datetime
import sys
sys.path.append('..')
from config import MODEL_SAVE_PATH

logger = logging.getLogger(__name__)

class BaseModel(ABC):
    """
    Abstract base class for all machine learning models
    """
    
    def __init__(self, name: str, symbol: str, timeframe: str, prediction_horizon: int = 1):
        """
        Initialize the model
        
        Args:
            name: Model name
            symbol: Trading pair symbol (e.g., "BTC/USDT")
            timeframe: Timeframe for data (e.g., "1h", "1d")
            prediction_horizon: Number of time periods to predict into the future
        """
        self.name = name
        self.symbol = symbol
        self.timeframe = timeframe
        self.prediction_horizon = prediction_horizon
        self.model = None
        self.is_trained = False
        self.feature_names = []
        self.target_name = "close"
        self.scaler = None
        
        # Create model save directory if it doesn't exist
        os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
        
        logger.info(f"Initialized {self.__class__.__name__} for {symbol} with timeframe {timeframe}")
    
    @abstractmethod
    def preprocess_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocess data for training or prediction
        
        Args:
            df: DataFrame with market data and technical indicators
            
        Returns:
            Tuple of (X, y) where X is features and y is target
        """
        pass
    
    @abstractmethod
    def train(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Train the model
        
        Args:
            df: DataFrame with market data and technical indicators
            
        Returns:
            Dictionary with training metrics
        """
        pass
    
    @abstractmethod
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Make predictions
        
        Args:
            df: DataFrame with market data and technical indicators
            
        Returns:
            DataFrame with predictions
        """
        pass
    
    @abstractmethod
    def evaluate(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Evaluate the model
        
        Args:
            df: DataFrame with market data and technical indicators
            
        Returns:
            Dictionary with evaluation metrics
        """
        pass
    
    def save(self) -> str:
        """
        Save the model to disk
        
        Returns:
            Path to saved model
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        # Create filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.name}_{self.symbol.replace('/', '_')}_{self.timeframe}_{timestamp}.joblib"
        filepath = os.path.join(MODEL_SAVE_PATH, filename)
        
        # Save model and metadata
        model_data = {
            "model": self.model,
            "name": self.name,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "prediction_horizon": self.prediction_horizon,
            "feature_names": self.feature_names,
            "target_name": self.target_name,
            "is_trained": self.is_trained,
            "scaler": self.scaler,
            "class_name": self.__class__.__name__,
            "timestamp": timestamp
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"Saved model to {filepath}")
        
        return filepath
    
    def load(self, filepath: str) -> None:
        """
        Load the model from disk
        
        Args:
            filepath: Path to saved model
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        # Load model and metadata
        model_data = joblib.load(filepath)
        
        # Check if model class matches
        if model_data["class_name"] != self.__class__.__name__:
            logger.warning(f"Loading model of class {model_data['class_name']} into {self.__class__.__name__}")
        
        # Load model attributes
        self.model = model_data["model"]
        self.name = model_data["name"]
        self.symbol = model_data["symbol"]
        self.timeframe = model_data["timeframe"]
        self.prediction_horizon = model_data["prediction_horizon"]
        self.feature_names = model_data["feature_names"]
        self.target_name = model_data["target_name"]
        self.is_trained = model_data["is_trained"]
        self.scaler = model_data["scaler"]
        
        logger.info(f"Loaded model from {filepath}")
    
    def _create_features_target(self, df: pd.DataFrame, target_column: str = "close", shift: int = 1) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Create features and target from DataFrame
        
        Args:
            df: DataFrame with market data and technical indicators
            target_column: Column to use as target
            shift: Number of periods to shift target for prediction
            
        Returns:
            Tuple of (X, y) where X is features DataFrame and y is target Series
        """
        # Make a copy to avoid modifying the original
        df = df.copy()
        
        # Create target by shifting the close price
        df[f"{target_column}_future_{shift}"] = df[target_column].shift(-shift)
        
        # Drop rows with NaN in target
        df = df.dropna(subset=[f"{target_column}_future_{shift}"])
        
        # Select features (all columns except target and timestamp)
        features = df.drop(columns=[f"{target_column}_future_{shift}"])
        if isinstance(features.index, pd.DatetimeIndex):
            # Keep the datetime index for later reference
            features_with_datetime = features.copy()
            # Reset index to get features only
            features = features.reset_index(drop=True)
        
        # Select target
        target = df[f"{target_column}_future_{shift}"]
        
        return features, target
