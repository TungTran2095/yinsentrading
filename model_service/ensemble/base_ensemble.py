"""
Base ensemble class for combining multiple models
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
from config import MODEL_SAVE_PATH, ENSEMBLE_METHODS
from models import BaseModel, RandomForestModel, XGBoostModel, LSTMModel, TransformerModel

logger = logging.getLogger(__name__)

class BaseEnsemble(ABC):
    """
    Abstract base class for all ensemble methods
    """
    
    def __init__(self, name: str, symbol: str, timeframe: str, prediction_horizon: int = 1):
        """
        Initialize the ensemble
        
        Args:
            name: Ensemble name
            symbol: Trading pair symbol (e.g., "BTC/USDT")
            timeframe: Timeframe for data (e.g., "1h", "1d")
            prediction_horizon: Number of time periods to predict into the future
        """
        self.name = name
        self.symbol = symbol
        self.timeframe = timeframe
        self.prediction_horizon = prediction_horizon
        self.models = []
        self.is_trained = False
        
        # Create model save directory if it doesn't exist
        os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
        
        logger.info(f"Initialized {self.__class__.__name__} for {symbol} with timeframe {timeframe}")
    
    @abstractmethod
    def add_model(self, model: BaseModel) -> None:
        """
        Add a model to the ensemble
        
        Args:
            model: Model to add
        """
        pass
    
    @abstractmethod
    def train(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Train the ensemble
        
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
        Evaluate the ensemble
        
        Args:
            df: DataFrame with market data and technical indicators
            
        Returns:
            Dictionary with evaluation metrics
        """
        pass
    
    def save(self) -> str:
        """
        Save the ensemble to disk
        
        Returns:
            Path to saved ensemble
        """
        if not self.is_trained:
            raise ValueError("Ensemble must be trained before saving")
        
        # Create filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.name}_{self.symbol.replace('/', '_')}_{self.timeframe}_{timestamp}.joblib"
        filepath = os.path.join(MODEL_SAVE_PATH, filename)
        
        # Save ensemble metadata
        ensemble_data = {
            "name": self.name,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "prediction_horizon": self.prediction_horizon,
            "is_trained": self.is_trained,
            "class_name": self.__class__.__name__,
            "timestamp": timestamp,
            "model_paths": []
        }
        
        # Save each model
        for i, model in enumerate(self.models):
            model_path = model.save()
            ensemble_data["model_paths"].append(model_path)
        
        # Save ensemble data
        joblib.dump(ensemble_data, filepath)
        logger.info(f"Saved ensemble to {filepath}")
        
        return filepath
    
    def load(self, filepath: str) -> None:
        """
        Load the ensemble from disk
        
        Args:
            filepath: Path to saved ensemble
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Ensemble file not found: {filepath}")
        
        # Load ensemble data
        ensemble_data = joblib.load(filepath)
        
        # Check if ensemble class matches
        if ensemble_data["class_name"] != self.__class__.__name__:
            logger.warning(f"Loading ensemble of class {ensemble_data['class_name']} into {self.__class__.__name__}")
        
        # Load ensemble attributes
        self.name = ensemble_data["name"]
        self.symbol = ensemble_data["symbol"]
        self.timeframe = ensemble_data["timeframe"]
        self.prediction_horizon = ensemble_data["prediction_horizon"]
        self.is_trained = ensemble_data["is_trained"]
        
        # Load models
        self.models = []
        for model_path in ensemble_data["model_paths"]:
            # Determine model type from filename
            if "random_forest" in model_path:
                model = RandomForestModel(self.symbol, self.timeframe, self.prediction_horizon)
            elif "xgboost" in model_path:
                model = XGBoostModel(self.symbol, self.timeframe, self.prediction_horizon)
            elif "lstm" in model_path:
                model = LSTMModel(self.symbol, self.timeframe, self.prediction_horizon)
            elif "transformer" in model_path:
                model = TransformerModel(self.symbol, self.timeframe, self.prediction_horizon)
            else:
                logger.warning(f"Unknown model type: {model_path}")
                continue
            
            # Load model
            model.load(model_path)
            self.models.append(model)
        
        logger.info(f"Loaded ensemble from {filepath} with {len(self.models)} models")
