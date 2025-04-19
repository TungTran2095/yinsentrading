"""
Ensemble-based strategy for trading
"""
import logging
from typing import Dict, List, Optional, Any, Tuple
import requests
import pandas as pd
import numpy as np
import json
from datetime import datetime
import sys
sys.path.append('..')
from config import (
    MODEL_SERVICE_URL, MODEL_SERVICE_API_PREFIX,
    STOP_LOSS_PERCENTAGE, TAKE_PROFIT_PERCENTAGE
)
from .base_strategy import BaseStrategy

logger = logging.getLogger(__name__)

class EnsembleStrategy(BaseStrategy):
    """
    Trading strategy based on ensemble learning predictions
    """
    
    def __init__(self, symbol: str, timeframe: str, 
                 ensemble_id: str,
                 threshold_buy: float = 0.6,
                 threshold_sell: float = 0.4,
                 position_size: float = 1.0,
                 stop_loss_pct: float = STOP_LOSS_PERCENTAGE,
                 take_profit_pct: float = TAKE_PROFIT_PERCENTAGE):
        """
        Initialize the ensemble strategy
        
        Args:
            symbol: Trading pair symbol (e.g., "BTC/USDT")
            timeframe: Timeframe for data (e.g., "1h", "1d")
            ensemble_id: ID of the ensemble model to use
            threshold_buy: Threshold for buy signals (0-1)
            threshold_sell: Threshold for sell signals (0-1)
            position_size: Position size as fraction of available capital
            stop_loss_pct: Stop loss percentage
            take_profit_pct: Take profit percentage
        """
        super().__init__(
            name=f"ensemble_{ensemble_id}",
            symbol=symbol,
            timeframe=timeframe,
            stop_loss_pct=stop_loss_pct,
            take_profit_pct=take_profit_pct
        )
        
        self.ensemble_id = ensemble_id
        self.threshold_buy = threshold_buy
        self.threshold_sell = threshold_sell
        self.position_size = position_size
        
        logger.info(f"Initialized EnsembleStrategy for {symbol} with ensemble_id {ensemble_id}")
    
    def generate_signal(self, data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Generate trading signal based on ensemble predictions
        
        Args:
            data: DataFrame with market data (optional, will fetch if not provided)
            
        Returns:
            Dictionary with signal details
        """
        try:
            # Get data if not provided
            if data is None:
                data = self.get_data()
            
            # Get latest price
            current_price = data["close"].iloc[-1]
            
            # Get prediction from Model Service
            prediction = self._get_prediction()
            
            # Default signal is hold
            action = "hold"
            size = 0
            confidence = prediction["confidence"]
            
            # Check if we should close existing position based on stop loss/take profit
            if self.current_position != 0 and self.should_close_position(current_price):
                action = "close"
                logger.info(f"Closing position at {current_price} due to stop loss/take profit")
            
            # Generate signal based on prediction
            elif prediction["prediction"] >= self.threshold_buy and self.current_position <= 0:
                # Buy signal
                action = "buy"
                size = self.position_size
                logger.info(f"Buy signal generated with confidence {confidence}")
            
            elif prediction["prediction"] <= self.threshold_sell and self.current_position >= 0:
                # Sell signal
                action = "sell"
                size = self.position_size
                logger.info(f"Sell signal generated with confidence {confidence}")
            
            # Update position state
            if action != "hold":
                self.update_position(action, current_price, size)
            
            # Create signal
            signal = {
                "strategy": self.name,
                "symbol": self.symbol,
                "timeframe": self.timeframe,
                "timestamp": datetime.now().isoformat(),
                "price": current_price,
                "action": action,
                "size": size,
                "confidence": confidence,
                "prediction": prediction["prediction"],
                "metadata": {
                    "ensemble_id": self.ensemble_id,
                    "threshold_buy": self.threshold_buy,
                    "threshold_sell": self.threshold_sell
                }
            }
            
            return signal
        
        except Exception as e:
            logger.error(f"Error generating signal: {e}")
            # Return hold signal on error
            return {
                "strategy": self.name,
                "symbol": self.symbol,
                "timeframe": self.timeframe,
                "timestamp": datetime.now().isoformat(),
                "price": data["close"].iloc[-1] if data is not None else 0,
                "action": "hold",
                "size": 0,
                "confidence": 0,
                "prediction": 0.5,
                "error": str(e)
            }
    
    def _get_prediction(self) -> Dict[str, Any]:
        """
        Get prediction from Model Service
        
        Returns:
            Dictionary with prediction details
        """
        try:
            # Build URL
            url = f"{MODEL_SERVICE_URL}{MODEL_SERVICE_API_PREFIX}/predict"
            
            # Build request data
            data = {
                "symbol": self.symbol,
                "timeframe": self.timeframe,
                "ensemble_id": self.ensemble_id
            }
            
            # Make request
            response = requests.post(url, json=data)
            response.raise_for_status()
            
            # Parse response
            prediction = response.json()
            
            logger.info(f"Got prediction for {self.symbol}: {prediction['prediction']}")
            
            return prediction
        
        except Exception as e:
            logger.error(f"Error getting prediction: {e}")
            # Return neutral prediction on error
            return {
                "prediction": 0.5,
                "confidence": 0,
                "error": str(e)
            }
