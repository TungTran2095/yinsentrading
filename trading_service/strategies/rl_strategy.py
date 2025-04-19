"""
RL-based strategy for trading
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
    RL_SERVICE_URL, RL_SERVICE_API_PREFIX,
    STOP_LOSS_PERCENTAGE, TAKE_PROFIT_PERCENTAGE
)
from .base_strategy import BaseStrategy

logger = logging.getLogger(__name__)

class RLStrategy(BaseStrategy):
    """
    Trading strategy based on reinforcement learning agents
    """
    
    def __init__(self, symbol: str, timeframe: str, 
                 agent_id: str,
                 confidence_threshold: float = 0.5,
                 stop_loss_pct: float = STOP_LOSS_PERCENTAGE,
                 take_profit_pct: float = TAKE_PROFIT_PERCENTAGE):
        """
        Initialize the RL strategy
        
        Args:
            symbol: Trading pair symbol (e.g., "BTC/USDT")
            timeframe: Timeframe for data (e.g., "1h", "1d")
            agent_id: ID of the RL agent to use
            confidence_threshold: Minimum confidence threshold for executing trades
            stop_loss_pct: Stop loss percentage
            take_profit_pct: Take profit percentage
        """
        super().__init__(
            name=f"rl_{agent_id}",
            symbol=symbol,
            timeframe=timeframe,
            stop_loss_pct=stop_loss_pct,
            take_profit_pct=take_profit_pct
        )
        
        self.agent_id = agent_id
        self.confidence_threshold = confidence_threshold
        
        logger.info(f"Initialized RLStrategy for {symbol} with agent_id {agent_id}")
    
    def generate_signal(self, data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Generate trading signal based on RL agent predictions
        
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
            
            # Get prediction from RL Service
            prediction = self._get_prediction(data)
            
            # Default signal is hold
            action = "hold"
            size = 0
            confidence = prediction.get("confidence", 0)
            
            # Check if we should close existing position based on stop loss/take profit
            if self.current_position != 0 and self.should_close_position(current_price):
                action = "close"
                logger.info(f"Closing position at {current_price} due to stop loss/take profit")
            
            # Generate signal based on prediction
            elif confidence >= self.confidence_threshold:
                action = prediction["action_type"]
                size = prediction["position_size"]
                logger.info(f"{action.capitalize()} signal generated with confidence {confidence}")
            
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
                "metadata": {
                    "agent_id": self.agent_id,
                    "confidence_threshold": self.confidence_threshold,
                    "rl_action": prediction.get("action", -1),
                    "rl_info": prediction.get("info", {})
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
                "error": str(e)
            }
    
    def _get_prediction(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Get prediction from RL Service
        
        Args:
            data: DataFrame with market data
            
        Returns:
            Dictionary with prediction details
        """
        try:
            # Build URL
            url = f"{RL_SERVICE_URL}{RL_SERVICE_API_PREFIX}/predict"
            
            # Create environment config
            environment = {
                "symbol": self.symbol,
                "timeframe": self.timeframe,
                "initial_balance": 10000.0,
                "max_steps": 1000,
                "window_size": 30,
                "commission": 0.001,
                "slippage": 0.001,
                "use_ensemble": False
            }
            
            # Build request data
            request_data = {
                "agent_id": self.agent_id,
                "environment": environment,
                "deterministic": True
            }
            
            # Make request
            response = requests.post(url, json=request_data)
            response.raise_for_status()
            
            # Parse response
            prediction = response.json()
            
            logger.info(f"Got prediction for {self.symbol}: {prediction['action_type']} with size {prediction['position_size']}")
            
            return prediction
        
        except Exception as e:
            logger.error(f"Error getting prediction: {e}")
            # Return hold prediction on error
            return {
                "action": -1,
                "action_type": "hold",
                "position_size": 0,
                "confidence": 0,
                "error": str(e)
            }
