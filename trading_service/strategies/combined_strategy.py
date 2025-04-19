"""
Combined strategy for trading using both Ensemble Learning and Reinforcement Learning
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
    RL_SERVICE_URL, RL_SERVICE_API_PREFIX,
    STOP_LOSS_PERCENTAGE, TAKE_PROFIT_PERCENTAGE
)
from .base_strategy import BaseStrategy
from .ensemble_strategy import EnsembleStrategy
from .rl_strategy import RLStrategy

logger = logging.getLogger(__name__)

class CombinedStrategy(BaseStrategy):
    """
    Trading strategy combining ensemble learning and reinforcement learning
    """
    
    def __init__(self, symbol: str, timeframe: str, 
                 ensemble_id: str, agent_id: str,
                 ensemble_weight: float = 0.5,
                 rl_weight: float = 0.5,
                 confidence_threshold: float = 0.6,
                 position_size: float = 1.0,
                 stop_loss_pct: float = STOP_LOSS_PERCENTAGE,
                 take_profit_pct: float = TAKE_PROFIT_PERCENTAGE):
        """
        Initialize the combined strategy
        
        Args:
            symbol: Trading pair symbol (e.g., "BTC/USDT")
            timeframe: Timeframe for data (e.g., "1h", "1d")
            ensemble_id: ID of the ensemble model to use
            agent_id: ID of the RL agent to use
            ensemble_weight: Weight for ensemble predictions (0-1)
            rl_weight: Weight for RL predictions (0-1)
            confidence_threshold: Minimum combined confidence threshold for executing trades
            position_size: Position size as fraction of available capital
            stop_loss_pct: Stop loss percentage
            take_profit_pct: Take profit percentage
        """
        super().__init__(
            name=f"combined_{ensemble_id}_{agent_id}",
            symbol=symbol,
            timeframe=timeframe,
            stop_loss_pct=stop_loss_pct,
            take_profit_pct=take_profit_pct
        )
        
        # Normalize weights
        total_weight = ensemble_weight + rl_weight
        self.ensemble_weight = ensemble_weight / total_weight
        self.rl_weight = rl_weight / total_weight
        
        self.ensemble_id = ensemble_id
        self.agent_id = agent_id
        self.confidence_threshold = confidence_threshold
        self.position_size = position_size
        
        # Create individual strategies
        self.ensemble_strategy = EnsembleStrategy(
            symbol=symbol,
            timeframe=timeframe,
            ensemble_id=ensemble_id,
            threshold_buy=0.6,
            threshold_sell=0.4,
            position_size=position_size,
            stop_loss_pct=stop_loss_pct,
            take_profit_pct=take_profit_pct
        )
        
        self.rl_strategy = RLStrategy(
            symbol=symbol,
            timeframe=timeframe,
            agent_id=agent_id,
            confidence_threshold=0.5,
            stop_loss_pct=stop_loss_pct,
            take_profit_pct=take_profit_pct
        )
        
        logger.info(f"Initialized CombinedStrategy for {symbol} with ensemble_id {ensemble_id} and agent_id {agent_id}")
    
    def generate_signal(self, data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Generate trading signal based on combined predictions
        
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
            
            # Get signals from individual strategies
            ensemble_signal = self.ensemble_strategy.generate_signal(data)
            rl_signal = self.rl_strategy.generate_signal(data)
            
            # Default signal is hold
            action = "hold"
            size = 0
            
            # Check if we should close existing position based on stop loss/take profit
            if self.current_position != 0 and self.should_close_position(current_price):
                action = "close"
                logger.info(f"Closing position at {current_price} due to stop loss/take profit")
            
            # Combine signals
            else:
                combined_signal = self._combine_signals(ensemble_signal, rl_signal)
                action = combined_signal["action"]
                size = combined_signal["size"]
                confidence = combined_signal["confidence"]
                
                if action != "hold" and confidence >= self.confidence_threshold:
                    logger.info(f"{action.capitalize()} signal generated with confidence {confidence}")
                else:
                    action = "hold"
                    size = 0
            
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
                "confidence": combined_signal.get("confidence", 0) if action != "close" else 1.0,
                "metadata": {
                    "ensemble_id": self.ensemble_id,
                    "agent_id": self.agent_id,
                    "ensemble_weight": self.ensemble_weight,
                    "rl_weight": self.rl_weight,
                    "ensemble_signal": ensemble_signal,
                    "rl_signal": rl_signal
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
    
    def _combine_signals(self, ensemble_signal: Dict[str, Any], rl_signal: Dict[str, Any]) -> Dict[str, Any]:
        """
        Combine signals from ensemble and RL strategies
        
        Args:
            ensemble_signal: Signal from ensemble strategy
            rl_signal: Signal from RL strategy
            
        Returns:
            Combined signal
        """
        # Extract actions and confidences
        ensemble_action = ensemble_signal["action"]
        ensemble_confidence = ensemble_signal.get("confidence", 0)
        ensemble_size = ensemble_signal.get("size", 0)
        
        rl_action = rl_signal["action"]
        rl_confidence = rl_signal.get("confidence", 0)
        rl_size = rl_signal.get("size", 0)
        
        # Convert actions to numeric values for easier comparison
        action_values = {"buy": 1, "hold": 0, "sell": -1, "close": 0}
        ensemble_value = action_values.get(ensemble_action, 0)
        rl_value = action_values.get(rl_action, 0)
        
        # Calculate weighted action value
        weighted_value = (ensemble_value * self.ensemble_weight * ensemble_confidence + 
                          rl_value * self.rl_weight * rl_confidence)
        
        # Determine combined action
        if weighted_value > 0.3:
            action = "buy"
        elif weighted_value < -0.3:
            action = "sell"
        else:
            action = "hold"
        
        # Calculate combined confidence
        if action == "hold":
            confidence = 0
        else:
            # Confidence is higher when both strategies agree
            if (ensemble_value > 0 and rl_value > 0) or (ensemble_value < 0 and rl_value < 0):
                agreement_bonus = 0.2
            else:
                agreement_bonus = 0
            
            confidence = (ensemble_confidence * self.ensemble_weight + 
                          rl_confidence * self.rl_weight + 
                          agreement_bonus)
            
            # Cap confidence at 1.0
            confidence = min(confidence, 1.0)
        
        # Calculate combined position size
        size = self.position_size
        
        # Create combined signal
        combined_signal = {
            "action": action,
            "size": size,
            "confidence": confidence,
            "weighted_value": weighted_value
        }
        
        return combined_signal
