"""
Base strategy class for trading
"""
from abc import ABC, abstractmethod
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
    DATA_SERVICE_URL, DATA_SERVICE_API_PREFIX, 
    MODEL_SERVICE_URL, MODEL_SERVICE_API_PREFIX,
    RL_SERVICE_URL, RL_SERVICE_API_PREFIX,
    STOP_LOSS_PERCENTAGE, TAKE_PROFIT_PERCENTAGE
)

logger = logging.getLogger(__name__)

class BaseStrategy(ABC):
    """
    Abstract base class for all trading strategies
    """
    
    def __init__(self, name: str, symbol: str, timeframe: str, 
                 stop_loss_pct: float = STOP_LOSS_PERCENTAGE,
                 take_profit_pct: float = TAKE_PROFIT_PERCENTAGE):
        """
        Initialize the strategy
        
        Args:
            name: Strategy name
            symbol: Trading pair symbol (e.g., "BTC/USDT")
            timeframe: Timeframe for data (e.g., "1h", "1d")
            stop_loss_pct: Stop loss percentage
            take_profit_pct: Take profit percentage
        """
        self.name = name
        self.symbol = symbol
        self.timeframe = timeframe
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        
        # State variables
        self.current_position = 0  # Current position size (negative for short)
        self.entry_price = 0  # Entry price for current position
        self.last_action = None  # Last action taken
        self.last_signal_time = None  # Timestamp of last signal
        
        logger.info(f"Initialized {self.__class__.__name__} strategy for {symbol} with timeframe {timeframe}")
    
    @abstractmethod
    def generate_signal(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate trading signal based on data
        
        Args:
            data: DataFrame with market data
            
        Returns:
            Dictionary with signal details
        """
        pass
    
    def get_data(self) -> pd.DataFrame:
        """
        Get market data from Data Service
        
        Returns:
            DataFrame with market data
        """
        try:
            # Build URL
            url = f"{DATA_SERVICE_URL}{DATA_SERVICE_API_PREFIX}/technical/{self.symbol}/{self.timeframe}"
            
            # Make request
            response = requests.get(url)
            response.raise_for_status()
            
            # Parse response
            data = response.json()["data"]
            
            if not data:
                raise ValueError(f"No data for {self.symbol} with timeframe {self.timeframe}")
            
            # Convert to DataFrame
            df = pd.DataFrame(data)
            
            # Convert timestamp to datetime
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df.set_index("timestamp")
            
            logger.info(f"Got {len(df)} rows of data for {self.symbol} with timeframe {self.timeframe}")
            
            return df
        
        except Exception as e:
            logger.error(f"Error getting data: {e}")
            raise
    
    def calculate_stop_loss(self, entry_price: float, is_long: bool) -> float:
        """
        Calculate stop loss price
        
        Args:
            entry_price: Entry price
            is_long: Whether position is long
            
        Returns:
            Stop loss price
        """
        if is_long:
            return entry_price * (1 - self.stop_loss_pct)
        else:
            return entry_price * (1 + self.stop_loss_pct)
    
    def calculate_take_profit(self, entry_price: float, is_long: bool) -> float:
        """
        Calculate take profit price
        
        Args:
            entry_price: Entry price
            is_long: Whether position is long
            
        Returns:
            Take profit price
        """
        if is_long:
            return entry_price * (1 + self.take_profit_pct)
        else:
            return entry_price * (1 - self.take_profit_pct)
    
    def update_position(self, action: str, price: float, size: float) -> None:
        """
        Update position state
        
        Args:
            action: Action taken ("buy", "sell", "hold")
            price: Current price
            size: Position size
        """
        if action == "buy":
            self.current_position = size
            self.entry_price = price
        elif action == "sell":
            self.current_position = -size
            self.entry_price = price
        elif action == "close":
            self.current_position = 0
            self.entry_price = 0
        
        self.last_action = action
        self.last_signal_time = datetime.now()
    
    def should_close_position(self, current_price: float) -> bool:
        """
        Check if position should be closed based on stop loss/take profit
        
        Args:
            current_price: Current price
            
        Returns:
            True if position should be closed
        """
        if self.current_position == 0:
            return False
        
        is_long = self.current_position > 0
        
        # Calculate stop loss and take profit prices
        stop_loss = self.calculate_stop_loss(self.entry_price, is_long)
        take_profit = self.calculate_take_profit(self.entry_price, is_long)
        
        # Check if price hit stop loss or take profit
        if is_long:
            return current_price <= stop_loss or current_price >= take_profit
        else:
            return current_price >= stop_loss or current_price <= take_profit
