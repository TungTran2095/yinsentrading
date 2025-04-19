"""
Base execution class for trading
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
    TRANSACTION_FEE, SLIPPAGE, EXECUTION_MODES
)

logger = logging.getLogger(__name__)

class BaseExecutor(ABC):
    """
    Abstract base class for all trade executors
    """
    
    def __init__(self, name: str, mode: str = "paper", 
                 transaction_fee: float = TRANSACTION_FEE,
                 slippage: float = SLIPPAGE):
        """
        Initialize the executor
        
        Args:
            name: Executor name
            mode: Execution mode ("paper" or "live")
            transaction_fee: Transaction fee rate
            slippage: Price slippage rate
        """
        if mode not in EXECUTION_MODES:
            raise ValueError(f"Invalid execution mode: {mode}. Must be one of {EXECUTION_MODES}")
        
        self.name = name
        self.mode = mode
        self.transaction_fee = transaction_fee
        self.slippage = slippage
        
        # State variables
        self.trades = []
        self.orders = []
        self.balance = 0
        self.positions = {}
        
        logger.info(f"Initialized {self.__class__.__name__} executor in {mode} mode")
    
    @abstractmethod
    def execute_signal(self, signal: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a trading signal
        
        Args:
            signal: Trading signal
            
        Returns:
            Dictionary with execution details
        """
        pass
    
    @abstractmethod
    def get_account_info(self) -> Dict[str, Any]:
        """
        Get account information
        
        Returns:
            Dictionary with account details
        """
        pass
    
    @abstractmethod
    def get_position(self, symbol: str) -> Dict[str, Any]:
        """
        Get position for a symbol
        
        Args:
            symbol: Trading pair symbol
            
        Returns:
            Dictionary with position details
        """
        pass
    
    @abstractmethod
    def get_market_price(self, symbol: str) -> float:
        """
        Get current market price for a symbol
        
        Args:
            symbol: Trading pair symbol
            
        Returns:
            Current price
        """
        pass
    
    def calculate_fee(self, amount: float) -> float:
        """
        Calculate transaction fee
        
        Args:
            amount: Transaction amount
            
        Returns:
            Fee amount
        """
        return amount * self.transaction_fee
    
    def apply_slippage(self, price: float, is_buy: bool) -> float:
        """
        Apply slippage to price
        
        Args:
            price: Original price
            is_buy: Whether it's a buy order
            
        Returns:
            Price with slippage applied
        """
        if is_buy:
            return price * (1 + self.slippage)
        else:
            return price * (1 - self.slippage)
    
    def record_trade(self, trade: Dict[str, Any]) -> None:
        """
        Record a trade
        
        Args:
            trade: Trade details
        """
        # Add timestamp if not present
        if "timestamp" not in trade:
            trade["timestamp"] = datetime.now().isoformat()
        
        # Add to trades list
        self.trades.append(trade)
        
        logger.info(f"Recorded trade: {trade}")
    
    def record_order(self, order: Dict[str, Any]) -> None:
        """
        Record an order
        
        Args:
            order: Order details
        """
        # Add timestamp if not present
        if "timestamp" not in order:
            order["timestamp"] = datetime.now().isoformat()
        
        # Add to orders list
        self.orders.append(order)
        
        logger.info(f"Recorded order: {order}")
    
    def update_position(self, symbol: str, amount: float, price: float) -> None:
        """
        Update position for a symbol
        
        Args:
            symbol: Trading pair symbol
            amount: Amount to add (positive) or subtract (negative)
            price: Current price
        """
        # Initialize position if not exists
        if symbol not in self.positions:
            self.positions[symbol] = {
                "amount": 0,
                "avg_price": 0,
                "value": 0
            }
        
        position = self.positions[symbol]
        
        if position["amount"] == 0:
            # New position
            position["amount"] = amount
            position["avg_price"] = price
            position["value"] = amount * price
        elif (position["amount"] > 0 and amount > 0) or (position["amount"] < 0 and amount < 0):
            # Increase position
            total_value = position["value"] + (amount * price)
            total_amount = position["amount"] + amount
            position["avg_price"] = total_value / total_amount
            position["amount"] = total_amount
            position["value"] = total_value
        elif abs(amount) >= abs(position["amount"]):
            # Close position and open new one in opposite direction
            remaining_amount = amount + position["amount"]
            position["amount"] = remaining_amount
            position["avg_price"] = price
            position["value"] = remaining_amount * price
        else:
            # Reduce position
            position["amount"] += amount
            position["value"] = position["amount"] * price
        
        logger.info(f"Updated position for {symbol}: {position}")
    
    def update_balance(self, amount: float) -> None:
        """
        Update account balance
        
        Args:
            amount: Amount to add (positive) or subtract (negative)
        """
        self.balance += amount
        logger.info(f"Updated balance: {self.balance}")
    
    def get_trade_history(self) -> List[Dict[str, Any]]:
        """
        Get trade history
        
        Returns:
            List of trades
        """
        return self.trades
    
    def get_order_history(self) -> List[Dict[str, Any]]:
        """
        Get order history
        
        Returns:
            List of orders
        """
        return self.orders
