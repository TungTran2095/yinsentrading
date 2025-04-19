"""
Paper trading executor for simulated trading
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
    DATA_SERVICE_URL, DATA_SERVICE_API_PREFIX,
    TRANSACTION_FEE, SLIPPAGE
)
from .base_executor import BaseExecutor

logger = logging.getLogger(__name__)

class PaperExecutor(BaseExecutor):
    """
    Paper trading executor for simulated trading
    """
    
    def __init__(self, initial_balance: float = 10000.0,
                 transaction_fee: float = TRANSACTION_FEE,
                 slippage: float = SLIPPAGE):
        """
        Initialize the paper trading executor
        
        Args:
            initial_balance: Initial account balance
            transaction_fee: Transaction fee rate
            slippage: Price slippage rate
        """
        super().__init__(
            name="paper_executor",
            mode="paper",
            transaction_fee=transaction_fee,
            slippage=slippage
        )
        
        self.balance = initial_balance
        self.initial_balance = initial_balance
        self.equity_history = []
        
        # Record initial equity
        self._record_equity()
        
        logger.info(f"Initialized PaperExecutor with initial balance: {initial_balance}")
    
    def execute_signal(self, signal: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a trading signal in paper trading mode
        
        Args:
            signal: Trading signal
            
        Returns:
            Dictionary with execution details
        """
        try:
            # Extract signal details
            symbol = signal["symbol"]
            action = signal["action"]
            size = signal.get("size", 0)
            price = signal.get("price", self.get_market_price(symbol))
            
            # Initialize execution result
            execution = {
                "executor": self.name,
                "symbol": symbol,
                "action": action,
                "timestamp": datetime.now().isoformat(),
                "status": "failed",
                "error": None
            }
            
            # Skip if action is hold
            if action == "hold":
                execution["status"] = "skipped"
                execution["message"] = "Hold signal, no action taken"
                return execution
            
            # Get current position
            position = self.get_position(symbol)
            position_amount = position["amount"]
            
            # Calculate order details based on action
            if action == "buy":
                # Apply slippage to buy price
                execution_price = self.apply_slippage(price, is_buy=True)
                
                # Calculate order amount based on size (fraction of balance)
                order_value = self.balance * size
                order_amount = order_value / execution_price
                
                # Calculate fee
                fee = self.calculate_fee(order_value)
                
                # Check if enough balance
                if order_value + fee > self.balance:
                    execution["error"] = "Insufficient balance"
                    return execution
                
                # Update balance and position
                self.update_balance(-(order_value + fee))
                self.update_position(symbol, order_amount, execution_price)
                
                # Record trade
                trade = {
                    "symbol": symbol,
                    "action": "buy",
                    "amount": order_amount,
                    "price": execution_price,
                    "value": order_value,
                    "fee": fee,
                    "timestamp": datetime.now().isoformat()
                }
                self.record_trade(trade)
                
                # Update execution result
                execution["status"] = "success"
                execution["amount"] = order_amount
                execution["price"] = execution_price
                execution["value"] = order_value
                execution["fee"] = fee
                
            elif action == "sell":
                # Apply slippage to sell price
                execution_price = self.apply_slippage(price, is_buy=False)
                
                # Calculate order amount based on size (fraction of balance)
                order_value = self.balance * size
                order_amount = order_value / execution_price
                
                # Make it negative for sell
                order_amount = -order_amount
                
                # Calculate fee
                fee = self.calculate_fee(abs(order_value))
                
                # Check if enough balance
                if order_value + fee > self.balance:
                    execution["error"] = "Insufficient balance"
                    return execution
                
                # Update balance and position
                self.update_balance(-(order_value + fee))
                self.update_position(symbol, order_amount, execution_price)
                
                # Record trade
                trade = {
                    "symbol": symbol,
                    "action": "sell",
                    "amount": abs(order_amount),
                    "price": execution_price,
                    "value": abs(order_value),
                    "fee": fee,
                    "timestamp": datetime.now().isoformat()
                }
                self.record_trade(trade)
                
                # Update execution result
                execution["status"] = "success"
                execution["amount"] = abs(order_amount)
                execution["price"] = execution_price
                execution["value"] = abs(order_value)
                execution["fee"] = fee
                
            elif action == "close":
                # Skip if no position
                if position_amount == 0:
                    execution["status"] = "skipped"
                    execution["message"] = "No position to close"
                    return execution
                
                # Determine if long or short position
                is_long = position_amount > 0
                
                # Apply slippage to close price
                execution_price = self.apply_slippage(price, is_buy=not is_long)
                
                # Calculate order value
                order_amount = abs(position_amount)
                order_value = order_amount * execution_price
                
                # Calculate fee
                fee = self.calculate_fee(order_value)
                
                # Calculate PnL
                if is_long:
                    pnl = (execution_price - position["avg_price"]) * order_amount - fee
                else:
                    pnl = (position["avg_price"] - execution_price) * order_amount - fee
                
                # Update balance and position
                self.update_balance(order_value - fee)
                self.update_position(symbol, -position_amount, execution_price)
                
                # Record trade
                trade = {
                    "symbol": symbol,
                    "action": "close",
                    "amount": order_amount,
                    "price": execution_price,
                    "value": order_value,
                    "fee": fee,
                    "pnl": pnl,
                    "timestamp": datetime.now().isoformat()
                }
                self.record_trade(trade)
                
                # Update execution result
                execution["status"] = "success"
                execution["amount"] = order_amount
                execution["price"] = execution_price
                execution["value"] = order_value
                execution["fee"] = fee
                execution["pnl"] = pnl
            
            # Record equity after execution
            self._record_equity()
            
            return execution
        
        except Exception as e:
            logger.error(f"Error executing signal: {e}")
            return {
                "executor": self.name,
                "symbol": signal.get("symbol", "unknown"),
                "action": signal.get("action", "unknown"),
                "timestamp": datetime.now().isoformat(),
                "status": "failed",
                "error": str(e)
            }
    
    def get_account_info(self) -> Dict[str, Any]:
        """
        Get account information
        
        Returns:
            Dictionary with account details
        """
        # Calculate total equity
        equity = self.balance
        for symbol, position in self.positions.items():
            if position["amount"] != 0:
                current_price = self.get_market_price(symbol)
                position_value = position["amount"] * current_price
                equity += position_value
        
        # Calculate performance metrics
        pnl = equity - self.initial_balance
        pnl_percentage = (pnl / self.initial_balance) * 100 if self.initial_balance > 0 else 0
        
        # Calculate drawdown
        peak_equity = self.initial_balance
        for point in self.equity_history:
            peak_equity = max(peak_equity, point["equity"])
        
        drawdown = (peak_equity - equity) / peak_equity * 100 if peak_equity > 0 else 0
        
        return {
            "balance": self.balance,
            "equity": equity,
            "initial_balance": self.initial_balance,
            "pnl": pnl,
            "pnl_percentage": pnl_percentage,
            "drawdown": drawdown,
            "positions": self.positions,
            "timestamp": datetime.now().isoformat()
        }
    
    def get_position(self, symbol: str) -> Dict[str, Any]:
        """
        Get position for a symbol
        
        Args:
            symbol: Trading pair symbol
            
        Returns:
            Dictionary with position details
        """
        if symbol in self.positions:
            position = self.positions[symbol].copy()
            
            # Add current market price and value
            current_price = self.get_market_price(symbol)
            position["current_price"] = current_price
            position["current_value"] = position["amount"] * current_price
            
            # Calculate unrealized PnL
            if position["amount"] != 0:
                if position["amount"] > 0:  # Long position
                    position["unrealized_pnl"] = (current_price - position["avg_price"]) * position["amount"]
                else:  # Short position
                    position["unrealized_pnl"] = (position["avg_price"] - current_price) * abs(position["amount"])
            else:
                position["unrealized_pnl"] = 0
            
            return position
        else:
            return {
                "amount": 0,
                "avg_price": 0,
                "value": 0,
                "current_price": self.get_market_price(symbol),
                "current_value": 0,
                "unrealized_pnl": 0
            }
    
    def get_market_price(self, symbol: str) -> float:
        """
        Get current market price for a symbol
        
        Args:
            symbol: Trading pair symbol
            
        Returns:
            Current price
        """
        try:
            # Build URL
            url = f"{DATA_SERVICE_URL}{DATA_SERVICE_API_PREFIX}/price/{symbol}"
            
            # Make request
            response = requests.get(url)
            response.raise_for_status()
            
            # Parse response
            data = response.json()
            price = data["price"]
            
            return price
        
        except Exception as e:
            logger.error(f"Error getting market price: {e}")
            
            # If we have trades for this symbol, use the last trade price
            for trade in reversed(self.trades):
                if trade["symbol"] == symbol:
                    return trade["price"]
            
            # Otherwise return a default price
            return 0.0
    
    def _record_equity(self) -> None:
        """
        Record current equity
        """
        account_info = self.get_account_info()
        
        equity_point = {
            "timestamp": datetime.now().isoformat(),
            "balance": account_info["balance"],
            "equity": account_info["equity"],
            "drawdown": account_info["drawdown"]
        }
        
        self.equity_history.append(equity_point)
    
    def get_equity_history(self) -> List[Dict[str, Any]]:
        """
        Get equity history
        
        Returns:
            List of equity points
        """
        return self.equity_history
    
    def reset(self, initial_balance: Optional[float] = None) -> None:
        """
        Reset the paper trading executor
        
        Args:
            initial_balance: New initial balance (optional)
        """
        if initial_balance is not None:
            self.initial_balance = initial_balance
        
        self.balance = self.initial_balance
        self.positions = {}
        self.trades = []
        self.orders = []
        self.equity_history = []
        
        # Record initial equity
        self._record_equity()
        
        logger.info(f"Reset PaperExecutor with initial balance: {self.initial_balance}")
