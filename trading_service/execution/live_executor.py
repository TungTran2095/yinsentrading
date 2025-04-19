"""
Live trading executor for real trading
"""
import logging
from typing import Dict, List, Optional, Any, Tuple
import requests
import pandas as pd
import numpy as np
import json
from datetime import datetime
import ccxt
import sys
sys.path.append('..')
from config import (
    EXCHANGE_CONFIGS, DEFAULT_EXCHANGE,
    TRANSACTION_FEE, SLIPPAGE
)
from .base_executor import BaseExecutor

logger = logging.getLogger(__name__)

class LiveExecutor(BaseExecutor):
    """
    Live trading executor for real trading
    """
    
    def __init__(self, exchange_name: str = DEFAULT_EXCHANGE,
                 transaction_fee: float = TRANSACTION_FEE,
                 slippage: float = SLIPPAGE):
        """
        Initialize the live trading executor
        
        Args:
            exchange_name: Name of the exchange to use
            transaction_fee: Transaction fee rate
            slippage: Price slippage rate
        """
        super().__init__(
            name=f"live_executor_{exchange_name}",
            mode="live",
            transaction_fee=transaction_fee,
            slippage=slippage
        )
        
        self.exchange_name = exchange_name
        
        # Get exchange config
        if exchange_name not in EXCHANGE_CONFIGS:
            raise ValueError(f"Invalid exchange name: {exchange_name}. Must be one of {list(EXCHANGE_CONFIGS.keys())}")
        
        exchange_config = EXCHANGE_CONFIGS[exchange_name]
        
        # Initialize exchange
        self._initialize_exchange(exchange_config)
        
        # Get account info
        account_info = self.get_account_info()
        self.balance = account_info["balance"]
        
        logger.info(f"Initialized LiveExecutor for {exchange_name}")
    
    def _initialize_exchange(self, config: Dict[str, Any]) -> None:
        """
        Initialize exchange connection
        
        Args:
            config: Exchange configuration
        """
        try:
            # Create exchange instance
            exchange_class = getattr(ccxt, self.exchange_name)
            
            self.exchange = exchange_class({
                'apiKey': config['api_key'],
                'secret': config['api_secret'],
                'enableRateLimit': True,
                'options': {
                    'defaultType': 'spot'
                }
            })
            
            # Use testnet if configured
            if config.get('testnet', False):
                if hasattr(self.exchange, 'set_sandbox_mode'):
                    self.exchange.set_sandbox_mode(True)
                    logger.info(f"Using testnet for {self.exchange_name}")
            
            # Load markets
            self.exchange.load_markets()
            
            logger.info(f"Initialized exchange: {self.exchange_name}")
        
        except Exception as e:
            logger.error(f"Error initializing exchange: {e}")
            raise
    
    def execute_signal(self, signal: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a trading signal in live trading mode
        
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
            
            # Get market price
            market_price = self.get_market_price(symbol)
            
            # Calculate order details based on action
            if action == "buy":
                # Calculate order amount based on size (fraction of balance)
                order_value = self.balance * size
                order_amount = order_value / market_price
                
                # Execute order
                order = self._create_order(symbol, "buy", order_amount)
                
                # Record order
                self.record_order(order)
                
                # Update execution result
                execution["status"] = "success"
                execution["order_id"] = order["id"]
                execution["amount"] = order["amount"]
                execution["price"] = order["price"]
                execution["value"] = order["cost"]
                execution["fee"] = order.get("fee", {}).get("cost", 0)
                
            elif action == "sell":
                # Calculate order amount based on size (fraction of balance)
                order_value = self.balance * size
                order_amount = order_value / market_price
                
                # Execute order
                order = self._create_order(symbol, "sell", order_amount)
                
                # Record order
                self.record_order(order)
                
                # Update execution result
                execution["status"] = "success"
                execution["order_id"] = order["id"]
                execution["amount"] = order["amount"]
                execution["price"] = order["price"]
                execution["value"] = order["cost"]
                execution["fee"] = order.get("fee", {}).get("cost", 0)
                
            elif action == "close":
                # Skip if no position
                if position_amount == 0:
                    execution["status"] = "skipped"
                    execution["message"] = "No position to close"
                    return execution
                
                # Determine if long or short position
                side = "sell" if position_amount > 0 else "buy"
                
                # Execute order
                order = self._create_order(symbol, side, abs(position_amount))
                
                # Record order
                self.record_order(order)
                
                # Update execution result
                execution["status"] = "success"
                execution["order_id"] = order["id"]
                execution["amount"] = order["amount"]
                execution["price"] = order["price"]
                execution["value"] = order["cost"]
                execution["fee"] = order.get("fee", {}).get("cost", 0)
            
            # Update account info after execution
            self._update_account_info()
            
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
    
    def _create_order(self, symbol: str, side: str, amount: float) -> Dict[str, Any]:
        """
        Create an order on the exchange
        
        Args:
            symbol: Trading pair symbol
            side: Order side ("buy" or "sell")
            amount: Order amount
            
        Returns:
            Order details
        """
        try:
            # Format symbol for exchange
            exchange_symbol = self._format_symbol(symbol)
            
            # Create market order
            order = self.exchange.create_market_order(
                symbol=exchange_symbol,
                side=side,
                amount=amount
            )
            
            # Fetch order to get filled details
            order = self.exchange.fetch_order(order['id'], exchange_symbol)
            
            logger.info(f"Created {side} order for {amount} {symbol} at market price")
            
            return order
        
        except Exception as e:
            logger.error(f"Error creating order: {e}")
            raise
    
    def _format_symbol(self, symbol: str) -> str:
        """
        Format symbol for exchange
        
        Args:
            symbol: Trading pair symbol
            
        Returns:
            Formatted symbol
        """
        # Different exchanges have different symbol formats
        if self.exchange_name == "binance":
            return symbol.replace("/", "")
        else:
            return symbol
    
    def _update_account_info(self) -> None:
        """
        Update account information
        """
        try:
            # Fetch account balance
            balance = self.exchange.fetch_balance()
            
            # Update balance
            self.balance = balance['total']['USDT']
            
            # Update positions
            for symbol in self.positions:
                position = self.get_position(symbol)
                self.positions[symbol] = position
            
            logger.info(f"Updated account info: balance={self.balance}")
        
        except Exception as e:
            logger.error(f"Error updating account info: {e}")
    
    def get_account_info(self) -> Dict[str, Any]:
        """
        Get account information
        
        Returns:
            Dictionary with account details
        """
        try:
            # Fetch account balance
            balance = self.exchange.fetch_balance()
            
            # Calculate total equity
            equity = balance['total']['USDT']
            
            # Get positions
            positions = {}
            for symbol in self.exchange.markets:
                try:
                    position = self.get_position(symbol)
                    if position["amount"] != 0:
                        positions[symbol] = position
                except:
                    pass
            
            return {
                "balance": equity,
                "equity": equity,
                "positions": positions,
                "timestamp": datetime.now().isoformat()
            }
        
        except Exception as e:
            logger.error(f"Error getting account info: {e}")
            return {
                "balance": 0,
                "equity": 0,
                "positions": {},
                "timestamp": datetime.now().isoformat(),
                "error": str(e)
            }
    
    def get_position(self, symbol: str) -> Dict[str, Any]:
        """
        Get position for a symbol
        
        Args:
            symbol: Trading pair symbol
            
        Returns:
            Dictionary with position details
        """
        try:
            # Format symbol for exchange
            exchange_symbol = self._format_symbol(symbol)
            
            # Get base currency
            base_currency = symbol.split('/')[0]
            
            # Fetch balance
            balance = self.exchange.fetch_balance()
            
            # Get amount
            amount = balance['total'].get(base_currency, 0)
            
            # Get current price
            current_price = self.get_market_price(symbol)
            
            # Calculate position value
            position_value = amount * current_price
            
            # Create position object
            position = {
                "amount": amount,
                "avg_price": 0,  # Not available from exchange
                "value": position_value,
                "current_price": current_price,
                "current_value": position_value,
                "unrealized_pnl": 0  # Not available without avg_price
            }
            
            return position
        
        except Exception as e:
            logger.error(f"Error getting position: {e}")
            return {
                "amount": 0,
                "avg_price": 0,
                "value": 0,
                "current_price": 0,
                "current_value": 0,
                "unrealized_pnl": 0,
                "error": str(e)
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
            # Format symbol for exchange
            exchange_symbol = self._format_symbol(symbol)
            
            # Fetch ticker
            ticker = self.exchange.fetch_ticker(exchange_symbol)
            
            return ticker['last']
        
        except Exception as e:
            logger.error(f"Error getting market price: {e}")
            return 0.0
