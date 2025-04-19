"""
Risk management module for trading
"""
import logging
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
import numpy as np
from datetime import datetime
import sys
sys.path.append('..')
from config import (
    MAX_POSITION_SIZE, MAX_DRAWDOWN,
    STOP_LOSS_PERCENTAGE, TAKE_PROFIT_PERCENTAGE
)

logger = logging.getLogger(__name__)

class RiskManager:
    """
    Risk management for trading
    """
    
    def __init__(self, max_position_size: float = MAX_POSITION_SIZE,
                 max_drawdown: float = MAX_DRAWDOWN,
                 stop_loss_pct: float = STOP_LOSS_PERCENTAGE,
                 take_profit_pct: float = TAKE_PROFIT_PERCENTAGE):
        """
        Initialize the risk manager
        
        Args:
            max_position_size: Maximum position size as fraction of account balance
            max_drawdown: Maximum drawdown before stopping trading
            stop_loss_pct: Default stop loss percentage
            take_profit_pct: Default take profit percentage
        """
        self.max_position_size = max_position_size
        self.max_drawdown = max_drawdown
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        
        logger.info(f"Initialized RiskManager with max_position_size={max_position_size}, max_drawdown={max_drawdown}")
    
    def validate_signal(self, signal: Dict[str, Any], account_info: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        """
        Validate a trading signal against risk management rules
        
        Args:
            signal: Trading signal
            account_info: Account information
            
        Returns:
            Tuple of (is_valid, modified_signal)
        """
        # Make a copy of the signal
        modified_signal = signal.copy()
        
        # Check if trading should be stopped due to drawdown
        if self._check_max_drawdown(account_info):
            return False, {
                "error": f"Trading stopped due to max drawdown ({self.max_drawdown*100}%) exceeded",
                "original_signal": signal
            }
        
        # Skip validation for hold or close signals
        if signal["action"] in ["hold", "close"]:
            return True, modified_signal
        
        # Check and adjust position size
        if "size" in signal and signal["size"] > 0:
            size = signal["size"]
            
            # Limit position size
            if size > self.max_position_size:
                logger.warning(f"Position size {size} exceeds maximum {self.max_position_size}, adjusting")
                modified_signal["size"] = self.max_position_size
                modified_signal["size_adjusted"] = True
                modified_signal["original_size"] = size
        
        # Add stop loss and take profit if not present
        if "stop_loss" not in modified_signal:
            price = signal.get("price", 0)
            if price > 0:
                if signal["action"] == "buy":
                    modified_signal["stop_loss"] = price * (1 - self.stop_loss_pct)
                elif signal["action"] == "sell":
                    modified_signal["stop_loss"] = price * (1 + self.stop_loss_pct)
        
        if "take_profit" not in modified_signal:
            price = signal.get("price", 0)
            if price > 0:
                if signal["action"] == "buy":
                    modified_signal["take_profit"] = price * (1 + self.take_profit_pct)
                elif signal["action"] == "sell":
                    modified_signal["take_profit"] = price * (1 - self.take_profit_pct)
        
        return True, modified_signal
    
    def _check_max_drawdown(self, account_info: Dict[str, Any]) -> bool:
        """
        Check if max drawdown has been exceeded
        
        Args:
            account_info: Account information
            
        Returns:
            True if max drawdown exceeded
        """
        if "drawdown" in account_info:
            current_drawdown = account_info["drawdown"] / 100  # Convert from percentage
            return current_drawdown > self.max_drawdown
        
        # If drawdown not available, calculate it
        if "equity_history" in account_info and len(account_info["equity_history"]) > 0:
            equity_history = account_info["equity_history"]
            current_equity = account_info["equity"]
            
            # Find peak equity
            peak_equity = max([point["equity"] for point in equity_history])
            
            # Calculate drawdown
            if peak_equity > 0:
                drawdown = (peak_equity - current_equity) / peak_equity
                return drawdown > self.max_drawdown
        
        return False
    
    def calculate_position_size(self, signal: Dict[str, Any], account_info: Dict[str, Any]) -> float:
        """
        Calculate optimal position size based on risk management rules
        
        Args:
            signal: Trading signal
            account_info: Account information
            
        Returns:
            Optimal position size
        """
        # Get account balance and equity
        balance = account_info.get("balance", 0)
        
        # Get signal confidence
        confidence = signal.get("confidence", 0.5)
        
        # Base position size on confidence and max position size
        position_size = confidence * self.max_position_size
        
        # Ensure position size is within limits
        position_size = min(position_size, self.max_position_size)
        
        return position_size
    
    def calculate_risk_metrics(self, trades: List[Dict[str, Any]], 
                              equity_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate risk metrics based on trade history
        
        Args:
            trades: List of trades
            equity_history: Equity history
            
        Returns:
            Dictionary with risk metrics
        """
        # Initialize metrics
        metrics = {
            "total_trades": len(trades),
            "winning_trades": 0,
            "losing_trades": 0,
            "win_rate": 0,
            "profit_factor": 0,
            "average_profit": 0,
            "average_loss": 0,
            "largest_profit": 0,
            "largest_loss": 0,
            "max_drawdown": 0,
            "max_drawdown_percentage": 0,
            "sharpe_ratio": 0,
            "sortino_ratio": 0
        }
        
        # Skip if no trades
        if len(trades) == 0:
            return metrics
        
        # Calculate trade metrics
        total_profit = 0
        total_loss = 0
        profits = []
        losses = []
        
        for trade in trades:
            if "pnl" in trade:
                pnl = trade["pnl"]
                if pnl > 0:
                    metrics["winning_trades"] += 1
                    total_profit += pnl
                    profits.append(pnl)
                    metrics["largest_profit"] = max(metrics["largest_profit"], pnl)
                elif pnl < 0:
                    metrics["losing_trades"] += 1
                    total_loss += abs(pnl)
                    losses.append(abs(pnl))
                    metrics["largest_loss"] = max(metrics["largest_loss"], abs(pnl))
        
        # Calculate win rate
        if metrics["total_trades"] > 0:
            metrics["win_rate"] = metrics["winning_trades"] / metrics["total_trades"]
        
        # Calculate profit factor
        if total_loss > 0:
            metrics["profit_factor"] = total_profit / total_loss
        
        # Calculate average profit and loss
        if len(profits) > 0:
            metrics["average_profit"] = sum(profits) / len(profits)
        
        if len(losses) > 0:
            metrics["average_loss"] = sum(losses) / len(losses)
        
        # Calculate drawdown metrics
        if len(equity_history) > 0:
            equity_values = [point["equity"] for point in equity_history]
            peak = equity_values[0]
            drawdown = 0
            drawdown_pct = 0
            
            for equity in equity_values:
                if equity > peak:
                    peak = equity
                
                current_drawdown = peak - equity
                if current_drawdown > drawdown:
                    drawdown = current_drawdown
                    drawdown_pct = drawdown / peak if peak > 0 else 0
            
            metrics["max_drawdown"] = drawdown
            metrics["max_drawdown_percentage"] = drawdown_pct * 100
        
        # Calculate Sharpe and Sortino ratios
        if len(equity_history) > 1:
            returns = []
            negative_returns = []
            
            for i in range(1, len(equity_history)):
                prev_equity = equity_history[i-1]["equity"]
                curr_equity = equity_history[i]["equity"]
                
                if prev_equity > 0:
                    ret = (curr_equity - prev_equity) / prev_equity
                    returns.append(ret)
                    
                    if ret < 0:
                        negative_returns.append(ret)
            
            if len(returns) > 0:
                avg_return = np.mean(returns)
                std_return = np.std(returns)
                
                if std_return > 0:
                    # Annualized Sharpe ratio (assuming daily returns)
                    metrics["sharpe_ratio"] = avg_return / std_return * np.sqrt(252)
                
                if len(negative_returns) > 0:
                    downside_deviation = np.std(negative_returns)
                    
                    if downside_deviation > 0:
                        # Annualized Sortino ratio (assuming daily returns)
                        metrics["sortino_ratio"] = avg_return / downside_deviation * np.sqrt(252)
        
        return metrics
