"""
API models for Trading service
"""
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum

class StrategyType(str, Enum):
    """Strategy types"""
    ENSEMBLE = "ensemble"
    RL = "rl"
    COMBINED = "combined"
    CUSTOM = "custom"

class ExecutionMode(str, Enum):
    """Execution modes"""
    PAPER = "paper"
    LIVE = "live"

class StrategyConfig(BaseModel):
    """Configuration for trading strategy"""
    type: StrategyType = Field(..., description="Type of strategy")
    symbol: str = Field(..., description="Trading pair symbol (e.g., 'BTC/USDT')")
    timeframe: str = Field(..., description="Timeframe for data (e.g., '1h', '1d')")
    ensemble_id: Optional[str] = Field(None, description="ID of ensemble model (for ensemble or combined strategy)")
    agent_id: Optional[str] = Field(None, description="ID of RL agent (for rl or combined strategy)")
    ensemble_weight: Optional[float] = Field(0.5, description="Weight for ensemble predictions (for combined strategy)")
    rl_weight: Optional[float] = Field(0.5, description="Weight for RL predictions (for combined strategy)")
    threshold_buy: Optional[float] = Field(0.6, description="Threshold for buy signals (for ensemble strategy)")
    threshold_sell: Optional[float] = Field(0.4, description="Threshold for sell signals (for ensemble strategy)")
    confidence_threshold: Optional[float] = Field(0.6, description="Confidence threshold for executing trades")
    position_size: Optional[float] = Field(1.0, description="Position size as fraction of available capital")
    stop_loss_pct: Optional[float] = Field(0.05, description="Stop loss percentage")
    take_profit_pct: Optional[float] = Field(0.1, description="Take profit percentage")
    custom_params: Optional[Dict[str, Any]] = Field(None, description="Custom parameters for custom strategy")

class ExecutorConfig(BaseModel):
    """Configuration for trade executor"""
    mode: ExecutionMode = Field(..., description="Execution mode")
    exchange: Optional[str] = Field("binance", description="Exchange name for live trading")
    initial_balance: Optional[float] = Field(10000.0, description="Initial balance for paper trading")
    transaction_fee: Optional[float] = Field(0.001, description="Transaction fee rate")
    slippage: Optional[float] = Field(0.001, description="Price slippage rate")

class RiskConfig(BaseModel):
    """Configuration for risk management"""
    max_position_size: Optional[float] = Field(0.5, description="Maximum position size as fraction of account balance")
    max_drawdown: Optional[float] = Field(0.2, description="Maximum drawdown before stopping trading")
    stop_loss_pct: Optional[float] = Field(0.05, description="Default stop loss percentage")
    take_profit_pct: Optional[float] = Field(0.1, description="Default take profit percentage")

class NotificationConfig(BaseModel):
    """Configuration for notifications"""
    email_enabled: Optional[bool] = Field(False, description="Whether to enable email notifications")
    email_recipients: Optional[List[str]] = Field(None, description="Email recipients")
    telegram_enabled: Optional[bool] = Field(False, description="Whether to enable Telegram notifications")
    telegram_chat_ids: Optional[List[str]] = Field(None, description="Telegram chat IDs")
    webhook_enabled: Optional[bool] = Field(False, description="Whether to enable webhook notifications")
    webhook_url: Optional[str] = Field(None, description="Webhook URL")

class TradingBotConfig(BaseModel):
    """Configuration for trading bot"""
    name: str = Field(..., description="Bot name")
    strategy: StrategyConfig = Field(..., description="Strategy configuration")
    executor: ExecutorConfig = Field(..., description="Executor configuration")
    risk: Optional[RiskConfig] = Field(None, description="Risk management configuration")
    notification: Optional[NotificationConfig] = Field(None, description="Notification configuration")

class CreateBotRequest(BaseModel):
    """Request model for creating a trading bot"""
    config: TradingBotConfig = Field(..., description="Bot configuration")

class UpdateBotRequest(BaseModel):
    """Request model for updating a trading bot"""
    config: TradingBotConfig = Field(..., description="Bot configuration")

class StartBotRequest(BaseModel):
    """Request model for starting a trading bot"""
    bot_id: str = Field(..., description="Bot ID")

class StopBotRequest(BaseModel):
    """Request model for stopping a trading bot"""
    bot_id: str = Field(..., description="Bot ID")

class BacktestRequest(BaseModel):
    """Request model for backtesting a strategy"""
    strategy: StrategyConfig = Field(..., description="Strategy configuration")
    start_time: Optional[str] = Field(None, description="Start time for backtest")
    end_time: Optional[str] = Field(None, description="End time for backtest")
    initial_balance: Optional[float] = Field(10000.0, description="Initial balance for backtest")

class Signal(BaseModel):
    """Trading signal"""
    strategy: str = Field(..., description="Strategy name")
    symbol: str = Field(..., description="Trading pair symbol")
    timeframe: str = Field(..., description="Timeframe")
    timestamp: str = Field(..., description="Signal timestamp")
    price: float = Field(..., description="Current price")
    action: str = Field(..., description="Action to take (buy, sell, hold, close)")
    size: float = Field(..., description="Position size")
    confidence: float = Field(..., description="Signal confidence")
    stop_loss: Optional[float] = Field(None, description="Stop loss price")
    take_profit: Optional[float] = Field(None, description="Take profit price")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")

class Execution(BaseModel):
    """Trade execution"""
    executor: str = Field(..., description="Executor name")
    symbol: str = Field(..., description="Trading pair symbol")
    action: str = Field(..., description="Action taken")
    timestamp: str = Field(..., description="Execution timestamp")
    status: str = Field(..., description="Execution status")
    amount: Optional[float] = Field(None, description="Executed amount")
    price: Optional[float] = Field(None, description="Execution price")
    value: Optional[float] = Field(None, description="Execution value")
    fee: Optional[float] = Field(None, description="Execution fee")
    pnl: Optional[float] = Field(None, description="Profit/loss")
    order_id: Optional[str] = Field(None, description="Order ID (for live trading)")
    error: Optional[str] = Field(None, description="Error message (if failed)")

class Trade(BaseModel):
    """Trade record"""
    id: str = Field(..., description="Trade ID")
    bot_id: str = Field(..., description="Bot ID")
    symbol: str = Field(..., description="Trading pair symbol")
    action: str = Field(..., description="Action taken")
    timestamp: str = Field(..., description="Trade timestamp")
    amount: float = Field(..., description="Trade amount")
    price: float = Field(..., description="Trade price")
    value: float = Field(..., description="Trade value")
    fee: float = Field(..., description="Trade fee")
    pnl: Optional[float] = Field(None, description="Profit/loss")

class Position(BaseModel):
    """Position record"""
    symbol: str = Field(..., description="Trading pair symbol")
    amount: float = Field(..., description="Position amount")
    avg_price: float = Field(..., description="Average entry price")
    value: float = Field(..., description="Position value")
    current_price: float = Field(..., description="Current market price")
    current_value: float = Field(..., description="Current position value")
    unrealized_pnl: float = Field(..., description="Unrealized profit/loss")

class AccountInfo(BaseModel):
    """Account information"""
    balance: float = Field(..., description="Account balance")
    equity: float = Field(..., description="Account equity")
    positions: Dict[str, Position] = Field(..., description="Open positions")
    timestamp: str = Field(..., description="Timestamp")

class BotStatus(BaseModel):
    """Trading bot status"""
    id: str = Field(..., description="Bot ID")
    name: str = Field(..., description="Bot name")
    status: str = Field(..., description="Bot status (running, stopped, error)")
    strategy: str = Field(..., description="Strategy name")
    executor: str = Field(..., description="Executor name")
    symbol: str = Field(..., description="Trading pair symbol")
    timeframe: str = Field(..., description="Timeframe")
    account_info: AccountInfo = Field(..., description="Account information")
    last_signal: Optional[Signal] = Field(None, description="Last signal")
    last_execution: Optional[Execution] = Field(None, description="Last execution")
    last_update: str = Field(..., description="Last update timestamp")
    error: Optional[str] = Field(None, description="Error message (if status is error)")

class BacktestResult(BaseModel):
    """Backtest result"""
    strategy: str = Field(..., description="Strategy name")
    symbol: str = Field(..., description="Trading pair symbol")
    timeframe: str = Field(..., description="Timeframe")
    start_time: str = Field(..., description="Start time")
    end_time: str = Field(..., description="End time")
    initial_balance: float = Field(..., description="Initial balance")
    final_balance: float = Field(..., description="Final balance")
    total_return: float = Field(..., description="Total return")
    total_return_pct: float = Field(..., description="Total return percentage")
    annualized_return: float = Field(..., description="Annualized return")
    max_drawdown: float = Field(..., description="Maximum drawdown")
    max_drawdown_pct: float = Field(..., description="Maximum drawdown percentage")
    sharpe_ratio: float = Field(..., description="Sharpe ratio")
    sortino_ratio: float = Field(..., description="Sortino ratio")
    total_trades: int = Field(..., description="Total number of trades")
    winning_trades: int = Field(..., description="Number of winning trades")
    losing_trades: int = Field(..., description="Number of losing trades")
    win_rate: float = Field(..., description="Win rate")
    profit_factor: float = Field(..., description="Profit factor")
    average_profit: float = Field(..., description="Average profit")
    average_loss: float = Field(..., description="Average loss")
    largest_profit: float = Field(..., description="Largest profit")
    largest_loss: float = Field(..., description="Largest loss")
    trades: List[Trade] = Field(..., description="List of trades")
    equity_curve: List[Dict[str, Any]] = Field(..., description="Equity curve")

class StatusResponse(BaseModel):
    """Response model for status"""
    status: str = Field(..., description="Status message")
    timestamp: datetime = Field(..., description="Timestamp of the response")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional details")
