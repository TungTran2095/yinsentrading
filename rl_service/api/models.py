"""
API models for RL service
"""
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime

class EnvironmentConfig(BaseModel):
    """Configuration for trading environment"""
    symbol: str = Field(..., description="Trading pair symbol (e.g., 'BTC/USDT')")
    timeframe: str = Field(..., description="Timeframe for data (e.g., '1h', '1d')")
    initial_balance: float = Field(10000.0, description="Initial balance for trading")
    max_steps: int = Field(1000, description="Maximum number of steps in an episode")
    window_size: int = Field(30, description="Size of observation window")
    commission: float = Field(0.001, description="Trading commission rate")
    slippage: float = Field(0.001, description="Price slippage rate")
    use_ensemble: bool = Field(False, description="Whether to use ensemble predictions")
    ensemble_id: Optional[str] = Field(None, description="ID of ensemble model to use for predictions")
    data_start_time: Optional[str] = Field(None, description="Start time for data")
    data_end_time: Optional[str] = Field(None, description="End time for data")

class AgentConfig(BaseModel):
    """Configuration for RL agent"""
    agent_type: str = Field(..., description="Type of RL agent (e.g., 'ppo', 'dqn', 'a2c', 'sac')")
    hyperparameters: Optional[Dict[str, Any]] = Field({}, description="Agent-specific hyperparameters")

class TrainRequest(BaseModel):
    """Request model for training an RL agent"""
    environment: EnvironmentConfig = Field(..., description="Environment configuration")
    agent: AgentConfig = Field(..., description="Agent configuration")
    total_timesteps: int = Field(100000, description="Total timesteps to train for")
    eval_freq: int = Field(10000, description="Frequency of evaluation during training")
    n_eval_episodes: int = Field(5, description="Number of episodes for evaluation")

class PredictRequest(BaseModel):
    """Request model for making predictions with an RL agent"""
    agent_id: str = Field(..., description="ID of the agent to use")
    observation: Optional[List[List[float]]] = Field(None, description="Current observation (if not using environment)")
    environment: Optional[EnvironmentConfig] = Field(None, description="Environment configuration (if not providing observation)")
    deterministic: bool = Field(True, description="Whether to use deterministic actions")

class EvaluateRequest(BaseModel):
    """Request model for evaluating an RL agent"""
    agent_id: str = Field(..., description="ID of the agent to evaluate")
    environment: EnvironmentConfig = Field(..., description="Environment configuration")
    num_episodes: int = Field(10, description="Number of episodes to evaluate for")
    deterministic: bool = Field(True, description="Whether to use deterministic actions")

class BacktestRequest(BaseModel):
    """Request model for backtesting an RL agent"""
    agent_id: str = Field(..., description="ID of the agent to backtest")
    environment: EnvironmentConfig = Field(..., description="Environment configuration")
    initial_balance: float = Field(10000.0, description="Initial balance for backtesting")

class AgentInfo(BaseModel):
    """Information about an RL agent"""
    id: str = Field(..., description="Agent ID")
    name: str = Field(..., description="Agent name")
    type: str = Field(..., description="Agent type")
    environment: Dict[str, Any] = Field(..., description="Environment configuration")
    hyperparameters: Dict[str, Any] = Field(..., description="Agent hyperparameters")
    created_at: datetime = Field(..., description="Creation timestamp")
    is_trained: bool = Field(..., description="Whether the agent is trained")
    metrics: Optional[Dict[str, Any]] = Field(None, description="Training metrics")

class PredictionResult(BaseModel):
    """Result of a prediction"""
    action: int = Field(..., description="Action to take")
    action_type: str = Field(..., description="Type of action (buy, sell, hold)")
    position_size: float = Field(..., description="Position size")
    confidence: Optional[float] = Field(None, description="Confidence in the prediction")
    info: Dict[str, Any] = Field(..., description="Additional information")

class EvaluationResult(BaseModel):
    """Result of an evaluation"""
    mean_reward: float = Field(..., description="Mean reward")
    std_reward: float = Field(..., description="Standard deviation of reward")
    mean_episode_length: float = Field(..., description="Mean episode length")
    total_trades: int = Field(..., description="Total number of trades")
    profitable_trades: int = Field(..., description="Number of profitable trades")
    win_rate: float = Field(..., description="Win rate")
    additional_metrics: Optional[Dict[str, Any]] = Field(None, description="Additional metrics")

class BacktestResult(BaseModel):
    """Result of a backtest"""
    initial_balance: float = Field(..., description="Initial balance")
    final_balance: float = Field(..., description="Final balance")
    total_return: float = Field(..., description="Total return")
    total_trades: int = Field(..., description="Total number of trades")
    profitable_trades: int = Field(..., description="Number of profitable trades")
    win_rate: float = Field(..., description="Win rate")
    max_drawdown: float = Field(..., description="Maximum drawdown")
    sharpe_ratio: float = Field(..., description="Sharpe ratio")
    trades: List[Dict[str, Any]] = Field(..., description="List of trades")
    equity_curve: List[Dict[str, Any]] = Field(..., description="Equity curve")

class StatusResponse(BaseModel):
    """Response model for status"""
    status: str = Field(..., description="Status message")
    timestamp: datetime = Field(..., description="Timestamp of the response")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional details")
