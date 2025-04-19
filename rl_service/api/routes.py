"""
API routes for RL service
"""
import logging
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
import requests
import json
from datetime import datetime
import os
import uuid
import sys
sys.path.append('..')
from config import (
    API_PREFIX, RL_ALGORITHMS, MODEL_SAVE_PATH, DATA_SERVICE_URL, 
    DATA_SERVICE_API_PREFIX, MODEL_SERVICE_URL, MODEL_SERVICE_API_PREFIX
)
from .models import (
    EnvironmentConfig, AgentConfig, TrainRequest, PredictRequest, EvaluateRequest,
    BacktestRequest, AgentInfo, PredictionResult, EvaluationResult, BacktestResult,
    StatusResponse
)
from environments.trading_env import TradingEnvironment
from agents import PPOAgent, DQNAgent, A2CAgent, SACAgent

logger = logging.getLogger(__name__)

router = APIRouter(prefix=API_PREFIX)

# In-memory storage for agents
# In a production system, this would be stored in a database
agents = {}

@router.post("/agents/train", response_model=AgentInfo, summary="Train an RL agent")
async def train_agent(request: TrainRequest, background_tasks: BackgroundTasks):
    """
    Train a reinforcement learning agent
    """
    try:
        # Validate agent type
        if request.agent.agent_type not in RL_ALGORITHMS:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid agent type. Available agent types: {', '.join(RL_ALGORITHMS)}"
            )
        
        # Create agent ID
        agent_id = f"{request.agent.agent_type}_{request.environment.symbol.replace('/', '_')}_{request.environment.timeframe}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Add task to background
        background_tasks.add_task(
            _train_agent_task,
            agent_id,
            request.agent.agent_type,
            request.environment,
            request.agent.hyperparameters,
            request.total_timesteps,
            request.eval_freq,
            request.n_eval_episodes
        )
        
        # Create agent info
        agent_info = AgentInfo(
            id=agent_id,
            name=request.agent.agent_type,
            type=request.agent.agent_type,
            environment=request.environment.dict(),
            hyperparameters=request.agent.hyperparameters,
            created_at=datetime.now(),
            is_trained=False
        )
        
        # Store agent info
        agents[agent_id] = agent_info
        
        return agent_info
    
    except Exception as e:
        logger.error(f"Error training agent: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/predict", response_model=PredictionResult, summary="Make a prediction with an RL agent")
async def predict(request: PredictRequest):
    """
    Make a prediction with an RL agent
    """
    try:
        # Check if agent exists
        if request.agent_id not in agents:
            raise HTTPException(
                status_code=404,
                detail=f"Agent not found: {request.agent_id}"
            )
        
        # Get agent info
        agent_info = agents[request.agent_id]
        
        # Check if agent is trained
        if not agent_info.is_trained:
            raise HTTPException(
                status_code=400,
                detail=f"Agent is not trained: {request.agent_id}"
            )
        
        # Create environment and agent
        if request.environment is not None:
            # Create environment
            env = TradingEnvironment(
                symbol=request.environment.symbol,
                timeframe=request.environment.timeframe,
                initial_balance=request.environment.initial_balance,
                max_steps=request.environment.max_steps,
                window_size=request.environment.window_size,
                commission=request.environment.commission,
                slippage=request.environment.slippage,
                use_ensemble=request.environment.use_ensemble,
                ensemble_id=request.environment.ensemble_id,
                data_start_time=request.environment.data_start_time,
                data_end_time=request.environment.data_end_time
            )
            
            # Reset environment to get observation
            observation, _ = env.reset()
        elif request.observation is not None:
            # Use provided observation
            observation = np.array(request.observation, dtype=np.float32)
            env = None
        else:
            raise HTTPException(
                status_code=400,
                detail="Either environment or observation must be provided"
            )
        
        # Create agent
        agent = _create_agent(agent_info.type, env or TradingEnvironment(**agent_info.environment))
        
        # Load agent
        agent_path = os.path.join(MODEL_SAVE_PATH, f"{agent_info.id}.zip")
        agent.load(agent_path)
        
        # Make prediction
        action, info = agent.predict(observation, deterministic=request.deterministic)
        
        # Decode action
        if env is not None:
            action_type, position_size = env._decode_action(action)
        else:
            # If no environment is provided, use default decoding
            action_idx = action // len([0.25, 0.5, 0.75, 1.0])
            size_idx = action % len([0.25, 0.5, 0.75, 1.0])
            
            action_types = ["buy", "sell", "hold"]
            position_sizes = [0.25, 0.5, 0.75, 1.0]
            
            action_type = action_types[action_idx]
            position_size = position_sizes[size_idx]
        
        # Create result
        result = PredictionResult(
            action=int(action),
            action_type=action_type,
            position_size=float(position_size),
            confidence=_calculate_confidence(info),
            info=_convert_numpy_to_python(info)
        )
        
        return result
    
    except Exception as e:
        logger.error(f"Error making prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/evaluate", response_model=EvaluationResult, summary="Evaluate an RL agent")
async def evaluate(request: EvaluateRequest):
    """
    Evaluate an RL agent
    """
    try:
        # Check if agent exists
        if request.agent_id not in agents:
            raise HTTPException(
                status_code=404,
                detail=f"Agent not found: {request.agent_id}"
            )
        
        # Get agent info
        agent_info = agents[request.agent_id]
        
        # Check if agent is trained
        if not agent_info.is_trained:
            raise HTTPException(
                status_code=400,
                detail=f"Agent is not trained: {request.agent_id}"
            )
        
        # Create environment
        env = TradingEnvironment(
            symbol=request.environment.symbol,
            timeframe=request.environment.timeframe,
            initial_balance=request.environment.initial_balance,
            max_steps=request.environment.max_steps,
            window_size=request.environment.window_size,
            commission=request.environment.commission,
            slippage=request.environment.slippage,
            use_ensemble=request.environment.use_ensemble,
            ensemble_id=request.environment.ensemble_id,
            data_start_time=request.environment.data_start_time,
            data_end_time=request.environment.data_end_time
        )
        
        # Create agent
        agent = _create_agent(agent_info.type, env)
        
        # Load agent
        agent_path = os.path.join(MODEL_SAVE_PATH, f"{agent_info.id}.zip")
        agent.load(agent_path)
        
        # Evaluate agent
        metrics = agent.evaluate(num_episodes=request.num_episodes, deterministic=request.deterministic)
        
        # Create result
        result = EvaluationResult(
            mean_reward=float(metrics["mean_reward"]),
            std_reward=float(metrics["std_reward"]),
            mean_episode_length=float(metrics["mean_episode_length"]),
            total_trades=int(metrics["total_trades"]),
            profitable_trades=int(metrics["profitable_trades"]),
            win_rate=float(metrics["win_rate"]),
            additional_metrics=_convert_numpy_to_python({k: v for k, v in metrics.items() if k not in [
                "mean_reward", "std_reward", "mean_episode_length", 
                "total_trades", "profitable_trades", "win_rate"
            ]})
        )
        
        return result
    
    except Exception as e:
        logger.error(f"Error evaluating agent: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/backtest", response_model=BacktestResult, summary="Backtest an RL agent")
async def backtest(request: BacktestRequest):
    """
    Backtest an RL agent
    """
    try:
        # Check if agent exists
        if request.agent_id not in agents:
            raise HTTPException(
                status_code=404,
                detail=f"Agent not found: {request.agent_id}"
            )
        
        # Get agent info
        agent_info = agents[request.agent_id]
        
        # Check if agent is trained
        if not agent_info.is_trained:
            raise HTTPException(
                status_code=400,
                detail=f"Agent is not trained: {request.agent_id}"
            )
        
        # Create environment
        env = TradingEnvironment(
            symbol=request.environment.symbol,
            timeframe=request.environment.timeframe,
            initial_balance=request.initial_balance,
            max_steps=request.environment.max_steps,
            window_size=request.environment.window_size,
            commission=request.environment.commission,
            slippage=request.environment.slippage,
            use_ensemble=request.environment.use_ensemble,
            ensemble_id=request.environment.ensemble_id,
            data_start_time=request.environment.data_start_time,
            data_end_time=request.environment.data_end_time
        )
        
        # Create agent
        agent = _create_agent(agent_info.type, env)
        
        # Load agent
        agent_path = os.path.join(MODEL_SAVE_PATH, f"{agent_info.id}.zip")
        agent.load(agent_path)
        
        # Run backtest
        backtest_results = _run_backtest(agent, env)
        
        return backtest_results
    
    except Exception as e:
        logger.error(f"Error backtesting agent: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/agents", response_model=List[AgentInfo], summary="Get all agents")
async def get_agents():
    """
    Get all agents
    """
    try:
        return list(agents.values())
    
    except Exception as e:
        logger.error(f"Error getting agents: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/agents/{agent_id}", response_model=AgentInfo, summary="Get an agent")
async def get_agent(agent_id: str):
    """
    Get an agent by ID
    """
    try:
        if agent_id not in agents:
            raise HTTPException(
                status_code=404,
                detail=f"Agent not found: {agent_id}"
            )
        
        return agents[agent_id]
    
    except Exception as e:
        logger.error(f"Error getting agent: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/status", summary="Get RL service status")
async def get_status():
    """
    Get RL service status
    """
    try:
        return StatusResponse(
            status="running",
            timestamp=datetime.now(),
            details={
                "available_algorithms": RL_ALGORITHMS,
                "agent_count": len(agents),
                "trained_agent_count": sum(1 for agent in agents.values() if agent.is_trained)
            }
        )
    
    except Exception as e:
        logger.error(f"Error getting status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

def _create_agent(agent_type: str, env: TradingEnvironment, **kwargs) -> Any:
    """
    Create an agent instance
    """
    if agent_type == "ppo":
        return PPOAgent(env=env, **kwargs)
    elif agent_type == "dqn":
        return DQNAgent(env=env, **kwargs)
    elif agent_type == "a2c":
        return A2CAgent(env=env, **kwargs)
    elif agent_type == "sac":
        return SACAgent(env=env, **kwargs)
    else:
        raise ValueError(f"Invalid agent type: {agent_type}")

def _calculate_confidence(info: Dict[str, Any]) -> float:
    """
    Calculate confidence from agent info
    """
    # Different agents provide different info
    if "action_probs" in info:
        # For PPO and A2C
        return float(np.max(info["action_probs"]))
    elif "q_values" in info:
        # For DQN
        q_values = info["q_values"]
        return float(np.max(q_values) / (np.sum(np.abs(q_values)) + 1e-6))
    elif "mean_actions" in info:
        # For SAC
        return 0.8  # SAC doesn't provide direct confidence, use a default value
    else:
        return 0.5  # Default confidence

def _convert_numpy_to_python(obj: Any) -> Any:
    """
    Convert numpy types to Python types for JSON serialization
    """
    if isinstance(obj, dict):
        return {k: _convert_numpy_to_python(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_convert_numpy_to_python(item) for item in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64,
                         np.uint8, np.uint16, np.uint32, np.uint64)):
        return int(obj)
    elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.bool_)):
        return bool(obj)
    else:
        return obj

def _run_backtest(agent: Any, env: TradingEnvironment) -> BacktestResult:
    """
    Run a backtest with an agent
    """
    # Reset environment
    observation, _ = env.reset()
    
    # Initialize variables
    done = False
    trades = []
    equity_curve = []
    
    # Record initial state
    equity_curve.append({
        "timestamp": env.data.index[env.current_step].isoformat(),
        "equity": env.balance,
        "position": env.position,
        "price": env.current_price
    })
    
    # Run until done
    while not done:
        # Get action from agent
        action, _ = agent.predict(observation, deterministic=True)
        
        # Take action in environment
        observation, reward, done, truncated, info = env.step(action)
        
        # Record trade if executed
        if info.get("trade_executed", False):
            trades.append({
                "timestamp": env.data.index[env.current_step].isoformat(),
                "action": info["trade_action"],
                "price": info["trade_price"],
                "amount": info["trade_amount"],
                "value": info["trade_value"],
                "fee": info["trade_fee"],
                "profit": info.get("trade_profit", 0)
            })
        
        # Record equity curve
        equity_curve.append({
            "timestamp": env.data.index[env.current_step].isoformat(),
            "equity": env.balance + env.position_value,
            "position": env.position,
            "price": env.current_price
        })
    
    # Calculate metrics
    initial_balance = env.initial_balance
    final_balance = env.balance
    total_return = (final_balance - initial_balance) / initial_balance
    
    # Calculate drawdown
    equity = [point["equity"] for point in equity_curve]
    max_drawdown = 0
    peak = equity[0]
    
    for value in equity:
        if value > peak:
            peak = value
        drawdown = (peak - value) / peak
        max_drawdown = max(max_drawdown, drawdown)
    
    # Calculate Sharpe ratio (simplified)
    returns = []
    for i in range(1, len(equity)):
        returns.append((equity[i] - equity[i-1]) / equity[i-1])
    
    sharpe_ratio = np.mean(returns) / (np.std(returns) + 1e-6) * np.sqrt(252)  # Annualized
    
    # Count profitable trades
    total_trades = len(trades)
    profitable_trades = sum(1 for trade in trades if trade["profit"] > 0)
    win_rate = profitable_trades / total_trades if total_trades > 0 else 0
    
    # Create result
    result = BacktestResult(
        initial_balance=initial_balance,
        final_balance=final_balance,
        total_return=total_return,
        total_trades=total_trades,
        profitable_trades=profitable_trades,
        win_rate=win_rate,
        max_drawdown=max_drawdown,
        sharpe_ratio=sharpe_ratio,
        trades=trades,
        equity_curve=equity_curve
    )
    
    return result

async def _train_agent_task(agent_id: str, agent_type: str, environment: EnvironmentConfig, 
                           hyperparameters: Dict[str, Any], total_timesteps: int,
                           eval_freq: int, n_eval_episodes: int):
    """
    Background task for training an agent
    """
    try:
        logger.info(f"Training agent {agent_id} of type {agent_type}")
        
        # Create environment
        env = TradingEnvironment(
            symbol=environment.symbol,
            timeframe=environment.timeframe,
            initial_balance=environment.initial_balance,
            max_steps=environment.max_steps,
            window_size=environment.window_size,
            commission=environment.commission,
            slippage=environment.slippage,
            use_ensemble=environment.use_ensemble,
            ensemble_id=environment.ensemble_id,
            data_start_time=environment.data_start_time,
            data_end_time=environment.data_end_time
        )
        
        # Create agent
        agent = _create_agent(agent_type, env, **hyperparameters)
        
        # Train agent
        metrics = agent.train(
            total_timesteps=total_timesteps,
            eval_freq=eval_freq,
            n_eval_episodes=n_eval_episodes
        )
        
        # Save agent
        agent_path = os.path.join(MODEL_SAVE_PATH, f"{agent_id}.zip")
        agent.save(agent_path)
        
        # Update agent info
        if agent_id in agents:
            agent_info = agents[agent_id]
            agent_info.is_trained = True
            agent_info.metrics = metrics
            agents[agent_id] = agent_info
        
        logger.info(f"Trained agent {agent_id} with metrics: {metrics}")
    
    except Exception as e:
        logger.error(f"Error in agent training task: {e}")
