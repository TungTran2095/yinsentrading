"""
API routes for Trading service
"""
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from typing import List, Dict, Any, Optional
import uuid
import logging
from datetime import datetime
import sys
sys.path.append('..')
from config import (
    API_PREFIX, DEFAULT_SYMBOL, DEFAULT_TIMEFRAME, 
    DEFAULT_STRATEGY_TYPE, DEFAULT_EXECUTION_MODE
)
from .models import (
    StrategyType, ExecutionMode, TradingBotConfig, CreateBotRequest, 
    UpdateBotRequest, StartBotRequest, StopBotRequest, BacktestRequest,
    Signal, Execution, Trade, Position, AccountInfo, BotStatus, BacktestResult,
    StatusResponse
)
from strategies import ( 
    BaseStrategy, EnsembleStrategy, RLStrategy, CombinedStrategy
)
from execution import (
    BaseExecutor, PaperExecutor, LiveExecutor
)
from risk import RiskManager

# Create router
router = APIRouter(prefix=API_PREFIX)

# Initialize logger
logger = logging.getLogger(__name__)

# In-memory storage for bots
bots = {}

# Helper function to create strategy
def create_strategy(config: Dict[str, Any]) -> BaseStrategy:
    """
    Create strategy based on configuration
    
    Args:
        config: Strategy configuration
        
    Returns:
        Strategy instance
    """
    strategy_type = config.get("type", DEFAULT_STRATEGY_TYPE)
    symbol = config.get("symbol", DEFAULT_SYMBOL)
    timeframe = config.get("timeframe", DEFAULT_TIMEFRAME)
    
    if strategy_type == StrategyType.ENSEMBLE:
        ensemble_id = config.get("ensemble_id")
        if not ensemble_id:
            raise ValueError("ensemble_id is required for ensemble strategy")
        
        return EnsembleStrategy(
            symbol=symbol,
            timeframe=timeframe,
            ensemble_id=ensemble_id,
            threshold_buy=config.get("threshold_buy", 0.6),
            threshold_sell=config.get("threshold_sell", 0.4),
            position_size=config.get("position_size", 1.0),
            stop_loss_pct=config.get("stop_loss_pct", 0.05),
            take_profit_pct=config.get("take_profit_pct", 0.1)
        )
    
    elif strategy_type == StrategyType.RL:
        agent_id = config.get("agent_id")
        if not agent_id:
            raise ValueError("agent_id is required for RL strategy")
        
        return RLStrategy(
            symbol=symbol,
            timeframe=timeframe,
            agent_id=agent_id,
            confidence_threshold=config.get("confidence_threshold", 0.5),
            stop_loss_pct=config.get("stop_loss_pct", 0.05),
            take_profit_pct=config.get("take_profit_pct", 0.1)
        )
    
    elif strategy_type == StrategyType.COMBINED:
        ensemble_id = config.get("ensemble_id")
        agent_id = config.get("agent_id")
        if not ensemble_id or not agent_id:
            raise ValueError("ensemble_id and agent_id are required for combined strategy")
        
        return CombinedStrategy(
            symbol=symbol,
            timeframe=timeframe,
            ensemble_id=ensemble_id,
            agent_id=agent_id,
            ensemble_weight=config.get("ensemble_weight", 0.5),
            rl_weight=config.get("rl_weight", 0.5),
            confidence_threshold=config.get("confidence_threshold", 0.6),
            position_size=config.get("position_size", 1.0),
            stop_loss_pct=config.get("stop_loss_pct", 0.05),
            take_profit_pct=config.get("take_profit_pct", 0.1)
        )
    
    else:
        raise ValueError(f"Invalid strategy type: {strategy_type}")

# Helper function to create executor
def create_executor(config: Dict[str, Any]) -> BaseExecutor:
    """
    Create executor based on configuration
    
    Args:
        config: Executor configuration
        
    Returns:
        Executor instance
    """
    mode = config.get("mode", DEFAULT_EXECUTION_MODE)
    
    if mode == ExecutionMode.PAPER:
        return PaperExecutor(
            initial_balance=config.get("initial_balance", 10000.0),
            transaction_fee=config.get("transaction_fee", 0.001),
            slippage=config.get("slippage", 0.001)
        )
    
    elif mode == ExecutionMode.LIVE:
        return LiveExecutor(
            exchange_name=config.get("exchange", "binance"),
            transaction_fee=config.get("transaction_fee", 0.001),
            slippage=config.get("slippage", 0.001)
        )
    
    else:
        raise ValueError(f"Invalid execution mode: {mode}")

# Helper function to create risk manager
def create_risk_manager(config: Dict[str, Any]) -> RiskManager:
    """
    Create risk manager based on configuration
    
    Args:
        config: Risk configuration
        
    Returns:
        RiskManager instance
    """
    if not config:
        return RiskManager()
    
    return RiskManager(
        max_position_size=config.get("max_position_size", 0.5),
        max_drawdown=config.get("max_drawdown", 0.2),
        stop_loss_pct=config.get("stop_loss_pct", 0.05),
        take_profit_pct=config.get("take_profit_pct", 0.1)
    )

# Helper function to create trading bot
def create_bot(config: TradingBotConfig) -> Dict[str, Any]:
    """
    Create trading bot based on configuration
    
    Args:
        config: Bot configuration
        
    Returns:
        Bot instance
    """
    try:
        # Generate bot ID
        bot_id = str(uuid.uuid4())
        
        # Create strategy
        strategy = create_strategy(config.strategy.dict())
        
        # Create executor
        executor = create_executor(config.executor.dict())
        
        # Create risk manager
        risk_manager = create_risk_manager(config.risk.dict() if config.risk else None)
        
        # Create bot
        bot = {
            "id": bot_id,
            "name": config.name,
            "status": "stopped",
            "strategy": strategy,
            "executor": executor,
            "risk_manager": risk_manager,
            "config": config,
            "last_signal": None,
            "last_execution": None,
            "last_update": datetime.now().isoformat(),
            "error": None
        }
        
        # Store bot
        bots[bot_id] = bot
        
        return bot
    
    except Exception as e:
        logger.error(f"Error creating bot: {e}")
        raise ValueError(f"Error creating bot: {e}")

# Helper function to get bot status
def get_bot_status(bot: Dict[str, Any]) -> BotStatus:
    """
    Get bot status
    
    Args:
        bot: Bot instance
        
    Returns:
        Bot status
    """
    # Get account info
    account_info = bot["executor"].get_account_info()
    
    # Create status
    status = BotStatus(
        id=bot["id"],
        name=bot["name"],
        status=bot["status"],
        strategy=bot["strategy"].name,
        executor=bot["executor"].name,
        symbol=bot["strategy"].symbol,
        timeframe=bot["strategy"].timeframe,
        account_info=AccountInfo(**account_info),
        last_signal=bot["last_signal"],
        last_execution=bot["last_execution"],
        last_update=bot["last_update"],
        error=bot["error"]
    )
    
    return status

# Background task for bot execution
def run_bot(bot_id: str):
    """
    Run bot in background
    
    Args:
        bot_id: Bot ID
    """
    try:
        # Get bot
        bot = bots.get(bot_id)
        if not bot:
            logger.error(f"Bot not found: {bot_id}")
            return
        
        # Update status
        bot["status"] = "running"
        bot["error"] = None
        bot["last_update"] = datetime.now().isoformat()
        
        # Get strategy and executor
        strategy = bot["strategy"]
        executor = bot["executor"]
        risk_manager = bot["risk_manager"]
        
        # Main loop
        while bot["status"] == "running":
            try:
                # Get data
                data = strategy.get_data()
                
                # Generate signal
                signal = strategy.generate_signal(data)
                bot["last_signal"] = signal
                
                # Validate signal with risk manager
                account_info = executor.get_account_info()
                is_valid, modified_signal = risk_manager.validate_signal(signal, account_info)
                
                if is_valid:
                    # Execute signal
                    execution = executor.execute_signal(modified_signal)
                    bot["last_execution"] = execution
                
                # Update last update time
                bot["last_update"] = datetime.now().isoformat()
                
                # Sleep for a while
                import time
                time.sleep(60)  # Sleep for 1 minute
            
            except Exception as e:
                logger.error(f"Error running bot {bot_id}: {e}")
                bot["error"] = str(e)
                bot["status"] = "error"
                bot["last_update"] = datetime.now().isoformat()
                break
    
    except Exception as e:
        logger.error(f"Error in run_bot: {e}")
        if bot_id in bots:
            bots[bot_id]["status"] = "error"
            bots[bot_id]["error"] = str(e)
            bots[bot_id]["last_update"] = datetime.now().isoformat()

@router.post("/bots", response_model=BotStatus)
async def create_trading_bot(request: CreateBotRequest):
    """
    Create a new trading bot
    """
    try:
        # Create bot
        bot = create_bot(request.config)
        
        # Return bot status
        return get_bot_status(bot)
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    except Exception as e:
        logger.error(f"Error creating trading bot: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")

@router.get("/bots", response_model=List[BotStatus])
async def list_trading_bots():
    """
    List all trading bots
    """
    try:
        # Get all bots
        bot_statuses = [get_bot_status(bot) for bot in bots.values()]
        
        return bot_statuses
    
    except Exception as e:
        logger.error(f"Error listing trading bots: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")

@router.get("/bots/{bot_id}", response_model=BotStatus)
async def get_trading_bot(bot_id: str):
    """
    Get trading bot by ID
    """
    try:
        # Get bot
        bot = bots.get(bot_id)
        if not bot:
            raise HTTPException(status_code=404, detail=f"Bot not found: {bot_id}")
        
        # Return bot status
        return get_bot_status(bot)
    
    except HTTPException:
        raise
    
    except Exception as e:
        logger.error(f"Error getting trading bot: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")

@router.put("/bots/{bot_id}", response_model=BotStatus)
async def update_trading_bot(bot_id: str, request: UpdateBotRequest):
    """
    Update trading bot
    """
    try:
        # Get bot
        bot = bots.get(bot_id)
        if not bot:
            raise HTTPException(status_code=404, detail=f"Bot not found: {bot_id}")
        
        # Check if bot is running
        if bot["status"] == "running":
            raise HTTPException(status_code=400, detail="Cannot update running bot")
        
        # Create new bot with updated config
        updated_bot = create_bot(request.config)
        
        # Update bot ID
        updated_bot["id"] = bot_id
        
        # Replace bot
        bots[bot_id] = updated_bot
        
        # Return bot status
        return get_bot_status(updated_bot)
    
    except HTTPException:
        raise
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    except Exception as e:
        logger.error(f"Error updating trading bot: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")

@router.post("/bots/{bot_id}/start", response_model=StatusResponse)
async def start_trading_bot(bot_id: str, background_tasks: BackgroundTasks):
    """
    Start trading bot
    """
    try:
        # Get bot
        bot = bots.get(bot_id)
        if not bot:
            raise HTTPException(status_code=404, detail=f"Bot not found: {bot_id}")
        
        # Check if bot is already running
        if bot["status"] == "running":
            return StatusResponse(
                status="success",
                timestamp=datetime.now(),
                details={"message": f"Bot {bot_id} is already running"}
            )
        
        # Start bot in background
        background_tasks.add_task(run_bot, bot_id)
        
        # Update status
        bot["status"] = "running"
        bot["error"] = None
        bot["last_update"] = datetime.now().isoformat()
        
        return StatusResponse(
            status="success",
            timestamp=datetime.now(),
            details={"message": f"Bot {bot_id} started"}
        )
    
    except HTTPException:
        raise
    
    except Exception as e:
        logger.error(f"Error starting trading bot: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")

@router.post("/bots/{bot_id}/stop", response_model=StatusResponse)
async def stop_trading_bot(bot_id: str):
    """
    Stop trading bot
    """
    try:
        # Get bot
        bot = bots.get(bot_id)
        if not bot:
            raise HTTPException(status_code=404, detail=f"Bot not found: {bot_id}")
        
        # Check if bot is already stopped
        if bot["status"] != "running":
            return StatusResponse(
                status="success",
                timestamp=datetime.now(),
                details={"message": f"Bot {bot_id} is not running"}
            )
        
        # Update status
        bot["status"] = "stopped"
        bot["last_update"] = datetime.now().isoformat()
        
        return StatusResponse(
            status="success",
            timestamp=datetime.now(),
            details={"message": f"Bot {bot_id} stopped"}
        )
    
    except HTTPException:
        raise
    
    except Exception as e:
        logger.error(f"Error stopping trading bot: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")

@router.delete("/bots/{bot_id}", response_model=StatusResponse)
async def delete_trading_bot(bot_id: str):
    """
    Delete trading bot
    """
    try:
        # Get bot
        bot = bots.get(bot_id)
        if not bot:
            raise HTTPException(status_code=404, detail=f"Bot not found: {bot_id}")
        
        # Check if bot is running
        if bot["status"] == "running":
            raise HTTPException(status_code=400, detail="Cannot delete running bot")
        
        # Delete bot
        del bots[bot_id]
        
        return StatusResponse(
            status="success",
            timestamp=datetime.now(),
            details={"message": f"Bot {bot_id} deleted"}
        )
    
    except HTTPException:
        raise
    
    except Exception as e:
        logger.error(f"Error deleting trading bot: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")

@router.post("/backtest", response_model=BacktestResult)
async def backtest_strategy(request: BacktestRequest):
    """
    Backtest a strategy
    """
    try:
        # Create strategy
        strategy = create_strategy(request.strategy.dict())
        
        # Create paper executor
        executor = PaperExecutor(
            initial_balance=request.initial_balance or 10000.0,
            transaction_fee=0.001,
            slippage=0.001
        )
        
        # Create risk manager
        risk_manager = RiskManager()
        
        # Get data
        import pandas as pd
        from datetime import datetime, timedelta
        
        # Parse start and end times
        if request.start_time:
            start_time = datetime.fromisoformat(request.start_time)
        else:
            start_time = datetime.now() - timedelta(days=30)
        
        if request.end_time:
            end_time = datetime.fromisoformat(request.end_time)
        else:
            end_time = datetime.now()
        
        # Get data from data service
        data = strategy.get_data()
        
        # Filter data by time range
        data = data[(data.index >= start_time) & (data.index <= end_time)]
        
        # Run backtest
        for _, row in data.iterrows():
            # Create data slice
            data_slice = data[data.index <= row.name]
            
            # Generate signal
            signal = strategy.generate_signal(data_slice)
            
            # Validate signal with risk manager
            account_info = executor.get_account_info()
            is_valid, modified_signal = risk_manager.validate_signal(signal, account_info)
            
            if is_valid:
                # Execute signal
                execution = executor.execute_signal(modified_signal)
        
        # Get account info
        account_info = executor.get_account_info()
        
        # Get trade history
        trades = executor.get_trade_history()
        
        # Get equity history
        equity_history = executor.get_equity_history()
        
        # Calculate risk metrics
        risk_metrics = risk_manager.calculate_risk_metrics(trades, equity_history)
        
        # Create backtest result
        result = BacktestResult(
            strategy=strategy.name,
            symbol=strategy.symbol,
            timeframe=strategy.timeframe,
            start_time=start_time.isoformat(),
            end_time=end_time.isoformat(),
            initial_balance=request.initial_balance or 10000.0,
            final_balance=account_info["equity"],
            total_return=account_info["equity"] - (request.initial_balance or 10000.0),
            total_return_pct=((account_info["equity"] - (request.initial_balance or 10000.0)) / (request.initial_balance or 10000.0)) * 100,
            annualized_return=0,  # TODO: Calculate annualized return
            max_drawdown=risk_metrics["max_drawdown"],
            max_drawdown_pct=risk_metrics["max_drawdown_percentage"],
            sharpe_ratio=risk_metrics["sharpe_ratio"],
            sortino_ratio=risk_metrics["sortino_ratio"],
            total_trades=risk_metrics["total_trades"],
            winning_trades=risk_metrics["winning_trades"],
            losing_trades=risk_metrics["losing_trades"],
            win_rate=risk_metrics["win_rate"],
            profit_factor=risk_metrics["profit_factor"],
            average_profit=risk_metrics["average_profit"],
            average_loss=risk_metrics["average_loss"],
            largest_profit=risk_metrics["largest_profit"],
            largest_loss=risk_metrics["largest_loss"],
            trades=trades,
            equity_curve=equity_history
        )
        
        return result
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    except Exception as e:
        logger.error(f"Error backtesting strategy: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")

@router.get("/health", response_model=StatusResponse)
async def health_check():
    """
    Health check endpoint
    """
    return StatusResponse(
        status="ok",
        timestamp=datetime.now(),
        details={"version": "0.1.0"}
    )
