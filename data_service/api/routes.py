"""
API routes for data service
"""
import logging
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from typing import List, Dict, Any, Optional
import pandas as pd
import asyncio
from datetime import datetime
import sys
sys.path.append('..')
from config import API_PREFIX, AVAILABLE_TIMEFRAMES, TECHNICAL_INDICATORS
from .models import MarketDataRequest, CollectDataRequest, IndicatorRequest, StatusResponse
from collectors import BinanceCollector, CCXTCollector, YahooCollector
from processors import DataCleaner, TechnicalIndicator
from storage import PostgresStorage, RedisStorage

logger = logging.getLogger(__name__)

router = APIRouter(prefix=API_PREFIX)

# Initialize storage
postgres = PostgresStorage()
redis = RedisStorage()

@router.get("/market/{symbol}/{timeframe}", summary="Get market data")
async def get_market_data(
    symbol: str,
    timeframe: str,
    limit: int = 1000,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None
):
    """
    Get market data for a symbol and timeframe
    """
    try:
        # Validate timeframe
        if timeframe not in AVAILABLE_TIMEFRAMES:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid timeframe. Available timeframes: {', '.join(AVAILABLE_TIMEFRAMES)}"
            )
        
        # Try to get data from Redis first (latest data point)
        latest_df = redis.get_latest_market_data(symbol, timeframe)
        
        # Get historical data from PostgreSQL
        historical_df = postgres.get_market_data(symbol, timeframe, limit, start_time, end_time)
        
        # Combine data
        if not latest_df.empty and not historical_df.empty:
            # Check if latest data is already in historical data
            if latest_df.index[0] not in historical_df.index:
                df = pd.concat([historical_df, latest_df])
            else:
                df = historical_df
        elif not latest_df.empty:
            df = latest_df
        else:
            df = historical_df
        
        # Convert to dict for JSON response
        if df.empty:
            return {"data": []}
        
        # Reset index to include timestamp in the response
        df = df.reset_index()
        
        # Convert to dict
        result = df.to_dict(orient="records")
        
        return {"data": result}
    
    except Exception as e:
        logger.error(f"Error getting market data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/technical/{symbol}/{timeframe}", summary="Get technical indicators")
async def get_technical_indicators(
    symbol: str,
    timeframe: str,
    limit: int = 1000,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None
):
    """
    Get technical indicators for a symbol and timeframe
    """
    try:
        # Validate timeframe
        if timeframe not in AVAILABLE_TIMEFRAMES:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid timeframe. Available timeframes: {', '.join(AVAILABLE_TIMEFRAMES)}"
            )
        
        # Get data from PostgreSQL
        df = postgres.get_technical_indicators(symbol, timeframe, limit, start_time, end_time)
        
        # Convert to dict for JSON response
        if df.empty:
            return {"data": []}
        
        # Reset index to include timestamp in the response
        df = df.reset_index()
        
        # Convert to dict
        result = df.to_dict(orient="records")
        
        return {"data": result}
    
    except Exception as e:
        logger.error(f"Error getting technical indicators: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/collect", summary="Collect market data")
async def collect_data(request: CollectDataRequest, background_tasks: BackgroundTasks):
    """
    Collect market data from a source
    """
    try:
        # Validate timeframe
        if request.timeframe not in AVAILABLE_TIMEFRAMES:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid timeframe. Available timeframes: {', '.join(AVAILABLE_TIMEFRAMES)}"
            )
        
        # Add task to background
        background_tasks.add_task(
            _collect_data_task,
            request.symbol,
            request.timeframe,
            request.source,
            request.exchange,
            request.limit,
            request.start_time,
            request.end_time,
            request.calculate_indicators,
            request.indicators
        )
        
        return StatusResponse(
            status="success",
            timestamp=datetime.now(),
            details={
                "message": f"Data collection for {request.symbol} with timeframe {request.timeframe} started in the background"
            }
        )
    
    except Exception as e:
        logger.error(f"Error collecting data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/indicators", summary="Calculate technical indicators")
async def calculate_indicators(request: IndicatorRequest, background_tasks: BackgroundTasks):
    """
    Calculate technical indicators for existing market data
    """
    try:
        # Validate timeframe
        if request.timeframe not in AVAILABLE_TIMEFRAMES:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid timeframe. Available timeframes: {', '.join(AVAILABLE_TIMEFRAMES)}"
            )
        
        # Validate indicators
        for indicator in request.indicators:
            if indicator not in TECHNICAL_INDICATORS:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid indicator: {indicator}. Available indicators: {', '.join(TECHNICAL_INDICATORS)}"
                )
        
        # Add task to background
        background_tasks.add_task(
            _calculate_indicators_task,
            request.symbol,
            request.timeframe,
            request.indicators,
            request.limit,
            request.start_time,
            request.end_time
        )
        
        return StatusResponse(
            status="success",
            timestamp=datetime.now(),
            details={
                "message": f"Technical indicators calculation for {request.symbol} with timeframe {request.timeframe} started in the background"
            }
        )
    
    except Exception as e:
        logger.error(f"Error calculating indicators: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/status", summary="Get data service status")
async def get_status():
    """
    Get data service status
    """
    try:
        return StatusResponse(
            status="running",
            timestamp=datetime.now(),
            details={
                "available_timeframes": AVAILABLE_TIMEFRAMES,
                "available_indicators": TECHNICAL_INDICATORS
            }
        )
    
    except Exception as e:
        logger.error(f"Error getting status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def _collect_data_task(
    symbol: str,
    timeframe: str,
    source: str,
    exchange: Optional[str],
    limit: int,
    start_time: Optional[datetime],
    end_time: Optional[datetime],
    calculate_indicators: bool,
    indicators: Optional[List[str]]
):
    """
    Background task for collecting data
    """
    try:
        logger.info(f"Starting data collection for {symbol} with timeframe {timeframe} from {source}")
        
        # Initialize collector based on source
        if source.lower() == "binance":
            collector = BinanceCollector(symbol, timeframe)
        elif source.lower() == "ccxt":
            if not exchange:
                exchange = "binance"
            collector = CCXTCollector(symbol, timeframe, exchange)
        elif source.lower() == "yahoo":
            collector = YahooCollector(symbol, timeframe)
        else:
            logger.error(f"Invalid source: {source}")
            return
        
        # Convert datetime to timestamp if provided
        start_timestamp = int(start_time.timestamp() * 1000) if start_time else None
        end_timestamp = int(end_time.timestamp() * 1000) if end_time else None
        
        # Fetch historical data
        df = await collector.fetch_historical_data(limit, start_timestamp, end_timestamp)
        
        # Clean data
        cleaner = DataCleaner(fill_missing=True, remove_outliers=True)
        df = cleaner.process(df)
        
        # Calculate technical indicators if requested
        if calculate_indicators:
            tech_indicator = TechnicalIndicator(indicators)
            df = tech_indicator.process(df)
        
        # Store data in PostgreSQL
        postgres.store_market_data(df, symbol, timeframe)
        
        # Store latest data in Redis
        redis.store_market_data(df, symbol, timeframe)
        
        # Store technical indicators in PostgreSQL if calculated
        if calculate_indicators:
            postgres.store_technical_indicators(df, symbol, timeframe)
        
        logger.info(f"Completed data collection for {symbol} with timeframe {timeframe} from {source}")
    
    except Exception as e:
        logger.error(f"Error in data collection task: {e}")

async def _calculate_indicators_task(
    symbol: str,
    timeframe: str,
    indicators: List[str],
    limit: int,
    start_time: Optional[datetime],
    end_time: Optional[datetime]
):
    """
    Background task for calculating technical indicators
    """
    try:
        logger.info(f"Starting technical indicators calculation for {symbol} with timeframe {timeframe}")
        
        # Get market data from PostgreSQL
        df = postgres.get_market_data(symbol, timeframe, limit, start_time, end_time)
        
        if df.empty:
            logger.warning(f"No market data found for {symbol} with timeframe {timeframe}")
            return
        
        # Calculate technical indicators
        tech_indicator = TechnicalIndicator(indicators)
        df = tech_indicator.process(df)
        
        # Store technical indicators in PostgreSQL
        postgres.store_technical_indicators(df, symbol, timeframe)
        
        # Store latest data in Redis
        redis.store_market_data(df, symbol, timeframe)
        
        logger.info(f"Completed technical indicators calculation for {symbol} with timeframe {timeframe}")
    
    except Exception as e:
        logger.error(f"Error in technical indicators calculation task: {e}")
