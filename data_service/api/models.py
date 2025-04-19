"""
API models for data service
"""
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime

class MarketDataRequest(BaseModel):
    """Request model for market data"""
    symbol: str = Field(..., description="Trading pair symbol (e.g., 'BTC/USDT')")
    timeframe: str = Field(..., description="Timeframe for data (e.g., '1h', '1d')")
    limit: Optional[int] = Field(1000, description="Maximum number of candles to fetch")
    start_time: Optional[datetime] = Field(None, description="Start timestamp")
    end_time: Optional[datetime] = Field(None, description="End timestamp")

class CollectDataRequest(BaseModel):
    """Request model for collecting data"""
    symbol: str = Field(..., description="Trading pair symbol (e.g., 'BTC/USDT')")
    timeframe: str = Field(..., description="Timeframe for data (e.g., '1h', '1d')")
    source: str = Field("binance", description="Data source (e.g., 'binance', 'ccxt', 'yahoo')")
    exchange: Optional[str] = Field(None, description="Exchange name for CCXT (e.g., 'coinbase', 'kraken')")
    limit: Optional[int] = Field(1000, description="Maximum number of candles to fetch")
    start_time: Optional[datetime] = Field(None, description="Start timestamp")
    end_time: Optional[datetime] = Field(None, description="End timestamp")
    calculate_indicators: Optional[bool] = Field(True, description="Whether to calculate technical indicators")
    indicators: Optional[List[str]] = Field(None, description="List of indicators to calculate")

class IndicatorRequest(BaseModel):
    """Request model for technical indicators"""
    symbol: str = Field(..., description="Trading pair symbol (e.g., 'BTC/USDT')")
    timeframe: str = Field(..., description="Timeframe for data (e.g., '1h', '1d')")
    indicators: List[str] = Field(..., description="List of indicators to calculate")
    limit: Optional[int] = Field(1000, description="Maximum number of candles to fetch")
    start_time: Optional[datetime] = Field(None, description="Start timestamp")
    end_time: Optional[datetime] = Field(None, description="End timestamp")

class StatusResponse(BaseModel):
    """Response model for status"""
    status: str = Field(..., description="Status message")
    timestamp: datetime = Field(..., description="Timestamp of the response")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional details")
