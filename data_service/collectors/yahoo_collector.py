"""
Yahoo Finance collector for data collection
"""
import pandas as pd
import logging
import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import yfinance as yf
from .base_collector import BaseCollector
import sys
sys.path.append('..')

logger = logging.getLogger(__name__)

class YahooCollector(BaseCollector):
    """
    Collector for Yahoo Finance
    """
    
    def __init__(self, symbol: str, timeframe: str):
        """
        Initialize the Yahoo Finance collector
        
        Args:
            symbol: Trading pair symbol (e.g., "BTC-USD")
            timeframe: Timeframe for data collection (e.g., "1h", "1d")
        """
        super().__init__(symbol, timeframe)
        
        # Convert symbol format from "BTC/USDT" to "BTC-USD" for Yahoo Finance
        self.yahoo_symbol = self.symbol.replace("/", "-")
        
        # Convert timeframe format from "1h" to "1h" for Yahoo Finance
        self.timeframe_map = {
            "1m": "1m",
            "5m": "5m",
            "15m": "15m",
            "30m": "30m",
            "1h": "1h",
            "4h": "4h",
            "1d": "1d"
        }
        self.yahoo_timeframe = self.timeframe_map.get(self.timeframe)
        if not self.yahoo_timeframe:
            raise ValueError(f"Unsupported timeframe: {self.timeframe}")
        
        # Map timeframe to period and interval for Yahoo Finance
        self.interval_map = {
            "1m": "1m",
            "5m": "5m",
            "15m": "15m",
            "30m": "30m",
            "1h": "1h",
            "4h": "4h",
            "1d": "1d"
        }
        self.yahoo_interval = self.interval_map.get(self.timeframe)
    
    async def fetch_historical_data(self, limit: int = 1000, start_time: Optional[int] = None, end_time: Optional[int] = None) -> pd.DataFrame:
        """
        Fetch historical market data from Yahoo Finance
        
        Args:
            limit: Maximum number of candles to fetch
            start_time: Start timestamp in milliseconds
            end_time: End timestamp in milliseconds
            
        Returns:
            DataFrame with market data
        """
        logger.info(f"Fetching historical data for {self.yahoo_symbol} with timeframe {self.yahoo_timeframe}")
        
        # Convert timestamps to datetime
        if start_time:
            start_date = datetime.fromtimestamp(start_time / 1000)
        else:
            # Default to 1000 candles back based on timeframe
            if self.timeframe == "1m":
                start_date = datetime.now() - timedelta(minutes=1000)
            elif self.timeframe == "5m":
                start_date = datetime.now() - timedelta(minutes=5 * 1000)
            elif self.timeframe == "15m":
                start_date = datetime.now() - timedelta(minutes=15 * 1000)
            elif self.timeframe == "30m":
                start_date = datetime.now() - timedelta(minutes=30 * 1000)
            elif self.timeframe == "1h":
                start_date = datetime.now() - timedelta(hours=1000)
            elif self.timeframe == "4h":
                start_date = datetime.now() - timedelta(hours=4 * 1000)
            else:  # "1d"
                start_date = datetime.now() - timedelta(days=1000)
        
        if end_time:
            end_date = datetime.fromtimestamp(end_time / 1000)
        else:
            end_date = datetime.now()
        
        # Use yfinance to fetch data (run in a separate thread to avoid blocking)
        loop = asyncio.get_event_loop()
        df = await loop.run_in_executor(
            None,
            lambda: yf.download(
                self.yahoo_symbol,
                start=start_date,
                end=end_date,
                interval=self.yahoo_interval,
                progress=False
            )
        )
        
        # Reset index to make timestamp a column
        df = df.reset_index()
        
        # Rename columns to match our standard format
        df = df.rename(columns={
            'Date': 'timestamp',
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume'
        })
        
        # Validate and standardize DataFrame
        df = self._validate_dataframe(df)
        
        # Limit the number of rows if needed
        if limit and len(df) > limit:
            df = df.iloc[-limit:]
        
        logger.info(f"Fetched {len(df)} historical data points for {self.yahoo_symbol}")
        
        return df
    
    async def fetch_latest_data(self) -> pd.DataFrame:
        """
        Fetch latest market data from Yahoo Finance
        
        Returns:
            DataFrame with latest market data
        """
        # For Yahoo Finance, we can just fetch the last candle
        df = await self.fetch_historical_data(limit=1)
        return df
    
    async def subscribe_to_live_data(self, callback: callable):
        """
        Subscribe to live market data from Yahoo Finance
        
        Args:
            callback: Function to call when new data is received
        """
        logger.info(f"Subscribing to live data for {self.yahoo_symbol} with timeframe {self.yahoo_timeframe}")
        
        # Yahoo Finance doesn't have a WebSocket API, so we'll use polling
        while True:
            try:
                df = await self.fetch_latest_data()
                await callback(df)
                
                # Sleep based on timeframe
                sleep_seconds = {
                    "1m": 60,
                    "5m": 60 * 5,
                    "15m": 60 * 15,
                    "30m": 60 * 30,
                    "1h": 60 * 60,
                    "4h": 60 * 60 * 4,
                    "1d": 60 * 60 * 24
                }.get(self.timeframe, 60)
                
                await asyncio.sleep(sleep_seconds)
            except Exception as e:
                logger.error(f"Error in live data subscription: {e}")
                await asyncio.sleep(10)  # Wait before retrying
    
    async def unsubscribe_from_live_data(self):
        """
        Unsubscribe from live market data
        """
        # Nothing to do for Yahoo Finance
        logger.info(f"Unsubscribed from live data for {self.yahoo_symbol}")
