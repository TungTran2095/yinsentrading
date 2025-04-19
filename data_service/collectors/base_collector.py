"""
Base collector class for data collection
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
import pandas as pd
import logging

logger = logging.getLogger(__name__)

class BaseCollector(ABC):
    """
    Abstract base class for all data collectors
    """
    
    def __init__(self, symbol: str, timeframe: str):
        """
        Initialize the collector
        
        Args:
            symbol: Trading pair symbol (e.g., "BTC/USDT")
            timeframe: Timeframe for data collection (e.g., "1h", "1d")
        """
        self.symbol = symbol
        self.timeframe = timeframe
        logger.info(f"Initialized {self.__class__.__name__} for {symbol} with timeframe {timeframe}")
    
    @abstractmethod
    async def fetch_historical_data(self, limit: int = 1000, start_time: Optional[int] = None, end_time: Optional[int] = None) -> pd.DataFrame:
        """
        Fetch historical market data
        
        Args:
            limit: Maximum number of candles to fetch
            start_time: Start timestamp in milliseconds
            end_time: End timestamp in milliseconds
            
        Returns:
            DataFrame with market data
        """
        pass
    
    @abstractmethod
    async def fetch_latest_data(self) -> pd.DataFrame:
        """
        Fetch latest market data
        
        Returns:
            DataFrame with latest market data
        """
        pass
    
    @abstractmethod
    async def subscribe_to_live_data(self, callback: callable):
        """
        Subscribe to live market data
        
        Args:
            callback: Function to call when new data is received
        """
        pass
    
    @abstractmethod
    async def unsubscribe_from_live_data(self):
        """
        Unsubscribe from live market data
        """
        pass
    
    def _validate_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate and standardize the DataFrame format
        
        Args:
            df: DataFrame to validate
            
        Returns:
            Standardized DataFrame
        """
        required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        
        # Check if all required columns are present
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Required column '{col}' not found in DataFrame")
        
        # Ensure timestamp is datetime
        if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        # Set timestamp as index
        df = df.set_index('timestamp')
        
        # Sort by timestamp
        df = df.sort_index()
        
        # Remove duplicates
        df = df[~df.index.duplicated(keep='last')]
        
        return df
