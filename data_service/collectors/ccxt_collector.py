"""
CCXT collector for data collection from multiple exchanges
"""
import pandas as pd
import logging
import asyncio
import ccxt.async_support as ccxt
from typing import Dict, List, Optional, Any
from datetime import datetime
from .base_collector import BaseCollector
import sys
sys.path.append('..')

logger = logging.getLogger(__name__)

class CCXTCollector(BaseCollector):
    """
    Collector for multiple exchanges using CCXT library
    """
    
    def __init__(self, symbol: str, timeframe: str, exchange_id: str = "binance"):
        """
        Initialize the CCXT collector
        
        Args:
            symbol: Trading pair symbol (e.g., "BTC/USDT")
            timeframe: Timeframe for data collection (e.g., "1h", "1d")
            exchange_id: Exchange ID (e.g., "binance", "coinbase", "kraken")
        """
        super().__init__(symbol, timeframe)
        self.exchange_id = exchange_id
        self.exchange = None
        self.ws = None
        
        # Validate timeframe
        self.timeframe_map = {
            "1m": "1m",
            "5m": "5m",
            "15m": "15m",
            "30m": "30m",
            "1h": "1h",
            "4h": "4h",
            "1d": "1d"
        }
        
        if self.timeframe not in self.timeframe_map:
            raise ValueError(f"Unsupported timeframe: {self.timeframe}")
    
    async def _init_exchange(self):
        """
        Initialize CCXT exchange
        """
        if self.exchange is None:
            # Create exchange instance
            exchange_class = getattr(ccxt, self.exchange_id)
            self.exchange = exchange_class({
                'enableRateLimit': True,
                'asyncio_loop': asyncio.get_event_loop(),
            })
            
            # Load markets
            await self.exchange.load_markets()
            
            # Check if symbol is supported
            if self.symbol not in self.exchange.symbols:
                supported_symbols = ', '.join(self.exchange.symbols[:5]) + '...'
                raise ValueError(f"Symbol {self.symbol} not supported by {self.exchange_id}. Supported symbols: {supported_symbols}")
            
            # Check if timeframe is supported
            if self.timeframe not in self.exchange.timeframes:
                supported_timeframes = ', '.join(self.exchange.timeframes)
                raise ValueError(f"Timeframe {self.timeframe} not supported by {self.exchange_id}. Supported timeframes: {supported_timeframes}")
            
            logger.info(f"Initialized CCXT exchange {self.exchange_id} for {self.symbol}")
    
    async def fetch_historical_data(self, limit: int = 1000, start_time: Optional[int] = None, end_time: Optional[int] = None) -> pd.DataFrame:
        """
        Fetch historical market data from exchange
        
        Args:
            limit: Maximum number of candles to fetch
            start_time: Start timestamp in milliseconds
            end_time: End timestamp in milliseconds
            
        Returns:
            DataFrame with market data
        """
        await self._init_exchange()
        
        logger.info(f"Fetching historical data for {self.symbol} with timeframe {self.timeframe} from {self.exchange_id}")
        
        # Fetch OHLCV data
        ohlcv = await self.exchange.fetch_ohlcv(
            symbol=self.symbol,
            timeframe=self.timeframe,
            limit=limit,
            since=start_time
        )
        
        # Convert to DataFrame
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        
        # Validate and standardize DataFrame
        df = self._validate_dataframe(df)
        
        logger.info(f"Fetched {len(df)} historical data points for {self.symbol} from {self.exchange_id}")
        
        return df
    
    async def fetch_latest_data(self) -> pd.DataFrame:
        """
        Fetch latest market data from exchange
        
        Returns:
            DataFrame with latest market data
        """
        # For CCXT, we can just fetch the last candle
        df = await self.fetch_historical_data(limit=1)
        return df
    
    async def subscribe_to_live_data(self, callback: callable):
        """
        Subscribe to live market data from exchange
        
        Args:
            callback: Function to call when new data is received
        """
        await self._init_exchange()
        
        # Check if exchange supports WebSocket
        if not hasattr(self.exchange, 'has') or not self.exchange.has['ws']:
            raise NotImplementedError(f"Exchange {self.exchange_id} does not support WebSocket")
        
        logger.info(f"Subscribing to live data for {self.symbol} with timeframe {self.timeframe} from {self.exchange_id}")
        
        # This is a simplified implementation
        # In a real system, you would need to use the exchange's WebSocket API
        # Since CCXT's WebSocket support varies by exchange, we'll use a polling approach
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
        if self.exchange:
            await self.exchange.close()
            self.exchange = None
        
        logger.info(f"Unsubscribed from live data for {self.symbol} from {self.exchange_id}")
