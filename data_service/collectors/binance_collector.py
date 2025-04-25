"""
Binance collector for data collection
"""
import pandas as pd
import logging
import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime
from binance import AsyncClient, BinanceSocketManager
from .base_collector import BaseCollector
import sys
sys.path.append('..')
from config import BINANCE_API_KEY, BINANCE_API_SECRET

logger = logging.getLogger(__name__)

class BinanceCollector(BaseCollector):
    """
    Collector for Binance exchange
    """
    
    def __init__(self, symbol: str, timeframe: str):
        """
        Initialize the Binance collector
        
        Args:
            symbol: Trading pair symbol (e.g., "BTC/USDT")
            timeframe: Timeframe for data collection (e.g., "1h", "1d")
        """
        super().__init__(symbol, timeframe)
        self.client = None
        self.socket_manager = None
        self.socket_connection = None
        self.callback = None
        
        # Convert symbol format from "BTC/USDT" to "BTCUSDT" for Binance
        self.binance_symbol = self.symbol.replace("/", "")
        
        # Convert timeframe format from "1h" to "1h" for Binance
        self.timeframe_map = {
            "1m": "1m",
            "5m": "5m",
            "15m": "15m",
            "30m": "30m",
            "1h": "1h",
            "4h": "4h",
            "1d": "1d"
        }
        self.binance_timeframe = self.timeframe_map.get(self.timeframe)
        if not self.binance_timeframe:
            raise ValueError(f"Unsupported timeframe: {self.timeframe}")
    
    async def _init_client(self):
        """
        Initialize Binance client
        """
        if self.client is None:
            self.client = await AsyncClient.create(BINANCE_API_KEY, BINANCE_API_SECRET)
            logger.info(f"Initialized Binance client for {self.symbol}")
    
    async def fetch_historical_data(self, limit: int = 1000, start_time: Optional[int] = None, end_time: Optional[int] = None) -> pd.DataFrame:
        """
        Fetch historical market data from Binance
        
        Args:
            limit: Maximum number of candles to fetch
            start_time: Start timestamp in milliseconds
            end_time: End timestamp in milliseconds
            
        Returns:
            DataFrame with market data
        """
        await self._init_client()
        
        logger.info(f"Fetching historical data for {self.binance_symbol} with timeframe {self.binance_timeframe}")
        
        klines = await self.client.get_klines(
            symbol=self.binance_symbol,
            interval=self.binance_timeframe,
            limit=limit,
            startTime=start_time,
            endTime=end_time
        )
        
        # Convert to DataFrame
        df = pd.DataFrame(klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ])
        
        # Convert string values to float
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        # Validate and standardize DataFrame
        df = self._validate_dataframe(df)
        
        logger.info(f"Fetched {len(df)} historical data points for {self.binance_symbol}")
        
        return df
    
    async def fetch_latest_data(self) -> pd.DataFrame:
        """
        Fetch latest market data from Binance
        
        Returns:
            DataFrame with latest market data
        """
        # For Binance, we can just fetch the last candle
        df = await self.fetch_historical_data(limit=1)
        return df
    
    async def subscribe_to_live_data(self, callback: callable):
        """
        Subscribe to live market data from Binance
        
        Args:
            callback: Function to call when new data is received
        """
        await self._init_client()
        
        self.callback = callback
        self.socket_manager = BinanceSocketManager(self.client)
        
        if self.binance_timeframe == "1m":
            # For 1-minute data, we can use the kline socket
            self.socket_connection = self.socket_manager.kline_socket(
                symbol=self.binance_symbol,
                interval=self.binance_timeframe
            )
        else:
            # For other timeframes, we need to use the trade socket and aggregate
            self.socket_connection = self.socket_manager.trade_socket(self.binance_symbol)
        
        # Start the socket
        async with self.socket_connection as socket:
            while True:
                msg = await socket.recv()
                await self._process_socket_message(msg)
    
    async def _process_socket_message(self, msg: Dict):
        """
        Process socket message from Binance
        
        Args:
            msg: Socket message
        """
        if 'e' in msg and msg['e'] == 'kline':
            # Kline message
            kline = msg['k']
            
            # Create DataFrame
            data = {
                'timestamp': [pd.to_datetime(kline['t'], unit='ms')],
                'open': [float(kline['o'])],
                'high': [float(kline['h'])],
                'low': [float(kline['l'])],
                'close': [float(kline['c'])],
                'volume': [float(kline['v'])]
            }
            df = pd.DataFrame(data)
            
            # Validate and standardize DataFrame
            df = self._validate_dataframe(df)
            
            # Call the callback
            if self.callback:
                await self.callback(df)
        
        elif 'e' in msg and msg['e'] == 'trade':
            # Trade message - we need to aggregate these
            # This is a simplified implementation
            # In a real system, you would need to aggregate trades into candles
            pass
    
    async def unsubscribe_from_live_data(self):
        """
        Unsubscribe from live market data
        """
        if self.socket_connection:
            await self.socket_connection.close()
            self.socket_connection = None
        
        if self.socket_manager:
            self.socket_manager = None
        
        if self.client:
            await self.client.close_connection()
            self.client = None
        
        self.callback = None
        logger.info(f"Unsubscribed from live data for {self.binance_symbol}")
