"""
Redis storage for real-time market data
"""
import logging
import pandas as pd
import json
import redis
from datetime import datetime
from typing import Dict, List, Optional, Any
import sys
sys.path.append('..')
from config import REDIS_URL

logger = logging.getLogger(__name__)

class RedisStorage:
    """
    Storage class for Redis database
    """
    
    def __init__(self):
        """
        Initialize Redis storage
        """
        self.redis = redis.from_url(REDIS_URL)
        logger.info("Initialized Redis storage")
    
    def store_market_data(self, df: pd.DataFrame, symbol: str, timeframe: str) -> None:
        """
        Store market data in Redis
        
        Args:
            df: DataFrame with market data
            symbol: Trading pair symbol
            timeframe: Timeframe for data
        """
        if df.empty:
            logger.warning(f"Empty DataFrame for {symbol} with timeframe {timeframe}")
            return
        
        # Get the latest data point
        if isinstance(df.index, pd.DatetimeIndex):
            latest = df.iloc[-1].copy()
        else:
            latest = df.iloc[-1].copy()
            if 'timestamp' in latest:
                latest.name = latest['timestamp']
        
        # Convert timestamp to string if it's a datetime
        timestamp = latest.name
        if isinstance(timestamp, pd.Timestamp):
            timestamp = timestamp.isoformat()
        
        # Create a hash with the latest data
        data = {
            'timestamp': timestamp,
            'open': float(latest['open']),
            'high': float(latest['high']),
            'low': float(latest['low']),
            'close': float(latest['close']),
            'volume': float(latest['volume'])
        }
        
        # Add technical indicators if available
        for indicator in ['rsi', 'macd', 'macd_signal', 'macd_histogram', 
                         'bb_upper', 'bb_middle', 'bb_lower',
                         'ema_9', 'ema_21', 'ema_50', 'ema_200']:
            if indicator in latest and not pd.isna(latest[indicator]):
                data[indicator] = float(latest[indicator])
        
        # Store in Redis
        key = f"market:{symbol}:{timeframe}:latest"
        self.redis.hmset(key, data)
        
        # Set expiration (24 hours)
        self.redis.expire(key, 60 * 60 * 24)
        
        # Publish update to channel
        channel = f"updates:market:{symbol}:{timeframe}"
        self.redis.publish(channel, json.dumps(data))
        
        logger.info(f"Stored latest market data for {symbol} with timeframe {timeframe}")
    
    def get_latest_market_data(self, symbol: str, timeframe: str) -> pd.DataFrame:
        """
        Get latest market data from Redis
        
        Args:
            symbol: Trading pair symbol
            timeframe: Timeframe for data
            
        Returns:
            DataFrame with latest market data
        """
        logger.info(f"Getting latest market data for {symbol} with timeframe {timeframe}")
        
        # Get data from Redis
        key = f"market:{symbol}:{timeframe}:latest"
        data = self.redis.hgetall(key)
        
        if not data:
            logger.warning(f"No data found for {symbol} with timeframe {timeframe}")
            return pd.DataFrame()
        
        # Convert to DataFrame
        df_data = {}
        for k, v in data.items():
            if k.decode() == 'timestamp':
                df_data[k.decode()] = [v.decode()]
            else:
                df_data[k.decode()] = [float(v)]
        
        df = pd.DataFrame(df_data)
        
        # Convert timestamp to datetime
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.set_index('timestamp')
        
        logger.info(f"Got latest market data for {symbol} with timeframe {timeframe}")
        
        return df
    
    def store_prediction(self, model: str, symbol: str, timeframe: str, value: float, confidence: float = None) -> None:
        """
        Store model prediction in Redis
        
        Args:
            model: Model name
            symbol: Trading pair symbol
            timeframe: Timeframe for data
            value: Prediction value
            confidence: Prediction confidence
        """
        # Create a hash with the prediction
        data = {
            'timestamp': datetime.now().isoformat(),
            'value': float(value)
        }
        
        if confidence is not None:
            data['confidence'] = float(confidence)
        
        # Store in Redis
        key = f"prediction:{model}:{symbol}:{timeframe}:latest"
        self.redis.hmset(key, data)
        
        # Set expiration (24 hours)
        self.redis.expire(key, 60 * 60 * 24)
        
        # Publish update to channel
        channel = f"updates:prediction:{symbol}:{timeframe}"
        self.redis.publish(channel, json.dumps(data))
        
        logger.info(f"Stored prediction for {symbol} with timeframe {timeframe} from model {model}")
    
    def get_latest_prediction(self, model: str, symbol: str, timeframe: str) -> Dict:
        """
        Get latest prediction from Redis
        
        Args:
            model: Model name
            symbol: Trading pair symbol
            timeframe: Timeframe for data
            
        Returns:
            Dictionary with prediction data
        """
        logger.info(f"Getting latest prediction for {symbol} with timeframe {timeframe} from model {model}")
        
        # Get data from Redis
        key = f"prediction:{model}:{symbol}:{timeframe}:latest"
        data = self.redis.hgetall(key)
        
        if not data:
            logger.warning(f"No prediction found for {symbol} with timeframe {timeframe} from model {model}")
            return {}
        
        # Convert to dictionary
        result = {}
        for k, v in data.items():
            if k.decode() == 'timestamp':
                result[k.decode()] = v.decode()
            else:
                result[k.decode()] = float(v)
        
        logger.info(f"Got latest prediction for {symbol} with timeframe {timeframe} from model {model}")
        
        return result
    
    def close(self):
        """
        Close Redis connection
        """
        self.redis.close()
        logger.info("Closed Redis connection")
