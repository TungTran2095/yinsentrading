"""
PostgreSQL storage for market data
"""
import logging
import pandas as pd
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, MetaData, Table, select, insert, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import sys
sys.path.append('..')
from config import DATABASE_URL

logger = logging.getLogger(__name__)

Base = declarative_base()

class PostgresStorage:
    """
    Storage class for PostgreSQL database
    """
    
    def __init__(self):
        """
        Initialize PostgreSQL storage
        """
        self.engine = create_engine(DATABASE_URL)
        self.metadata = MetaData()
        self.Session = sessionmaker(bind=self.engine)
        self.session = self.Session()
        
        # Define tables
        self.market_data = Table(
            'market_data',
            self.metadata,
            Column('id', Integer, primary_key=True),
            Column('symbol', String(20), nullable=False),
            Column('timestamp', DateTime, nullable=False),
            Column('open', Float, nullable=False),
            Column('high', Float, nullable=False),
            Column('low', Float, nullable=False),
            Column('close', Float, nullable=False),
            Column('volume', Float, nullable=False),
            Column('timeframe', String(10), nullable=False),
        )
        
        self.technical_indicators = Table(
            'technical_indicators',
            self.metadata,
            Column('id', Integer, primary_key=True),
            Column('market_data_id', Integer, nullable=False),
            Column('rsi', Float),
            Column('macd', Float),
            Column('macd_signal', Float),
            Column('macd_histogram', Float),
            Column('bb_upper', Float),
            Column('bb_middle', Float),
            Column('bb_lower', Float),
            Column('ema_short', Float),
            Column('ema_medium', Float),
            Column('ema_long', Float),
        )
        
        # Create tables if they don't exist
        self.metadata.create_all(self.engine)
        
        logger.info("Initialized PostgreSQL storage")
    
    def store_market_data(self, df: pd.DataFrame, symbol: str, timeframe: str) -> None:
        """
        Store market data in PostgreSQL
        
        Args:
            df: DataFrame with market data
            symbol: Trading pair symbol
            timeframe: Timeframe for data
        """
        logger.info(f"Storing {len(df)} market data points for {symbol} with timeframe {timeframe}")
        
        # Reset index to make timestamp a column if it's the index
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index()
        
        # Prepare data for insertion
        data = []
        for _, row in df.iterrows():
            data.append({
                'symbol': symbol,
                'timestamp': row['timestamp'],
                'open': float(row['open']),
                'high': float(row['high']),
                'low': float(row['low']),
                'close': float(row['close']),
                'volume': float(row['volume']),
                'timeframe': timeframe
            })
        
        # Insert data
        if data:
            with self.engine.begin() as conn:
                conn.execute(
                    self.market_data.insert().prefix_with('OR REPLACE'),
                    data
                )
        
        logger.info(f"Stored {len(data)} market data points for {symbol}")
    
    def store_technical_indicators(self, df: pd.DataFrame, symbol: str, timeframe: str) -> None:
        """
        Store technical indicators in PostgreSQL
        
        Args:
            df: DataFrame with technical indicators
            symbol: Trading pair symbol
            timeframe: Timeframe for data
        """
        logger.info(f"Storing technical indicators for {symbol} with timeframe {timeframe}")
        
        # Reset index to make timestamp a column if it's the index
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index()
        
        # Get market data IDs
        market_data_ids = {}
        with self.engine.begin() as conn:
            for _, row in df.iterrows():
                timestamp = row['timestamp']
                result = conn.execute(
                    select([self.market_data.c.id])
                    .where(self.market_data.c.symbol == symbol)
                    .where(self.market_data.c.timestamp == timestamp)
                    .where(self.market_data.c.timeframe == timeframe)
                ).fetchone()
                
                if result:
                    market_data_ids[timestamp] = result[0]
        
        # Prepare data for insertion
        data = []
        for _, row in df.iterrows():
            timestamp = row['timestamp']
            if timestamp in market_data_ids:
                indicator_data = {
                    'market_data_id': market_data_ids[timestamp],
                    'rsi': float(row['rsi']) if 'rsi' in row else None,
                    'macd': float(row['macd']) if 'macd' in row else None,
                    'macd_signal': float(row['macd_signal']) if 'macd_signal' in row else None,
                    'macd_histogram': float(row['macd_histogram']) if 'macd_histogram' in row else None,
                    'bb_upper': float(row['bb_upper']) if 'bb_upper' in row else None,
                    'bb_middle': float(row['bb_middle']) if 'bb_middle' in row else None,
                    'bb_lower': float(row['bb_lower']) if 'bb_lower' in row else None,
                    'ema_short': float(row['ema_9']) if 'ema_9' in row else None,
                    'ema_medium': float(row['ema_21']) if 'ema_21' in row else None,
                    'ema_long': float(row['ema_50']) if 'ema_50' in row else None,
                }
                data.append(indicator_data)
        
        # Insert data
        if data:
            with self.engine.begin() as conn:
                conn.execute(
                    self.technical_indicators.insert().prefix_with('OR REPLACE'),
                    data
                )
        
        logger.info(f"Stored {len(data)} technical indicators for {symbol}")
    
    def get_market_data(self, symbol: str, timeframe: str, limit: int = 1000, start_time: datetime = None, end_time: datetime = None) -> pd.DataFrame:
        """
        Get market data from PostgreSQL
        
        Args:
            symbol: Trading pair symbol
            timeframe: Timeframe for data
            limit: Maximum number of candles to fetch
            start_time: Start timestamp
            end_time: End timestamp
            
        Returns:
            DataFrame with market data
        """
        logger.info(f"Getting market data for {symbol} with timeframe {timeframe}")
        
        # Build query
        query = select([self.market_data]).where(
            self.market_data.c.symbol == symbol
        ).where(
            self.market_data.c.timeframe == timeframe
        )
        
        if start_time:
            query = query.where(self.market_data.c.timestamp >= start_time)
        
        if end_time:
            query = query.where(self.market_data.c.timestamp <= end_time)
        
        query = query.order_by(self.market_data.c.timestamp.desc()).limit(limit)
        
        # Execute query
        with self.engine.begin() as conn:
            result = conn.execute(query).fetchall()
        
        # Convert to DataFrame
        df = pd.DataFrame(result, columns=self.market_data.columns.keys())
        
        # Set timestamp as index
        df = df.set_index('timestamp')
        
        logger.info(f"Got {len(df)} market data points for {symbol}")
        
        return df
    
    def get_technical_indicators(self, symbol: str, timeframe: str, limit: int = 1000, start_time: datetime = None, end_time: datetime = None) -> pd.DataFrame:
        """
        Get technical indicators from PostgreSQL
        
        Args:
            symbol: Trading pair symbol
            timeframe: Timeframe for data
            limit: Maximum number of candles to fetch
            start_time: Start timestamp
            end_time: End timestamp
            
        Returns:
            DataFrame with technical indicators
        """
        logger.info(f"Getting technical indicators for {symbol} with timeframe {timeframe}")
        
        # Build query
        query = text("""
            SELECT 
                md.timestamp, md.open, md.high, md.low, md.close, md.volume,
                ti.rsi, ti.macd, ti.macd_signal, ti.macd_histogram,
                ti.bb_upper, ti.bb_middle, ti.bb_lower,
                ti.ema_short, ti.ema_medium, ti.ema_long
            FROM 
                market_data md
            LEFT JOIN 
                technical_indicators ti ON md.id = ti.market_data_id
            WHERE 
                md.symbol = :symbol AND md.timeframe = :timeframe
        """)
        
        params = {'symbol': symbol, 'timeframe': timeframe}
        
        if start_time:
            query = query.text(query.text + " AND md.timestamp >= :start_time")
            params['start_time'] = start_time
        
        if end_time:
            query = query.text(query.text + " AND md.timestamp <= :end_time")
            params['end_time'] = end_time
        
        query = query.text(query.text + " ORDER BY md.timestamp DESC LIMIT :limit")
        params['limit'] = limit
        
        # Execute query
        with self.engine.begin() as conn:
            result = conn.execute(query, params).fetchall()
        
        # Convert to DataFrame
        columns = [
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'rsi', 'macd', 'macd_signal', 'macd_histogram',
            'bb_upper', 'bb_middle', 'bb_lower',
            'ema_short', 'ema_medium', 'ema_long'
        ]
        df = pd.DataFrame(result, columns=columns)
        
        # Set timestamp as index
        df = df.set_index('timestamp')
        
        logger.info(f"Got {len(df)} technical indicators for {symbol}")
        
        return df
    
    def close(self):
        """
        Close database connection
        """
        self.session.close()
        logger.info("Closed PostgreSQL connection")
