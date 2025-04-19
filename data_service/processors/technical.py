"""
Technical indicators processor for calculating technical indicators
"""
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Any
import ta
from .base_processor import BaseProcessor
import sys
sys.path.append('..')
from config import TECHNICAL_INDICATORS

logger = logging.getLogger(__name__)

class TechnicalIndicator(BaseProcessor):
    """
    Processor for calculating technical indicators
    """
    
    def __init__(self, indicators: List[str] = None):
        """
        Initialize the technical indicator processor
        
        Args:
            indicators: List of indicators to calculate, if None, use all from config
        """
        super().__init__()
        self.indicators = indicators or TECHNICAL_INDICATORS
        logger.info(f"Initialized TechnicalIndicator with indicators={self.indicators}")
    
    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate technical indicators for the DataFrame
        
        Args:
            df: DataFrame to process
            
        Returns:
            DataFrame with technical indicators
        """
        # Validate DataFrame
        df = self._validate_dataframe(df)
        
        # Make a copy to avoid modifying the original
        df = df.copy()
        
        # Calculate indicators
        for indicator in self.indicators:
            if indicator.lower() == 'rsi':
                df = self._add_rsi(df)
            elif indicator.lower() == 'macd':
                df = self._add_macd(df)
            elif indicator.lower() == 'bollinger_bands':
                df = self._add_bollinger_bands(df)
            elif indicator.lower() == 'ema':
                df = self._add_ema(df)
            else:
                logger.warning(f"Unknown indicator: {indicator}")
        
        logger.info(f"Calculated technical indicators for DataFrame with {len(df)} rows")
        
        return df
    
    def _add_rsi(self, df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
        """
        Add Relative Strength Index (RSI) to the DataFrame
        
        Args:
            df: DataFrame to process
            window: RSI window
            
        Returns:
            DataFrame with RSI
        """
        logger.info(f"Calculating RSI with window={window}")
        
        # Calculate RSI using ta library
        df['rsi'] = ta.momentum.RSIIndicator(close=df['close'], window=window).rsi()
        
        return df
    
    def _add_macd(self, df: pd.DataFrame, window_slow: int = 26, window_fast: int = 12, window_sign: int = 9) -> pd.DataFrame:
        """
        Add Moving Average Convergence Divergence (MACD) to the DataFrame
        
        Args:
            df: DataFrame to process
            window_slow: Slow window
            window_fast: Fast window
            window_sign: Signal window
            
        Returns:
            DataFrame with MACD
        """
        logger.info(f"Calculating MACD with window_slow={window_slow}, window_fast={window_fast}, window_sign={window_sign}")
        
        # Calculate MACD using ta library
        macd = ta.trend.MACD(
            close=df['close'],
            window_slow=window_slow,
            window_fast=window_fast,
            window_sign=window_sign
        )
        
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_histogram'] = macd.macd_diff()
        
        return df
    
    def _add_bollinger_bands(self, df: pd.DataFrame, window: int = 20, window_dev: int = 2) -> pd.DataFrame:
        """
        Add Bollinger Bands to the DataFrame
        
        Args:
            df: DataFrame to process
            window: Window size
            window_dev: Standard deviation multiplier
            
        Returns:
            DataFrame with Bollinger Bands
        """
        logger.info(f"Calculating Bollinger Bands with window={window}, window_dev={window_dev}")
        
        # Calculate Bollinger Bands using ta library
        bollinger = ta.volatility.BollingerBands(
            close=df['close'],
            window=window,
            window_dev=window_dev
        )
        
        df['bb_upper'] = bollinger.bollinger_hband()
        df['bb_middle'] = bollinger.bollinger_mavg()
        df['bb_lower'] = bollinger.bollinger_lband()
        
        return df
    
    def _add_ema(self, df: pd.DataFrame, windows: List[int] = [9, 21, 50, 200]) -> pd.DataFrame:
        """
        Add Exponential Moving Average (EMA) to the DataFrame
        
        Args:
            df: DataFrame to process
            windows: List of EMA windows
            
        Returns:
            DataFrame with EMA
        """
        logger.info(f"Calculating EMA with windows={windows}")
        
        # Calculate EMA for each window
        for window in windows:
            df[f'ema_{window}'] = ta.trend.EMAIndicator(close=df['close'], window=window).ema_indicator()
        
        return df
