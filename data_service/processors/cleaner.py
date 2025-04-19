"""
Data cleaner processor for cleaning raw data
"""
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Any
from .base_processor import BaseProcessor

logger = logging.getLogger(__name__)

class DataCleaner(BaseProcessor):
    """
    Processor for cleaning raw market data
    """
    
    def __init__(self, fill_missing: bool = True, remove_outliers: bool = True, normalize: bool = False):
        """
        Initialize the data cleaner
        
        Args:
            fill_missing: Whether to fill missing values
            remove_outliers: Whether to remove outliers
            normalize: Whether to normalize data
        """
        super().__init__()
        self.fill_missing = fill_missing
        self.remove_outliers = remove_outliers
        self.normalize = normalize
        logger.info(f"Initialized DataCleaner with fill_missing={fill_missing}, remove_outliers={remove_outliers}, normalize={normalize}")
    
    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean the DataFrame
        
        Args:
            df: DataFrame to clean
            
        Returns:
            Cleaned DataFrame
        """
        # Validate DataFrame
        df = self._validate_dataframe(df)
        
        # Make a copy to avoid modifying the original
        df = df.copy()
        
        # Fill missing values
        if self.fill_missing:
            df = self._fill_missing_values(df)
        
        # Remove outliers
        if self.remove_outliers:
            df = self._remove_outliers(df)
        
        # Normalize data
        if self.normalize:
            df = self._normalize_data(df)
        
        logger.info(f"Cleaned DataFrame with {len(df)} rows")
        
        return df
    
    def _fill_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fill missing values in the DataFrame
        
        Args:
            df: DataFrame to fill
            
        Returns:
            DataFrame with filled values
        """
        # Check for missing values
        missing_count = df.isna().sum().sum()
        if missing_count > 0:
            logger.info(f"Filling {missing_count} missing values")
            
            # Fill missing values in OHLC with forward fill
            for col in ['open', 'high', 'low', 'close']:
                df[col] = df[col].fillna(method='ffill')
            
            # Fill any remaining missing values with backward fill
            for col in ['open', 'high', 'low', 'close']:
                df[col] = df[col].fillna(method='bfill')
            
            # Fill missing volume with 0
            df['volume'] = df['volume'].fillna(0)
        
        return df
    
    def _remove_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove outliers from the DataFrame
        
        Args:
            df: DataFrame to process
            
        Returns:
            DataFrame without outliers
        """
        # Use rolling median and standard deviation to identify outliers
        # This is a simple approach, more sophisticated methods could be used
        
        # Calculate rolling median and standard deviation
        window = 20
        if len(df) < window:
            logger.warning(f"DataFrame has less than {window} rows, skipping outlier removal")
            return df
        
        for col in ['open', 'high', 'low', 'close']:
            rolling_median = df[col].rolling(window=window, center=True).median()
            rolling_std = df[col].rolling(window=window, center=True).std()
            
            # Define outlier bounds (3 standard deviations from median)
            lower_bound = rolling_median - 3 * rolling_std
            upper_bound = rolling_median + 3 * rolling_std
            
            # Identify outliers
            outliers = (df[col] < lower_bound) | (df[col] > upper_bound)
            outlier_count = outliers.sum()
            
            if outlier_count > 0:
                logger.info(f"Replacing {outlier_count} outliers in {col}")
                
                # Replace outliers with rolling median
                df.loc[outliers, col] = rolling_median[outliers]
        
        return df
    
    def _normalize_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize data in the DataFrame
        
        Args:
            df: DataFrame to normalize
            
        Returns:
            Normalized DataFrame
        """
        # Create a new DataFrame for normalized values
        df_norm = df.copy()
        
        # Normalize OHLC using min-max scaling
        for col in ['open', 'high', 'low', 'close']:
            min_val = df[col].min()
            max_val = df[col].max()
            
            if max_val > min_val:
                df_norm[col] = (df[col] - min_val) / (max_val - min_val)
            else:
                df_norm[col] = df[col]
        
        # Normalize volume
        min_vol = df['volume'].min()
        max_vol = df['volume'].max()
        
        if max_vol > min_vol:
            df_norm['volume'] = (df['volume'] - min_vol) / (max_vol - min_vol)
        
        # Add normalization parameters to the DataFrame as attributes
        df_norm.attrs['normalization'] = {
            'open': {'min': float(min_val), 'max': float(max_val)},
            'high': {'min': float(min_val), 'max': float(max_val)},
            'low': {'min': float(min_val), 'max': float(max_val)},
            'close': {'min': float(min_val), 'max': float(max_val)},
            'volume': {'min': float(min_vol), 'max': float(max_vol)}
        }
        
        logger.info(f"Normalized DataFrame with {len(df_norm)} rows")
        
        return df_norm
