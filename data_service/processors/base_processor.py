"""
Base processor class for data processing
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
import pandas as pd
import logging

logger = logging.getLogger(__name__)

class BaseProcessor(ABC):
    """
    Abstract base class for all data processors
    """
    
    def __init__(self):
        """
        Initialize the processor
        """
        logger.info(f"Initialized {self.__class__.__name__}")
    
    @abstractmethod
    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process the DataFrame
        
        Args:
            df: DataFrame to process
            
        Returns:
            Processed DataFrame
        """
        pass
    
    def _validate_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate the DataFrame format
        
        Args:
            df: DataFrame to validate
            
        Returns:
            Validated DataFrame
        """
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        
        # Check if all required columns are present
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Required column '{col}' not found in DataFrame")
        
        # Ensure DataFrame is sorted by index
        df = df.sort_index()
        
        return df
