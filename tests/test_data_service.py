import unittest
import requests
import os
import sys
import json
from unittest.mock import patch, MagicMock

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import from data_service
try:
    from data_service.collectors.base_collector import BaseCollector
    from data_service.processors.base_processor import BaseProcessor
    from data_service.storage.postgres import PostgresStorage
    IMPORTS_SUCCESSFUL = True
except ImportError:
    IMPORTS_SUCCESSFUL = False

class TestDataService(unittest.TestCase):
    """Test cases for data service"""
    
    def test_imports(self):
        """Test that imports are successful"""
        self.assertTrue(IMPORTS_SUCCESSFUL, "Failed to import modules from data_service")
    
    @unittest.skipIf(not IMPORTS_SUCCESSFUL, "Imports failed")
    def test_base_collector(self):
        """Test BaseCollector class"""
        collector = BaseCollector()
        self.assertIsNotNone(collector)
        self.assertTrue(hasattr(collector, 'collect'), "BaseCollector should have collect method")
    
    @unittest.skipIf(not IMPORTS_SUCCESSFUL, "Imports failed")
    def test_base_processor(self):
        """Test BaseProcessor class"""
        processor = BaseProcessor()
        self.assertIsNotNone(processor)
        self.assertTrue(hasattr(processor, 'process'), "BaseProcessor should have process method")
    
    @unittest.skipIf(not IMPORTS_SUCCESSFUL, "Imports failed")
    @patch('data_service.storage.postgres.psycopg2')
    def test_postgres_storage(self, mock_psycopg2):
        """Test PostgresStorage class"""
        # Mock connection and cursor
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_psycopg2.connect.return_value = mock_conn
        mock_conn.cursor.return_value = mock_cursor
        
        # Create storage instance
        storage = PostgresStorage(uri="mock_uri")
        self.assertIsNotNone(storage)
        
        # Test store method
        data = {"symbol": "BTC/USDT", "timestamp": "2023-01-01T00:00:00", "close": 50000}
        storage.store("market_data", data)
        
        # Verify that execute was called
        mock_cursor.execute.assert_called()

if __name__ == '__main__':
    unittest.main()
