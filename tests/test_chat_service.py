import unittest
import os
import sys
from unittest.mock import patch, MagicMock

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import from chat_service
try:
    from chat_service.services.intent import IntentRecognitionService
    from chat_service.services.response import ResponseService
    from chat_service.services.database import MongoDBService
    IMPORTS_SUCCESSFUL = True
except ImportError:
    IMPORTS_SUCCESSFUL = False

class TestChatService(unittest.TestCase):
    """Test cases for chat service"""
    
    def test_imports(self):
        """Test that imports are successful"""
        self.assertTrue(IMPORTS_SUCCESSFUL, "Failed to import modules from chat_service")
    
    @unittest.skipIf(not IMPORTS_SUCCESSFUL, "Imports failed")
    @patch('chat_service.services.intent.SentenceTransformer')
    @patch('chat_service.services.intent.spacy')
    def test_intent_recognition_service(self, mock_spacy, mock_transformer):
        """Test IntentRecognitionService class"""
        # Mock necessary components
        mock_spacy.load.return_value = MagicMock()
        mock_transformer.return_value = MagicMock()
        
        # Create service instance
        service = IntentRecognitionService()
        self.assertIsNotNone(service)
        
        # Test required methods
        self.assertTrue(hasattr(service, 'recognize_intent'), "IntentRecognitionService should have recognize_intent method")
    
    @unittest.skipIf(not IMPORTS_SUCCESSFUL, "Imports failed")
    def test_response_service(self):
        """Test ResponseService class"""
        service = ResponseService()
        self.assertIsNotNone(service)
        self.assertTrue(hasattr(service, 'generate_response'), "ResponseService should have generate_response method")
    
    @unittest.skipIf(not IMPORTS_SUCCESSFUL, "Imports failed")
    @patch('chat_service.services.database.MongoClient')
    def test_mongodb_service(self, mock_mongo_client):
        """Test MongoDBService class"""
        # Mock necessary components
        mock_mongo_client.return_value = MagicMock()
        
        # Create service instance
        service = MongoDBService()
        self.assertIsNotNone(service)
        
        # Test required methods
        self.assertTrue(hasattr(service, 'create_conversation'), "MongoDBService should have create_conversation method")
        self.assertTrue(hasattr(service, 'add_message'), "MongoDBService should have add_message method")

if __name__ == '__main__':
    unittest.main()
