#!/bin/bash

# Test script for Trading System
# This script runs unit tests for each service

echo "=== Running tests for Trading System ==="
echo

# Create test directory if it doesn't exist
mkdir -p /home/ubuntu/trading_system/tests

# Create test file for data_service
cat > /home/ubuntu/trading_system/tests/test_data_service.py << 'EOF'
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
EOF

# Create test file for model_service
cat > /home/ubuntu/trading_system/tests/test_model_service.py << 'EOF'
import unittest
import os
import sys
from unittest.mock import patch, MagicMock

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import from model_service
try:
    from model_service.models.base_model import BaseModel
    from model_service.ensemble.base_ensemble import BaseEnsemble
    IMPORTS_SUCCESSFUL = True
except ImportError:
    IMPORTS_SUCCESSFUL = False

class TestModelService(unittest.TestCase):
    """Test cases for model service"""
    
    def test_imports(self):
        """Test that imports are successful"""
        self.assertTrue(IMPORTS_SUCCESSFUL, "Failed to import modules from model_service")
    
    @unittest.skipIf(not IMPORTS_SUCCESSFUL, "Imports failed")
    def test_base_model(self):
        """Test BaseModel class"""
        model = BaseModel()
        self.assertIsNotNone(model)
        self.assertTrue(hasattr(model, 'train'), "BaseModel should have train method")
        self.assertTrue(hasattr(model, 'predict'), "BaseModel should have predict method")
    
    @unittest.skipIf(not IMPORTS_SUCCESSFUL, "Imports failed")
    def test_base_ensemble(self):
        """Test BaseEnsemble class"""
        ensemble = BaseEnsemble()
        self.assertIsNotNone(ensemble)
        self.assertTrue(hasattr(ensemble, 'train'), "BaseEnsemble should have train method")
        self.assertTrue(hasattr(ensemble, 'predict'), "BaseEnsemble should have predict method")

if __name__ == '__main__':
    unittest.main()
EOF

# Create test file for rl_service
cat > /home/ubuntu/trading_system/tests/test_rl_service.py << 'EOF'
import unittest
import os
import sys
from unittest.mock import patch, MagicMock

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import from rl_service
try:
    from rl_service.agents.base_agent import BaseAgent
    from rl_service.environments.trading_env import TradingEnvironment
    IMPORTS_SUCCESSFUL = True
except ImportError:
    IMPORTS_SUCCESSFUL = False

class TestRLService(unittest.TestCase):
    """Test cases for RL service"""
    
    def test_imports(self):
        """Test that imports are successful"""
        self.assertTrue(IMPORTS_SUCCESSFUL, "Failed to import modules from rl_service")
    
    @unittest.skipIf(not IMPORTS_SUCCESSFUL, "Imports failed")
    def test_base_agent(self):
        """Test BaseAgent class"""
        agent = BaseAgent()
        self.assertIsNotNone(agent)
        self.assertTrue(hasattr(agent, 'train'), "BaseAgent should have train method")
        self.assertTrue(hasattr(agent, 'predict'), "BaseAgent should have predict method")
    
    @unittest.skipIf(not IMPORTS_SUCCESSFUL, "Imports failed")
    @patch('rl_service.environments.trading_env.gym')
    def test_trading_environment(self, mock_gym):
        """Test TradingEnvironment class"""
        # Mock necessary components
        mock_gym.Env = MagicMock
        
        # Create environment instance
        env = TradingEnvironment()
        self.assertIsNotNone(env)
        
        # Test required methods
        self.assertTrue(hasattr(env, 'reset'), "TradingEnvironment should have reset method")
        self.assertTrue(hasattr(env, 'step'), "TradingEnvironment should have step method")

if __name__ == '__main__':
    unittest.main()
EOF

# Create test file for trading_service
cat > /home/ubuntu/trading_system/tests/test_trading_service.py << 'EOF'
import unittest
import os
import sys
from unittest.mock import patch, MagicMock

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import from trading_service
try:
    from trading_service.strategies.base_strategy import BaseStrategy
    from trading_service.execution.base_executor import BaseExecutor
    from trading_service.risk.risk_manager import RiskManager
    IMPORTS_SUCCESSFUL = True
except ImportError:
    IMPORTS_SUCCESSFUL = False

class TestTradingService(unittest.TestCase):
    """Test cases for trading service"""
    
    def test_imports(self):
        """Test that imports are successful"""
        self.assertTrue(IMPORTS_SUCCESSFUL, "Failed to import modules from trading_service")
    
    @unittest.skipIf(not IMPORTS_SUCCESSFUL, "Imports failed")
    def test_base_strategy(self):
        """Test BaseStrategy class"""
        strategy = BaseStrategy()
        self.assertIsNotNone(strategy)
        self.assertTrue(hasattr(strategy, 'generate_signal'), "BaseStrategy should have generate_signal method")
    
    @unittest.skipIf(not IMPORTS_SUCCESSFUL, "Imports failed")
    def test_base_executor(self):
        """Test BaseExecutor class"""
        executor = BaseExecutor()
        self.assertIsNotNone(executor)
        self.assertTrue(hasattr(executor, 'execute'), "BaseExecutor should have execute method")
    
    @unittest.skipIf(not IMPORTS_SUCCESSFUL, "Imports failed")
    def test_risk_manager(self):
        """Test RiskManager class"""
        risk_manager = RiskManager()
        self.assertIsNotNone(risk_manager)
        self.assertTrue(hasattr(risk_manager, 'check_risk'), "RiskManager should have check_risk method")

if __name__ == '__main__':
    unittest.main()
EOF

# Create test file for chat_service
cat > /home/ubuntu/trading_system/tests/test_chat_service.py << 'EOF'
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
EOF

# Create integration test file
cat > /home/ubuntu/trading_system/tests/test_integration.py << 'EOF'
import unittest
import os
import sys
import requests
from unittest.mock import patch

class TestIntegration(unittest.TestCase):
    """Integration tests for Trading System"""
    
    def test_docker_compose_file(self):
        """Test that docker-compose.yml exists and is valid"""
        docker_compose_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docker-compose.yml")
        self.assertTrue(os.path.exists(docker_compose_path), "docker-compose.yml should exist")
        
        # Check file size to ensure it's not empty
        self.assertGreater(os.path.getsize(docker_compose_path), 100, "docker-compose.yml should not be empty")
    
    def test_dockerfiles(self):
        """Test that Dockerfiles exist for all services"""
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        services = ["data_service", "model_service", "rl_service", "trading_service", "chat_service", "frontend"]
        
        for service in services:
            dockerfile_path = os.path.join(base_dir, service, "Dockerfile")
            self.assertTrue(os.path.exists(dockerfile_path), f"Dockerfile should exist for {service}")
            
            # Check file size to ensure it's not empty
            self.assertGreater(os.path.getsize(dockerfile_path), 50, f"Dockerfile for {service} should not be empty")
    
    def test_env_file(self):
        """Test that .env file exists and is valid"""
        env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
        self.assertTrue(os.path.exists(env_path), ".env file should exist")
        
        # Check file size to ensure it's not empty
        self.assertGreater(os.path.getsize(env_path), 100, ".env file should not be empty")
    
    def test_nginx_config(self):
        """Test that nginx.conf exists and is valid"""
        nginx_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend", "nginx.conf")
        self.assertTrue(os.path.exists(nginx_path), "nginx.conf should exist")
        
        # Check file size to ensure it's not empty
        self.assertGreater(os.path.getsize(nginx_path), 100, "nginx.conf should not be empty")
        
        # Check that it contains proxy configurations for all services
        with open(nginx_path, 'r') as f:
            content = f.read()
            self.assertIn("proxy_pass", content, "nginx.conf should contain proxy_pass directives")
            self.assertIn("data_service", content, "nginx.conf should proxy to data_service")
            self.assertIn("model_service", content, "nginx.conf should proxy to model_service")
            self.assertIn("trading_service", content, "nginx.conf should proxy to trading_service")
            self.assertIn("chat_service", content, "nginx.conf should proxy to chat_service")

if __name__ == '__main__':
    unittest.main()
EOF

# Run the tests
echo "Running unit tests for data_service..."
python3 /home/ubuntu/trading_system/tests/test_data_service.py

echo "Running unit tests for model_service..."
python3 /home/ubuntu/trading_system/tests/test_model_service.py

echo "Running unit tests for rl_service..."
python3 /home/ubuntu/trading_system/tests/test_rl_service.py

echo "Running unit tests for trading_service..."
python3 /home/ubuntu/trading_system/tests/test_trading_service.py

echo "Running unit tests for chat_service..."
python3 /home/ubuntu/trading_system/tests/test_chat_service.py

echo "Running integration tests..."
python3 /home/ubuntu/trading_system/tests/test_integration.py

echo
echo "=== Testing completed ==="
