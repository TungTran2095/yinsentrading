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
