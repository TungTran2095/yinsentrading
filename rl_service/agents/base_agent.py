"""
Base agent class for Reinforcement Learning
"""
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
import gymnasium as gym
import logging
from typing import Dict, List, Optional, Any, Tuple
import os
import sys
sys.path.append('..')
from config import MODEL_SAVE_PATH, RL_ALGORITHMS
from environments.trading_env import TradingEnvironment

logger = logging.getLogger(__name__)

class BaseRLAgent(ABC):
    """
    Abstract base class for all RL agents
    """
    
    def __init__(self, name: str, env: TradingEnvironment):
        """
        Initialize the RL agent
        
        Args:
            name: Agent name
            env: Trading environment
        """
        self.name = name
        self.env = env
        self.model = None
        self.is_trained = False
        
        # Create model save directory if it doesn't exist
        os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
        
        logger.info(f"Initialized {self.__class__.__name__} agent")
    
    @abstractmethod
    def train(self, total_timesteps: int, **kwargs) -> Dict[str, Any]:
        """
        Train the agent
        
        Args:
            total_timesteps: Total number of timesteps to train for
            **kwargs: Additional training parameters
            
        Returns:
            Dictionary with training metrics
        """
        pass
    
    @abstractmethod
    def predict(self, observation: np.ndarray, **kwargs) -> Tuple[int, Dict[str, Any]]:
        """
        Make a prediction (choose an action) based on the observation
        
        Args:
            observation: Current observation
            **kwargs: Additional prediction parameters
            
        Returns:
            Tuple of (action, info)
        """
        pass
    
    @abstractmethod
    def evaluate(self, num_episodes: int = 10) -> Dict[str, Any]:
        """
        Evaluate the agent
        
        Args:
            num_episodes: Number of episodes to evaluate for
            
        Returns:
            Dictionary with evaluation metrics
        """
        pass
    
    def save(self, path: Optional[str] = None) -> str:
        """
        Save the agent to disk
        
        Args:
            path: Path to save to (optional)
            
        Returns:
            Path to saved agent
        """
        if not self.is_trained:
            raise ValueError("Agent must be trained before saving")
        
        if path is None:
            # Create default path
            path = os.path.join(MODEL_SAVE_PATH, f"{self.name}_{self.env.symbol.replace('/', '_')}_{self.env.timeframe}.zip")
        
        # Save model
        self._save_model(path)
        
        logger.info(f"Saved agent to {path}")
        
        return path
    
    def load(self, path: str) -> None:
        """
        Load the agent from disk
        
        Args:
            path: Path to load from
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Agent file not found: {path}")
        
        # Load model
        self._load_model(path)
        
        self.is_trained = True
        
        logger.info(f"Loaded agent from {path}")
    
    @abstractmethod
    def _save_model(self, path: str) -> None:
        """
        Save the model to disk
        
        Args:
            path: Path to save to
        """
        pass
    
    @abstractmethod
    def _load_model(self, path: str) -> None:
        """
        Load the model from disk
        
        Args:
            path: Path to load from
        """
        pass
