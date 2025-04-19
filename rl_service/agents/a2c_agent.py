"""
A2C (Advantage Actor-Critic) agent for Reinforcement Learning
"""
import numpy as np
import pandas as pd
import gymnasium as gym
import logging
from typing import Dict, List, Optional, Any, Tuple
import os
from stable_baselines3 import A2C
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import sys
sys.path.append('..')
from config import MODEL_SAVE_PATH
from environments.trading_env import TradingEnvironment
from .base_agent import BaseRLAgent

logger = logging.getLogger(__name__)

class A2CAgent(BaseRLAgent):
    """
    A2C agent for Reinforcement Learning
    """
    
    def __init__(self, env: TradingEnvironment, policy: str = "MlpPolicy", 
                 learning_rate: float = 7e-4, n_steps: int = 5, 
                 gamma: float = 0.99, gae_lambda: float = 1.0, 
                 ent_coef: float = 0.0, vf_coef: float = 0.5,
                 max_grad_norm: float = 0.5, rms_prop_eps: float = 1e-5,
                 use_rms_prop: bool = True, normalize_advantage: bool = False,
                 verbose: int = 0):
        """
        Initialize the A2C agent
        
        Args:
            env: Trading environment
            policy: Policy network architecture
            learning_rate: Learning rate
            n_steps: Number of steps to run for each environment per update
            gamma: Discount factor
            gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator
            ent_coef: Entropy coefficient for the loss calculation
            vf_coef: Value function coefficient for the loss calculation
            max_grad_norm: Maximum value for gradient clipping
            rms_prop_eps: RMSProp epsilon
            use_rms_prop: Whether to use RMSProp or Adam as optimizer
            normalize_advantage: Whether to normalize or not the advantage
            verbose: Verbosity level
        """
        super().__init__("a2c", env)
        
        # Create a vectorized environment
        self.vec_env = DummyVecEnv([lambda: env])
        self.vec_env = VecNormalize(self.vec_env, norm_obs=True, norm_reward=True, clip_obs=10.)
        
        # Initialize A2C model
        self.model = A2C(
            policy=policy,
            env=self.vec_env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            gamma=gamma,
            gae_lambda=gae_lambda,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            rms_prop_eps=rms_prop_eps,
            use_rms_prop=use_rms_prop,
            normalize_advantage=normalize_advantage,
            verbose=verbose
        )
        
        # Store hyperparameters
        self.hyperparams = {
            "policy": policy,
            "learning_rate": learning_rate,
            "n_steps": n_steps,
            "gamma": gamma,
            "gae_lambda": gae_lambda,
            "ent_coef": ent_coef,
            "vf_coef": vf_coef,
            "max_grad_norm": max_grad_norm,
            "rms_prop_eps": rms_prop_eps,
            "use_rms_prop": use_rms_prop,
            "normalize_advantage": normalize_advantage
        }
        
        logger.info(f"Initialized A2C agent with hyperparameters: {self.hyperparams}")
    
    def train(self, total_timesteps: int, eval_freq: int = 10000, 
              n_eval_episodes: int = 5, **kwargs) -> Dict[str, Any]:
        """
        Train the agent
        
        Args:
            total_timesteps: Total number of timesteps to train for
            eval_freq: Evaluate the agent every eval_freq timesteps
            n_eval_episodes: Number of episodes to evaluate for
            **kwargs: Additional training parameters
            
        Returns:
            Dictionary with training metrics
        """
        logger.info(f"Training A2C agent for {total_timesteps} timesteps")
        
        # Create evaluation environment
        eval_env = DummyVecEnv([lambda: self.env])
        eval_env = VecNormalize(
            eval_env,
            norm_obs=True,
            norm_reward=True,
            clip_obs=10.,
            training=False
        )
        
        # Sync eval env with training env
        eval_env.obs_rms = self.vec_env.obs_rms
        eval_env.ret_rms = self.vec_env.ret_rms
        
        # Create evaluation callback
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=os.path.join(MODEL_SAVE_PATH, "best_model"),
            log_path=os.path.join(MODEL_SAVE_PATH, "logs"),
            eval_freq=eval_freq,
            n_eval_episodes=n_eval_episodes,
            deterministic=True,
            render=False
        )
        
        # Train the model
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=eval_callback,
            **kwargs
        )
        
        # Set trained flag
        self.is_trained = True
        
        # Get training metrics
        metrics = {
            "total_timesteps": total_timesteps,
            "mean_reward": eval_callback.best_mean_reward,
            "best_model_path": eval_callback.best_model_path
        }
        
        logger.info(f"Trained A2C agent with metrics: {metrics}")
        
        return metrics
    
    def predict(self, observation: np.ndarray, deterministic: bool = True, **kwargs) -> Tuple[int, Dict[str, Any]]:
        """
        Make a prediction (choose an action) based on the observation
        
        Args:
            observation: Current observation
            deterministic: Whether to use deterministic actions
            **kwargs: Additional prediction parameters
            
        Returns:
            Tuple of (action, info)
        """
        if not self.is_trained:
            raise ValueError("Agent must be trained before making predictions")
        
        # Normalize observation
        obs = self.vec_env.normalize_obs(observation)
        
        # Get action from model
        action, _states = self.model.predict(obs, deterministic=deterministic)
        
        # Get action probabilities
        action_probs = self.model.policy.get_distribution(obs).distribution.probs.detach().numpy()
        
        # Create info dictionary
        info = {
            "action_probs": action_probs,
            "value": self.model.policy.predict_values(obs).item()
        }
        
        return action, info
    
    def evaluate(self, num_episodes: int = 10, deterministic: bool = True) -> Dict[str, Any]:
        """
        Evaluate the agent
        
        Args:
            num_episodes: Number of episodes to evaluate for
            deterministic: Whether to use deterministic actions
            
        Returns:
            Dictionary with evaluation metrics
        """
        if not self.is_trained:
            raise ValueError("Agent must be trained before evaluation")
        
        logger.info(f"Evaluating A2C agent for {num_episodes} episodes")
        
        # Create evaluation environment
        eval_env = DummyVecEnv([lambda: self.env])
        eval_env = VecNormalize(
            eval_env,
            norm_obs=True,
            norm_reward=True,
            clip_obs=10.,
            training=False
        )
        
        # Sync eval env with training env
        eval_env.obs_rms = self.vec_env.obs_rms
        eval_env.ret_rms = self.vec_env.ret_rms
        
        # Run evaluation
        episode_rewards = []
        episode_lengths = []
        total_trades = 0
        profitable_trades = 0
        
        for i in range(num_episodes):
            obs, _ = eval_env.reset()
            done = False
            episode_reward = 0
            episode_length = 0
            
            while not done:
                action, _states = self.model.predict(obs, deterministic=deterministic)
                obs, reward, done, truncated, info = eval_env.step(action)
                
                episode_reward += reward[0]
                episode_length += 1
                
                # Track trades
                if info[0].get("trade_executed", False):
                    total_trades += 1
                    if info[0].get("trade_profit", 0) > 0:
                        profitable_trades += 1
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
        
        # Calculate metrics
        mean_reward = np.mean(episode_rewards)
        std_reward = np.std(episode_rewards)
        mean_length = np.mean(episode_lengths)
        win_rate = profitable_trades / total_trades if total_trades > 0 else 0
        
        # Create metrics dictionary
        metrics = {
            "mean_reward": mean_reward,
            "std_reward": std_reward,
            "mean_episode_length": mean_length,
            "total_trades": total_trades,
            "profitable_trades": profitable_trades,
            "win_rate": win_rate
        }
        
        logger.info(f"Evaluated A2C agent with metrics: {metrics}")
        
        return metrics
    
    def _save_model(self, path: str) -> None:
        """
        Save the model to disk
        
        Args:
            path: Path to save to
        """
        # Save the model
        self.model.save(path)
        
        # Save the vectorized environment statistics
        vec_normalize_path = path.replace(".zip", "_vecnormalize.pkl")
        self.vec_env.save(vec_normalize_path)
    
    def _load_model(self, path: str) -> None:
        """
        Load the model from disk
        
        Args:
            path: Path to load from
        """
        # Load the model
        self.model = A2C.load(path, env=self.vec_env)
        
        # Load the vectorized environment statistics
        vec_normalize_path = path.replace(".zip", "_vecnormalize.pkl")
        if os.path.exists(vec_normalize_path):
            self.vec_env = VecNormalize.load(vec_normalize_path, self.vec_env)
            # Don't update the normalization statistics during prediction
            self.vec_env.training = False
