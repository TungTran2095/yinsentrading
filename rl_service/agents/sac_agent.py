"""
SAC (Soft Actor-Critic) agent for Reinforcement Learning
"""
import numpy as np
import pandas as pd
import gymnasium as gym
import logging
from typing import Dict, List, Optional, Any, Tuple
import os
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import sys
sys.path.append('..')
from config import MODEL_SAVE_PATH
from environments.trading_env import TradingEnvironment
from .base_agent import BaseRLAgent

logger = logging.getLogger(__name__)

class SACAgent(BaseRLAgent):
    """
    SAC agent for Reinforcement Learning
    """
    
    def __init__(self, env: TradingEnvironment, policy: str = "MlpPolicy", 
                 learning_rate: float = 3e-4, buffer_size: int = 100000, 
                 learning_starts: int = 100, batch_size: int = 256, 
                 tau: float = 0.005, gamma: float = 0.99, 
                 train_freq: int = 1, gradient_steps: int = 1,
                 action_noise: Optional[Any] = None, ent_coef: str = "auto",
                 target_update_interval: int = 1, target_entropy: str = "auto",
                 use_sde: bool = False, sde_sample_freq: int = -1,
                 use_sde_at_warmup: bool = False, verbose: int = 0):
        """
        Initialize the SAC agent
        
        Args:
            env: Trading environment
            policy: Policy network architecture
            learning_rate: Learning rate
            buffer_size: Size of the replay buffer
            learning_starts: How many steps before learning starts
            batch_size: Minibatch size
            tau: Target network update rate
            gamma: Discount factor
            train_freq: Update the model every train_freq steps
            gradient_steps: How many gradient steps to do after each rollout
            action_noise: Action noise
            ent_coef: Entropy coefficient or "auto"
            target_update_interval: Update the target network every target_update_interval steps
            target_entropy: Target entropy when using "auto" entropy coefficient
            use_sde: Whether to use State Dependent Exploration
            sde_sample_freq: Sample a new noise matrix every sde_sample_freq steps
            use_sde_at_warmup: Whether to use SDE at warmup
            verbose: Verbosity level
        """
        super().__init__("sac", env)
        
        # Create a vectorized environment
        self.vec_env = DummyVecEnv([lambda: env])
        self.vec_env = VecNormalize(self.vec_env, norm_obs=True, norm_reward=True, clip_obs=10.)
        
        # Initialize SAC model
        self.model = SAC(
            policy=policy,
            env=self.vec_env,
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            learning_starts=learning_starts,
            batch_size=batch_size,
            tau=tau,
            gamma=gamma,
            train_freq=train_freq,
            gradient_steps=gradient_steps,
            action_noise=action_noise,
            ent_coef=ent_coef,
            target_update_interval=target_update_interval,
            target_entropy=target_entropy,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            use_sde_at_warmup=use_sde_at_warmup,
            verbose=verbose
        )
        
        # Store hyperparameters
        self.hyperparams = {
            "policy": policy,
            "learning_rate": learning_rate,
            "buffer_size": buffer_size,
            "learning_starts": learning_starts,
            "batch_size": batch_size,
            "tau": tau,
            "gamma": gamma,
            "train_freq": train_freq,
            "gradient_steps": gradient_steps,
            "ent_coef": ent_coef,
            "target_update_interval": target_update_interval,
            "target_entropy": target_entropy,
            "use_sde": use_sde,
            "sde_sample_freq": sde_sample_freq,
            "use_sde_at_warmup": use_sde_at_warmup
        }
        
        logger.info(f"Initialized SAC agent with hyperparameters: {self.hyperparams}")
    
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
        logger.info(f"Training SAC agent for {total_timesteps} timesteps")
        
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
        
        logger.info(f"Trained SAC agent with metrics: {metrics}")
        
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
        
        # Get action probabilities and values
        # Note: SAC uses a stochastic policy, so we get the mean and log_std
        mean_actions, log_std = self.model.actor.get_action_dist_params(self.model.actor.obs_to_tensor(obs)[0])
        mean_actions = mean_actions.detach().numpy()
        log_std = log_std.detach().numpy()
        
        # Create info dictionary
        info = {
            "mean_actions": mean_actions,
            "log_std": log_std,
            "value": self.model.critic.q1_forward(
                self.model.critic.obs_to_tensor(obs)[0], 
                self.model.actor.obs_to_tensor(obs)[0]
            )[0].detach().numpy()
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
        
        logger.info(f"Evaluating SAC agent for {num_episodes} episodes")
        
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
        
        logger.info(f"Evaluated SAC agent with metrics: {metrics}")
        
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
        self.model = SAC.load(path, env=self.vec_env)
        
        # Load the vectorized environment statistics
        vec_normalize_path = path.replace(".zip", "_vecnormalize.pkl")
        if os.path.exists(vec_normalize_path):
            self.vec_env = VecNormalize.load(vec_normalize_path, self.vec_env)
            # Don't update the normalization statistics during prediction
            self.vec_env.training = False
