"""
LSTM model for market prediction
"""
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Any, Tuple
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import sys
sys.path.append('..')
from config import TRAIN_TEST_SPLIT_RATIO, RANDOM_SEED, BATCH_SIZE, EPOCHS, EARLY_STOPPING_PATIENCE
from .base_model import BaseModel

logger = logging.getLogger(__name__)

class LSTMModel(BaseModel):
    """
    LSTM model for market prediction
    """
    
    def __init__(self, symbol: str, timeframe: str, prediction_horizon: int = 1, 
                 units: int = 50, dropout: float = 0.2, sequence_length: int = 10):
        """
        Initialize the LSTM model
        
        Args:
            symbol: Trading pair symbol (e.g., "BTC/USDT")
            timeframe: Timeframe for data (e.g., "1h", "1d")
            prediction_horizon: Number of time periods to predict into the future
            units: Number of LSTM units
            dropout: Dropout rate
            sequence_length: Length of input sequences
        """
        super().__init__("lstm", symbol, timeframe, prediction_horizon)
        self.units = units
        self.dropout = dropout
        self.sequence_length = sequence_length
        
        # Initialize model
        self.model = None  # Will be built during training
        
        # Initialize scalers
        self.scaler_x = MinMaxScaler()
        self.scaler_y = MinMaxScaler()
        
        logger.info(f"Initialized LSTMModel with units={units}, dropout={dropout}, sequence_length={sequence_length}")
    
    def _build_model(self, input_shape: Tuple[int, int]) -> None:
        """
        Build the LSTM model
        
        Args:
            input_shape: Shape of input data (sequence_length, n_features)
        """
        model = Sequential()
        model.add(LSTM(units=self.units, return_sequences=True, input_shape=input_shape))
        model.add(Dropout(self.dropout))
        model.add(LSTM(units=self.units))
        model.add(Dropout(self.dropout))
        model.add(Dense(1))
        
        model.compile(optimizer='adam', loss='mse')
        
        self.model = model
        
        logger.info(f"Built LSTM model with input shape {input_shape}")
    
    def _create_sequences(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for LSTM input
        
        Args:
            X: Features array
            y: Target array
            
        Returns:
            Tuple of (X_seq, y_seq) where X_seq is sequences of features and y_seq is target values
        """
        X_seq, y_seq = [], []
        
        for i in range(len(X) - self.sequence_length):
            X_seq.append(X[i:i+self.sequence_length])
            y_seq.append(y[i+self.sequence_length])
        
        return np.array(X_seq), np.array(y_seq)
    
    def preprocess_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocess data for training or prediction
        
        Args:
            df: DataFrame with market data and technical indicators
            
        Returns:
            Tuple of (X, y) where X is features and y is target
        """
        # Create features and target
        features, target = self._create_features_target(df, self.target_name, self.prediction_horizon)
        
        # Store feature names
        self.feature_names = features.columns.tolist()
        
        # Convert to numpy arrays
        X = features.values
        y = target.values.reshape(-1, 1)  # Reshape for scaler
        
        # Scale features and target
        if not self.is_trained:
            X = self.scaler_x.fit_transform(X)
            y = self.scaler_y.fit_transform(y)
        else:
            X = self.scaler_x.transform(X)
            y = self.scaler_y.transform(y)
        
        # Create sequences
        X_seq, y_seq = self._create_sequences(X, y)
        
        return X_seq, y_seq
    
    def train(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Train the model
        
        Args:
            df: DataFrame with market data and technical indicators
            
        Returns:
            Dictionary with training metrics
        """
        logger.info(f"Training LSTMModel for {self.symbol} with timeframe {self.timeframe}")
        
        # Preprocess data
        X, y = self.preprocess_data(df)
        
        # Split data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=1-TRAIN_TEST_SPLIT_RATIO, random_state=RANDOM_SEED
        )
        
        # Build model if not already built
        if self.model is None:
            self._build_model((self.sequence_length, X.shape[2]))
        
        # Early stopping callback
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=EARLY_STOPPING_PATIENCE,
            restore_best_weights=True
        )
        
        # Train model
        history = self.model.fit(
            X_train, y_train,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            validation_data=(X_test, y_test),
            callbacks=[early_stopping],
            verbose=0
        )
        
        # Make predictions on test set
        y_pred = self.model.predict(X_test)
        
        # Inverse transform predictions and actual values
        y_test_inv = self.scaler_y.inverse_transform(y_test)
        y_pred_inv = self.scaler_y.inverse_transform(y_pred)
        
        # Calculate metrics
        mse = mean_squared_error(y_test_inv, y_pred_inv)
        mae = mean_absolute_error(y_test_inv, y_pred_inv)
        r2 = r2_score(y_test_inv, y_pred_inv)
        
        # Calculate directional accuracy
        direction_correct = np.sum(np.sign(y_test_inv[1:] - y_test_inv[:-1]) == np.sign(y_pred_inv[1:] - y_pred_inv[:-1]))
        direction_accuracy = direction_correct / (len(y_test_inv) - 1)
        
        # Set trained flag
        self.is_trained = True
        
        # Return metrics
        metrics = {
            "mse": mse,
            "mae": mae,
            "r2": r2,
            "direction_accuracy": direction_accuracy,
            "training_history": {
                "loss": history.history['loss'],
                "val_loss": history.history['val_loss']
            }
        }
        
        logger.info(f"Trained LSTMModel with metrics: MSE={mse:.6f}, MAE={mae:.6f}, R²={r2:.6f}, Direction Accuracy={direction_accuracy:.6f}")
        
        return metrics
    
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Make predictions
        
        Args:
            df: DataFrame with market data and technical indicators
            
        Returns:
            DataFrame with predictions
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        logger.info(f"Making predictions with LSTMModel for {self.symbol} with timeframe {self.timeframe}")
        
        # Make a copy to avoid modifying the original
        df_copy = df.copy()
        
        # Preprocess data
        X, _ = self.preprocess_data(df_copy)
        
        # Make predictions
        y_pred = self.model.predict(X)
        
        # Inverse transform predictions
        y_pred_inv = self.scaler_y.inverse_transform(y_pred)
        
        # Create DataFrame with predictions
        # Note: We need to account for sequence_length in the index
        timestamps = df_copy.index[self.sequence_length:] if isinstance(df_copy.index, pd.DatetimeIndex) else df_copy.index[self.sequence_length:]
        
        df_pred = pd.DataFrame({
            "timestamp": timestamps,
            "prediction": y_pred_inv.flatten(),
            "model": self.name,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "prediction_horizon": self.prediction_horizon
        })
        
        # Set timestamp as index if it's not already
        if not isinstance(df_pred.index, pd.DatetimeIndex) and "timestamp" in df_pred.columns:
            df_pred = df_pred.set_index("timestamp")
        
        logger.info(f"Made {len(df_pred)} predictions with LSTMModel")
        
        return df_pred
    
    def evaluate(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Evaluate the model
        
        Args:
            df: DataFrame with market data and technical indicators
            
        Returns:
            Dictionary with evaluation metrics
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")
        
        logger.info(f"Evaluating LSTMModel for {self.symbol} with timeframe {self.timeframe}")
        
        # Preprocess data
        X, y = self.preprocess_data(df)
        
        # Make predictions
        y_pred = self.model.predict(X)
        
        # Inverse transform predictions and actual values
        y_inv = self.scaler_y.inverse_transform(y)
        y_pred_inv = self.scaler_y.inverse_transform(y_pred)
        
        # Calculate metrics
        mse = mean_squared_error(y_inv, y_pred_inv)
        mae = mean_absolute_error(y_inv, y_pred_inv)
        r2 = r2_score(y_inv, y_pred_inv)
        
        # Calculate directional accuracy
        direction_correct = np.sum(np.sign(y_inv[1:] - y_inv[:-1]) == np.sign(y_pred_inv[1:] - y_pred_inv[:-1]))
        direction_accuracy = direction_correct / (len(y_inv) - 1)
        
        # Calculate sharpe ratio (simplified)
        returns_actual = np.diff(y_inv.flatten()) / y_inv[:-1].flatten()
        returns_pred = np.diff(y_pred_inv.flatten()) / y_pred_inv[:-1].flatten()
        sharpe_ratio = np.mean(returns_pred) / np.std(returns_pred) if np.std(returns_pred) > 0 else 0
        
        # Return metrics
        metrics = {
            "mse": mse,
            "mae": mae,
            "r2": r2,
            "direction_accuracy": direction_accuracy,
            "sharpe_ratio": sharpe_ratio
        }
        
        logger.info(f"Evaluated LSTMModel with metrics: MSE={mse:.6f}, MAE={mae:.6f}, R²={r2:.6f}, Direction Accuracy={direction_accuracy:.6f}, Sharpe Ratio={sharpe_ratio:.6f}")
        
        return metrics
    
    def save(self) -> str:
        """
        Save the model to disk
        
        Returns:
            Path to saved model
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        # Create filename
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.name}_{self.symbol.replace('/', '_')}_{self.timeframe}_{timestamp}"
        filepath = f"{filename}.h5"
        
        # Save Keras model
        self.model.save(filepath)
        
        # Save metadata using parent class method
        super().save()
        
        logger.info(f"Saved LSTM model to {filepath}")
        
        return filepath
