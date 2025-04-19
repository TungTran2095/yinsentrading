"""
API models for model service
"""
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime

class ModelTrainRequest(BaseModel):
    """Request model for training a model"""
    symbol: str = Field(..., description="Trading pair symbol (e.g., 'BTC/USDT')")
    timeframe: str = Field(..., description="Timeframe for data (e.g., '1h', '1d')")
    model_type: str = Field(..., description="Type of model to train (e.g., 'random_forest', 'xgboost', 'lstm', 'transformer')")
    prediction_horizon: int = Field(1, description="Number of time periods to predict into the future")
    model_params: Optional[Dict[str, Any]] = Field(None, description="Model-specific parameters")

class EnsembleTrainRequest(BaseModel):
    """Request model for training an ensemble"""
    symbol: str = Field(..., description="Trading pair symbol (e.g., 'BTC/USDT')")
    timeframe: str = Field(..., description="Timeframe for data (e.g., '1h', '1d')")
    ensemble_type: str = Field(..., description="Type of ensemble to train (e.g., 'weighted_avg', 'stacking')")
    prediction_horizon: int = Field(1, description="Number of time periods to predict into the future")
    model_types: List[str] = Field(..., description="Types of models to include in the ensemble")
    ensemble_params: Optional[Dict[str, Any]] = Field(None, description="Ensemble-specific parameters")

class PredictionRequest(BaseModel):
    """Request model for making predictions"""
    symbol: str = Field(..., description="Trading pair symbol (e.g., 'BTC/USDT')")
    timeframe: str = Field(..., description="Timeframe for data (e.g., '1h', '1d')")
    model_id: Optional[str] = Field(None, description="ID of the model to use for prediction")
    ensemble_id: Optional[str] = Field(None, description="ID of the ensemble to use for prediction")
    limit: Optional[int] = Field(100, description="Maximum number of data points to predict")
    start_time: Optional[datetime] = Field(None, description="Start timestamp")
    end_time: Optional[datetime] = Field(None, description="End timestamp")

class EvaluationRequest(BaseModel):
    """Request model for evaluating a model or ensemble"""
    symbol: str = Field(..., description="Trading pair symbol (e.g., 'BTC/USDT')")
    timeframe: str = Field(..., description="Timeframe for data (e.g., '1h', '1d')")
    model_id: Optional[str] = Field(None, description="ID of the model to evaluate")
    ensemble_id: Optional[str] = Field(None, description="ID of the ensemble to evaluate")
    metrics: Optional[List[str]] = Field(None, description="Metrics to calculate")
    start_time: Optional[datetime] = Field(None, description="Start timestamp")
    end_time: Optional[datetime] = Field(None, description="End timestamp")

class ModelInfo(BaseModel):
    """Model information"""
    id: str = Field(..., description="Model ID")
    name: str = Field(..., description="Model name")
    type: str = Field(..., description="Model type")
    symbol: str = Field(..., description="Trading pair symbol")
    timeframe: str = Field(..., description="Timeframe for data")
    prediction_horizon: int = Field(..., description="Number of time periods to predict into the future")
    created_at: datetime = Field(..., description="Creation timestamp")
    metrics: Optional[Dict[str, float]] = Field(None, description="Model metrics")

class EnsembleInfo(BaseModel):
    """Ensemble information"""
    id: str = Field(..., description="Ensemble ID")
    name: str = Field(..., description="Ensemble name")
    type: str = Field(..., description="Ensemble type")
    symbol: str = Field(..., description="Trading pair symbol")
    timeframe: str = Field(..., description="Timeframe for data")
    prediction_horizon: int = Field(..., description="Number of time periods to predict into the future")
    created_at: datetime = Field(..., description="Creation timestamp")
    models: List[str] = Field(..., description="IDs of models in the ensemble")
    metrics: Optional[Dict[str, float]] = Field(None, description="Ensemble metrics")

class PredictionResult(BaseModel):
    """Prediction result"""
    timestamp: datetime = Field(..., description="Prediction timestamp")
    symbol: str = Field(..., description="Trading pair symbol")
    timeframe: str = Field(..., description="Timeframe for data")
    prediction: float = Field(..., description="Predicted value")
    confidence: Optional[float] = Field(None, description="Prediction confidence")
    model_id: Optional[str] = Field(None, description="ID of the model used for prediction")
    ensemble_id: Optional[str] = Field(None, description="ID of the ensemble used for prediction")

class StatusResponse(BaseModel):
    """Response model for status"""
    status: str = Field(..., description="Status message")
    timestamp: datetime = Field(..., description="Timestamp of the response")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional details")
