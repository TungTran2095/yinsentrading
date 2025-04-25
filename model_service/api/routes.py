"""
API routes for model service
"""
import logging
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
import requests
import json
from datetime import datetime
import os
import sys
sys.path.append('..')
from config import API_PREFIX, ENSEMBLE_MODELS, ENSEMBLE_METHODS, DATA_SERVICE_URL, DATA_SERVICE_API_PREFIX
from .models import (
    ModelTrainRequest, EnsembleTrainRequest, PredictionRequest, EvaluationRequest,
    ModelInfo, EnsembleInfo, PredictionResult, StatusResponse
)
from models import RandomForestModel, XGBoostModel, LSTMModel, TransformerModel
from ensemble.base_ensemble import BaseEnsemble
from ensemble.weighted_average import WeightedAverageEnsemble
from ensemble.stacking import StackingEnsemble
from models.base_model import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(prefix=API_PREFIX)

# In-memory storage for models and ensembles
# In a production system, this would be stored in a database
models = {}
ensembles = {}

@router.post("/models/train", response_model=ModelInfo, summary="Train a model")
async def train_model(request: ModelTrainRequest, background_tasks: BackgroundTasks):
    """
    Train a machine learning model
    """
    try:
        # Validate model type
        if request.model_type not in ENSEMBLE_MODELS:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid model type. Available model types: {', '.join(ENSEMBLE_MODELS)}"
            )
        
        # Create model ID
        model_id = f"{request.model_type}_{request.symbol.replace('/', '_')}_{request.timeframe}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Add task to background
        background_tasks.add_task(
            _train_model_task,
            model_id,
            request.model_type,
            request.symbol,
            request.timeframe,
            request.prediction_horizon,
            request.model_params
        )
        
        # Create model info
        model_info = ModelInfo(
            id=model_id,
            name=request.model_type,
            type=request.model_type,
            symbol=request.symbol,
            timeframe=request.timeframe,
            prediction_horizon=request.prediction_horizon,
            created_at=datetime.now()
        )
        
        # Store model info
        models[model_id] = model_info
        
        return model_info
    
    except Exception as e:
        logger.error(f"Error training model: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/ensembles/train", response_model=EnsembleInfo, summary="Train an ensemble")
async def train_ensemble(request: EnsembleTrainRequest, background_tasks: BackgroundTasks):
    """
    Train an ensemble of models
    """
    try:
        # Validate ensemble type
        if request.ensemble_type not in ENSEMBLE_METHODS:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid ensemble type. Available ensemble types: {', '.join(ENSEMBLE_METHODS)}"
            )
        
        # Validate model types
        for model_type in request.model_types:
            if model_type not in ENSEMBLE_MODELS:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid model type: {model_type}. Available model types: {', '.join(ENSEMBLE_MODELS)}"
                )
        
        # Create ensemble ID
        ensemble_id = f"{request.ensemble_type}_{request.symbol.replace('/', '_')}_{request.timeframe}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Add task to background
        background_tasks.add_task(
            _train_ensemble_task,
            ensemble_id,
            request.ensemble_type,
            request.symbol,
            request.timeframe,
            request.prediction_horizon,
            request.model_types,
            request.ensemble_params
        )
        
        # Create ensemble info
        ensemble_info = EnsembleInfo(
            id=ensemble_id,
            name=request.ensemble_type,
            type=request.ensemble_type,
            symbol=request.symbol,
            timeframe=request.timeframe,
            prediction_horizon=request.prediction_horizon,
            created_at=datetime.now(),
            models=[]  # Will be populated during training
        )
        
        # Store ensemble info
        ensembles[ensemble_id] = ensemble_info
        
        return ensemble_info
    
    except Exception as e:
        logger.error(f"Error training ensemble: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/predict", response_model=List[PredictionResult], summary="Make predictions")
async def predict(request: PredictionRequest):
    """
    Make predictions with a model or ensemble
    """
    try:
        # Validate request
        if request.model_id is None and request.ensemble_id is None:
            raise HTTPException(
                status_code=400,
                detail="Either model_id or ensemble_id must be provided"
            )
        
        if request.model_id is not None and request.ensemble_id is not None:
            raise HTTPException(
                status_code=400,
                detail="Only one of model_id or ensemble_id can be provided"
            )
        
        # Get data from data service
        data = await _get_data(
            request.symbol,
            request.timeframe,
            request.limit,
            request.start_time,
            request.end_time
        )
        
        if data.empty:
            return []
        
        # Make predictions
        if request.model_id is not None:
            # Check if model exists
            if request.model_id not in models:
                raise HTTPException(
                    status_code=404,
                    detail=f"Model not found: {request.model_id}"
                )
            
            # Get model info
            model_info = models[request.model_id]
            
            # Load model
            model = _load_model(model_info.type, model_info.symbol, model_info.timeframe, model_info.prediction_horizon)
            
            # Make predictions
            pred_df = model.predict(data)
            
            # Convert to response format
            results = []
            for idx, row in pred_df.iterrows():
                results.append(
                    PredictionResult(
                        timestamp=idx,
                        symbol=request.symbol,
                        timeframe=request.timeframe,
                        prediction=float(row["prediction"]),
                        model_id=request.model_id
                    )
                )
            
            return results
        
        else:  # ensemble_id is not None
            # Check if ensemble exists
            if request.ensemble_id not in ensembles:
                raise HTTPException(
                    status_code=404,
                    detail=f"Ensemble not found: {request.ensemble_id}"
                )
            
            # Get ensemble info
            ensemble_info = ensembles[request.ensemble_id]
            
            # Load ensemble
            ensemble = _load_ensemble(ensemble_info.type, ensemble_info.symbol, ensemble_info.timeframe, ensemble_info.prediction_horizon)
            
            # Make predictions
            pred_df = ensemble.predict(data)
            
            # Convert to response format
            results = []
            for idx, row in pred_df.iterrows():
                results.append(
                    PredictionResult(
                        timestamp=idx,
                        symbol=request.symbol,
                        timeframe=request.timeframe,
                        prediction=float(row["prediction"]),
                        ensemble_id=request.ensemble_id
                    )
                )
            
            return results
    
    except Exception as e:
        logger.error(f"Error making predictions: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/evaluate", summary="Evaluate a model or ensemble")
async def evaluate(request: EvaluationRequest):
    """
    Evaluate a model or ensemble
    """
    try:
        # Validate request
        if request.model_id is None and request.ensemble_id is None:
            raise HTTPException(
                status_code=400,
                detail="Either model_id or ensemble_id must be provided"
            )
        
        if request.model_id is not None and request.ensemble_id is not None:
            raise HTTPException(
                status_code=400,
                detail="Only one of model_id or ensemble_id can be provided"
            )
        
        # Get data from data service
        data = await _get_data(
            request.symbol,
            request.timeframe,
            None,  # No limit for evaluation
            request.start_time,
            request.end_time
        )
        
        if data.empty:
            raise HTTPException(
                status_code=400,
                detail="No data available for evaluation"
            )
        
        # Evaluate
        if request.model_id is not None:
            # Check if model exists
            if request.model_id not in models:
                raise HTTPException(
                    status_code=404,
                    detail=f"Model not found: {request.model_id}"
                )
            
            # Get model info
            model_info = models[request.model_id]
            
            # Load model
            model = _load_model(model_info.type, model_info.symbol, model_info.timeframe, model_info.prediction_horizon)
            
            # Evaluate model
            metrics = model.evaluate(data)
            
            # Update model info
            model_info.metrics = metrics
            models[request.model_id] = model_info
            
            return metrics
        
        else:  # ensemble_id is not None
            # Check if ensemble exists
            if request.ensemble_id not in ensembles:
                raise HTTPException(
                    status_code=404,
                    detail=f"Ensemble not found: {request.ensemble_id}"
                )
            
            # Get ensemble info
            ensemble_info = ensembles[request.ensemble_id]
            
            # Load ensemble
            ensemble = _load_ensemble(ensemble_info.type, ensemble_info.symbol, ensemble_info.timeframe, ensemble_info.prediction_horizon)
            
            # Evaluate ensemble
            metrics = ensemble.evaluate(data)
            
            # Update ensemble info
            ensemble_info.metrics = metrics
            ensembles[request.ensemble_id] = ensemble_info
            
            return metrics
    
    except Exception as e:
        logger.error(f"Error evaluating: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/models", response_model=List[ModelInfo], summary="Get all models")
async def get_models():
    """
    Get all models
    """
    try:
        return list(models.values())
    
    except Exception as e:
        logger.error(f"Error getting models: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/models/{model_id}", response_model=ModelInfo, summary="Get a model")
async def get_model(model_id: str):
    """
    Get a model by ID
    """
    try:
        if model_id not in models:
            raise HTTPException(
                status_code=404,
                detail=f"Model not found: {model_id}"
            )
        
        return models[model_id]
    
    except Exception as e:
        logger.error(f"Error getting model: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/ensembles", response_model=List[EnsembleInfo], summary="Get all ensembles")
async def get_ensembles():
    """
    Get all ensembles
    """
    try:
        return list(ensembles.values())
    
    except Exception as e:
        logger.error(f"Error getting ensembles: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/ensembles/{ensemble_id}", response_model=EnsembleInfo, summary="Get an ensemble")
async def get_ensemble(ensemble_id: str):
    """
    Get an ensemble by ID
    """
    try:
        if ensemble_id not in ensembles:
            raise HTTPException(
                status_code=404,
                detail=f"Ensemble not found: {ensemble_id}"
            )
        
        return ensembles[ensemble_id]
    
    except Exception as e:
        logger.error(f"Error getting ensemble: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/status", summary="Get model service status")
async def get_status():
    """
    Get model service status
    """
    try:
        return StatusResponse(
            status="running",
            timestamp=datetime.now(),
            details={
                "available_models": ENSEMBLE_MODELS,
                "available_ensemble_methods": ENSEMBLE_METHODS,
                "model_count": len(models),
                "ensemble_count": len(ensembles)
            }
        )
    
    except Exception as e:
        logger.error(f"Error getting status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def _get_data(symbol: str, timeframe: str, limit: Optional[int] = None, 
                   start_time: Optional[datetime] = None, end_time: Optional[datetime] = None) -> pd.DataFrame:
    """
    Get data from data service
    """
    try:
        # Build URL
        url = f"{DATA_SERVICE_URL}{DATA_SERVICE_API_PREFIX}/technical/{symbol}/{timeframe}"
        
        # Build query parameters
        params = {}
        if limit is not None:
            params["limit"] = limit
        if start_time is not None:
            params["start_time"] = start_time.isoformat()
        if end_time is not None:
            params["end_time"] = end_time.isoformat()
        
        # Make request
        response = requests.get(url, params=params)
        response.raise_for_status()
        
        # Parse response
        data = response.json()["data"]
        
        if not data:
            return pd.DataFrame()
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        # Convert timestamp to datetime
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.set_index("timestamp")
        
        return df
    
    except Exception as e:
        logger.error(f"Error getting data from data service: {e}")
        raise

def _create_model(model_type: str, symbol: str, timeframe: str, prediction_horizon: int, params: Optional[Dict[str, Any]] = None) -> BaseModel:
    """
    Create a model instance
    """
    params = params or {}
    
    if model_type == "random_forest":
        return RandomForestModel(
            symbol=symbol,
            timeframe=timeframe,
            prediction_horizon=prediction_horizon,
            **params
        )
    elif model_type == "xgboost":
        return XGBoostModel(
            symbol=symbol,
            timeframe=timeframe,
            prediction_horizon=prediction_horizon,
            **params
        )
    elif model_type == "lstm":
        return LSTMModel(
            symbol=symbol,
            timeframe=timeframe,
            prediction_horizon=prediction_horizon,
            **params
        )
    elif model_type == "transformer":
        return TransformerModel(
            symbol=symbol,
            timeframe=timeframe,
            prediction_horizon=prediction_horizon,
            **params
        )
    else:
        raise ValueError(f"Invalid model type: {model_type}")

def _create_ensemble(ensemble_type: str, symbol: str, timeframe: str, prediction_horizon: int, params: Optional[Dict[str, Any]] = None) -> BaseEnsemble:
    """
    Create an ensemble instance
    """
    params = params or {}
    
    if ensemble_type == "weighted_avg":
        return WeightedAverageEnsemble(
            symbol=symbol,
            timeframe=timeframe,
            prediction_horizon=prediction_horizon
        )
    elif ensemble_type == "stacking":
        return StackingEnsemble(
            symbol=symbol,
            timeframe=timeframe,
            prediction_horizon=prediction_horizon,
            **params
        )
    else:
        raise ValueError(f"Invalid ensemble type: {ensemble_type}")

def _load_model(model_type: str, symbol: str, timeframe: str, prediction_horizon: int) -> BaseModel:
    """
    Load a model from disk
    """
    # In a real system, this would load the model from disk or database
    # For simplicity, we'll create a new instance
    return _create_model(model_type, symbol, timeframe, prediction_horizon)

def _load_ensemble(ensemble_type: str, symbol: str, timeframe: str, prediction_horizon: int) -> BaseEnsemble:
    """
    Load an ensemble from disk
    """
    # In a real system, this would load the ensemble from disk or database
    # For simplicity, we'll create a new instance
    return _create_ensemble(ensemble_type, symbol, timeframe, prediction_horizon)

async def _train_model_task(model_id: str, model_type: str, symbol: str, timeframe: str, 
                           prediction_horizon: int, params: Optional[Dict[str, Any]] = None):
    """
    Background task for training a model
    """
    try:
        logger.info(f"Training model {model_id} of type {model_type} for {symbol} with timeframe {timeframe}")
        
        # Get data from data service
        data = await _get_data(symbol, timeframe)
        
        if data.empty:
            logger.error(f"No data available for training model {model_id}")
            return
        
        # Create model
        model = _create_model(model_type, symbol, timeframe, prediction_horizon, params)
        
        # Train model
        metrics = model.train(data)
        
        # Save model
        model.save()
        
        # Update model info
        if model_id in models:
            model_info = models[model_id]
            model_info.metrics = metrics
            models[model_id] = model_info
        
        logger.info(f"Trained model {model_id} with metrics: {metrics}")
    
    except Exception as e:
        logger.error(f"Error in model training task: {e}")

async def _train_ensemble_task(ensemble_id: str, ensemble_type: str, symbol: str, timeframe: str, 
                              prediction_horizon: int, model_types: List[str], 
                              params: Optional[Dict[str, Any]] = None):
    """
    Background task for training an ensemble
    """
    try:
        logger.info(f"Training ensemble {ensemble_id} of type {ensemble_type} for {symbol} with timeframe {timeframe}")
        
        # Get data from data service
        data = await _get_data(symbol, timeframe)
        
        if data.empty:
            logger.error(f"No data available for training ensemble {ensemble_id}")
            return
        
        # Create ensemble
        ensemble = _create_ensemble(ensemble_type, symbol, timeframe, prediction_horizon, params)
        
        # Create and train models
        model_ids = []
        for model_type in model_types:
            # Create model ID
            model_id = f"{model_type}_{symbol.replace('/', '_')}_{timeframe}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Create model
            model = _create_model(model_type, symbol, timeframe, prediction_horizon)
            
            # Train model
            metrics = model.train(data)
            
            # Save model
            model.save()
            
            # Create model info
            model_info = ModelInfo(
                id=model_id,
                name=model_type,
                type=model_type,
                symbol=symbol,
                timeframe=timeframe,
                prediction_horizon=prediction_horizon,
                created_at=datetime.now(),
                metrics=metrics
            )
            
            # Store model info
            models[model_id] = model_info
            
            # Add model to ensemble
            ensemble.add_model(model)
            
            # Add model ID to list
            model_ids.append(model_id)
        
        # Train ensemble
        metrics = ensemble.train(data)
        
        # Save ensemble
        ensemble.save()
        
        # Update ensemble info
        if ensemble_id in ensembles:
            ensemble_info = ensembles[ensemble_id]
            ensemble_info.models = model_ids
            ensemble_info.metrics = metrics
            ensembles[ensemble_id] = ensemble_info
        
        logger.info(f"Trained ensemble {ensemble_id} with metrics: {metrics}")
    
    except Exception as e:
        logger.error(f"Error in ensemble training task: {e}")
