"""
Configuration settings for the Chat AI service
"""
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Project info
PROJECT_NAME = "Trading System - Chat AI Service"
VERSION = "0.1.0"
API_PREFIX = "/api/chat"

# Server settings
HOST = os.getenv("CHAT_SERVICE_HOST", "0.0.0.0")
PORT = int(os.getenv("CHAT_SERVICE_PORT", "8004"))
DEBUG = os.getenv("DEBUG", "False").lower() in ("true", "1", "t")

# MongoDB settings
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
MONGODB_DB = os.getenv("MONGODB_DB", "trading_system")
MONGODB_COLLECTION = os.getenv("MONGODB_COLLECTION", "chat_history")

# NLP model settings
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
SPACY_MODEL = os.getenv("SPACY_MODEL", "en_core_web_sm")

# API endpoints for other services
DATA_SERVICE_URL = os.getenv("DATA_SERVICE_URL", "http://localhost:8001/api/data")
MODEL_SERVICE_URL = os.getenv("MODEL_SERVICE_URL", "http://localhost:8002/api/models")
TRADING_SERVICE_URL = os.getenv("TRADING_SERVICE_URL", "http://localhost:8003/api/trading")

# Intent classification thresholds
INTENT_THRESHOLD = float(os.getenv("INTENT_THRESHOLD", "0.7"))

# Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
