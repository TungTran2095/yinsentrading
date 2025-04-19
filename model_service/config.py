"""
Cấu hình cho Model Service
"""
import os
from dotenv import load_dotenv

# Tải biến môi trường từ file .env nếu có
load_dotenv()

# Cấu hình chung
DEBUG = os.getenv("DEBUG", "True").lower() in ("true", "1", "t")
API_PREFIX = "/api/models"
PROJECT_NAME = "Trading System - Model Service"
VERSION = "0.1.0"

# Cấu hình cơ sở dữ liệu
POSTGRES_USER = os.getenv("POSTGRES_USER", "trading_user")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "trading_password")
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT = os.getenv("POSTGRES_PORT", "5432")
POSTGRES_DB = os.getenv("POSTGRES_DB", "trading_db")
DATABASE_URL = f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"

# Cấu hình Redis
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = os.getenv("REDIS_PORT", "6379")
REDIS_DB = os.getenv("REDIS_DB", "0")
REDIS_URL = f"redis://{REDIS_HOST}:{REDIS_PORT}/{REDIS_DB}"

# Cấu hình Data Service API
DATA_SERVICE_URL = os.getenv("DATA_SERVICE_URL", "http://localhost:8000")
DATA_SERVICE_API_PREFIX = os.getenv("DATA_SERVICE_API_PREFIX", "/api/data")

# Cấu hình Celery
CELERY_BROKER_URL = os.getenv("CELERY_BROKER_URL", REDIS_URL)
CELERY_RESULT_BACKEND = os.getenv("CELERY_RESULT_BACKEND", REDIS_URL)

# Cấu hình mô hình
DEFAULT_SYMBOL = os.getenv("DEFAULT_SYMBOL", "BTC/USDT")
DEFAULT_TIMEFRAME = os.getenv("DEFAULT_TIMEFRAME", "1h")
AVAILABLE_TIMEFRAMES = ["1m", "5m", "15m", "30m", "1h", "4h", "1d"]
PREDICTION_HORIZONS = [1, 3, 5, 10, 24]  # Số khoảng thời gian dự đoán trong tương lai

# Cấu hình mô hình ensemble
ENSEMBLE_MODELS = ["random_forest", "xgboost", "lstm", "transformer"]
ENSEMBLE_METHODS = ["weighted_avg", "stacking"]
DEFAULT_ENSEMBLE_METHOD = "weighted_avg"

# Cấu hình đánh giá mô hình
EVALUATION_METRICS = ["mse", "mae", "r2", "sharpe_ratio"]

# Cấu hình huấn luyện mô hình
TRAIN_TEST_SPLIT_RATIO = 0.8
VALIDATION_SPLIT_RATIO = 0.2
RANDOM_SEED = 42
BATCH_SIZE = 32
EPOCHS = 100
EARLY_STOPPING_PATIENCE = 10

# Cấu hình lưu trữ mô hình
MODEL_SAVE_PATH = os.getenv("MODEL_SAVE_PATH", "/home/ubuntu/trading_system/model_service/saved_models")
