"""
Cấu hình cho Reinforcement Learning Service
"""
import os
from dotenv import load_dotenv

# Tải biến môi trường từ file .env nếu có
load_dotenv()

# Cấu hình chung
DEBUG = os.getenv("DEBUG", "True").lower() in ("true", "1", "t")
API_PREFIX = "/api/rl"
PROJECT_NAME = "Trading System - Reinforcement Learning Service"
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

# Cấu hình Model Service API
MODEL_SERVICE_URL = os.getenv("MODEL_SERVICE_URL", "http://localhost:8001")
MODEL_SERVICE_API_PREFIX = os.getenv("MODEL_SERVICE_API_PREFIX", "/api/models")

# Cấu hình Celery
CELERY_BROKER_URL = os.getenv("CELERY_BROKER_URL", REDIS_URL)
CELERY_RESULT_BACKEND = os.getenv("CELERY_RESULT_BACKEND", REDIS_URL)

# Cấu hình môi trường RL
DEFAULT_SYMBOL = os.getenv("DEFAULT_SYMBOL", "BTC/USDT")
DEFAULT_TIMEFRAME = os.getenv("DEFAULT_TIMEFRAME", "1h")
AVAILABLE_TIMEFRAMES = ["1m", "5m", "15m", "30m", "1h", "4h", "1d"]

# Cấu hình agent RL
RL_ALGORITHMS = ["dqn", "ppo", "a2c", "sac"]
DEFAULT_RL_ALGORITHM = "ppo"
OBSERVATION_WINDOW = 30  # Số khoảng thời gian trong quá khứ để quan sát
MAX_EPISODE_STEPS = 1000  # Số bước tối đa trong một episode
TRAIN_EPISODES = 1000  # Số episode để huấn luyện

# Cấu hình reward
TRANSACTION_FEE = 0.001  # Phí giao dịch 0.1%
SLIPPAGE = 0.001  # Trượt giá 0.1%

# Cấu hình action space
ACTIONS = ["buy", "sell", "hold"]
POSITION_SIZES = [0.25, 0.5, 0.75, 1.0]  # Kích thước vị thế (% của vốn)

# Cấu hình lưu trữ mô hình
MODEL_SAVE_PATH = os.getenv("RL_MODEL_SAVE_PATH", "/home/ubuntu/trading_system/rl_service/saved_models")

# Cấu hình đánh giá
EVALUATION_METRICS = ["total_return", "sharpe_ratio", "max_drawdown", "win_rate"]
BACKTEST_INITIAL_BALANCE = 10000  # USD
