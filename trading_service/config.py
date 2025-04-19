"""
Cấu hình cho Trading Service
"""
import os
from dotenv import load_dotenv

# Tải biến môi trường từ file .env nếu có
load_dotenv()

# Cấu hình chung
DEBUG = os.getenv("DEBUG", "True").lower() in ("true", "1", "t")
API_PREFIX = "/api/trading"
PROJECT_NAME = "Trading System - Trading Service"
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

# Cấu hình RL Service API
RL_SERVICE_URL = os.getenv("RL_SERVICE_URL", "http://localhost:8002")
RL_SERVICE_API_PREFIX = os.getenv("RL_SERVICE_API_PREFIX", "/api/rl")

# Cấu hình Celery
CELERY_BROKER_URL = os.getenv("CELERY_BROKER_URL", REDIS_URL)
CELERY_RESULT_BACKEND = os.getenv("CELERY_RESULT_BACKEND", REDIS_URL)

# Cấu hình giao dịch
DEFAULT_SYMBOL = os.getenv("DEFAULT_SYMBOL", "BTC/USDT")
DEFAULT_TIMEFRAME = os.getenv("DEFAULT_TIMEFRAME", "1h")
AVAILABLE_TIMEFRAMES = ["1m", "5m", "15m", "30m", "1h", "4h", "1d"]

# Cấu hình chiến lược
STRATEGY_TYPES = ["ensemble", "rl", "combined", "custom"]
DEFAULT_STRATEGY_TYPE = "combined"

# Cấu hình quản lý rủi ro
MAX_POSITION_SIZE = 0.5  # Tối đa 50% vốn cho một vị thế
MAX_DRAWDOWN = 0.2  # Dừng giao dịch nếu drawdown vượt quá 20%
STOP_LOSS_PERCENTAGE = 0.05  # Stop loss 5%
TAKE_PROFIT_PERCENTAGE = 0.1  # Take profit 10%

# Cấu hình thực thi
EXECUTION_MODES = ["paper", "live"]
DEFAULT_EXECUTION_MODE = "paper"
SLIPPAGE = 0.001  # Trượt giá 0.1%
TRANSACTION_FEE = 0.001  # Phí giao dịch 0.1%

# Cấu hình sàn giao dịch
EXCHANGE_CONFIGS = {
    "binance": {
        "api_key": os.getenv("BINANCE_API_KEY", ""),
        "api_secret": os.getenv("BINANCE_API_SECRET", ""),
        "testnet": os.getenv("BINANCE_TESTNET", "True").lower() in ("true", "1", "t")
    },
    "ftx": {
        "api_key": os.getenv("FTX_API_KEY", ""),
        "api_secret": os.getenv("FTX_API_SECRET", ""),
        "testnet": os.getenv("FTX_TESTNET", "True").lower() in ("true", "1", "t")
    },
    "bybit": {
        "api_key": os.getenv("BYBIT_API_KEY", ""),
        "api_secret": os.getenv("BYBIT_API_SECRET", ""),
        "testnet": os.getenv("BYBIT_TESTNET", "True").lower() in ("true", "1", "t")
    }
}
DEFAULT_EXCHANGE = "binance"

# Cấu hình webhook
WEBHOOK_ENABLED = os.getenv("WEBHOOK_ENABLED", "False").lower() in ("true", "1", "t")
WEBHOOK_URL = os.getenv("WEBHOOK_URL", "")

# Cấu hình thông báo
NOTIFICATION_CHANNELS = ["email", "telegram", "webhook"]
EMAIL_ENABLED = os.getenv("EMAIL_ENABLED", "False").lower() in ("true", "1", "t")
EMAIL_HOST = os.getenv("EMAIL_HOST", "smtp.gmail.com")
EMAIL_PORT = int(os.getenv("EMAIL_PORT", "587"))
EMAIL_USERNAME = os.getenv("EMAIL_USERNAME", "")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD", "")
EMAIL_RECIPIENTS = os.getenv("EMAIL_RECIPIENTS", "").split(",")

TELEGRAM_ENABLED = os.getenv("TELEGRAM_ENABLED", "False").lower() in ("true", "1", "t")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_IDS = os.getenv("TELEGRAM_CHAT_IDS", "").split(",")

# Cấu hình n8n
N8N_ENABLED = os.getenv("N8N_ENABLED", "False").lower() in ("true", "1", "t")
N8N_URL = os.getenv("N8N_URL", "http://localhost:5678")
N8N_API_KEY = os.getenv("N8N_API_KEY", "")
