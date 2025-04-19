"""
Main application for Trading Service
"""
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from api import router
from config import PROJECT_NAME, VERSION, API_PREFIX

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title=PROJECT_NAME,
    version=VERSION,
    description="Trading System - Trading Service API",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include router
app.include_router(router)

# Root endpoint
@app.get("/")
async def root():
    return {
        "name": PROJECT_NAME,
        "version": VERSION,
        "api_prefix": API_PREFIX,
    }

if __name__ == "__main__":
    logger.info(f"Starting {PROJECT_NAME} v{VERSION}")
    uvicorn.run("main:app", host="0.0.0.0", port=8003, reload=True)
