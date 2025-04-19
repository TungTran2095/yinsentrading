"""
Main application for model service
"""
import logging
import uvicorn
from fastapi import FastAPI
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
    description="API for trading system model service",
)

# Include API router
app.include_router(router, prefix=API_PREFIX)

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "name": PROJECT_NAME,
        "version": VERSION,
        "description": "API for trading system model service",
    }

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "healthy"}

if __name__ == "__main__":
    logger.info(f"Starting {PROJECT_NAME} v{VERSION}")
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True)
