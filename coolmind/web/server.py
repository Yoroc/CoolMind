"""
CoolMind Web Server
Handles server lifecycle and deployment options
"""
import logging
import sys
from typing import Optional

try:
    import uvicorn
    UVICORN_AVAILABLE = True
except ImportError:
    UVICORN_AVAILABLE = False
    logging.warning("Uvicorn not installed. Install with: pip install uvicorn")

from .api import create_app

logger = logging.getLogger(__name__)

def run_server(host: str = "127.0.0.1", port: int = 8000, 
               reload: bool = False, workers: int = 1,
               log_level: str = "info") -> None:
    """Run the CoolMind web API server"""
    if not UVICORN_AVAILABLE:
        logger.error("Uvicorn is required to run the web server")
        logger.error("Install it with: pip install uvicorn")
        sys.exit(1)
    
    try:
        app = create_app()
        if app is None:
            logger.error("Failed to create CoolMind application")
            sys.exit(1)
        
        logger.info(f"Starting CoolMind Web API on {host}:{port}")
        logger.info(f"API Documentation: http://{host}:{port}/docs")
        logger.info(f"Alternative Docs: http://{host}:{port}/redoc")
        
        uvicorn.run(
            app,
            host=host,
            port=port,
            reload=reload,
            workers=workers,
            log_level=log_level,
            access_log=True
        )
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        sys.exit(1)

def create_app_instance():
    """Create app instance for external use (e.g., gunicorn)"""
    try:
        return create_app()
    except Exception as e:
        logger.error(f"Failed to create app instance: {e}")
        return None

if __name__ == "__main__":
    run_server()

