"""
CoolMind REST API
Provides HTTP endpoints for all pipeline types with thermal monitoring
"""
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, Union
import logging
import uuid
from datetime import datetime

# Import our pipeline system
try:
    from ..pipelines import get_pipeline
    from ..core.engine import ThermalConfig
import ThermalConfig
    PIPELINE_AVAILABLE = True
except ImportError:
    PIPELINE_AVAILABLE = False
    logging.warning("Pipeline system not available - API will be limited")

logger = logging.getLogger(__name__)
