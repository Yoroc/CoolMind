"""
CoolMind Package
Thermal-aware local AI inference engine with pipeline support and web API
"""
from .cli import main
from .core.engine import CoolEngine, ThermalConfig
from .pipelines import get_pipeline, PIPELINE_REGISTRY
from .web import app as web_app, create_app as create_web_app

__all__ = ["main", "CoolEngine", "ThermalConfig", "get_pipeline", "PIPELINE_REGISTRY", "web_app", "create_web_app"]

