"""
CoolMind Package
Thermal-aware local AI inference engine with pipeline support
"""
from .cli import main
from .core.engine import CoolEngine, ThermalConfig

__version__ = "0.1.0"
from .pipelines import get_pipeline, PIPELINE_REGISTRY

__all__ = ["main", "CoolEngine", "ThermalConfig", "get_pipeline", "PIPELINE_REGISTRY"]
