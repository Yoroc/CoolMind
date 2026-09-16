"""
CoolMind Package
Thermal-aware local AI inference engine with pipeline support
"""
from .cli import main
from .core.engine import CoolEngine, ThermalConfig
from .pipelines import get_pipeline, PIPELINE_REGISTRY

__all__ = ["main", "CoolEngine", "ThermalConfig", "get_pipeline", "PIPELINE_REGISTRY"]
