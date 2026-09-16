"""
Base Pipeline Class
Defines the interface for all CoolMind pipelines
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import logging

logger = logging.getLogger(__name__)

class BasePipeline(ABC):
    """Base class for all AI pipelines in CoolMind"""
    
    def __init__(self, model_name: str, device: str = "auto", **kwargs):
        self.model_name = model_name
        self.device = device
        self.kwargs = kwargs
        self._model = None
        self._tokenizer = None
        logger.info(f"Initializing {self.__class__.__name__} with model: {model_name}")
    
    @abstractmethod
    def load_model(self):
        """Load the model and tokenizer"""
        pass
    
    @abstractmethod
    def __call__(self, inputs: Any, **kwargs) -> Any:
        """Process inputs and return results"""
        pass
    
    def ensure_loaded(self):
        """Ensure model is loaded before use"""
        if self._model is None:
            self.load_model()
    
    def get_model_info(self) -> Dict[str, str]:
        """Get information about the loaded model"""
        return {
            "model_name": self.model_name,
            "device": str(self.device) if self.device else "unknown",
            "pipeline_type": self.__class__.__name__
        }

__all__ = ["BasePipeline"]
