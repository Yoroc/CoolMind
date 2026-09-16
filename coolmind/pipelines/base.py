"""
Base Pipeline Class with Model Caching
Defines the interface for all CoolMind pipelines with intelligent caching
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple
import logging
import threading
import weakref

logger = logging.getLogger(__name__)

# Global model cache to prevent reloading the same models
_model_cache: Dict[Tuple[str, str], weakref.ref] = {}
_cache_lock = threading.RLock()

class BasePipeline(ABC):
    """Base class for all AI pipelines in CoolMind with intelligent model caching"""
    
    def __init__(self, model_name: str, device: str = "auto", **kwargs):
        self.model_name = model_name
        self.device = device
        self.kwargs = kwargs
        self._model = None
        self._tokenizer = None
        self._cache_key = (model_name, device)
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
        """Ensure model is loaded before use, using cache when possible"""
        if self._model is not None:
            return
            
        # Check cache first
        with _cache_lock:
            cached_ref = _model_cache.get(self._cache_key)
            if cached_ref is not None:
                cached_model = cached_ref()
                if cached_model is not None:
                    self._model, self._tokenizer = cached_model
                    logger.info(f"Using cached model for {self.model_name}")
                    return
        
        # Load model if not in cache
        logger.info(f"Loading model {self.model_name} (not in cache)")
        self.load_model()
        
        # Cache the loaded model
        with _cache_lock:
            _model_cache[self._cache_key] = weakref.ref((self._model, self._tokenizer))
    
    def get_model_info(self) -> Dict[str, str]:
        """Get information about the loaded model"""
        return {
            "model_name": self.model_name,
            "device": str(self.device) if self.device else "unknown",
            "pipeline_type": self.__class__.__name__,
            "cached": self._model is not None
        }
    
    @classmethod
    def get_cache_stats(cls) -> Dict[str, Any]:
        """Get statistics about the model cache"""
        with _cache_lock:
            total_entries = len(_model_cache)
            valid_entries = sum(1 for ref in _model_cache.values() if ref() is not None)
            return {
                "total_cached_entries": total_entries,
                "valid_entries": valid_entries,
                "expired_entries": total_entries - valid_entries,
                "cache_keys": [str(key) for key in _model_cache.keys()]
            }
    
    @classmethod
    def clear_cache(cls):
        """Clear the model cache to free memory"""
        with _cache_lock:
            _model_cache.clear()
        logger.info("Model cache cleared")

__all__ = ["BasePipeline"]

