from typing import Any, Dict, Optional, Tuple
"""
Base Pipeline Class for CoolMind
Foundation for all AI model pipelines with LRU caching
"""
from abc import ABC, abstractmethod
from typing import Any, Dict
import logging
from collections import OrderedDict
import threading

logger = logging.getLogger(__name__)

# Simple LRU Cache for model instances
_model_cache = OrderedDict()
_cache_lock = threading.RLock()
_cache_hits = 0
_cache_misses = 0
_max_cache_size = 10


def _update_cache_stats(hit: bool = None):
    """Update cache hit/miss statistics"""
    global _cache_hits, _cache_misses
    with _cache_lock:
        if hit is True:
            _cache_hits += 1
        elif hit is False:
            _cache_misses += 1


class LRUModelCache:
    """Simple LRU cache for ML models"""
    
    def __init__(self, max_size: int = 10):
        self.max_size = max_size
        self._cache: OrderedDict[Tuple[str, str], Tuple[Any, Any]] = OrderedDict()
        self._lock = threading.RLock()
    
    def get(self, key: Tuple[str, str]) -> Optional[Tuple[Any, Any]]:
        """Get item from cache"""
        with self._lock:
            if key in self._cache:
                # Move to end (most recently used)
                self._cache.move_to_end(key)
                _update_cache_stats(True)
                return self._cache[key]
            else:
                _update_cache_stats(False)
                return None
    
    def put(self, key: Tuple[str, str], value: Tuple[Any, Any]) -> None:
        """Put item in cache"""
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
            else:
                if len(self._cache) >= self.max_size:
                    self._cache.popitem(last=False)
                self._cache[key] = value
    
    def stats(self) -> Dict[str, int]:
        """Get cache statistics"""
        with self._lock:
            total = _cache_hits + _cache_misses
            hit_rate = (_cache_hits / total * 100) if total > 0 else 0
            return {
                "size": len(self._cache),
                "max_size": self.max_size,
                "hits": _cache_hits,
                "misses": _cache_misses,
                "hit_rate_percent": round(hit_rate, 2)
            }
    
    def clear(self) -> None:
        """Clear the cache"""
        with self._lock:
            self._cache.clear()


class BasePipeline(ABC):
    """Base class for all AI pipelines in CoolMind"""
    
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
        """Ensure model is loaded before use, using LRU cache when possible"""
        if self._model is not None:
            return
        
        # Check LRU cache first
        cached_result = _model_cache.get(self._cache_key)
        if cached_result is not None:
            self._model, self._tokenizer = cached_result
            logger.info(f"Using LRU-cached model for {self.model_name}")
            return
        
        # Load model if not in cache
        logger.info(f"Loading model {self.model_name} (not in cache or evicted)")
        self.load_model()
        
        # Cache the loaded model using LRU strategy
        _model_cache.put(self._cache_key, (self._model, self._tokenizer))
    
    def get_model_info(self) -> Dict[str, str]:
        """Get information about the loaded model"""
        cache_stats = _model_cache.stats()
        return {
            "model_name": self.model_name,
            "device": str(self.device) if self.device else "unknown",
            "pipeline_type": self.__class__.__name__,
            "cached": self._model is not None,
            "cache_hit_rate": f"{cache_stats['hit_rate_percent']}%",
            "cache_size": f"{cache_stats['size']}/{cache_stats['max_size']}"
        }
    
    @classmethod
    def get_cache_stats(cls) -> Dict[str, Any]:
        """Get statistics about the model cache"""
        return _model_cache.stats()
    
    @classmethod
    def clear_cache(cls):
        """Clear the model cache to free memory"""
        _model_cache.clear()
        logger.info("Model cache cleared via BasePipeline.clear_cache()")
    
    @classmethod
    def set_cache_size(cls, size: int):
        """Adjust the maximum cache size"""
        global _model_cache, _max_cache_size
        old_cache = _model_cache
        _model_cache = LRUModelCache(max_size=size)
        logger.info(f"Cache size changed from {old_cache.max_size} to {size}")

