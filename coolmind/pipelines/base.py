"""
Base Pipeline Class with Advanced LRU Caching
Implements sophisticated caching strategies for production deployment
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple, List
import logging
import threading
import time
from collections import OrderedDict

logger = logging.getLogger(__name__)

# Advanced LRU Cache with size limits and eviction policies
class LRUModelCache:
    """Thread-safe LRU cache for ML models with size limits and statistics"""
    
    def __init__(self, max_size: int = 10):
        self.max_size = max_size
        self._cache: OrderedDict[Tuple[str, str], Tuple[Any, Any, float]] = OrderedDict()
        self._lock = threading.RLock()
        self._hits = 0
        self._misses = 0
        logger.info(f"Initialized LRU model cache with max_size={max_size}")
    
    def get(self, key: Tuple[str, str]) -> Optional[Tuple[Any, Any]]:
        """Get item from cache, marking it as recently used"""
        with self._lock:
            if key in self._cache:
                # Move to end (most recently used)
                model, tokenizer, _ = self._cache.pop(key)
                self._cache[key] = (model, tokenizer, time.time())
                self._hits += 1
                logger.debug(f"Cache hit for {key}")
                return model, tokenizer
            else:
                self._misses += 1
                logger.debug(f"Cache miss for {key}")
                return None
    
    def put(self, key: Tuple[str, str], model: Any, tokenizer: Any):
        """Put item in cache, evicting LRU if necessary"""
        with self._lock:
            # If key already exists, remove it first
            if key in self._cache:
                self._cache.pop(key)
            # If at capacity, remove least recently used item
            elif len(self._cache) >= self.max_size:
                evicted_key, (_, _, _) = self._cache.popitem(last=False)
                logger.debug(f"Evicted LRU model: {evicted_key}")
            
            # Add new item
            self._cache[key] = (model, tokenizer, time.time())
            logger.debug(f"Cached model: {key} (cache size: {len(self._cache)}/{self.max_size})")
    
    def clear(self):
        """Clear all cached items"""
        with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0
        logger.info("Model cache cleared")
    
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self._lock:
            total_requests = self._hits + self._misses
            hit_rate = (self._hits / total_requests * 100) if total_requests > 0 else 0
            return {
                "cache_size": len(self._cache),
                "max_size": self.max_size,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate_percent": round(hit_rate, 2),
                "total_requests": total_requests,
                "cached_models": [str(key) for key in self._cache.keys()]
            }

# Global cache instance
_model_cache = LRUModelCache(max_size=5)  # Default to 5 models to prevent memory issues

class BasePipeline(ABC):
    """Base class for all AI pipelines in CoolMind with advanced LRU caching"""
    
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
        _model_cache.put(self._cache_key, self._model, self._tokenizer)
    
    def get_model_info(self) -> Dict[str, str]:
        """Get information about the loaded model"""
        cache_stats = _model_cache.stats()
        return {
            "model_name": self.model_name,
            "device": str(self.device) if self.device else "unknown",
            "pipeline_type": self.__class__.__name__,
            "cached": self._model is not None,
            "cache_hit_rate": f"{cache_stats[hit_rate_percent]}%",
            "cache_size": f"{cache_stats[cache_size]}/{cache_stats[max_size]}"
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
        global _model_cache
        old_cache = _model_cache
        _model_cache = LRUModelCache(max_size=size)
        # Transfer existing entries (simplified - in production might want smarter transfer)
        logger.info(f"Cache size changed from {old_cache.max_size} to {size}")

__all__ = ["BasePipeline"]

