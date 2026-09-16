"""
Async Pipeline Support
Provides async/await interfaces for high-concurrency applications
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Union, Coroutine
import logging
import asyncio

logger = logging.getLogger(__name__)

class AsyncBasePipeline(ABC):
    """Base class for async AI pipelines in CoolMind"""
    
    def __init__(self, model_name: str, device: str = "auto", **kwargs):
        self.model_name = model_name
        self.device = device
        self.kwargs = kwargs
        self._model = None
        self._tokenizer = None
        self._lock = asyncio.Lock()  # For thread-safe async initialization
        self._loaded = False
        logger.info(f"Initializing Async{self.__class__.__name__} with model: {model_name}")
    
    @abstractmethod
    def load_model(self):
        """Load the model and tokenizer (synchronous)"""
        pass
    
    @abstractmethod
    def _process_inputs(self, inputs: Any, **kwargs) -> Any:
        """Process inputs and return results (synchronous core logic)"""
        pass
    
    async def ensure_loaded(self):
        """Ensure model is loaded before use (async-safe)"""
        if self._loaded:
            return
            
        async with self._lock:
            # Double-check pattern for async safety
            if self._loaded:
                return
                
            logger.info(f"Async loading model {self.model_name}")
            # Run synchronous model loading in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self.load_model)
            self._loaded = True
            logger.info(f"Async model {self.model_name} loaded successfully")
    
    @abstractmethod
    async def __call__(self, inputs: Any, **kwargs) -> Any:
        """Process inputs and return results (async interface)"""
        pass
    
    async def abatch(self, inputs: List[Any], **kwargs) -> List[Any]:
        """Process a batch of inputs asynchronously"""
        await self.ensure_loaded()
        
        # Process batch concurrently where possible
        if len(inputs) <= 1:
            return [await self.__call__(inputs[0], **kwargs)]
        
        # For now, process sequentially - subclasses can optimize
        # Override this method for true batch processing in subclasses
        results = []
        for inp in inputs:
            result = await self.__call__(inp, **kwargs)
            results.append(result)
        return results
    
    def get_model_info(self) -> Dict[str, str]:
        """Get information about the loaded model"""
        return {
            "model_name": self.model_name,
            "device": str(self.device) if self.device else "unknown",
            "pipeline_type": self.__class__.__name__,
            "async_support": True,
            "loaded": self._loaded
        }

__all__ = ["AsyncBasePipeline"]

