from typing import Optional, Union, List
"""
Summarization Pipeline
For text summarization tasks
"""
from typing import Union, List
import logging
from .base import BasePipeline

logger = logging.getLogger(__name__)

class SummarizationPipeline(BasePipeline):
    """Pipeline for text summarization tasks"""
    
    def __init__(self, model_name: str, device: str = "auto", 
                 max_length: int = 150, min_length: int = 30,
                 do_sample: bool = False, **kwargs):
        super().__init__(model_name, device, **kwargs)
        self.max_length = max_length
        self.min_length = min_length
        self.do_sample = do_sample
        
    def load_model(self):
        """Load the summarization model and tokenizer"""
        try:
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
            import torch
            
            logger.info(f"Loading summarization model: {self.model_name}")
            
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                **self.kwargs.get("tokenizer_kwargs", {})
            )
            
            self._model = AutoModelForSeq2SeqLM.from_pretrained(
                self.model_name,
                **self.kwargs.get("model_kwargs", {})
            )
            
            # Set device
            if self.device == "auto":
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            elif self.device == "mps" and torch.backends.mps.is_available():
                self.device = "mps"
                
            self._model.to(self.device)
            self._model.eval()
            
            logger.info(f"Summarization model loaded on {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to load summarization model {self.model_name}: {e}")
            raise
    
    def __call__(self, text: Union[str, List[str]],
                 max_length: Optional[int] = None,
                 min_length: Optional[int] = None,
                 do_sample: Optional[bool] = None,
                 **kwargs) -> Union[str, List[str]]:
        """
        Summarize text
        
        Args:
            text: Input text to summarize or list of texts
            max_length: Maximum length of summary
            min_length: Minimum length of summary  
            do_sample: Whether to use sampling
            
        Returns:
            Summarized text string or list of strings
        """
        self.ensure_loaded()
        
        # Use provided values or fall back to instance defaults
        max_length = max_length if max_length is not None else self.max_length
        min_length = min_length if min_length is not None else self.min_length
        do_sample = do_sample if do_sample is not None else self.do_sample
        
        # Handle single text vs batch
        is_batch = isinstance(text, list)
        texts = text if is_batch else [text]
        
        results = []
        for t in texts:
            # Tokenize input
            inputs = self._tokenizer(
                t,
                return_tensors="pt",
                truncation=True,
                max_length=1024
            ).to(self.device)
            
            # Generate summary
            with torch.no_grad():
                outputs = self._model.generate(
                    **inputs,
                    max_length=max_length,
                    min_length=min_length,
                    do_sample=do_sample,
                    num_beams=4 if not do_sample else 1,
                    early_stopping=True
                )
            
            # Decode output
            summary = self._tokenizer.decode(
                outputs[0], 
                skip_special_tokens=True
            )
            results.append(summary.strip())
        
        return results if is_batch else results[0]

__all__ = ["SummarizationPipeline"]
