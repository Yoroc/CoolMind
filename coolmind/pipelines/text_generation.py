from typing import Optional, Union, List
"""
Text Generation Pipeline
Extends CoolMind's core functionality for text generation tasks
"""
from typing import Optional, Union, List
import logging
from .base import BasePipeline

logger = logging.getLogger(__name__)

class TextGenerationPipeline(BasePipeline):
    """Pipeline for text generation tasks"""
    
    def __init__(self, model_name: str, device: str = "auto", 
                 max_length: int = 50, temperature: float = 0.7,
                 top_p: float = 0.9, do_sample: bool = True, **kwargs):
        super().__init__(model_name, device, **kwargs)
        self.max_length = max_length
        self.temperature = temperature
        self.top_p = top_p
        self.do_sample = do_sample
        
    def load_model(self):
        """Load the text generation model and tokenizer"""
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            import torch
            
            logger.info(f"Loading text generation model: {self.model_name}")
            
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                **self.kwargs.get("tokenizer_kwargs", {})
            )
            
            # Handle padding token
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token
                
            self._model = AutoModelForCausalLM.from_pretrained(
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
            
            logger.info(f"Model loaded on {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to load model {self.model_name}: {e}")
            raise
    
    def __call__(self, prompt: Union[str, List[str]], 
                 max_length: Optional[int] = None,
                 temperature: Optional[float] = None,
                 top_p: Optional[float] = None,
                 do_sample: Optional[bool] = None,
                 **kwargs) -> Union[str, List[str]]:
        """
        Generate text from prompt(s)
        
        Args:
            prompt: Input text prompt or list of prompts
            max_length: Maximum length of generated text
            temperature: Sampling temperature (0.0 to 1.0)
            top_p: Nucleus sampling parameter
            do_sample: Whether to use sampling vs greedy decoding
            
        Returns:
            Generated text string or list of strings
        """
        self.ensure_loaded()
        
        # Use provided values or fall back to instance defaults
        max_length = max_length if max_length is not None else self.max_length
        temperature = temperature if temperature is not None else self.temperature
        top_p = top_p if top_p is not None else self.top_p
        do_sample = do_sample if do_sample is not None else self.do_sample
        
        # Handle single prompt vs batch
        is_batch = isinstance(prompt, list)
        prompts = prompt if is_batch else [prompt]
        
        results = []
        for p in prompts:
            # Tokenize input
            inputs = self._tokenizer(
                p, 
                return_tensors="pt",
                padding=True,
                truncation=True
            ).to(self.device)
            
            # Generate
            with torch.no_grad():
                outputs = self._model.generate(
                    **inputs,
                    max_length=max_length,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=do_sample,
                    pad_token_id=self._tokenizer.pad_token_id,
                    eos_token_id=self._tokenizer.eos_token_id,
                    **kwargs
                )
            
            # Decode output (remove input prompt)
            generated_tokens = outputs[0][inputs["input_ids"].shape[-1]:]
            generated_text = self._tokenizer.decode(
                generated_tokens, 
                skip_special_tokens=True
            )
            results.append(generated_text.strip())
        
        return results if is_batch else results[0]

__all__ = ["TextGenerationPipeline"]
