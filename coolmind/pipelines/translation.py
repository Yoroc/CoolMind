"""
Translation Pipeline
Extends CoolMind's core functionality for translation tasks
"""
from typing import Optional, Union
import logging
from .base import BasePipeline

logger = logging.getLogger(__name__)

class TranslationPipeline(BasePipeline):
    """Pipeline for translation tasks"""
    
    def __init__(self, model_name: str, device: str = "auto", 
                 max_length: int = 100, **kwargs):
        super().__init__(model_name, device, **kwargs)
        self.max_length = max_length
        
    def load_model(self):
        """Load the translation model and tokenizer"""
        try:
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
            import torch
            
            logger.info(f"Loading translation model: {self.model_name}")
            
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
            logger.info(f"Translation model loaded successfully on {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to load translation model {self.model_name}: {e}")
            raise
            
    def translate(self, text: str, src_lang: str = None, tgt_lang: str = None) -> str:
        """Translate text from source language to target language"""
        if not self._model or not self._tokenizer:
            self.load_model()
            
        try:
            # Prepare input text
            if src_lang and tgt_lang:
                # Some models require language tags
                input_text = f">>{tgt_lang}<< {text}"
            else:
                input_text = text
                
            # Tokenize
            inputs = self._tokenizer(
                input_text,
                return_tensors="pt",
                max_length=self.max_length,
                truncation=True
            ).to(self.device)
            
            # Generate translation
            with torch.no_grad():
                outputs = self._model.generate(
                    **inputs,
                    max_length=self.max_length,
                    num_beams=4,
                    early_stopping=True
                )
            
            # Decode output
            translation = self._tokenizer.decode(
                outputs[0], 
                skip_special_tokens=True
            )
            
            return translation
            
        except Exception as e:
            logger.error(f"Translation failed: {e}")
            raise
            
    def __call__(self, text: str, src_lang: str = None, tgt_lang: str = None) -> str:
        """Make the pipeline callable"""
        return self.translate(text, src_lang, tgt_lang)
