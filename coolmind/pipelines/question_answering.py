from typing import Optional, Union, List, Dict
"""
Question Answering Pipeline
For extractive question answering tasks
"""
from typing import Union, List, Dict
import logging
from .base import BasePipeline

logger = logging.getLogger(__name__)

class QuestionAnsweringPipeline(BasePipeline):
    """Pipeline for question answering tasks"""
    
    def __init__(self, model_name: str, device: str = "auto", **kwargs):
        super().__init__(model_name, device, **kwargs)
        
    def load_model(self):
        """Load the question answering model and tokenizer"""
        try:
            from transformers import AutoTokenizer, AutoModelForQuestionAnswering
            import torch
            
            logger.info(f"Loading QA model: {self.model_name}")
            
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                **self.kwargs.get("tokenizer_kwargs", {})
            )
            
            self._model = AutoModelForQuestionAnswering.from_pretrained(
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
            
            logger.info(f"QA model loaded on {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to load QA model {self.model_name}: {e}")
            raise
    
    def __call__(self, question: str, context: str,
                 **kwargs) -> Dict[str, Union[str, float]]:
        """
        Answer a question based on context
        
        Args:
            question: The question to answer
            context: The context to search for the answer
            
        Returns:
            Dictionary with 'answer', 'score', 'start', 'end' keys
        """
        self.ensure_loaded()
        
        # Tokenize inputs
        inputs = self._tokenizer(
            question,
            context,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(self.device)
        
        # Get model outputs
        with torch.no_grad():
            outputs = self._model(**inputs)
            start_logits = outputs.start_logits
            end_logits = outputs.end_logits
        
        # Find the most likely answer span
        start_idx = start_logits.argmax(dim=-1).item()
        end_idx = end_logits.argmax(dim=-1).item()
        
        # Ensure valid span
        if end_idx < start_idx:
            end_idx = start_idx
        
        # Extract answer tokens
        answer_tokens = inputs["input_ids"][0][start_idx:end_idx+1]
        answer = self._tokenizer.decode(answer_tokens, skip_special_tokens=True)
        
        # Calculate confidence score (simplified)
        start_prob = start_logits.softmax(dim=-1)[0, start_idx].item()
        end_prob = end_logits.softmax(dim=-1)[0, end_idx].item()
        score = (start_prob + end_prob) / 2
        
        return {
            "answer": answer.strip(),
            "score": score,
            "start": start_idx,
            "end": end_idx
        }

__all__ = ["QuestionAnsweringPipeline"]
