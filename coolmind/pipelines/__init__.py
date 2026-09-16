"""
CoolMind Pipelines Module
Supports various AI model pipelines with thermal-aware capabilities
"""
from .base import BasePipeline
from .text_generation import TextGenerationPipeline
from .question_answering import QuestionAnsweringPipeline  
from .summarization import SummarizationPipeline

# Registry of available pipelines
PIPELINE_REGISTRY = {
    "text-generation": TextGenerationPipeline,
    "qa": QuestionAnsweringPipeline,
    "question-answering": QuestionAnsweringPipeline,
    "summarization": SummarizationPipeline,
    "summarize": SummarizationPipeline,
}

def get_pipeline(pipeline_name: str, **kwargs):
    """Get a pipeline instance by name"""
    pipeline_class = PIPELINE_REGISTRY.get(pipeline_name.lower())
    if pipeline_class is None:
        available = ", ".join(PIPELINE_REGISTRY.keys())
        raise ValueError(f"Unknown pipeline '{pipeline_name}'. Available: {available}")
    return pipeline_class(**kwargs)

__all__ = ["BasePipeline", "TextGenerationPipeline", "QuestionAnsweringPipeline", 
           "SummarizationPipeline", "get_pipeline", "PIPELINE_REGISTRY"]
