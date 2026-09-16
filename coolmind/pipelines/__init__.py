"""
CoolMind Pipelines Module
Supports various AI model pipelines with thermal-aware capabilities
Includes both sync and async pipeline options
"""
from typing import Dict, List
from .base import BasePipeline
from .text_generation import TextGenerationPipeline
from .question_answering import QuestionAnsweringPipeline
from .summarization import SummarizationPipeline
from .async_base import AsyncBasePipeline
from .async_text_generation import AsyncTextGenerationPipeline

# Registry of available synchronous pipelines
SYNC_PIPELINE_REGISTRY = {
    "text-generation": TextGenerationPipeline,
    "qa": QuestionAnsweringPipeline,
    "question-answering": QuestionAnsweringPipeline,
    "summarization": SummarizationPipeline,
    "summarize": SummarizationPipeline,
}

# Registry of available asynchronous pipelines
ASYNC_PIPELINE_REGISTRY = {
    "async-text-generation": AsyncTextGenerationPipeline,
    # More async pipelines can be added here
}

# Combined registry for easy access
PIPELINE_REGISTRY = {**SYNC_PIPELINE_REGISTRY, **ASYNC_PIPELINE_REGISTRY}

def get_pipeline(pipeline_name: str, **kwargs):
    """Get a pipeline instance by name"""
    pipeline_class = PIPELINE_REGISTRY.get(pipeline_name.lower())
    if pipeline_class is None:
        available = ", ".join(PIPELINE_REGISTRY.keys())
        raise ValueError(f"Unknown pipeline {pipeline_name}. Available: {available}")
    return pipeline_class(**kwargs)

def get_async_pipeline(pipeline_name: str, **kwargs):
    """Get an async pipeline instance by name"""
    pipeline_class = ASYNC_PIPELINE_REGISTRY.get(pipeline_name.lower())
    if pipeline_class is None:
        available = ", ".join(ASYNC_PIPELINE_REGISTRY.keys())
        raise ValueError(f"Unknown async pipeline {pipeline_name}. Available: {available}")
    return pipeline_class(**kwargs)

def list_pipelines(async_included: bool = False) -> Dict[str, List[str]]:
    """List available pipelines"""
    result = {
        "sync": list(SYNC_PIPELINE_REGISTRY.keys())
    }
    if async_included:
        result["async"] = list(ASYNC_PIPELINE_REGISTRY.keys())
        result["all"] = list(PIPELINE_REGISTRY.keys())
    return result

__all__ = [
    "BasePipeline", 
    "TextGenerationPipeline", 
    "QuestionAnsweringPipeline", 
    "SummarizationPipeline",
    "AsyncBasePipeline",
    "AsyncTextGenerationPipeline",
    "get_pipeline", 
    "get_async_pipeline",
    "list_pipelines",
    "PIPELINE_REGISTRY",
    "SYNC_PIPELINE_REGISTRY",
    "ASYNC_PIPELINE_REGISTRY"
]

