"""
CoolMind Web API - Minimal Viable Version
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import logging

logger = logging.getLogger(__name__)

# Try to import our pipeline system
try:
    from ..pipelines import get_pipeline
    from ..core.engine import ThermalConfig
    PIPELINE_AVAILABLE = True
except ImportError as e:
    PIPELINE_AVAILABLE = False
    logger.warning(f"Pipeline system not available: {e}")

# Pydantic models
class GenerationRequest(BaseModel):
    prompt: str
    max_length: Optional[int] = 50
    temperature: Optional[float] = 0.7

class QARequest(BaseModel):
    question: str
    context: str

class SummarizationRequest(BaseModel):
    text: str
    max_length: Optional[int] = 50

class APIResponse(BaseModel):
    success: bool
    result: Optional[str] = None
    error: Optional[str] = None

def create_app() -> FastAPI:
    """Create and configure the FastAPI application"""
    if not PIPELINE_AVAILABLE:
        # Return a minimal API that explains the limitation
        app = FastAPI(title="CoolMind API - Limited Mode")
        
        @app.get("/")
        async def root():
            return {
                "error": "Pipeline system not available",
                "message": "The CoolMind API requires the pipeline system to be installed.",
                "suggestion": "Please ensure all dependencies are installed."
            }
        
        @app.get("/health")
        async def health():
            return {"status": "limited", "reason": "pipeline_system_unavailable"}
        
        return app
    
    app = FastAPI(
        title="CoolMind API",
        description="Thermal-aware local AI inference engine",
        version="0.1.0"
    )
    
    # Simple in-memory pipeline cache
    _pipelines = {}
    
    def get_pipeline_cached(pipeline_type: str, model_name: str, 
                           max_length: int, temperature: float):
        """Get or create a pipeline instance"""
        key = f"{pipeline_type}:{model_name}:{max_length}:{temperature}"
        if key not in _pipelines:
            logger.info(f"Creating pipeline: {key}")
            pipeline = get_pipeline(
                pipeline_type,
                model_name=model_name,
                max_length=max_length,
                temperature=temperature,
                top_p=0.9,
                do_sample=False,
                thermal_config=ThermalConfig()
            )
            _pipelines[key] = pipeline
        return _pipelines[key]
    
    @app.post("/generate")
    async def generate_text(request: GenerationRequest):
        """Generate text from prompt"""
        try:
            pipeline = get_pipeline_cached(
                "text-generation", 
                getattr(request, 'model_name', 'sshleifer/tiny-gpt2'),
                request.max_length,
                request.temperature
            )
            
            result = pipeline(
                request.prompt,
                max_length=request.max_length,
                temperature=request.temperature
            )
            
            return APIResponse(success=True, result=result)
        except Exception as e:
            logger.error(f"Generation error: {e}")
            return APIResponse(success=False, error=str(e))
    
    @app.post("/qa")
    async def answer_question(request: QARequest):
        """Answer question based on context"""
        try:
            pipeline = get_pipeline_cached(
                "qa",
                getattr(request, 'model_name', 'distilbert-base-uncased-distilled-squad'),
                50,  # max_length for QA
                0.0  # temperature for QA (usually deterministic)
            )
            
            result = pipeline(
                question=request.question,
                context=request.context
            )
            
            answer = result.get("answer", "") if isinstance(result, dict) else str(result)
            return APIResponse(success=True, result=answer)
        except Exception as e:
            logger.error(f"QA error: {e}")
            return APIResponse(success=False, error=str(e))
    
    @app.post("/summarize")
    async def summarize_text(request: SummarizationRequest):
        """Summarize text"""
        try:
            pipeline = get_pipeline_cached(
                "summarization",
                getattr(request, 'model_name', 'sshleifer/distilbart-cnn-12-6'),
                request.max_length,
                0.7  # temperature
            )
            
            result = pipeline(
                text=request.text,
                max_length=request.max_length
            )
            
            return APIResponse(success=True, result=result)
        except Exception as e:
            logger.error(f"Summarization error: {e}")
            return APIResponse(success=False, error=str(e))
    
    @app.get("/health")
    async def health_check():
        """Health check"""
        return {
            "status": "healthy", 
            "service": "coolmind-api",
            "pipelines_loaded": len(_pipelines)
        }
    
    @app.get("/")
    async def root():
        """Root endpoint"""
        return {
            "name": "CoolMind API",
            "version": "0.1.0",
            "description": "Thermal-aware local AI inference engine with pipeline support",
            "endpoints": {
                "POST /generate": "Text generation",
                "POST /qa": "Question answering", 
                "POST /summarize": "Text summarization",
                "GET /health": "Health check",
                "GET /docs": "API documentation"
            }
        }
    
    return app

# Create app instance
app = create_app()
