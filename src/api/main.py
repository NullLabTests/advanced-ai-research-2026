"""
Advanced AI Research API
FastAPI REST API for disinformation analysis and manifold diffusion

Features:
- RESTful endpoints
- Async processing
- Request validation
- Rate limiting
- Authentication
- Monitoring
"""

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import asyncio
import time
import logging
from datetime import datetime, timedelta
import uuid
from contextlib import asynccontextmanager

# Add src to path
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.models.advanced_disinformation_analyzer import create_analyzer
from src.models.manifold_diffusion_model import create_manifold_diffusion
from src.utils.monitoring import MetricsCollector
from src.utils.cache import CacheManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables
app_state = {}

# Pydantic models
class TextInput(BaseModel):
    text: str = Field(..., min_length=1, max_length=10000, description="Text to analyze")
    human_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Optional human judge score")
    human_weight: Optional[float] = Field(0.7, ge=0.0, le=1.0, description="Weight for human judge")
    include_explanation: Optional[bool] = Field(True, description="Include detailed explanation")

class BatchTextInput(BaseModel):
    texts: List[str] = Field(..., min_items=1, max_items=100, description="List of texts to analyze")
    human_scores: Optional[List[float]] = Field(None, description="Optional human scores for each text")
    human_weight: Optional[float] = Field(0.7, ge=0.0, le=1.0)
    include_explanation: Optional[bool] = Field(True)

class ManifoldGenerationRequest(BaseModel):
    n_samples: int = Field(100, ge=1, le=1000, description="Number of samples to generate")
    data_dim: int = Field(2, ge=2, le=10, description="Data dimensionality")
    diffusion_steps: int = Field(100, ge=10, le=1000, description="Number of diffusion steps")
    manifold_constraint: bool = Field(True, description="Apply manifold constraint")

class AnalysisResponse(BaseModel):
    success: bool
    data: Dict[str, Any]
    processing_time: float
    timestamp: str

class HealthResponse(BaseModel):
    status: str
    version: str
    uptime: float
    models_loaded: bool
    memory_usage: Dict[str, float]

# Security
security = HTTPBearer(auto_error=False)

# Metrics
metrics = MetricsCollector()

# Cache
cache = CacheManager()

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Get current user (simplified authentication)"""
    if credentials is None:
        return None
    # In production, validate token here
    return {"user_id": "demo_user", "permissions": ["read", "write"]}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    logger.info("Starting Advanced AI Research API...")
    
    # Load models
    try:
        app_state["analyzer"] = create_analyzer(enable_explanations=True)
        app_state["manifold_model"] = create_manifold_diffusion(data_dim=2, diffusion_steps=100)
        app_state["models_loaded"] = True
        logger.info("Models loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load models: {e}")
        app_state["models_loaded"] = False
    
    app_state["start_time"] = time.time()
    
    yield
    
    # Shutdown
    logger.info("Shutting down Advanced AI Research API...")

# Create FastAPI app
app = FastAPI(
    title="Advanced AI Research API",
    description="REST API for disinformation analysis and manifold diffusion",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure properly in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(GZipMiddleware, minimum_size=1000)

# Rate limiting
@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    """Simple rate limiting middleware"""
    client_ip = request.client.host
    current_time = time.time()
    
    # Check rate limit (simplified)
    key = f"rate_limit:{client_ip}"
    requests = cache.get(key) or []
    requests = [req_time for req_time in requests if current_time - req_time < 60]  # 1 minute window
    
    if len(requests) > 100:  # 100 requests per minute
        return JSONResponse(
            status_code=429,
            content={"detail": "Rate limit exceeded"}
        )
    
    requests.append(current_time)
    cache.set(key, requests, ttl=60)
    
    response = await call_next(request)
    return response

# Routes
@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint"""
    return {
        "message": "Advanced AI Research API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    uptime = time.time() - app_state.get("start_time", time.time())
    
    # Get memory usage (simplified)
    import psutil
    memory = psutil.virtual_memory()
    
    return HealthResponse(
        status="healthy" if app_state.get("models_loaded", False) else "degraded",
        version="1.0.0",
        uptime=uptime,
        models_loaded=app_state.get("models_loaded", False),
        memory_usage={
            "total": memory.total / 1024**3,  # GB
            "available": memory.available / 1024**3,
            "percent": memory.percent
        }
    )

@app.post("/analyze/text", response_model=AnalysisResponse)
async def analyze_text(
    request: TextInput,
    background_tasks: BackgroundTasks,
    current_user: Optional[Dict] = Depends(get_current_user)
):
    """Analyze text for disinformation"""
    start_time = time.time()
    
    try:
        # Check cache
        cache_key = f"analysis:{hash(request.text)}"
        cached_result = cache.get(cache_key)
        if cached_result:
            logger.info(f"Cache hit for text analysis")
            return AnalysisResponse(
                success=True,
                data=cached_result,
                processing_time=time.time() - start_time,
                timestamp=datetime.utcnow().isoformat()
            )
        
        # Perform analysis
        analyzer = app_state.get("analyzer")
        if not analyzer:
            raise HTTPException(status_code=503, detail="Models not loaded")
        
        result = analyzer.analyze_text(
            request.text,
            human_score=request.human_score,
            human_weight=request.human_weight,
            return_explanation=request.include_explanation
        )
        
        # Convert to dict
        result_dict = {
            "text": result.text,
            "final_risk_score": result.final_risk_score,
            "llm_judge_score": result.llm_judge_score,
            "human_judge_score": result.human_judge_score,
            "confidence": result.confidence,
            "explanation": result.explanation,
            "risk_factors": result.risk_factors,
            "emotional_intensity": result.emotional_intensity,
            "logical_coherence": result.logical_coherence,
            "source_credibility": result.source_credibility,
            "timestamp": result.timestamp
        }
        
        # Cache result
        cache.set(cache_key, result_dict, ttl=3600)  # 1 hour
        
        # Log metrics
        background_tasks.add_task(
            metrics.record_analysis,
            "text_analysis",
            time.time() - start_time,
            result.final_risk_score
        )
        
        return AnalysisResponse(
            success=True,
            data=result_dict,
            processing_time=time.time() - start_time,
            timestamp=datetime.utcnow().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Text analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze/batch", response_model=AnalysisResponse)
async def analyze_batch(
    request: BatchTextInput,
    background_tasks: BackgroundTasks,
    current_user: Optional[Dict] = Depends(get_current_user)
):
    """Analyze multiple texts in batch"""
    start_time = time.time()
    
    try:
        analyzer = app_state.get("analyzer")
        if not analyzer:
            raise HTTPException(status_code=503, detail="Models not loaded")
        
        # Validate inputs
        if request.human_scores and len(request.human_scores) != len(request.texts):
            raise HTTPException(status_code=400, detail="Number of human scores must match number of texts")
        
        # Process batch
        human_scores = request.human_scores if request.human_scores else [None] * len(request.texts)
        results = []
        
        for text, human_score in zip(request.texts, human_scores):
            result = analyzer.analyze_text(
                text,
                human_score=human_score,
                human_weight=request.human_weight,
                return_explanation=request.include_explanation
            )
            
            results.append({
                "text": result.text,
                "final_risk_score": result.final_risk_score,
                "llm_judge_score": result.llm_judge_score,
                "human_judge_score": result.human_judge_score,
                "confidence": result.confidence,
                "explanation": result.explanation,
                "risk_factors": result.risk_factors,
                "emotional_intensity": result.emotional_intensity,
                "logical_coherence": result.logical_coherence,
                "source_credibility": result.source_credibility,
                "timestamp": result.timestamp
            })
        
        # Calculate batch statistics
        risk_scores = [r["final_risk_score"] for r in results]
        batch_stats = {
            "total_texts": len(results),
            "avg_risk_score": sum(risk_scores) / len(risk_scores),
            "max_risk_score": max(risk_scores),
            "min_risk_score": min(risk_scores),
            "high_risk_count": sum(1 for score in risk_scores if score > 0.7),
            "results": results
        }
        
        # Log metrics
        background_tasks.add_task(
            metrics.record_batch_analysis,
            "batch_analysis",
            time.time() - start_time,
            len(request.texts),
            sum(risk_scores) / len(risk_scores)
        )
        
        return AnalysisResponse(
            success=True,
            data=batch_stats,
            processing_time=time.time() - start_time,
            timestamp=datetime.utcnow().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Batch analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate/manifold", response_model=AnalysisResponse)
async def generate_manifold_samples(
    request: ManifoldGenerationRequest,
    background_tasks: BackgroundTasks,
    current_user: Optional[Dict] = Depends(get_current_user)
):
    """Generate samples using manifold diffusion"""
    start_time = time.time()
    
    try:
        manifold_model = app_state.get("manifold_model")
        if not manifold_model:
            raise HTTPException(status_code=503, detail="Models not loaded")
        
        # Generate samples
        samples = manifold_model.sample(
            shape=(request.n_samples, request.data_dim),
            n_steps=request.diffusion_steps
        )
        
        # Convert to list for JSON serialization
        samples_list = samples.tolist()
        
        # Calculate sample statistics
        samples_np = samples.detach().cpu().numpy()
        stats = {
            "mean": samples_np.mean(axis=0).tolist(),
            "std": samples_np.std(axis=0).tolist(),
            "min": samples_np.min(axis=0).tolist(),
            "max": samples_np.max(axis=0).tolist()
        }
        
        result_data = {
            "samples": samples_list,
            "n_samples": request.n_samples,
            "data_dim": request.data_dim,
            "diffusion_steps": request.diffusion_steps,
            "statistics": stats,
            "manifold_constraint": request.manifold_constraint
        }
        
        # Log metrics
        background_tasks.add_task(
            metrics.record_generation,
            "manifold_generation",
            time.time() - start_time,
            request.n_samples
        )
        
        return AnalysisResponse(
            success=True,
            data=result_data,
            processing_time=time.time() - start_time,
            timestamp=datetime.utcnow().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Manifold generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics")
async def get_metrics():
    """Get application metrics"""
    return metrics.get_all_metrics()

@app.get("/models/status")
async def get_models_status():
    """Get model status information"""
    return {
        "analyzer_loaded": "analyzer" in app_state,
        "manifold_model_loaded": "manifold_model" in app_state,
        "models_loaded": app_state.get("models_loaded", False)
    }

@app.post("/feedback")
async def submit_feedback(
    feedback: Dict[str, Any],
    current_user: Optional[Dict] = Depends(get_current_user)
):
    """Submit feedback for model improvement"""
    try:
        # Store feedback (in production, save to database)
        feedback_id = str(uuid.uuid4())
        feedback_data = {
            "id": feedback_id,
            "user_id": current_user.get("user_id") if current_user else "anonymous",
            "feedback": feedback,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Log feedback
        logger.info(f"Feedback received: {feedback_id}")
        
        return {
            "success": True,
            "feedback_id": feedback_id,
            "message": "Feedback submitted successfully"
        }
        
    except Exception as e:
        logger.error(f"Feedback submission error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# WebSocket for real-time updates
@app.websocket("/ws")
async def websocket_endpoint(websocket):
    """WebSocket endpoint for real-time updates"""
    await websocket.accept()
    
    try:
        while True:
            # Send periodic updates
            await websocket.send_json({
                "type": "metrics_update",
                "data": metrics.get_all_metrics(),
                "timestamp": datetime.utcnow().isoformat()
            })
            
            await asyncio.sleep(10)  # Send updates every 10 seconds
            
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        await websocket.close()

# Exception handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "timestamp": datetime.utcnow().isoformat()
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "status_code": 500,
            "timestamp": datetime.utcnow().isoformat()
        }
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
