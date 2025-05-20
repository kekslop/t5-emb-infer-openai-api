from fastapi import APIRouter, Depends, HTTPException
import time

from src.models.schemas import EmbeddingRequest, EmbeddingResponse, ModelList, ModelData
from src.services.model_service import process_embeddings
from src.services.queue_service import enqueue_request
from src.api.auth import verify_token
from src.core.config import MODEL_NAME

router = APIRouter()

@router.get("/v1/models")
async def list_models(token: str = Depends(verify_token)):
    """List available models in OpenAI format."""
    models = [
        ModelData(id=MODEL_NAME)
    ]
    return ModelList(data=models)

@router.get("/v1/models/{model_id}")
async def get_model(model_id: str, token: str = Depends(verify_token)):
    """Get model information in OpenAI format."""
    # Accept both the specified model name and a generic name for compatibility
    if model_id != MODEL_NAME and model_id != "text-embedding-3-small":
        raise HTTPException(status_code=404, detail=f"Model {model_id} not found")
    
    return ModelData(id=MODEL_NAME)

@router.post("/v1/embeddings", response_model=EmbeddingResponse)
async def create_embeddings(request: EmbeddingRequest, token: str = Depends(verify_token)):
    """Generate embeddings for texts - OpenAI compatible endpoint."""
    from src.main import app
    return await enqueue_request(process_embeddings, request, app.state.encoder)

@router.post("/v1/embedding", response_model=EmbeddingResponse)
async def create_embedding(request: EmbeddingRequest, token: str = Depends(verify_token)):
    """Generate embeddings for texts - alternative endpoint."""
    from src.main import app
    return await enqueue_request(process_embeddings, request, app.state.encoder)

@router.get("/")
async def root():
    """Root endpoint with basic API information."""
    return {
        "name": "Embedding Model API",
        "description": "API for getting text embeddings",
        "version": "1.0",
        "model": MODEL_NAME,
        "docs_url": "/docs",
        "openapi_url": "/openapi.json",
        "endpoints": {
            "embedding": "/v1/embedding",
            "embeddings": "/v1/embeddings",
            "models": "/v1/models",
            "health": "/health"
        }
    }

@router.get("/health")
async def health_check():
    """Health check endpoint."""
    from src.main import app
    
    # Check if model is loaded
    if not hasattr(app.state, "encoder") or app.state.encoder is None:
        return {
            "status": "error",
            "message": "Model not loaded",
            "timestamp": time.time()
        }
    
    return {
        "status": "ok",
        "model": MODEL_NAME,
        "queue_size": app.state.queue.qsize() if hasattr(app.state, "queue") else "unknown",
        "timestamp": time.time()
    } 