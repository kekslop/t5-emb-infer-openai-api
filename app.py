from sentence_transformers import SentenceTransformer
from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
from typing import List, Union, Dict, Optional, Any
import time
import os
import gc
import torch
import numpy as np
import asyncio
from contextlib import asynccontextmanager

# Constants - using environment variables
LOCAL_MODEL_PATH = os.environ.get("MODEL_PATH", "./FRIDA")  # Local path to your existing model
MODEL_NAME = os.environ.get("MODEL_NAME", "ai-forever/FRIDA")  # Model name can be customized via env
API_TOKEN = os.environ.get("API_TOKEN", "default_token_change_me")
MAX_QUEUE_SIZE = int(os.environ.get("MAX_QUEUE_SIZE", 10))

# Request queue for handling concurrent requests
request_queue = asyncio.Queue(maxsize=MAX_QUEUE_SIZE)
processing_semaphore = asyncio.Semaphore(1)  # Only process one request at a time

# Updated Request/Response Models for exact OpenAI compatibility
class EmbeddingRequest(BaseModel):
    model: str = MODEL_NAME
    input: Union[str, List[str], List[int]] = []  # Default to empty list as requested
    encoding_format: str = "float"
    dimensions: Optional[int] = None
    user: Optional[str] = None
    # Additional OpenAI parameters
    truncate_prompt_tokens: Optional[int] = None
    additional_data: Optional[str] = None
    add_special_tokens: Optional[bool] = None
    priority: Optional[int] = None
    # Allow for any additional properties
    class Config:
        extra = "allow"

class EmbeddingData(BaseModel):
    object: str = "embedding"
    index: int
    embedding: List[float]

class EmbeddingResponse(BaseModel):
    data: List[EmbeddingData]
    model: str
    object: str = "list"
    usage: Dict[str, int]

# Model information
class ModelData(BaseModel):
    id: str
    object: str = "model"
    created: int = int(time.time())
    owned_by: str = "organization"
    permission: List[Dict[str, Any]] = []
    root: str = MODEL_NAME
    parent: Optional[str] = None

class ModelList(BaseModel):
    object: str = "list"
    data: List[ModelData]

# Authentication dependency
api_key_header = APIKeyHeader(name="Authorization", auto_error=False)

async def verify_token(api_key: Optional[str] = Depends(api_key_header)):
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API token"
        )
    if api_key != f"Bearer {API_TOKEN}":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API token"
        )
    return api_key

async def queue_manager():
    """Process requests from the queue to avoid overloading the GPU."""
    while True:
        try:
            # Get the next request from the queue
            func, args, kwargs, future = await request_queue.get()
            
            # Process the request
            try:
                async with processing_semaphore:
                    result = await func(*args, **kwargs)
                future.set_result(result)
            except Exception as e:
                future.set_exception(e)
            finally:
                request_queue.task_done()
        except Exception as e:
            print(f"Error in queue manager: {e}")
            await asyncio.sleep(1)  # Prevent tight loop on persistent errors

async def enqueue_request(func, *args, **kwargs):
    """Add a request to the queue and return a future for the result."""
    if request_queue.full():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Server is currently handling maximum number of requests. Please try again later."
        )
    
    future = asyncio.Future()
    await request_queue.put((func, args, kwargs, future))
    return await future

def clear_gpu_memory():
    """Clear GPU memory cache."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        if hasattr(torch.cuda, 'ipc_collect'):
            torch.cuda.ipc_collect()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the model at startup
    print("Loading model from local path...")
    start_load_time = time.time()
    clear_gpu_memory()  # Clear memory before loading

    try:
        # Verify the local model path exists
        if not os.path.exists(LOCAL_MODEL_PATH):
            raise FileNotFoundError(f"Local model path not found: {LOCAL_MODEL_PATH}")
        
        # Use GPU if available, otherwise CPU
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        print(f"Loading model from: {LOCAL_MODEL_PATH}")
        
        # Load the model from local path
        app.state.encoder = SentenceTransformer(
            LOCAL_MODEL_PATH,  # Use local path instead of model ID
            device=device,
            trust_remote_code=True
        )
        
        load_time = time.time() - start_load_time
        print(f"Model loaded successfully in {load_time:.2f} seconds on {device}!")
        print(f"Model will be served as: {MODEL_NAME}")
        
        # Start the queue manager
        app.state.queue_task = asyncio.create_task(queue_manager())
        print(f"Request queue initialized with max size: {MAX_QUEUE_SIZE}")
    except Exception as e:
        print(f"Error loading model: {e}")
        raise

    yield  # API is ready to serve requests

    # Clean up resources on shutdown
    print("Releasing resources...")
    if hasattr(app.state, "queue_task"):
        app.state.queue_task.cancel()
        try:
            await app.state.queue_task
        except asyncio.CancelledError:
            pass
    
    if hasattr(app.state, "encoder") and app.state.encoder is not None:
        del app.state.encoder
    clear_gpu_memory()
    print("Resources released.")
    
app = FastAPI(
    lifespan=lifespan,
    title="Embedding Model API",
    description="API for text embeddings",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# OpenAI-compatible routes
@app.get("/v1/models")
async def list_models(token: str = Depends(verify_token)):
    """List available models in OpenAI format."""
    models = [
        ModelData(id=MODEL_NAME)
    ]
    return ModelList(data=models)

@app.get("/v1/models/{model_id}")
async def get_model(model_id: str, token: str = Depends(verify_token)):
    """Get model information in OpenAI format."""
    # Accept both the specified model name and a generic name for compatibility
    if model_id != MODEL_NAME and model_id != "text-embedding-3-small":
        raise HTTPException(status_code=404, detail=f"Model {model_id} not found")
    
    return ModelData(id=MODEL_NAME)

# Actual embedding processing function
async def process_embeddings(request: EmbeddingRequest, encoder):
    # Process input
    try:
        # Handle different types of input
        if isinstance(request.input, str):
            inputs = [request.input]
        elif isinstance(request.input, list):
            if not request.input:  # Handle empty list case
                return EmbeddingResponse(
                    data=[],
                    model=request.model,
                    usage={"prompt_tokens": 0, "total_tokens": 0}
                )
            elif all(isinstance(item, int) for item in request.input):
                # Handle token IDs - convert to string (simplified approach)
                inputs = [" ".join(map(str, request.input))]
            else:
                inputs = request.input
        else:
            raise HTTPException(status_code=400, detail="Invalid input format")
        
        # Extract prompt_name from additionalProp if exists
        prompt_name = None
        for key, value in request.dict(exclude_unset=True).items():
            if key.startswith("additionalProp") and isinstance(value, dict) and "prompt_name" in value:
                prompt_name = value["prompt_name"]
                break
        
        # Generate embeddings
        start_time = time.time()
        if prompt_name:
            embeddings = encoder.encode(inputs, prompt_name=prompt_name)
        else:
            embeddings = encoder.encode(inputs)
        
        # Count tokens (approximate)
        total_tokens = sum(len(text.split()) * 1.3 for text in inputs)  # Rough estimate
        
        # Format response
        data = [
            EmbeddingData(index=i, embedding=emb.tolist())
            for i, emb in enumerate(embeddings)
        ]
        
        return EmbeddingResponse(
            data=data,
            model=request.model,
            usage={"prompt_tokens": int(total_tokens), "total_tokens": int(total_tokens)}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing embeddings: {str(e)}")

@app.post("/v1/embeddings", response_model=EmbeddingResponse)
async def create_embeddings(request: EmbeddingRequest, token: str = Depends(verify_token)):
    """Create embeddings using the model (OpenAI-compatible endpoint)."""
    encoder = app.state.encoder
    return await enqueue_request(process_embeddings, request, encoder)

@app.post("/v1/embedding", response_model=EmbeddingResponse)
async def create_embedding(request: EmbeddingRequest, token: str = Depends(verify_token)):
    """Create embeddings using the model (custom endpoint with empty list as default input)."""
    encoder = app.state.encoder
    return await enqueue_request(process_embeddings, request, encoder)

# Root endpoint for API info
@app.get("/")
async def root():
    return JSONResponse(content={
        "message": "Welcome to the Embedding API!",
        "model": MODEL_NAME,
        "model_path": LOCAL_MODEL_PATH,
        "endpoints": {
            "OpenAI-compatible": [
                "/v1/models",
                "/v1/models/{model_id}",
                "/v1/embeddings"
            ],
            "Custom": [
                "/v1/embedding"
            ]
        },
        "documentation": "/docs",
        "authentication": "Bearer token required in Authorization header"
    })

@app.get("/health")
async def health_check():
    """Health check endpoint to verify the API is running."""
    return {
        "status": "ok",
        "model": MODEL_NAME,
        "queue_size": request_queue.qsize(),
        "max_queue_size": MAX_QUEUE_SIZE
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)