from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import asyncio
from contextlib import asynccontextmanager

from src.services.model_service import load_model, clear_gpu_memory
from src.services.queue_service import queue_manager, request_queue
from src.api.routes import router

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the model at startup
    app.state.encoder = await load_model()
    
    # Start the queue manager
    app.state.queue = request_queue
    app.state.queue_task = asyncio.create_task(queue_manager())
    print(f"Request queue initialized")
    
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

# Include API routes
app.include_router(router) 