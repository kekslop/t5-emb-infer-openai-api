import time
import os
import gc
import torch
import asyncio

from src.core.config import LOCAL_MODEL_PATH, MODEL_NAME
from src.models.schemas import EmbeddingRequest, EmbeddingData, EmbeddingResponse

def clear_gpu_memory():
    """Clear GPU memory cache."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        if hasattr(torch.cuda, 'ipc_collect'):
            torch.cuda.ipc_collect()

async def load_model():
    """Load the model at startup."""
    from sentence_transformers import SentenceTransformer
    
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
        encoder = SentenceTransformer(
            LOCAL_MODEL_PATH,  # Use local path instead of model ID
            device=device,
            trust_remote_code=True
        )
        
        load_time = time.time() - start_load_time
        print(f"Model loaded successfully in {load_time:.2f} seconds on {device}!")
        print(f"Model will be served as: {MODEL_NAME}")
        
        return encoder
    except Exception as e:
        print(f"Error loading model: {e}")
        raise

async def process_embeddings(request: EmbeddingRequest, encoder):
    """Process embeddings for the given request."""
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
            raise ValueError("Invalid input format")
        
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
        
        compute_time = time.time() - start_time
        print(f"Generated {len(data)} embeddings in {compute_time:.2f}s")
        
        return EmbeddingResponse(
            data=data,
            model=request.model,
            usage={
                "prompt_tokens": int(total_tokens),
                "total_tokens": int(total_tokens)
            }
        )
    except Exception as e:
        print(f"Error generating embeddings: {e}")
        raise 