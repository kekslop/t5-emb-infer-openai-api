from pydantic import BaseModel
from typing import List, Union, Dict, Optional, Any
import time

from src.core.config import MODEL_NAME

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