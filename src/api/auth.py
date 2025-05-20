from fastapi import HTTPException, status, Depends
from fastapi.security import APIKeyHeader
from typing import Optional

from src.core.config import API_TOKEN

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