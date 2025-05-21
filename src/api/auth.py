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
    
    # Обработка различных форматов токена
    # 1. Если токен начинается с "Bearer ", проверяем его как обычно
    # 2. Если нет префикса, проверяем сам токен
    if api_key.startswith("Bearer "):
        # Формат "Bearer TOKEN"
        provided_token = api_key[7:].strip()  # Удаляем 'Bearer ' и пробелы
    else:
        # Формат просто "TOKEN"
        provided_token = api_key.strip()
    
    if provided_token != API_TOKEN:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API token"
        )
    
    return api_key 