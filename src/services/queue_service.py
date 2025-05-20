import asyncio
from fastapi import HTTPException, status

from src.core.config import MAX_QUEUE_SIZE

# Request queue for handling concurrent requests
request_queue = asyncio.Queue(maxsize=MAX_QUEUE_SIZE)
processing_semaphore = asyncio.Semaphore(1)  # Only process one request at a time

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