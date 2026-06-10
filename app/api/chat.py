import asyncio

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from src.graphs.chains import graph

router = APIRouter()

class ChatRequest(BaseModel):
    user_id: int
    query: str

@router.post("/chat")
async def chat_endpoint(request: ChatRequest):
    queue = asyncio.Queue()

    async def generate_response():
        try:
            initial_state = {
                "query": request.query,
                "original_query": request.query,
                "namespace": f"user_{request.user_id}",
                "retries": 0,
                "stream_queue": queue,
                "user_id": request.user_id
            }
            async def run_graph():
                try:
                    await graph.ainvoke(initial_state) # type: ignore
                except Exception as e:
                    await queue.put(f"\n[Graph Error: {str(e)}]")
                finally:
                    await queue.put(None)
                    
            # Run graph in background task
            asyncio.create_task(run_graph())
            
            while True:
                chunk = await queue.get()
                if chunk is None:
                    break
                yield chunk
        except Exception as e:
            yield f"Error: {str(e)}"

    return StreamingResponse(generate_response(), media_type="text/plain")
