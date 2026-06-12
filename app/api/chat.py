import asyncio

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage
from app.schemas.workflow_schema import ChatRequest

router = APIRouter()

@router.post("/chat/stream")
async def chat_endpoint(request: Request, payload: ChatRequest):
    """Stream an answer from the LangGraph RAG pipeline."""
    # Graph is compiled at startup with the real AsyncPostgresSaver and stored
    # on app.state; fall back to the module-level graph if something went wrong.
    import src.graphs.chains as chains_module
    from src.graphs.chains import _stream_queue_var
    graph = getattr(request.app.state, "graph", chains_module.graph)
    config = {
        "configurable":{
            "thread_id": payload.chat_id,
            "user_id": payload.user_id
        }
    }
    queue: asyncio.Queue = asyncio.Queue()

    initial_state = {
        "query": payload.query,
        "original_query": payload.query,
        "namespace": f"user_{payload.user_id}",
        "retries": 0,
        "user_id": payload.user_id,
        "messages": [HumanMessage(content=payload.query)],
    }

    async def generate_response():
        try:
            async def run_graph():
                try:
                    # Bind the queue to the ContextVar so generate_answer can
                    # stream chunks without putting the Queue in GraphState
                    # (which would fail msgpack serialisation in the checkpointer).
                    _stream_queue_var.set(queue)
                    final_state = await graph.ainvoke(initial_state, config=config)  # type: ignore
                    
                    from src.utils.citations import format_citations
                    citations_text = format_citations(final_state)
                    if citations_text:
                        await queue.put(citations_text)
                    usage = final_state.get("usage", {})
                    if usage:
                        import json
                        await queue.put(f"\n__META__{json.dumps(usage)}")
                except Exception as e:
                    await queue.put(f"\n[Graph Error: {str(e)}]")
                finally:
                    _stream_queue_var.set(None)
                    await queue.put(None)

            asyncio.create_task(run_graph())

            while True:
                chunk = await queue.get()
                if chunk is None:
                    break
                yield chunk
        except Exception as e:
            yield f"Error: {str(e)}"

    return StreamingResponse(generate_response(), media_type="text/plain")


@router.get("/chat/history/{user_id}/{chat_id}")
async def get_chat_history(request: Request, user_id: int | str, chat_id: str):
    import src.graphs.chains as chains_module
    graph = getattr(request.app.state, "graph", chains_module.graph)
    config = {"configurable": {"thread_id": chat_id, "user_id": user_id}}

    # Fetch existing state directly out of Postgres tables using LangGraph engine
    state = await graph.aget_state(config) # type: ignore

    messages = []

    if state and state.values:
        if "messages" in state.values:
            for msg in state.values["messages"]:
                role = "user" if msg.type == "human" else "assistant"
                messages.append({"role": role, "content": msg.content})
        
        token_usage = state.values.get("token_usage", {})

    return {"messages": messages, "token_usage": token_usage}