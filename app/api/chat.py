import asyncio
import uuid

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage
from app.schemas.workflow_schema import ChatRequest
from app.db.manager import db_manager

router = APIRouter()

@router.post("/stream")
async def chat_endpoint(request: Request, payload: ChatRequest):
    """Stream an answer from the LangGraph RAG pipeline."""
    # Graph is compiled at startup with the real AsyncPostgresSaver and stored
    # on app.state; fall back to the module-level graph if something went wrong.
    import src.graphs.chains as chains_module
    from src.graphs.chains import _stream_queue_var
    graph = getattr(request.app.state, "graph", chains_module.graph)
    chat_id = payload.chat_id or str(uuid.uuid4())
    await db_manager.execute_command(
        """
        INSERT INTO chat_conversations (user_id, chat_id, title)
        VALUES ($1, $2, $3)
        ON CONFLICT (user_id, chat_id)
        DO UPDATE SET updated_at = CURRENT_TIMESTAMP
        """,
        payload.user_id,
        chat_id,
        payload.query[:80],
    )
    config = {
        "configurable":{"thread_id": chat_id, "user_id": payload.user_id}
    }
    queue: asyncio.Queue = asyncio.Queue()

    initial_state = {
        "user_id": payload.user_id,
        "chat_id": chat_id,
        
        "original_query": payload.query,
        "query": payload.query,
        "retries": 0,
        "messages": [HumanMessage(content=payload.query)],
        
        "final_context": "",
        
        "evidence_sufficient": False,
        
        "raw_documents": [],
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
                    usage = final_state.get("token_usage", {})
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


@router.get("/history/{user_id}/{chat_id}")
async def get_chat_history(request: Request, user_id: int | str, chat_id: str):
    owner = await db_manager.fetch_rows(
        "SELECT 1 FROM chat_conversations WHERE user_id = $1 AND chat_id = $2",
        int(user_id),
        chat_id,
    )
    if not owner:
        raise HTTPException(status_code=404, detail="Conversation not found")

    import src.graphs.chains as chains_module
    graph = getattr(request.app.state, "graph", chains_module.graph)
    config = {"configurable": {"thread_id": chat_id, "user_id": user_id}}

    state = await graph.aget_state(config) # type: ignore

    messages = []
    token_usage = {}

    if state and state.values:
        if "messages" in state.values:
            for msg in state.values["messages"]:
                role = "user" if msg.type == "human" else "assistant"
                messages.append({"role": role, "content": msg.content})
        
        token_usage = state.values.get("token_usage", {})

    return {"messages": messages, "token_usage": token_usage}


@router.get("/conversations/{user_id}")
async def list_conversations(user_id: int):
    conversations = await db_manager.fetch_rows(
        """
        SELECT chat_id AS id, title, created_at, updated_at
        FROM chat_conversations
        WHERE user_id = $1
        ORDER BY updated_at DESC
        """,
        user_id,
    )
    return {"conversations": conversations}
