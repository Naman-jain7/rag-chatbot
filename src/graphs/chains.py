import asyncio
from asyncio import Queue as AsyncQueue
from contextvars import ContextVar
from typing import Optional

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_huggingface import HuggingFaceEmbeddings
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode
from langsmith import traceable

from app.core.config import settings
from app.schemas.workflow_schema import EvidenceEvaluation, QueryRouter, MemoryDecision
from src.graphs.state import GraphState
from src.graphs.tools import tools
from src.llm.providers import fast_providers, slow_providers
from src.llm.resilience import ResilientLLMManager
from src.memory.long_term_memory import long_term_memory as memory_service
from src.memory.vector_db_manager import vector_db_manager
from src.utils.token_usage import get_token_usage
from src.prompts.prompts import EVALUATE_EVIDENCE_PROMPT,REWRITE_QUERY_PROMPT,ROUTER_PROMPT,MEMORY_PROMPT,GENERATE_ANSWER_PROMPT
from src.utils.logger import LLM_LOGGER, MEMORY_LOGGER, APP_LOGGER

# ContextVar that carries the streaming queue into generate_answer without
# touching the checkpointed GraphState (asyncio.Queue is not serialisable).
_stream_queue_var: ContextVar[Optional[AsyncQueue]] = ContextVar("stream_queue", default=None)

_query_embeddings: HuggingFaceEmbeddings | None = None

def _get_query_embeddings() -> HuggingFaceEmbeddings:
    global _query_embeddings
    if _query_embeddings is None:
        LLM_LOGGER.info("Loading HuggingFaceEmbeddings for query encoding...")
        _query_embeddings = HuggingFaceEmbeddings(model_name=settings.embedding.EMBEDDING_LLM)
    return _query_embeddings

load_dotenv()

# ==========================================================================================

fast_llm = ResilientLLMManager(providers=fast_providers)  # type: ignore
slow_llm = ResilientLLMManager(providers=slow_providers)  # type: ignore

# ==========================================================================================

async def retrieve_memories(state: GraphState):
    print("--- RETRIEVING MEMORIES ---")
    user_id = state.get("user_id")
    query = state.get("query") or state.get("original_query")
    memories = await memory_service.get_relevant(user_id, query) if user_id else ""
    return {"memories": memories}

# ==========================================================================================

_DOCUMENT_KEYWORDS = {
    "document", "documents", "file", "files", "resume", "cv",
    "report", "pdf", "uploaded", "upload", "attachment", "notes",
    "according to", "based on", "in my", "from my", "from the",
    "what does", "what did", "what is in", "summarise", "summarize",
    "paper", "thesis", "contract", "invoice", "letter",
}

@traceable(run_type="llm", name="Route query")
async def route_query(state: GraphState):
    print("--- ROUTING QUERY ---")
    query = state.get("query")
    memory_context = state.get("memories", "")
    prompt = ROUTER_PROMPT.format(query=query, memory_context=memory_context)
    msg = [{"role": "user", "content": prompt}]

    try:
        response = await fast_llm.generate_structured(messages=msg, schema=QueryRouter)
        route = response.route.value  # type: ignore
        confidence = response.confidence_score  # type: ignore
    except Exception as e:
        LLM_LOGGER.error(f"Error routing query: {e}")
        route = "retrieve"
        confidence = 1.0

    print(f"--- ROUTE: {route} | CONFIDENCE: {confidence} ---")
    return {"route": route, "confidence": confidence}

def route_decision(state: GraphState):
    query_lower = (state.get("original_query") or state.get("query", "")).lower()
    route = state.get("route", "retrieve")
    confidence = state.get("confidence", 0.5)
    retries = state.get("retries", 0)

    # Hard override: any query referencing document content must go to retrieve.
    # This prevents memory from being used when the user explicitly asks about
    # an uploaded file, even if the LLM router picked "memory".
    if any(kw in query_lower for kw in _DOCUMENT_KEYWORDS):
        print("--- ROUTE OVERRIDE → retrieve (document keyword detected) ---")
        return "retrieve"

    if confidence < 0.75 and retries < 1:
        return "rewrite"

    if route == "memory":
        return "generate"
    elif route == "retrieve":
        return "retrieve"
    elif route == "hybrid":
        return "retrieve"
    elif route == "tools":
        return "generate"
    else:
        return "retrieve"

def route_evaluation(state: GraphState):
    if state.get("evidence_sufficient", False) or state.get("retries", 0) >= 1:
        return "generate"
    return "rewrite"

def route_tools(state: GraphState):
    messages = state.get("messages", [])
    if messages:
        last_message = messages[-1]
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tools"
    return "end"

# ==========================================================================================

async def store_memories(state: GraphState):
    print("--- STORING MEMORIES ---")
    user_id = state.get("user_id")
    query = state.get("original_query")
    response = state.get("response")
    if not user_id or not response:
        return {}
    
    try:
        existing_raw_memories = await memory_service.get_all_raw_texts(user_id)
        if existing_raw_memories:
            user_details = "\n".join([f"- {text}" for text in existing_raw_memories])
        else:
            user_details = "No existing memories"
        
        prompt = MEMORY_PROMPT.format(query=query, memory_context=user_details)
        user_msg = f"User Query: {query}\nAssistant Response: {response}"
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user","content": f"Please review this interaction and extract any new memories:\n{user_msg}"},
        ]

        decision = await fast_llm.generate_structured(messages=messages, schema=MemoryDecision)

        if decision and decision.should_write:
            new_memories = [m.text for m in decision.memories if m.is_new]
            if new_memories:
                await memory_service.store_batch(user_id, new_memories)
    except Exception as e:
        MEMORY_LOGGER.error(f"Error in store_memories node workflow: {e}")
    
    return {}

# ==========================================================================================

async def retrieve_docs(state: GraphState):
    print("--- RETRIEVING DOCS FROM POSTGRES ---")
    user_id = state.get("user_id")
    query = state.get("query") or state.get("original_query")
    retries = state.get("retries", 0)

    if not user_id:
        return {"raw_documents": [], "final_context": "", "retries": retries}

    matched_filenames = await vector_db_manager.find_matching_filenames(user_id, query)
    if matched_filenames:
        # Whole-document requests such as "summarise resume 1" need all chunks
        # in source order, not only the five chunks preferred by a reranker.
        raw_docs = await vector_db_manager.get_document_chunks(user_id, matched_filenames)
    else:
        embeddings_model = await asyncio.to_thread(_get_query_embeddings)
        query_embedding = await asyncio.to_thread(embeddings_model.embed_query, query)
        raw_docs = await vector_db_manager.retrieve_and_rerank(
            user_id=user_id,
            query_text=query,
            query_embedding=query_embedding,
            top_k=5,
        )

    if not raw_docs:
        return {"raw_documents": [], "final_context": "", "retries": retries}

    # Normalise to the dict shape the rest of the graph expects
    docs = [
        {
            "content": d["chunk_text"],
            "metadata": {
                "filename": d.get("filename", "unknown"),
                "page_number": d.get("page_number", "unknown"),
                "section_id": d.get("section_id", ""),
            },
        }
        for d in raw_docs
    ]

    context = "\n\n".join([
        f"Document [{d['metadata']['filename']} - Page {d['metadata']['page_number']}]:\n{d['content']}"
        for d in docs
    ])
    return {"raw_documents": docs, "final_context": context, "retries": retries}

# ==========================================================================================

@traceable(run_type="llm", name="Evaluate evidence")
async def evaluate_evidence(state: GraphState):
    print("--- EVALUATING EVIDENCE ---")
    query = state["query"]
    context = state.get("final_context", "")

    # If no docs returned, fail fast
    if not state.get("raw_documents"):
        return {"evidence_sufficient": False}

    prompt = EVALUATE_EVIDENCE_PROMPT.format(query=query, context=context)
    msg = [{"role": "user", "content": prompt}]

    try:
        response = await fast_llm.generate_structured(messages=msg, schema=EvidenceEvaluation)
        evidence_sufficient = response.evidence_sufficient  # type: ignore
    except Exception as e:
        LLM_LOGGER.error(f"Error evaluating evidence: {e}")
        # Default to True to avoid endless looping on error
        evidence_sufficient = True

    return {"evidence_sufficient": evidence_sufficient}

@traceable(run_type="llm", name="Rewrite query")
async def rewrite_query(state: GraphState):
    print(f"--- REWRITING QUERY (Retry {state.get('retries', 0) + 1}/1) ---")
    query = state.get("query") or state.get("original_query")
    retries = state.get("retries", 0)

    prompt = REWRITE_QUERY_PROMPT.format(query=query)
    msg = [{"role": "user", "content": prompt}]

    try:
        chunks = []
        async for chunk in slow_llm.generate_stream(messages=msg):
            chunks.append(chunk.content if hasattr(chunk, "content") else str(chunk))
        rewritten_query = "".join(chunks).strip()
    except Exception as e:
        LLM_LOGGER.error(f"Error rewriting query: {e}")
        rewritten_query = query

    return {"query": rewritten_query, "retries": retries + 1}

# ==========================================================================================

@traceable(run_type='llm', name="Answer generation")
async def generate_answer(state: GraphState):
    print("--- GENERATING ANSWER ---")
    query = state.get("query") or state.get("original_query")
    context = state.get("final_context", "")
    memory_context = state.get("memories", "")
    chat_id = state.get("chat_id", "")

    formatted_system_prompt = GENERATE_ANSWER_PROMPT.format(query=query, context=context, memory_context=memory_context)

    system_message = SystemMessage(content=formatted_system_prompt)
    
    messages_list = state.get("messages", [])
    if not messages_list:
        messages_list = [HumanMessage(content=query)]

    llm_messages = [system_message] + messages_list

    stream_queue = _stream_queue_var.get()
    try:
        chunks = []
        async for chunk in fast_llm.generate_stream(messages=llm_messages):  # type: ignore
            chunks.append(chunk)
            if chunk.content and stream_queue is not None:
                await stream_queue.put(chunk.content)

        if chunks:
            final_message = chunks[0]
            for chunk in chunks[1:]:
                final_message += chunk
        else:
            final_message = AIMessage(content="")

        content = final_message.content
    except Exception as e:
        content = f"Error during generation: {str(e)}"
        final_message = AIMessage(content=content)
        if stream_queue is not None:
            await stream_queue.put(content)
    
    try:
        usage = get_token_usage(session_id=chat_id)
    except Exception as e:
        APP_LOGGER.error(f"Failed fetching external token metrics: {e}")
        usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
    
    return {"messages": [final_message], "response": content, "token_usage": usage}

# ==========================================================================================

workflow = StateGraph(GraphState)

workflow.add_node("route_query", route_query)
workflow.add_node("retrieve_memories", retrieve_memories)
workflow.add_node("retrieve", retrieve_docs)
workflow.add_node("evaluate", evaluate_evidence)
workflow.add_node("rewrite", rewrite_query)
workflow.add_node("generate", generate_answer)
workflow.add_node("tools", ToolNode(tools))
workflow.add_node("store_memories", store_memories)


workflow.add_edge(START, "retrieve_memories")
workflow.add_edge("retrieve_memories", "route_query")

workflow.add_conditional_edges(
    "route_query", 
    route_decision, 
    {
        "rewrite": "rewrite",
        "retrieve": "retrieve",
        "generate": "generate"
    }
)

workflow.add_edge("retrieve", "evaluate")
workflow.add_conditional_edges("evaluate", route_evaluation, {"generate": "generate", "rewrite": "rewrite"})
workflow.add_edge("rewrite", "retrieve_memories")

workflow.add_conditional_edges("generate",route_tools,{"tools": "tools", "end": "store_memories"})
workflow.add_edge("tools", "generate")
workflow.add_edge("store_memories", END)

graph = workflow.compile()  # compiled without checkpointer as a safe default
