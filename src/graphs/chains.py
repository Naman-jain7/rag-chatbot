import asyncio
from asyncio import Queue as AsyncQueue
from contextvars import ContextVar

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_huggingface import HuggingFaceEmbeddings
from langgraph.graph import END, START, StateGraph
from langsmith import traceable

from app.core.config import settings
from app.schemas.workflow_schema import EvidenceEvaluation, ShortTermSufficiency
from src.graphs.state import GraphState
from src.llm.providers import (  # noqa: F401
    gemini_llm,
    ollama_llm,
    ollama_local_llm,
    openrouter_llm,
)
from src.memory.vector_db_manager import vector_db_manager
from src.prompts.prompts import (
    CHECK_SHORT_TERM_PROMPT,
    EVALUATE_EVIDENCE_PROMPT,
    GENERATE_ANSWER_PROMPT,
    REWRITE_QUERY_PROMPT,
)
from src.utils.logger import APP_LOGGER, EMBEDDING_LOGGER, LLM_LOGGER
from src.utils.token_usage import get_token_usage

# ContextVar that carries the streaming queue into generate_answer without touching the checkpointed GraphState (asyncio.Queue is not serialisable).
_stream_queue_var: ContextVar[AsyncQueue | None] = ContextVar("stream_queue", default=None)

load_dotenv()

# ===================================== HELPER FUNCTIONS =====================================================

_query_embeddings: HuggingFaceEmbeddings | None = None

def _get_query_embeddings() -> HuggingFaceEmbeddings:
    global _query_embeddings
    if _query_embeddings is None:
        LLM_LOGGER.info("Loading HuggingFaceEmbeddings for query encoding...")
        _query_embeddings = HuggingFaceEmbeddings(model_name=settings.embedding.EMBEDDING_MODEL)
    return _query_embeddings

_DOCUMENT_KEYWORDS = {
    "document", "documents", "file", "files", "resume", "cv",
    "report", "pdf", "uploaded", "upload", "attachment", "notes",
    "according to", "based on",
    "what does", "what did", "what is in", "summarise", "summarize",
    "paper", "thesis", "contract", "invoice", "letter",
}


def _contains_any_keyword(text: str, keywords: set[str]) -> bool:
    return any(keyword in text for keyword in keywords)


def _is_document_query(state: GraphState) -> bool:
    query = state.get("query") or state.get("original_query") or ""
    query_lower = query.lower()
    return _contains_any_keyword(query_lower, _DOCUMENT_KEYWORDS)


def _route_after_evaluation(state: GraphState):
    if state.get("evidence_sufficient", False) or state.get("retries", 0) >= 1:
        return "generate"
    if not _is_document_query(state):
        return "generate"
    return "rewrite"

def _route_after_short_term_check(state: GraphState):
    if state.get("short_term_sufficient", False):
        return "generate"
    return "retrieve"

# ==========================================================================================

@traceable(run_type="llm", name="Check short term context")
async def check_short_term(state: GraphState):
    print("--- CHECKING SHORT TERM CONTEXT ---")
    
    if _is_document_query(state):
        print("    -> Document query detected, bypassing to retrieval.")
        return {"short_term_sufficient": False}

    query = state.get("query") or state.get("original_query")
    messages = state.get("messages", [])
    
    # Exclude the very last message which is the current query itself
    history_msgs = messages[:-1] if messages else []
    if not history_msgs:
        return {"short_term_sufficient": False}

    chat_history = ""
    for msg in history_msgs:
        role = "User" if isinstance(msg, HumanMessage) else "Assistant"
        chat_history += f"{role}: {msg.content}\n"
    
    prompt = CHECK_SHORT_TERM_PROMPT.format(chat_history=chat_history.strip(), query=query)
    msg = [{"role": "user", "content": prompt}]
    
    try:
        response = await ollama_local_llm.generate_structured(messages=msg, schema=ShortTermSufficiency)
        is_sufficient = response.is_sufficient  # type: ignore
    except Exception as e:
        LLM_LOGGER.error(f"Error checking short term context: {e}")
        is_sufficient = False
        
    return {"short_term_sufficient": is_sufficient}

# ==========================================================================================

@traceable(run_type="chain", name="Retrieve documents")
async def retrieve_docs(state: GraphState):
    print("--- RETRIEVING DOCS ---")
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
        EMBEDDING_LOGGER.info('The LLM received no documents')
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

    def _format_doc(d: dict) -> str:
        filename = d['metadata']['filename']
        page_num = d['metadata']['page_number']
        is_audio = str(filename).lower().endswith(('.m4a', '.mp3', '.wav', '.ogg', '.flac'))
        
        if is_audio or page_num in (None, "unknown", "", 0, -1):
            return f"Document [{filename}]:\n{d['content']}"
        else:
            return f"Document [{filename} - Page {page_num}]:\n{d['content']}"

    context = "\n\n".join([_format_doc(d) for d in docs])
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
        response = await ollama_local_llm.generate_structured(messages=msg, schema=EvidenceEvaluation)
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
        async for chunk in ollama_local_llm.generate_stream(messages=msg):
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
    chat_id = state.get("chat_id", "")
    user_id = state.get("user_id", "")

    formatted_system_prompt = GENERATE_ANSWER_PROMPT.format(query=query, context=context)

    system_message = SystemMessage(content=formatted_system_prompt)
    
    messages_list = state.get("messages", [])
    if not messages_list:
        messages_list = [HumanMessage(content=query)]

    llm_messages = [system_message] + messages_list

    stream_queue = _stream_queue_var.get()
    try:
        chunks = []
        async for chunk in ollama_local_llm.generate_stream(messages=llm_messages):  # type: ignore
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
        usage = get_token_usage(user_id = user_id, session_id=chat_id)
    except Exception as e:
        APP_LOGGER.error(f"Failed fetching external token metrics: {e}")
        usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
    
    return {"messages": [final_message], "response": content, "token_usage": usage}

# ===================================== GRAPH =====================================================

workflow = StateGraph(GraphState)

workflow.add_node("check_short_term", check_short_term)
workflow.add_node("retrieve", retrieve_docs)
workflow.add_node("evaluate", evaluate_evidence)
workflow.add_node("rewrite", rewrite_query)
workflow.add_node("generate", generate_answer)


workflow.add_edge(START, "check_short_term")
workflow.add_conditional_edges(
    "check_short_term",
    _route_after_short_term_check,
    {"generate": "generate", "retrieve": "retrieve"}
)
workflow.add_edge("retrieve", "evaluate")
workflow.add_conditional_edges(
    "evaluate", 
    _route_after_evaluation, 
    {"generate": "generate", "rewrite": "rewrite"}
)
workflow.add_edge("rewrite", "retrieve")
workflow.add_edge("generate", END)

graph = workflow.compile()  # compiled without checkpointer as a safe default
