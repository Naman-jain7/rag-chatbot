import tiktoken
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode
from langsmith import traceable

from app.core.config import RAW_DB_PATH, settings
from app.schemas.workflow import EvidenceEvaluation, QueryRouter
from app.vector_db.chroma_db import ChromaManager
from app.services.memory_service import memory_service
from src.graphs.state import GraphState
from src.graphs.tools import tools
from src.llm.providers import fast_providers, slow_providers
from src.llm.resilience import ResilientLLMManager
from src.prompts.prompts import (
    EVALUATE_EVIDENCE_PROMPT,
    GENERATE_ANSWER_PROMPT,
    REWRITE_QUERY_PROMPT,
    ROUTER_PROMPT,
)
from src.utils.logger import LLM_LOGGER

load_dotenv()


fast_llm = ResilientLLMManager(providers=fast_providers)  # type: ignore
slow_llm = ResilientLLMManager(providers=slow_providers)  # type: ignore


async def route_query(state: GraphState):
    print("--- ROUTING QUERY ---")
    query = state["query"]
    memory_context = state.get("memories", "")
    prompt = ROUTER_PROMPT.format(query=query, memory_context=memory_context)
    msg = [{"role": "user", "content": prompt}]
    
    try:
        response = await fast_llm.generate_structured(messages=msg, schema=QueryRouter)
        route = response.route.value # type: ignore
        confidence = response.confidence_score # type: ignore
    except Exception as e:
        LLM_LOGGER.error(f"Error routing query: {e}")
        route = "retrieve"
        confidence = 1.0

    print(f"--- ROUTE: {route} | CONFIDENCE: {confidence} ---")
    return {"route": route, "confidence": confidence}


async def retrieve_memories(state: GraphState):
    print("--- RETRIEVING MEMORIES ---")
    user_id = state.get("user_id")
    query = state.get("original_query") or state.get("query")
    memories = memory_service.get_relevant(user_id, query) if user_id else ""
    return {"memories": memories}


async def retrieve_docs(state: GraphState):
    print("--- RETRIEVING DOCS FROM CHROMA ---")
    query = state["query"]
    retries = state.get("retries", 0)

    chroma_manager = ChromaManager(
        collection_name=state.get("namespace", "default"),
        persist_dir=str(RAW_DB_PATH),
        model_name=settings.embedding.EMBEDDING_LLM,
    )
    docs = chroma_manager.retrieve_and_rerank(
        query=query, 
        initial_k=15, 
        final_top_k=3, 
        collection_name=state.get("namespace", "default")
    )
    if not docs:
        return {"raw_documents": [], "final_context": "", "retries": retries}
    context = "\n\n".join([f"Document:\n{d['content']}" for d in docs])
    return {"raw_documents": docs, "final_context": context, "retries": retries}


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
    query = state["query"]
    retries = state.get("retries", 0)

    prompt = REWRITE_QUERY_PROMPT.format(query=query)
    msg = [{"role": "user", "content": prompt}]

    try:
        chunks = []
        async for chunk in slow_llm.generate_stream(messages=msg):
            chunks.append(chunk)
        rewritten_query = "".join(chunks).strip()
    except Exception as e:
        LLM_LOGGER.error(f"Error rewriting query: {e}")
        rewritten_query = query

    return {"query": rewritten_query, "retries": retries + 1}

@traceable(run_type='llm', name="Answer generation")
async def generate_answer(state: GraphState):
    print("--- GENERATING ANSWER ---")
    query = state.get("original_query") or state["query"]
    context = state.get("final_context", "")

    messages = state.get("messages", [])
    if not messages:
        memory_context = state.get("memories", "")
        prompt = GENERATE_ANSWER_PROMPT.format(query=query, context=context, memory_context=memory_context)
        messages = [HumanMessage(content=prompt)]

    try:
        chunks = []
        async for chunk in fast_llm.generate_stream(messages=messages): # type:ignore
            chunks.append(chunk)
            if chunk.content and "stream_queue" in state and state["stream_queue"] is not None:
                await state["stream_queue"].put(chunk.content)
        
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
        if "stream_queue" in state and state["stream_queue"] is not None:
            await state["stream_queue"].put(content)

    try:
        enc = tiktoken.get_encoding("cl100k_base")
        prompt_tokens = len(enc.encode(str(messages[-1].content))) if hasattr(messages[-1], "content") else 0
        response_tokens = len(enc.encode(content)) if content else 0 # type:ignore
    except Exception:
        prompt_tokens = 0
        response_tokens = 0

    usage = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": response_tokens,
        "total_tokens": prompt_tokens + response_tokens,
    }

    if len(state.get("messages", [])) == 0:
        return {"messages": [messages[0], final_message], "response": content, "usage": usage}
    else:
        return {"messages": [final_message], "response": content, "usage": usage}


async def store_memories(state: GraphState):
    print("--- STORING MEMORIES ---")
    user_id = state.get("user_id")
    query = state.get("original_query") or state.get("query")
    response = state.get("response")
    if user_id and response:
        memory_service.store(user_id, query, response)
    return {}


def route_decision(state: GraphState):
    confidence = state.get("confidence")
    route = state.get("route", "retrieve")
    retries = state.get("retries", 0)
    
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

graph = workflow.compile()
