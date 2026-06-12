from typing import List, Annotated, TypedDict, Any
from langgraph.graph.message import add_messages
import operator

class GraphState(TypedDict):
    user_id: int
    chat_id: str
    original_query: str
    query: str
    model_name: str
    namespace: str
    raw_documents: Annotated[List[dict], operator.add]
    final_context: str
    response: str
    token_usage: dict
    retries: int
    evidence_sufficient: bool
    messages: Annotated[list[Any], add_messages]
    memories: str
    route: str
    confidence: float
