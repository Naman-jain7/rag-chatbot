from typing import List, Annotated, TypedDict, Any
from langgraph.graph.message import add_messages
import operator

class GraphState(TypedDict):
    query: str
    original_query: str
    model_name: str
    namespace: str
    raw_documents: Annotated[List[dict], operator.add]
    final_context: str
    response: str
    usage: dict
    retries: int
    evidence_sufficient: bool
    stream_queue: Any
    messages: Annotated[list[Any], add_messages]
    user_id: int
    memories: str
