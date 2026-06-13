from typing import Annotated, TypedDict, Any
from langgraph.graph.message import add_messages

class GraphState(TypedDict):
    user_id: int
    chat_id: str

    route: str

    original_query: str
    query: str
    retries: int
    messages: Annotated[list[Any], add_messages]

    memories: str
    final_context: str

    evidence_sufficient: bool
    confidence: float

    raw_documents: list[dict]

    response: str
    token_usage: dict
