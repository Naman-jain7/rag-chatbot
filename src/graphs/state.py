from typing import Annotated, Any, TypedDict

from langgraph.graph.message import add_messages


class GraphState(TypedDict):
    user_id: int
    chat_id: str

    original_query: str
    query: str
    retries: int
    messages: Annotated[list[Any], add_messages]

    final_context: str

    short_term_sufficient: bool
    evidence_sufficient: bool
    confidence: float

    raw_documents: list[dict]

    response: str
    token_usage: dict
