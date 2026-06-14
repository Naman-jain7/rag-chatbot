from pydantic import BaseModel, Field
from typing import Optional, List
from enum import Enum

class EvidenceEvaluation(BaseModel):
    """Schema for evaluating if retrieved evidence is sufficient."""
    evidence_sufficient: bool = Field(..., description="True if the context contains enough information to comprehensively answer the query, False otherwise.")

class ChatRequest(BaseModel):
    user_id: int
    chat_id: Optional[str] = None
    query: str

class MemoryItem(BaseModel):
    text: str = Field(description="Atomic user memory as a short sentence")
    is_new: bool = Field(description="True if new, false if duplicate")

class MemoryDecision(BaseModel):
    should_write: bool = Field(description="Whether to store any memories")
    memories: List[MemoryItem] = Field(default_factory=list)



class RouteDestination(str, Enum):
    MEMORY = "memory"
    RETRIEVE = "retrieve"
    TOOLS = "tools"

class QueryRouter(BaseModel):
    """Schema for checking if query can be answered using memory context"""

    route: RouteDestination = Field(...,description=(
            "Select the best path to answer the query: "
            "'memory' for stored user context; "
            "'retrieve' for vector DB/RAG lookup; "
            "'tools' for external APIs/functions; "
        ),
    )
    
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Confidence score of the routing decision, ranging from 0.0 (no confidence) to 1.0 (absolute certainty).")
    
    # reasoning: str = Field(..., description="Brief justification explaining why the memory context is or is not sufficient to fulfill the query.")