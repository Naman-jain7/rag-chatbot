from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum

class EvidenceEvaluation(BaseModel):
    """Schema for evaluating if retrieved evidence is sufficient."""
    evidence_sufficient: bool = Field(..., description="True if the context contains enough information to comprehensively answer the query, False otherwise.")

class ChatRequest(BaseModel):
    user_id: int
    chat_id: Optional[str] = None
    query: str