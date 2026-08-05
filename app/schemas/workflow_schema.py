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

class ShortTermSufficiency(BaseModel):
    """Schema for determining if short-term conversation context is sufficient to answer the query."""
    is_sufficient: bool = Field(..., description="True if the conversation history contains enough information to answer the latest query, False otherwise.")