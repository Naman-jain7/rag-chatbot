from pydantic import BaseModel, Field

class EvidenceEvaluation(BaseModel):
    """Schema for evaluating if retrieved evidence is sufficient."""
    evidence_sufficient: bool = Field(..., description="True if the context contains enough information to comprehensively answer the query, False otherwise.")

class QueryRouter(BaseModel):
    """Schema for checking if query can be answered using memory context"""

    can_answer: bool = Field(..., description="True if the available memory context contains enough information to accurately answer the user's query. False otherwise.")
    
    # confidence_score: float = Field(..., ge=0.0, le=1.0, description="Confidence score of the routing decision, ranging from 0.0 (no confidence) to 1.0 (absolute certainty).")
    
    # reasoning: str = Field(..., description="Brief justification explaining why the memory context is or is not sufficient to fulfill the query.")