from pydantic import BaseModel, Field

class EvidenceEvaluation(BaseModel):
    """Schema for evaluating if retrieved evidence is sufficient."""
    evidence_sufficient: bool = Field(..., description="True if the context contains enough information to comprehensively answer the query, False otherwise.")
