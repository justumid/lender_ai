# app/schemas/scoring_response.py

from typing import Dict, Optional
from pydantic import BaseModel


class ScoringResponse(BaseModel):
    pinfl: str
    credit_score: float
    decision: str
    loan_limit: Optional[float]
    fraud_risk: float
    segment_id: str
    volatility_profile: Dict[str, float]
    explanations: Dict[str, float]
