from pydantic import BaseModel
from typing import Optional, Dict

class ScoringResponse(BaseModel):
    pinfl: str
    approval_decision: Optional[str]
    final_score: float
    loan_limit: float
    fraud_score: float
    vae_anomaly: float
    confidence_level: str
    segment_id: str
    fraud_alert: str
    explanations: Dict[str, float]
