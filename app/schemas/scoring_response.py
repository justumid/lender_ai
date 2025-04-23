from pydantic import BaseModel, Field
from typing import Optional, Dict, Literal

class ScoringResponse(BaseModel):
    pinfl: str = Field(..., description="Personal Identification Number")
    approval_decision: Optional[Literal["APPROVED", "REJECTED"]] = Field(
        None, description="Final decision based on scoring and fraud risk"
    )
    final_score: float = Field(..., ge=0.0, le=100.0, description="Final composite risk score (0-100)")
    loan_limit: float = Field(..., ge=0.0, description="Predicted loan limit in UZS")
    fraud_score: float = Field(..., ge=0.0, le=1.0, description="Predicted fraud probability")
    vae_anomaly: float = Field(..., ge=0.0, le=1.0, description="Anomaly score from VAE model")
    confidence_level: Literal["LOW", "MEDIUM", "HIGH"] = Field(..., description="Model confidence level")
    segment_id: str = Field(..., description="Customer segmentation ID (from SimCLR clustering)")
    fraud_alert: Literal["LOW", "HIGH"] = Field(..., description="Fraud alert level")
    explanations: Dict[str, float] = Field(..., description="Attention-based feature importance map")

    class Config:
        schema_extra = {
            "example": {
                "pinfl": "30202836860013",
                "approval_decision": "APPROVED",
                "final_score": 86.72,
                "loan_limit": 45000000,
                "fraud_score": 0.18,
                "vae_anomaly": 0.07,
                "confidence_level": "HIGH",
                "segment_id": "SEG_A1",
                "fraud_alert": "LOW",
                "explanations": {
                    "salary_avg": 0.21,
                    "credit_amount": 0.18,
                    "overdue_sum": 0.12,
                    "inpsSum": 0.07
                }
            }
        }
