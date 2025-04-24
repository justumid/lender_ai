from pydantic import BaseModel, Field
from typing import Optional, Dict, Literal


class ScoringResponse(BaseModel):
    pinfl: str = Field(..., description="Personal Identification Number")

    approval_decision: Optional[Literal["APPROVED", "REJECTED"]] = Field(
        None, description="Final loan decision based on risk and fraud analysis"
    )
    final_score: float = Field(..., ge=0.0, le=100.0, description="Final AI-generated risk score (0-100)")
    loan_limit: float = Field(..., ge=0.0, description="Predicted loan limit in UZS")
    fraud_score: float = Field(..., ge=0.0, le=1.0, description="Fraud probability score from SimCLR")
    vae_anomaly: float = Field(..., ge=0.0, le=1.0, description="VAE-based anomaly score")
    confidence_level: Literal["LOW", "MEDIUM", "HIGH"] = Field(..., description="Model confidence rating")
    segment_id: str = Field(..., description="User cluster ID from SimCLR segmentation")
    fraud_alert: Literal["LOW", "HIGH"] = Field(..., description="Fraud alert level derived from fraud score")

    explanations: Dict[str, float] = Field(
        ..., description="Key feature contributions (e.g. attention or scoring weights)"
    )
    human_readable_summary: str = Field(..., description="Human-friendly explanation of the score result")
    salary_analysis: str = Field(..., description="Summary of income pattern and trend over time")
    credit_analysis: str = Field(..., description="Summary of credit activity and overdue behavior")
 
    class Config:
        schema_extra = {
            "example": {
                "pinfl": "30202836860013",
                "approval_decision": "APPROVED",
                "final_score": 86.72,
                "loan_limit": 45000000,
                "fraud_score": 0.12,
                "vae_anomaly": 0.07,
                "confidence_level": "HIGH",
                "segment_id": "SEG_A1",
                "fraud_alert": "LOW",
                "explanations": {
                    "vae_mse": 0.07,
                    "simclr_std": 0.12
                },
                "human_readable_summary": "Applicant is likely eligible. Risk score is 86.72, anomaly level is low, fraud signal is low. Loan limit: 45,000,000 UZS.",
                "salary_analysis": "Average salary over last 6 periods: 4,200,000 UZS. Stable income trend.",
                "credit_analysis": "3 contracts found. 1 had overdue payments."
            }
        }
