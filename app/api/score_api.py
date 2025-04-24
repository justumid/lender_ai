from fastapi import APIRouter, Query, HTTPException
from app.services.data_ingestion import get_applicant_by_pinfl
from app.services.static_scoring_engine import compute_static_score
from training.train_pipeline import run_full_scoring_pipeline
from app.schemas.scoring_response import ScoringResponse

router = APIRouter(tags=["Scoring"])

@router.get("/score", response_model=ScoringResponse)
def get_score(pinfl: str = Query(..., min_length=10, max_length=20)):
    """
    Run the full AI-based credit scoring pipeline for a given applicant by PINFL.

    Returns:
        ScoringResponse: Includes final_score, fraud_score, loan_limit, anomaly score,
                         confidence level, explanations, and fraud alert.
    """
    applicant = get_applicant_by_pinfl(pinfl)
    if not applicant:
        raise HTTPException(status_code=404, detail=f"No applicant found for PINFL: {pinfl}")

    # Ensure static_score is computed
    try:
        static_score_result = compute_static_score(applicant)
        applicant["static_score"] = static_score_result.get("static_score", 0.0)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to compute static score: {str(e)}")

    # Run full scoring pipeline
    result = run_full_scoring_pipeline(applicant)
    return result
